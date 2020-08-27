/**
 * \file src/rocm/convolution/forward/inplace_matmul_impl.cpp.hip
 *
 * This file is part of MegDNN, a deep neural network run-time library
 * developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2019 Megvii Inc. All rights reserved.
 */
#include "./inplace_matmul_impl.h.hip"
#include "src/rocm/utils.h.hip"

using namespace megdnn;
using namespace rocm;

namespace {

struct BufferFetcherTexture {
    hipTextureObject_t tex;

    __device__ __forceinline__ float get(uint32_t offset) {
        return tex1Dfetch<float>(tex, offset);
    }
};

struct BufferFetcherRaw {
    const float* ptr;

    __device__ __forceinline__ float get(uint32_t offset) {
        return ptr[offset];
    }
};

struct BufferFetcherTextureHost {
    bool init_succ;
    BufferFetcherTexture val;

    BufferFetcherTextureHost(float* p, const size_t n);

    ~BufferFetcherTextureHost() { reset(); }

    void reset() {
        if (init_succ) {
            hip_check(hipDestroyTextureObject(val.tex));
            init_succ = false;
        }
    }
};

BufferFetcherTextureHost::BufferFetcherTextureHost(float* p, const size_t n) {
    init_succ = false;
    hipTextureObject_t tex_obj;

    hipResourceDesc res_desc;
    memset(&res_desc, 0, sizeof(hipResourceDesc));
    res_desc.resType = hipResourceTypeLinear;
    res_desc.res.linear.devPtr = static_cast<void*>(p);
    res_desc.res.linear.sizeInBytes = n * sizeof(float);
    res_desc.res.linear.desc =
            hipCreateChannelDesc(32, 0, 0, 0, hipChannelFormatKindFloat);
    hipTextureDesc tex_desc;
    memset(&tex_desc, 0, sizeof(hipTextureDesc));
    if (hipCreateTextureObject(&tex_obj, &res_desc, &tex_desc, NULL) ==
        hipSuccess) {
        val.tex = tex_obj;
        init_succ = true;
    } else {
        hipGetLastError();  // reset error
    }
}

template <class BufferFetcher>
struct KernelPtr {
    typedef void (*type)(BufferFetcher, BufferFetcher, float*, uint32_t,
                         uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                         uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                         uint32_t, uint32_t, uint32_t);
};

//! 1 -> 0xffffffff, 0 -> 0x00000000
__device__ __forceinline__ uint32_t bool_as_mask(uint32_t cond) {
    return (!cond) - 1u;
}

union FloatAndU32 {
    float f;
    uint32_t u;
};

//! \p mask must be either all 1 or 0 bits
template <class BufferFetcher>
__device__ __forceinline__ float visit_with_mask(BufferFetcher buf,
                                                 uint32_t offset,
                                                 uint32_t mask) {
    FloatAndU32 f;
    f.f = buf.get(offset & mask);
    f.u &= mask;
    return f.f;
}

template <uint32_t BY, uint32_t BX, bool is_xcorr, class BufferFetcher>
__global__ void conv_kernel(BufferFetcher src, BufferFetcher filter, float* dst,
                            const uint32_t INP_BS, const uint32_t OUT_BS,
                            const uint32_t IC, const uint32_t IH,
                            const uint32_t IW, const uint32_t OC,
                            const uint32_t OH, const uint32_t OW,
                            const uint32_t FH, const uint32_t FW,
                            const uint32_t SH, const uint32_t SW,
                            const uint32_t PH, const uint32_t PW) {
    const uint32_t BM = BY < BX ? BY : BX;
    const uint32_t n = blockIdx.z;
    const uint32_t tidx = threadIdx.x;
    const uint32_t tidy = threadIdx.y;
    const uint32_t posx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t posy = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t posx2 = posx << 2;
    const uint32_t posy2 = posy << 2;
    const uint32_t heightA = OC;
    const uint32_t widthA = IC * FH * FW;
    const uint32_t heightB = widthA;
    const uint32_t widthB = OH * OW;
    const uint32_t oh0 = (posx2 + 0) / OW * SH;
    const uint32_t ow0 = (posx2 + 0) % OW * SW;
    const uint32_t op0 = oh0 * IW + ow0;
    const uint32_t oh1 = (posx2 + 1) / OW * SH;
    const uint32_t ow1 = (posx2 + 1) % OW * SW;
    const uint32_t op1 = oh1 * IW + ow1;
    const uint32_t oh2 = (posx2 + 2) / OW * SH;
    const uint32_t ow2 = (posx2 + 2) % OW * SW;
    const uint32_t op2 = oh2 * IW + ow2;
    const uint32_t oh3 = (posx2 + 3) / OW * SH;
    const uint32_t ow3 = (posx2 + 3) % OW * SW;
    const uint32_t op3 = oh3 * IW + ow3;
    const uint32_t FP = FH * FW;
    __shared__ float4 localA[BY][BM];
    __shared__ float4 localB[BM][BX];
    uint32_t i = 0u;
    uint32_t offsetA = posy2 * widthA + tidx;
    uint32_t offsetB = n * INP_BS - PH * IW - PW;
    float4 sum0 = {0.0f, 0.0f, 0.0f, 0.0f}, sum1 = {0.0f, 0.0f, 0.0f, 0.0f},
           sum2 = {0.0f, 0.0f, 0.0f, 0.0f}, sum3 = {0.0f, 0.0f, 0.0f, 0.0f};
    uint32_t fh = tidy / FW % FH;
    uint32_t fw = tidy % FW;
    uint32_t ic = tidy / (FH * FW);
    uint32_t icm = tidy % (FH * FW);

    const uint32_t fhs = BM / FW % FH;
    const uint32_t fws = BM % FW;
    const uint32_t ics = BM / (FH * FW);
    const uint32_t icms = BM % (FH * FW);

    for (; i < widthA; i += BM, offsetA += BM) {
        // load localA
        if (tidx < BM) {
            localA[tidy][tidx].x = filter.get(offsetA + 0 * widthA);
            localA[tidy][tidx].y = filter.get(offsetA + 1 * widthA);
            localA[tidy][tidx].z = filter.get(offsetA + 2 * widthA);
            localA[tidy][tidx].w = filter.get(offsetA + 3 * widthA);
        }

        // load localB
        uint32_t fh2, fw2;
        if (is_xcorr) {
            fh2 = fh;
            fw2 = fw;
        } else {
            fh2 = FH - fh - 1;
            fw2 = FW - fw - 1;
        }

        if (tidy < BM) {
            uint32_t tmp = offsetB + (ic * IH + (fh2)) * IW + (fw2),
                     ok = bool_as_mask(tidy + i < heightB),
                     p0 = bool_as_mask(fh2 + oh0 >= PH && fh2 + oh0 < IH + PH &&
                                       fw2 + ow0 >= PW && fw2 + ow0 < IW + PW),
                     p1 = bool_as_mask(fh2 + oh1 >= PH && fh2 + oh1 < IH + PH &&
                                       fw2 + ow1 >= PW && fw2 + ow1 < IW + PW),
                     p2 = bool_as_mask(fh2 + oh2 >= PH && fh2 + oh2 < IH + PH &&
                                       fw2 + ow2 >= PW && fw2 + ow2 < IW + PW),
                     p3 = bool_as_mask(fh2 + oh3 >= PH && fh2 + oh3 < IH + PH &&
                                       fw2 + ow3 >= PW && fw2 + ow3 < IW + PW);
            localB[tidy][tidx].x = visit_with_mask(src, tmp + op0, ok & p0);
            localB[tidy][tidx].y = visit_with_mask(src, tmp + op1, ok & p1);
            localB[tidy][tidx].z = visit_with_mask(src, tmp + op2, ok & p2);
            localB[tidy][tidx].w = visit_with_mask(src, tmp + op3, ok & p3);
        }

        __syncthreads();

        for (uint32_t j = 0u; j < BM; ++j) {
            float4 tmpA = localA[tidy][j];
            float4 tmpB = localB[j][tidx];
            sum0.x += tmpA.x * tmpB.x;
            sum0.y += tmpA.x * tmpB.y;
            sum0.z += tmpA.x * tmpB.z;
            sum0.w += tmpA.x * tmpB.w;
            sum1.x += tmpA.y * tmpB.x;
            sum1.y += tmpA.y * tmpB.y;
            sum1.z += tmpA.y * tmpB.z;
            sum1.w += tmpA.y * tmpB.w;
            sum2.x += tmpA.z * tmpB.x;
            sum2.y += tmpA.z * tmpB.y;
            sum2.z += tmpA.z * tmpB.z;
            sum2.w += tmpA.z * tmpB.w;
            sum3.x += tmpA.w * tmpB.x;
            sum3.y += tmpA.w * tmpB.y;
            sum3.z += tmpA.w * tmpB.z;
            sum3.w += tmpA.w * tmpB.w;
        }

        fw += fws;
        fh += fhs;
        fh += (fw >= FW);
        fh -= (fh >= FH) * FH;
        fw -= (fw >= FW) * FW;

        ic += ics;
        icm += icms;
        ic += (icm >= FP);
        icm -= (icm >= FP) * FP;
        __syncthreads();
    }
    const uint32_t dst_idx = n * OUT_BS + posy2 * widthB + posx2;
    bool y0 = (posy2 + 0 < heightA);
    bool y1 = (posy2 + 1 < heightA);
    bool y2 = (posy2 + 2 < heightA);
    bool y3 = (posy2 + 3 < heightA);
    bool x0 = (posx2 + 0 < widthB);
    bool x1 = (posx2 + 1 < widthB);
    bool x2 = (posx2 + 2 < widthB);
    bool x3 = (posx2 + 3 < widthB);
    if (y0) {
        if (x0)
            dst[dst_idx + 0 * widthB + 0] = sum0.x;
        if (x1)
            dst[dst_idx + 0 * widthB + 1] = sum0.y;
        if (x2)
            dst[dst_idx + 0 * widthB + 2] = sum0.z;
        if (x3)
            dst[dst_idx + 0 * widthB + 3] = sum0.w;
    }
    if (y1) {
        if (x0)
            dst[dst_idx + 1 * widthB + 0] = sum1.x;
        if (x1)
            dst[dst_idx + 1 * widthB + 1] = sum1.y;
        if (x2)
            dst[dst_idx + 1 * widthB + 2] = sum1.z;
        if (x3)
            dst[dst_idx + 1 * widthB + 3] = sum1.w;
    }
    if (y2) {
        if (x0)
            dst[dst_idx + 2 * widthB + 0] = sum2.x;
        if (x1)
            dst[dst_idx + 2 * widthB + 1] = sum2.y;
        if (x2)
            dst[dst_idx + 2 * widthB + 2] = sum2.z;
        if (x3)
            dst[dst_idx + 2 * widthB + 3] = sum2.w;
    }
    if (y3) {
        if (x0)
            dst[dst_idx + 3 * widthB + 0] = sum3.x;
        if (x1)
            dst[dst_idx + 3 * widthB + 1] = sum3.y;
        if (x2)
            dst[dst_idx + 3 * widthB + 2] = sum3.z;
        if (x3)
            dst[dst_idx + 3 * widthB + 3] = sum3.w;
    }
}

}  // anonymous namespace

void convolution::exec_inplace_matmul_fwd(
        const float* src, const float* filter, float* dst, size_t N,
        size_t INP_BS, size_t OUT_BS, size_t IC, size_t IH, size_t IW,
        size_t OC, size_t OH, size_t OW, size_t FH, size_t FW, size_t PH,
        size_t PW, size_t SH, size_t SW, bool is_xcorr, hipStream_t stream) {
    BufferFetcherTextureHost src_tex(const_cast<float*>(src), N * INP_BS),
            filter_tex(const_cast<float*>(filter), OC * IC * FH * FW);

    BufferFetcherRaw src_buf, filter_buf;
    src_buf.ptr = src;
    filter_buf.ptr = filter;
    if (!src_tex.init_succ || !filter_tex.init_succ) {
        src_tex.reset();
        filter_tex.reset();
    }
    int m = OC;
    int n = OH * OW;
    int BY = 1;
    int BX = 1;
    if (m <= 64) {
        while (BY < 16 && (BY << 2) < m)
            BY <<= 1;
        BX = 256 / BY;
    } else if (n <= 64) {
        while (BX < 16 && (BX << 2) < n)
            BX <<= 1;
        BY = 256 / BX;
    } else {
        BX = BY = 16;
    }
    dim3 blocks((OH * OW + BX * 4 - 1) / (BX * 4), (OC + BY * 4 - 1) / (BY * 4),
                N);
    dim3 threads(BX, BY);
#define DISPATCH_BX_BY(BX, BY)                                                \
    do {                                                                      \
        if (src_tex.init_succ) {                                              \
            KernelPtr<BufferFetcherTexture>::type kptr;                       \
            if (is_xcorr) {                                                   \
                kptr = conv_kernel<BY, BX, true, BufferFetcherTexture>;       \
            } else {                                                          \
                kptr = conv_kernel<BY, BX, false, BufferFetcherTexture>;      \
            }                                                                 \
            kptr<<<blocks, threads, 0, stream>>>(                             \
                    src_tex.val, filter_tex.val, dst, INP_BS, OUT_BS, IC, IH, \
                    IW, OC, OH, OW, FH, FW, SH, SW, PH, PW);                  \
        } else {                                                              \
            KernelPtr<BufferFetcherRaw>::type kptr;                           \
            if (is_xcorr) {                                                   \
                kptr = conv_kernel<BY, BX, true, BufferFetcherRaw>;           \
            } else {                                                          \
                kptr = conv_kernel<BY, BX, false, BufferFetcherRaw>;          \
            }                                                                 \
            kptr<<<blocks, threads, 0, stream>>>(                             \
                    src_buf, filter_buf, dst, INP_BS, OUT_BS, IC, IH, IW, OC, \
                    OH, OW, FH, FW, SH, SW, PH, PW);                          \
        }                                                                     \
    } while (0)
#define DISPATCH_BX(BX)               \
    do {                              \
        DISPATCH_BX_BY(BX, 256 / BX); \
    } while (0)
#define DISPATCH()                                \
    do {                                          \
        switch (BX) {                             \
            case 1:                               \
                DISPATCH_BX(1);                   \
                break;                            \
            case 2:                               \
                DISPATCH_BX(2);                   \
                break;                            \
            case 4:                               \
                DISPATCH_BX(4);                   \
                break;                            \
            case 8:                               \
                DISPATCH_BX(8);                   \
                break;                            \
            case 16:                              \
                DISPATCH_BX(16);                  \
                break;                            \
            case 32:                              \
                DISPATCH_BX(32);                  \
                break;                            \
            case 64:                              \
                DISPATCH_BX(64);                  \
                break;                            \
            case 128:                             \
                DISPATCH_BX(128);                 \
                break;                            \
            case 256:                             \
                DISPATCH_BX(256);                 \
                break;                            \
            default:                              \
                report_error("no usable kernel"); \
        }                                         \
    } while (0)
    DISPATCH();
#undef DISPATCH
#undef DISPATCH_BX
#undef DISPATCH_BX_BY
    after_kernel_launch();
}

// vim: syntax=cpp.doxygen
