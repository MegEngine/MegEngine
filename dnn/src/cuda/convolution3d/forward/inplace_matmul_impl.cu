/**
 * \file dnn/src/cuda/convolution3d/forward/inplace_matmul_impl.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "./inplace_matmul_impl.cuh"
#include "src/cuda/utils.cuh"
#include <iostream>
#include <stdio.h>
using namespace megdnn;
using namespace cuda;

namespace {

struct BufferFetcherTexture {
    cudaTextureObject_t tex;

    __device__ __forceinline__ float get(uint32_t offset) {
        return tex1Dfetch<float>(tex, offset);
    }
};

struct BufferFetcherRaw {
    const float *ptr;

    __device__ __forceinline__ float get(uint32_t offset) {
        return ptr[offset];
    }
};

struct BufferFetcherTextureHost {
    bool init_succ;
    BufferFetcherTexture val;

    BufferFetcherTextureHost(float *p, const size_t n);

    ~BufferFetcherTextureHost() {
        reset();
    }

    void reset() {
        if (init_succ) {
            cuda_check(cudaDestroyTextureObject(val.tex));
            init_succ = false;
        }
    }
};

BufferFetcherTextureHost::BufferFetcherTextureHost(float *p, const size_t n) {
    init_succ = false;
    cudaTextureObject_t tex_obj;

    cudaResourceDesc res_desc;
    memset(&res_desc, 0, sizeof(cudaResourceDesc));
    res_desc.resType = cudaResourceTypeLinear;
    res_desc.res.linear.devPtr = static_cast<void *>(p);
    res_desc.res.linear.sizeInBytes = n*sizeof(float);
    res_desc.res.linear.desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat); 
    cudaTextureDesc tex_desc; 
    memset(&tex_desc, 0, sizeof(cudaTextureDesc));
    if (cudaCreateTextureObject(&tex_obj, &res_desc, &tex_desc, NULL) == cudaSuccess) {
        val.tex = tex_obj;
        init_succ = true;
    } else {
        cudaGetLastError(); // reset error
    }
}

template<class BufferFetcher>
struct KernelPtr {
    typedef void(*type)(BufferFetcher, BufferFetcher, float*,
            uint32_t, uint32_t,
            uint32_t, uint32_t, uint32_t, uint32_t,
            uint32_t, uint32_t, uint32_t, uint32_t,
            uint32_t, uint32_t, uint32_t,
            uint32_t, uint32_t, uint32_t,
            uint32_t, uint32_t, uint32_t,
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
template<class BufferFetcher>
__device__ __forceinline__ float visit_with_mask(
        BufferFetcher buf, uint32_t offset, uint32_t mask) {
    FloatAndU32 f;
    f.f = buf.get(offset & mask);
    f.u &= mask;
    return f.f;
}

template <uint32_t BY, uint32_t BX, bool is_xcorr, class BufferFetcher>
__global__ void conv_kernel(BufferFetcher src, BufferFetcher filter,
        float *dst,
        const uint32_t INP_BS, const uint32_t OUT_BS,
        const uint32_t IC, const uint32_t ID, const uint32_t IH, const uint32_t IW,
        const uint32_t OC, const uint32_t OD, const uint32_t OH, const uint32_t OW,
        const uint32_t FD, const uint32_t FH, const uint32_t FW,
        const uint32_t SD, const uint32_t SH, const uint32_t SW,
        const uint32_t PD, const uint32_t PH, const uint32_t PW,
        const uint32_t DD, const uint32_t DH, const uint32_t DW)
{
    const uint32_t BM = BY < BX ? BY : BX;
    // BY*BX == 256
    // (OC) * (IC*FD*FH*FW) * (OD*OH*OW)
    const uint32_t n = blockIdx.z;
    const uint32_t tidx = threadIdx.x;
    const uint32_t tidy = threadIdx.y;
    const uint32_t posx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t posy = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t posx2 = posx<<2;
    const uint32_t posy2 = posy<<2;
    const uint32_t heightA = OC;
    const uint32_t widthA = IC*FD*FH*FW;
    const uint32_t heightB = widthA;
    const uint32_t widthB = OD*OH*OW;
    const uint32_t od0 = (posx2+0) / OW / OH * SD;
    const uint32_t oh0 = (posx2+0) / OW % OH * SH;
    const uint32_t ow0 = (posx2+0) % OW      * SW;
    const uint32_t op0 = od0 * IH * IW + oh0 * IW + ow0;

    const uint32_t od1 = (posx2+1) / OW / OH * SD;
    const uint32_t oh1 = (posx2+1) / OW % OH * SH;
    const uint32_t ow1 = (posx2+1) % OW      * SW;
    const uint32_t op1 = od1 * IH * IW + oh1 * IW + ow1;

    const uint32_t od2 = (posx2+2) / OW / OH * SD;
    const uint32_t oh2 = (posx2+2) / OW % OH * SH;
    const uint32_t ow2 = (posx2+2) % OW      * SW;
    const uint32_t op2 = od2 * IH * IW + oh2 * IW + ow2;

    const uint32_t od3 = (posx2+3) / OW / OH * SD;
    const uint32_t oh3 = (posx2+3) / OW % OH * SH;
    const uint32_t ow3 = (posx2+3) % OW      * SW;
    const uint32_t op3 = od3 * IH * IW + oh3 * IW + ow3;
    const uint32_t FP = FD*FH*FW;
    // OC % (BLOCK*4) == 0
    // IC*FD*FH*FW % BLOCK == 0
    // OD*OH*OW % (BLOCK*4) == 0
    __shared__ float4 localA[BY][BM];
    __shared__ float4 localB[BM][BX];
    uint32_t i = 0u;
    uint32_t offsetA = posy2 * widthA + tidx;
    uint32_t offsetB = n*INP_BS - PD*IH*IW - PH*IW - PW;
    float4 sum0 = {0.0f, 0.0f, 0.0f, 0.0f},
           sum1 = {0.0f, 0.0f, 0.0f, 0.0f},
           sum2 = {0.0f, 0.0f, 0.0f, 0.0f},
           sum3 = {0.0f, 0.0f, 0.0f, 0.0f};
    uint32_t fd = tidy / FW / FH % FD;
    uint32_t fh = tidy / FW % FH;
    uint32_t fw = tidy % FW;
    uint32_t ic = tidy / (FD*FH*FW);
    uint32_t icm = tidy % (FD*FH*FW);

    const uint32_t fds = BM / FW / FH % FD;
    const uint32_t fhs = BM / FW % FH;
    const uint32_t fws = BM % FW;
    const uint32_t ics = BM / (FD*FH*FW);
    const uint32_t icms = BM % (FD*FH*FW);

    for (; i < widthA; i += BM, offsetA += BM) {
        // load localA
        if (tidx < BM) {
            localA[tidy][tidx].x = filter.get(offsetA + 0*widthA);
            localA[tidy][tidx].y = filter.get(offsetA + 1*widthA);
            localA[tidy][tidx].z = filter.get(offsetA + 2*widthA);
            localA[tidy][tidx].w = filter.get(offsetA + 3*widthA);
        }

        // load localB
        uint32_t fd2, fh2, fw2;
        if (is_xcorr) {
            fd2 = fd;
            fh2 = fh;
            fw2 = fw;
        } else {
            fd2 = FD-fd-1;
            fh2 = FH-fh-1;
            fw2 = FW-fw-1;
        }

        if (tidy < BM) {
            uint32_t fd2d = fd2 * DD, 
                     fh2d = fh2 * DH,
                     fw2d = fw2 * DW;
            uint32_t tmp = offsetB+ic*ID*IH*IW+fd2d*IH*IW+fh2d*IW+fw2d,
                     ok = bool_as_mask(tidy+i < heightB),
                     p0 = bool_as_mask(
                             fd2d+od0 >= PD && fd2d+od0 < ID+PD &&
                             fh2d+oh0 >= PH && fh2d+oh0 < IH+PH &&
                             fw2d+ow0 >= PW && fw2d+ow0 < IW+PW),
                     p1 = bool_as_mask(
                             fd2d+od1 >= PD && fd2d+od1 < ID+PD &&
                             fh2d+oh1 >= PH && fh2d+oh1 < IH+PH &&
                             fw2d+ow1 >= PW && fw2d+ow1 < IW+PW),
                     p2 = bool_as_mask(
                             fd2d+od2 >= PD && fd2d+od2 < ID+PD &&
                             fh2d+oh2 >= PH && fh2d+oh2 < IH+PH &&
                             fw2d+ow2 >= PW && fw2d+ow2 < IW+PW),
                     p3 = bool_as_mask(
                             fd2d+od3 >= PD && fd2d+od3 < ID+PD &&
                             fh2d+oh3 >= PH && fh2d+oh3 < IH+PH &&
                             fw2d+ow3 >= PW && fw2d+ow3 < IW+PW);
            localB[tidy][tidx].x = visit_with_mask(src, tmp+op0, ok & p0);
            localB[tidy][tidx].y = visit_with_mask(src, tmp+op1, ok & p1);
            localB[tidy][tidx].z = visit_with_mask(src, tmp+op2, ok & p2);
            localB[tidy][tidx].w = visit_with_mask(src, tmp+op3, ok & p3); 
        }
        __syncthreads(); // die without this sync()..
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
        fd += fds;
        fw += fws;
        fh += fhs;

        fh += (fw >= FW);
        fw -= (fw >= FW) * FW;
        fd += (fh >= FH);
        fh -= (fh >= FH) * FH;
        fd -= (fd >= FD) * FD;

        ic += ics;
        icm += icms;
        ic += (icm >= FP);
        icm -= (icm >= FP) * FP;

        __syncthreads();
    }
    const uint32_t dst_idx = n*OUT_BS + posy2*widthB + posx2;
    bool y0 = (posy2+0 < heightA);
    bool y1 = (posy2+1 < heightA);
    bool y2 = (posy2+2 < heightA);
    bool y3 = (posy2+3 < heightA);
    bool x0 = (posx2+0 < widthB);
    bool x1 = (posx2+1 < widthB);
    bool x2 = (posx2+2 < widthB);
    bool x3 = (posx2+3 < widthB);
   if (y0) {
        if (x0) dst[dst_idx + 0*widthB + 0] = sum0.x;
        if (x1) dst[dst_idx + 0*widthB + 1] = sum0.y;
        if (x2) dst[dst_idx + 0*widthB + 2] = sum0.z;
        if (x3) dst[dst_idx + 0*widthB + 3] = sum0.w;
    }
    if (y1) {
        if (x0) dst[dst_idx + 1*widthB + 0] = sum1.x;
        if (x1) dst[dst_idx + 1*widthB + 1] = sum1.y;
        if (x2) dst[dst_idx + 1*widthB + 2] = sum1.z;
        if (x3) dst[dst_idx + 1*widthB + 3] = sum1.w;
    }
    if (y2) {
        if (x0) dst[dst_idx + 2*widthB + 0] = sum2.x;
        if (x1) dst[dst_idx + 2*widthB + 1] = sum2.y;
        if (x2) dst[dst_idx + 2*widthB + 2] = sum2.z;
        if (x3) dst[dst_idx + 2*widthB + 3] = sum2.w;
    }
    if (y3) {
        if (x0) dst[dst_idx + 3*widthB + 0] = sum3.x;
        if (x1) dst[dst_idx + 3*widthB + 1] = sum3.y;
        if (x2) dst[dst_idx + 3*widthB + 2] = sum3.z;
        if (x3) dst[dst_idx + 3*widthB + 3] = sum3.w;
    }
}

} // anonymous namespace

void convolution3d::exec_inplace_matmul_fwd(
        const float *src, const float *filter, float *dst,
        size_t N, size_t INP_BS, size_t OUT_BS,
        size_t IC, size_t ID, size_t IH, size_t IW,
        size_t OC, size_t OD, size_t OH, size_t OW,
        size_t FD, size_t FH, size_t FW,
        size_t PD, size_t PH, size_t PW,
        size_t SD, size_t SH, size_t SW,
        size_t DD, size_t DH, size_t DW,
        bool is_xcorr,
        cudaStream_t stream)
{
    BufferFetcherTextureHost src_tex(const_cast<float *>(src), N * INP_BS),
                             filter_tex(const_cast<float *>(filter), OC*IC*FD*FH*FW);
    BufferFetcherRaw src_buf, filter_buf;
    src_buf.ptr = src;
    filter_buf.ptr = filter;
    if (!src_tex.init_succ || !filter_tex.init_succ) {
        src_tex.reset();
        filter_tex.reset();
    }
    int m = OC;
    int n = OD*OH*OW;
    int BY = 1;
    int BX = 1;
    if (m <= 64) {
        while (BY < 16 && (BY<<2) < m) BY <<= 1;
        BX = 256 / BY;
    } else if (n <= 64) {
        while (BX < 16 && (BX<<2) < n) BX <<= 1;
        BY = 256 / BX;
    } else {
        BX = BY = 16;
    }
    dim3 blocks(DIVUP(OD*OH*OW, 4*BX), DIVUP(OC, 4*BY), N);
    dim3 threads(BX, BY);
#define DISPATCH_BX_BY(BX, BY) do { \
    if (src_tex.init_succ) { \
        KernelPtr<BufferFetcherTexture>::type kptr; \
        if (is_xcorr) { \
            kptr = conv_kernel<BY, BX, true, BufferFetcherTexture>; \
        } else  { \
            kptr = conv_kernel<BY, BX, false, BufferFetcherTexture>; \
        } \
        kptr<<<blocks, threads, 0, stream>>>( \
                src_tex.val, filter_tex.val, dst, \
                INP_BS, OUT_BS, \
                IC, ID, IH, IW, \
                OC, OD, OH, OW, \
                FD, FH, FW, \
                SD, SH, SW, \
                PD, PH, PW, \
                DD, DH, DW); \
    } else { \
        KernelPtr<BufferFetcherRaw>::type kptr; \
        if (is_xcorr) { \
            kptr = conv_kernel<BY, BX, true, BufferFetcherRaw>; \
        } else  { \
            kptr = conv_kernel<BY, BX, false, BufferFetcherRaw>; \
        } \
        kptr<<<blocks, threads, 0, stream>>>( \
                src_buf, filter_buf, dst, \
                INP_BS, OUT_BS, \
                IC, ID, IH, IW, \
                OC, OD, OH, OW, \
                FD, FH, FW, \
                SD, SH, SW, \
                PD, PH, PW, \
                DD, DH, DW); \
    } \
} while (0)
#define DISPATCH_BX(BX) do { \
    DISPATCH_BX_BY(BX, 256/BX); \
} while (0)
#define DISPATCH() do { \
    switch (BX) { \
        case 1: DISPATCH_BX(1); break; \
        case 2: DISPATCH_BX(2); break; \
        case 4: DISPATCH_BX(4); break; \
        case 8: DISPATCH_BX(8); break; \
        case 16: DISPATCH_BX(16); break; \
        case 32: DISPATCH_BX(32); break; \
        case 64: DISPATCH_BX(64); break; \
        case 128: DISPATCH_BX(128); break; \
        case 256: DISPATCH_BX(256); break; \
        default: \
            report_error("no usable kernel"); \
    } \
} while (0)
    DISPATCH();
#undef DISPATCH
#undef DISPATCH_BX
#undef DISPATCH_BX_BY
    after_kernel_launch();
}

// vim: syntax=cpp.doxygen

