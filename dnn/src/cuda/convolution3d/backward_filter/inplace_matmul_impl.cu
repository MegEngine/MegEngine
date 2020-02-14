/**
 * \file dnn/src/cuda/convolution3d/backward_filter/inplace_matmul_impl.cu
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
            uint32_t, uint32_t, uint32_t,
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

__device__ __forceinline__ uint32_t with_dilation(
        const uint32_t origin, const uint32_t D) {
    return origin * D;
}

template <uint32_t BY, uint32_t BX, bool is_xcorr, class BufferFetcher>
__global__ void conv_kernel(BufferFetcher diff, BufferFetcher src,
        float *grad,
        const uint32_t N, const uint32_t INP_BS, const uint32_t OUT_BS,
        const uint32_t IC, const uint32_t ID, const uint32_t IH, const uint32_t IW,
        const uint32_t OC, const uint32_t OD, const uint32_t OH, const uint32_t OW,
        const uint32_t FD, const uint32_t FH, const uint32_t FW,
        const uint32_t SD, const uint32_t SH, const uint32_t SW,
        const uint32_t PD, const uint32_t PH, const uint32_t PW,
        const uint32_t DD, const uint32_t DH, const uint32_t DW)
{
    const uint32_t BM = BY < BX ? BY : BX;

    uint32_t n = blockIdx.z;

    const uint32_t tidx = threadIdx.x;
    const uint32_t tidy = threadIdx.y;
    const uint32_t posx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t posy = blockIdx.y * blockDim.y + threadIdx.y;
    const uint32_t posx2 = posx<<2;
    const uint32_t posy2 = posy<<2;
    
    const uint32_t heightA = OC;
    const uint32_t widthA = OD*OH*OW;
    const uint32_t heightB = widthA;
    const uint32_t widthB = IC*FD*FH*FW;
    
    uint32_t ic0 = (posx2+0) / FW / FH / FD;
    uint32_t fd0 = (posx2+0) / FW / FH % FD;
    uint32_t fh0 = (posx2+0) / FW % FH;
    uint32_t fw0 = (posx2+0) % FW;
  
    uint32_t ic1 = (posx2+1) / FW / FH / FD; 
    uint32_t fd1 = (posx2+1) / FW / FH % FD;
    uint32_t fh1 = (posx2+1) / FW % FH;
    uint32_t fw1 = (posx2+1) % FW;

    uint32_t ic2 = (posx2+2) / FW / FH / FD;
    uint32_t fd2 = (posx2+2) / FW / FH % FD;
    uint32_t fh2 = (posx2+2) / FW % FH;
    uint32_t fw2 = (posx2+2) % FW;

    uint32_t ic3 = (posx2+3) / FW / FH / FD;
    uint32_t fd3 = (posx2+3) / FW / FH % FD;
    uint32_t fh3 = (posx2+3) / FW % FH;
    uint32_t fw3 = (posx2+3) % FW;

    if (!is_xcorr) {
        fd0 = FD - fd0 - 1;
        fd1 = FD - fd1 - 1;
        fd2 = FD - fd2 - 1;
        fd3 = FD - fd3 - 1;
        fh0 = FH - fh0 - 1;
        fh1 = FH - fh1 - 1;
        fh2 = FH - fh2 - 1;
        fh3 = FH - fh3 - 1;
        fw0 = FW - fw0 - 1;
        fw1 = FW - fw1 - 1;
        fw2 = FW - fw2 - 1;
        fw3 = FW - fw3 - 1;
    }

    const uint32_t fd0d = with_dilation(fd0, DD);
    const uint32_t fd1d = with_dilation(fd1, DD);
    const uint32_t fd2d = with_dilation(fd2, DD);
    const uint32_t fd3d = with_dilation(fd3, DD);
    
    const uint32_t fh0d = with_dilation(fh0, DH);
    const uint32_t fh1d = with_dilation(fh1, DH);
    const uint32_t fh2d = with_dilation(fh2, DH);
    const uint32_t fh3d = with_dilation(fh3, DH);

    const uint32_t fw0d = with_dilation(fw0, DW);
    const uint32_t fw1d = with_dilation(fw1, DW);
    const uint32_t fw2d = with_dilation(fw2, DW);
    const uint32_t fw3d = with_dilation(fw3, DW);

    const uint32_t fp0 = ic0 * ID*IH*IW + fd0d * IH*IW + fh0d * IW + fw0d;  
    const uint32_t fp1 = ic1 * ID*IH*IW + fd1d * IH*IW + fh1d * IW + fw1d; 
    const uint32_t fp2 = ic2 * ID*IH*IW + fd2d * IH*IW + fh2d * IW + fw2d;  
    const uint32_t fp3 = ic3 * ID*IH*IW + fd3d * IH*IW + fh3d * IW + fw3d;  

    const uint32_t OP = OH*OW;

    __shared__ float4 localA[BY][BM];
    __shared__ float4 localB[BM][BX];
    uint32_t i = 0u;

    uint32_t offsetA = n * OUT_BS + posy2 * widthA + tidx;
    uint32_t offsetB = n * INP_BS - PD*IH*IW - PH*IW - PW; 

    float4 sum0 = {0.0f, 0.0f, 0.0f, 0.0f},
           sum1 = {0.0f, 0.0f, 0.0f, 0.0f},
           sum2 = {0.0f, 0.0f, 0.0f, 0.0f},
           sum3 = {0.0f, 0.0f, 0.0f, 0.0f};
    
    uint32_t od = tidy / (OW*OH);
    uint32_t oh = tidy / (OW) % OH;
    uint32_t ow = tidy % OW;
    uint32_t odm = tidy % (OW*OH);

    const uint32_t ods = BM / (OW*OH);
    const uint32_t ohs = BM / (OW) % OH;
    const uint32_t ows = BM % OW;
    const uint32_t odms = BM % (OW*OH);

    for (; i < widthA; i += BM, offsetA += BM) {
        // load localA
        if (tidx < BM) {
            localA[tidy][tidx].x = diff.get(offsetA + 0*widthA);
            localA[tidy][tidx].y = diff.get(offsetA + 1*widthA);
            localA[tidy][tidx].z = diff.get(offsetA + 2*widthA);
            localA[tidy][tidx].w = diff.get(offsetA + 3*widthA);
        }
        if (tidy < BM) {
            uint32_t tmp = offsetB + od*SD*IH*IW + oh*SH*IW + ow*SW,
                     ok = bool_as_mask(tidy+i < heightB),
                     p0 = bool_as_mask(
                             fd0d+od*SD >= PD && fd0d+od*SD < ID+PD &&
                             fh0d+oh*SH >= PH && fh0d+oh*SH < IH+PH &&
                             fw0d+ow*SW >= PW && fw0d+ow*SW < IW+PW),
                     p1 = bool_as_mask(
                             fd1d+od*SD >= PD && fd1d+od*SD < ID+PD &&
                             fh1d+oh*SH >= PH && fh1d+oh*SH < IH+PH &&
                             fw1d+ow*SW >= PW && fw1d+ow*SW < IW+PW),
                     p2 = bool_as_mask(
                             fd2d+od*SD >= PD && fd2d+od*SD < ID+PD &&
                             fh2d+oh*SH >= PH && fh2d+oh*SH < IH+PH &&
                             fw2d+ow*SW >= PW && fw2d+ow*SW < IW+PW),
                     p3 = bool_as_mask(
                             fd3d+od*SD >= PD && fd3d+od*SD < ID+PD &&
                             fh3d+oh*SH >= PH && fh3d+oh*SH < IH+PH &&
                             fw3d+ow*SW >= PW && fw3d+ow*SW < IW+PW);

            localB[tidy][tidx].x = visit_with_mask(src, tmp+fp0, ok & p0);
            localB[tidy][tidx].y = visit_with_mask(src, tmp+fp1, ok & p1);
            localB[tidy][tidx].z = visit_with_mask(src, tmp+fp2, ok & p2);
            localB[tidy][tidx].w = visit_with_mask(src, tmp+fp3, ok & p3); 
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
        oh += ohs;
        ow += ows;
        oh += (ow >= OW);
        ow -= (ow >= OW) * OW;
        oh -= (oh >= OH) * OH;

        od += ods;
        odm += odms;
        od += (odm >= OP);
        odm -= (odm >= OP) * OP;
        __syncthreads();
    }
    
    // widthB == IC*FD*FH*FW, heightA == OC
    const uint32_t grad_idx = posy2 * widthB + posx2;
    bool y0 = (posy2+0 < heightA);
    bool y1 = (posy2+1 < heightA);
    bool y2 = (posy2+2 < heightA);
    bool y3 = (posy2+3 < heightA);
    bool x0 = (posx2+0 < widthB);
    bool x1 = (posx2+1 < widthB);
    bool x2 = (posx2+2 < widthB);
    bool x3 = (posx2+3 < widthB);
    if (y0) {
        if (x0) atomicAdd(&grad[grad_idx + 0*widthB + 0], sum0.x);
        if (x1) atomicAdd(&grad[grad_idx + 0*widthB + 1], sum0.y);
        if (x2) atomicAdd(&grad[grad_idx + 0*widthB + 2], sum0.z);
        if (x3) atomicAdd(&grad[grad_idx + 0*widthB + 3], sum0.w);
    }
    if (y1) {
        if (x0) atomicAdd(&grad[grad_idx + 1*widthB + 0], sum1.x);
        if (x1) atomicAdd(&grad[grad_idx + 1*widthB + 1], sum1.y);
        if (x2) atomicAdd(&grad[grad_idx + 1*widthB + 2], sum1.z);
        if (x3) atomicAdd(&grad[grad_idx + 1*widthB + 3], sum1.w);
    }
    if (y2) {
        if (x0) atomicAdd(&grad[grad_idx + 2*widthB + 0], sum2.x);
        if (x1) atomicAdd(&grad[grad_idx + 2*widthB + 1], sum2.y);
        if (x2) atomicAdd(&grad[grad_idx + 2*widthB + 2], sum2.z);
        if (x3) atomicAdd(&grad[grad_idx + 2*widthB + 3], sum2.w);
    }   
    if (y3) {
        if (x0) atomicAdd(&grad[grad_idx + 3*widthB + 0], sum3.x);
        if (x1) atomicAdd(&grad[grad_idx + 3*widthB + 1], sum3.y);
        if (x2) atomicAdd(&grad[grad_idx + 3*widthB + 2], sum3.z);
        if (x3) atomicAdd(&grad[grad_idx + 3*widthB + 3], sum3.w);
    }
}

} // anonymous namespace

void convolution3d::exec_inplace_matmul_bwd_filter(
        const float *diff, const float *src, float *grad,
        size_t N, size_t INP_BS, size_t OUT_BS,
        size_t IC, size_t ID, size_t IH, size_t IW,
        size_t OC, size_t OD, size_t OH, size_t OW,
        size_t FD, size_t FH, size_t FW,
        size_t PD, size_t PH, size_t PW,
        size_t SD, size_t SH, size_t SW,
        size_t DD, size_t DH, size_t DW,
        bool is_xcorr,
        cudaStream_t stream) {
    BufferFetcherTextureHost diff_tex(const_cast<float *>(diff), OC*OD*OH*OW*N),
                             src_tex(const_cast<float *>(src), N * INP_BS);
    BufferFetcherRaw diff_buf, src_buf;
    src_buf.ptr = src;
    diff_buf.ptr = diff;
    if (!src_tex.init_succ || !diff_tex.init_succ) {
        src_tex.reset();
        diff_tex.reset();
    }
    int m = OC;
    int n = IC*FD*FH*FW;
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
    cudaMemset(grad, 0, OC * IC * FD * FH * FW * sizeof(float));
    dim3 blocks(DIVUP(n, 4*BX), DIVUP(m, 4*BY), N);
    dim3 threads(BX, BY);
#define DISPATCH_BX_BY(BX, BY) do { \
    if (diff_tex.init_succ) { \
        KernelPtr<BufferFetcherTexture>::type kptr; \
        if (is_xcorr) { \
            kptr = conv_kernel<BY, BX, true, BufferFetcherTexture>; \
        } else  { \
            kptr = conv_kernel<BY, BX, false, BufferFetcherTexture>; \
        } \
        kptr<<<blocks, threads, 0, stream>>>( \
                diff_tex.val, src_tex.val, grad, \
                N, INP_BS, OUT_BS, \
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
                diff_buf, src_buf, grad, \
                N, INP_BS, OUT_BS, \
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

