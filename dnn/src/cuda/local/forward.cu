/**
 * \file dnn/src/cuda/local/forward.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/cuda/local/local.cuh"

#include "src/cuda/utils.cuh"
#include "src/cuda/local/cuda-convnet2/nvmatrix.cuh"
#include "src/cuda/local/cuda-convnet2/cudaconv2.cuh"

namespace megdnn {
namespace cuda {
namespace local {

constexpr size_t Ns = 4, ICs = 4;

size_t forward_proxy_default_share_mem_in_bytes(size_t IH, size_t IW) {
    return Ns * ICs * sizeof(float) * IH * IW;
}

// blockIdx.y is OC*OH*OW/1024
// blockIdx.x is N/4
// threadIdx.x is [0, 1024)
template <uint32_t Ns, uint32_t ICs, bool is_xcorr>
__global__ void forward_kernel(const float * __restrict__ src,
        const float * __restrict__ filter,
        float * __restrict__ dst,
        uint32_t N,
        uint32_t IC, uint32_t IH, uint32_t IW,
        uint32_t OC, uint32_t OH, uint32_t OW,
        uint32_t FH, uint32_t FW,
        uint32_t INs, size_t ONs,
        uint32_t PH, uint32_t PW,
        uint32_t SH, uint32_t SW)
{
    // Ns*ICs*sizeof(float)*IH*IW
    extern __shared__ float shared_mem[];
    float *src_cache = shared_mem;
    uint32_t tid = threadIdx.x;
    uint32_t tstride = blockDim.x;
    uint32_t oid = tid + blockIdx.y * tstride;
    src += blockIdx.x*Ns * INs;
    dst += blockIdx.x*Ns * ONs;
    uint32_t op = oid / OC;
    uint32_t oc = oid % OC;
    uint32_t oh = op / OW;
    uint32_t ow = op % OW;
    float dst_reg[Ns];
    for (uint32_t no = 0; no < Ns; ++no) dst_reg[no] = 0.0f;
    uint32_t Nb = min(N-blockIdx.x*Ns, Ns);
    for (uint32_t ic = 0; ic < IC; ic += ICs) {
        // read ICs-channel src
        // (Ns, ICs, IHs, IWs)
        uint32_t ICb = min(ICs, IC-ic);
        for (uint32_t i = tid; i < Nb*ICs*IH*IW; i += tstride) {
            uint32_t ip = i % (IH*IW);
            uint32_t ico = i / (IH*IW) % ICs;
            uint32_t no = i / (IH*IW) / ICs;
            src_cache[i] =
                (ico < ICb) * src[no*INs + min(IC-1, (ic+ico))*IH*IW + ip];
        }
        __syncthreads();
        if (oid < OC*OH*OW)
        for (uint32_t fh = 0; fh < FH; ++fh)
        {
        uint32_t ih;
        if (is_xcorr) ih = oh*SH + fh - PH; else ih = oh*SH + (FH-fh-1) - PH;
        if (ih < IH)
        for (uint32_t fw = 0; fw < FW; ++fw)
        {
            uint32_t iw;
            if (is_xcorr) iw = ow*SW + fw - PW; else iw = ow*SW + (FW-fw-1) - PW;
            if (iw < IW)
            for (uint32_t ico = 0; ico < ICb; ++ico) {
                uint32_t fid = op*IC*FH*FW*OC + (ic+ico)*FH*FW*OC +
                    fh*FW*OC + fw*OC + oc;
                float fval = filter[fid];
                float src_reg[Ns];
#pragma unroll
                for (uint32_t no = 0; no < Ns; ++no) {
                    src_reg[no] = src_cache[no*ICs*IH*IW + ico*IH*IW + ih*IW + iw];
                }
#pragma unroll
                for (uint32_t no = 0; no < Ns; ++no) {
                    dst_reg[no] += src_reg[no]*fval;
                }
            }
        }
        }
        __syncthreads();
    }
    if (oid < OC*OH*OW) {
        for (uint32_t no = 0; no < Nb; ++no) {
            dst[no*ONs + oc*OH*OW + op] = dst_reg[no];
        }
    }
}

void forward_proxy_default(const float *src, const float *filter, float *dst,
        size_t N,
        size_t IC, size_t IH, size_t IW,
        size_t OC, size_t OH, size_t OW,
        size_t FH, size_t FW,
        size_t INs, size_t ONs,
        size_t PH, size_t PW,
        size_t SH, size_t SW,
        bool is_xcorr,
        cudaStream_t stream)
{
    size_t threads = 256;
    dim3 blocks = dim3(DIVUP(N, Ns), DIVUP(OC*OH*OW, threads));
    if (is_xcorr) {
        forward_kernel<Ns, ICs, true><<<blocks, threads,
            Ns*ICs*sizeof(float)*IH*IW, stream>>>(src, filter, dst,
                    N,
                    IC, IH, IW,
                    OC, OH, OW,
                    FH, FW,
                    INs, ONs,
                    PH, PW,
                    SH, SW);
    } else {
        forward_kernel<Ns, ICs, false><<<blocks, threads,
            Ns*ICs*sizeof(float)*IH*IW, stream>>>(src, filter, dst,
                    N,
                    IC, IH, IW,
                    OC, OH, OW,
                    FH, FW,
                    INs, ONs,
                    PH, PW,
                    SH, SW);
    }
    after_kernel_launch();
}

bool can_forward_proxy_convnet(size_t N,
        size_t IC, size_t /* IH */, size_t /* IW */,
        size_t /*OC*/, size_t /* OH */, size_t /* OW */,
        size_t FH, size_t FW,
        size_t /* INs */, size_t /* ONs */,
        size_t PH, size_t PW,
        size_t SH, size_t SW)
{
    bool flag = true;
    // check pad
    flag &= (PH == PW);
    // check stride
    flag &= (SH == SW);
    // megdnn_assert(numGroups > 1 || (numImgColors > 0 && (numImgColors <= 3 || numImgColors % 4 == 0)));
    flag &= (IC <= 3 || IC % 4 == 0);
    // megdnn_assert(numFilters % (16 * numGroups) == 0);
    //flag &= (OC % 16 == 0);
    // megdnn_assert(filterSize * filterSize == filterPixels);
    flag &= (FH == FW);
    flag &= (SH <= FH);
    flag &= (N % 32 == 0);
    return flag;
}

size_t get_workspace_in_floats_forward_proxy_convnet(size_t N,
        size_t IC, size_t IH, size_t IW,
        size_t OC, size_t OH, size_t OW,
        size_t /* FH */, size_t /* FW */,
        size_t /* INs */, size_t /* ONs */,
        size_t /* PH */, size_t /* PW */,
        size_t /* SH */, size_t /* SW */)
{
    return N*IC*IH*IW + N*OC*OH*OW;
}

void forward_proxy_convnet(const float *src, const float *filter, float *dst,
        float *workspace,
        size_t N,
        size_t IC, size_t IH, size_t IW,
        size_t OC, size_t OH, size_t OW,
        size_t FH, size_t FW,
        size_t INs, size_t ONs, // IN stride and ON stride
        size_t PH, size_t /* PW */,
        size_t SH, size_t /* SW */,
        cublasHandle_t cublas_handle,
        cudaStream_t stream,
        float *one, float *zero)

{
    MemorySegment msrc_n(const_cast<float *>(src)),
                  mdst_n(dst),
                  mfilter(const_cast<float *>(filter)),
                  msrc_t(workspace+0),
                  mdst_t(workspace+N*IC*IH*IW);
    NVMatrix nvimage_n(&msrc_n, N, IC*IH*IW, INs);
    NVMatrix nvtarget_n(&mdst_n, N, OC*OH*OW, ONs);
    NVMatrix nvimage_t(&msrc_t, IC*IH*IW, N);
    NVMatrix nvfilter(&mfilter, OH*OW*IC*FH*FW, OC);
    NVMatrix nvtarget_t(&mdst_t, OC*OH*OW, N);

    nvimage_n.transpose(nvimage_t, cublas_handle, one, zero);

    localFilterActs(stream, nvimage_t, nvfilter, nvtarget_t,
            IH, OH, OW, -static_cast<int>(PH), SH, IC, 1);
    after_kernel_launch();

    nvtarget_t.transpose(nvtarget_n, cublas_handle, one, zero);
}

} // namespace local
} // namespace cuda
} // namespace megdnn

// vim: syntax=cpp.doxygen
