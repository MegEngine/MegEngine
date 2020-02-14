/**
 * \file dnn/src/cuda/mask_conv/mask_conv.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include <cstdio>
#include "./mask_conv.cuh"
#include "megdnn/dtype.h"
#include "src/cuda/utils.cuh"

namespace {
template <typename ctype>
__global__ void set_zero_by_mask_kernel(float* dst, const ctype* mask, size_t N,
                                        size_t mask_size) {
    int dst_offset = blockIdx.x * blockDim.x + threadIdx.x;
    int mask_idx = blockIdx.y * blockDim.y + threadIdx.y;
    if (dst_offset >= N || mask_idx >= mask_size) {
        return;
    }
    if (mask[mask_idx] == 0) {
        dst[dst_offset * mask_size + mask_idx] = 0;
    }
}

template <typename ctype>
__global__ void mask_propagate_kernel(const ctype* src, ctype* dst, size_t IH,
                                      size_t IW, size_t OH, size_t OW,
                                      size_t FH, size_t FW, size_t SH,
                                      size_t SW, size_t PH, size_t PW,
                                      size_t DH, size_t DW) {
    int dst_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (dst_idx >= OH * OW) {
        return;
    }
    int oh = dst_idx / OW;
    int ow = dst_idx - (OW * oh);
    dst[dst_idx] = 0;
    for (int fh = 0; fh < FH; ++fh) {
        for (int fw = 0; fw < FW; ++fw) {
            int ih = oh * SH + fh * DH - PH;
            int iw = ow * SW + fw * DW - PW;
            if (ih < 0 || ih >= IH || iw < 0 || iw >= IW ||
                src[ih * IW + iw] == 0) {
                continue;
            }
            dst[dst_idx] = 1;
            return;
        }
    }
}

}  // namespace

namespace megdnn {
namespace cuda {
namespace mask_conv {

template <typename ctype>
void set_zero_by_mask_proxy(float* dst, const ctype* mask, size_t N, size_t OC,
                            size_t OH, size_t OW, cudaStream_t stream) {
    dim3 threads(NR_THREADS_X, NR_THREADS_Y);
    dim3 blocks(DIVUP(N * OC, threads.x), DIVUP(OH * OW, threads.y));
    set_zero_by_mask_kernel<ctype>
            <<<blocks, threads, 0, stream>>>(dst, mask, N * OC, OH * OW);
}

template <typename ctype>
void mask_propagate_exec_proxy(const ctype* src, ctype* dst, size_t IH,
                               size_t IW, size_t OH, size_t OW, size_t FH,
                               size_t FW, size_t SH, size_t SW, size_t PH,
                               size_t PW, size_t DH, size_t DW,
                               cudaStream_t stream) {
    mask_propagate_kernel<ctype>
            <<<DIVUP(OH * OW, NR_THREADS), NR_THREADS, 0, stream>>>(
                    src, dst, IH, IW, OH, OW, FH, FW, SH, SW, PH, PW, DH, DW);
}

#define INST(ctype)                                                           \
    template void mask_propagate_exec_proxy<ctype>(                           \
            const ctype* src, ctype* dst, size_t IH, size_t IW, size_t OH,    \
            size_t OW, size_t FH, size_t FW, size_t SH, size_t SW, size_t PH, \
            size_t PW, size_t DH, size_t DW, cudaStream_t stream);            \
                                                                              \
    template void set_zero_by_mask_proxy<ctype>(                              \
            float* dst, const ctype* mask, size_t N, size_t OC, size_t OH,    \
            size_t OW, cudaStream_t stream);

#define cb(DType) INST(DTypeTrait<DType>::ctype)
MEGDNN_FOREACH_COMPUTING_DTYPE_INT(cb)
#undef cb

#undef INST

}  // namespace mask_conv
}  // namespace cuda
}  // namespace megdnn
