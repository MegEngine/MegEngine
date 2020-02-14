/**
 * \file dnn/src/cuda/flip/flip.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "./flip.cuh"

#include "megdnn/dtype.h"
#include "src/cuda/utils.cuh"

namespace megdnn {
namespace cuda {

static const int BX = 16;
static const int BY = 16;

namespace {

#define rep(i, n) for (size_t i = 0; i < (n); ++i)

template <typename T, bool vertical, bool horizontal, size_t IC>
__global__ void flip_kern(const T *src, T *dst, size_t N, size_t H, size_t W,
                          size_t stride1, size_t stride2, size_t stride3) {
    __shared__ T cache[BX][BY][IC];
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    if (ow < W && oh < H) {

        int iw = horizontal ? W - ow - 1 : ow;
        int ih = vertical ? H - oh - 1 : oh;
#pragma unroll
        rep(c, IC) {
            cache[threadIdx.y][threadIdx.x][c] =
                src[blockIdx.z * stride1 + ih * stride2 + iw * stride3 + c];
        }
        __syncthreads();
#pragma unroll
        rep(c, IC) {
            dst[blockIdx.z * stride1 + oh * stride2 + ow * stride3 + c] =
                cache[threadIdx.y][threadIdx.x][c];
        }
    }
}

#undef rep
} // anonymous namespace

namespace flip {

template <typename T, bool vertical, bool horizontal>
void flip(const T *src, T *dst, size_t N, size_t H, size_t W, size_t IC,
          size_t stride1, size_t stride2, size_t stride3, cudaStream_t stream) {
    dim3 threads(BX, BY);
    dim3 blocks(DIVUP(W, BX), DIVUP(H, BY), N);
    megdnn_assert(IC == 1 || IC == 3);
    if (IC == 1)
        flip_kern<T, vertical, horizontal, 1><<<blocks, threads, 0, stream>>>(
            src, dst, N, H, W, stride1, stride2, stride3);
    else
        flip_kern<T, vertical, horizontal, 3><<<blocks, threads, 0, stream>>>(
            src, dst, N, H, W, stride1, stride2, stride3);
    after_kernel_launch();
}

#define INST(T, vertical, horizontal)                                    \
    template void flip<T, vertical, horizontal>(                         \
        const T *src, T *dst, size_t N, size_t H, size_t W, size_t IC, \
        size_t stride1, size_t stride2, size_t stride3, cudaStream_t);

#define cb(DType)                                        \
    INST(typename DTypeTrait<DType>::ctype, true, true)  \
    INST(typename DTypeTrait<DType>::ctype, true, false) \
    INST(typename DTypeTrait<DType>::ctype, false, true) \
    INST(typename DTypeTrait<DType>::ctype, false, false)

MEGDNN_FOREACH_COMPUTING_DTYPE(cb)

#undef cb
#undef INST

}  // namespace flip
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
