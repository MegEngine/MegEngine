/**
 * \file dnn/src/cuda/transpose/transpose.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/cuda/transpose/transpose.cuh"

#include "megdnn/dtype.h"
#include "src/cuda/utils.cuh"

namespace {

// launch (16, 16) threads
template <typename T>
__global__ void kernel(const T *A, T *B, uint32_t m, uint32_t n,
        uint32_t LDA, uint32_t LDB)
{
    __shared__ T cache[16][16];
    {
        uint32_t y = threadIdx.y + blockIdx.y * 16;
        uint32_t x = threadIdx.x + blockIdx.x * 16;
        if (y < m && x < n) cache[threadIdx.y][threadIdx.x] = A[y*LDA + x];
    }
    __syncthreads();
    {
        // variable is idx wrt B rather than A (so x/y is swapped)
        uint32_t x = threadIdx.x + blockIdx.y * 16;
        uint32_t y = threadIdx.y + blockIdx.x * 16;
        if (y < n && x < m) B[y*LDB + x] = cache[threadIdx.x][threadIdx.y];
    }
}

} // anonymous namespace

namespace megdnn {
namespace cuda {

template <typename T>
void transpose(const T *A, T *B, size_t m, size_t n,
        size_t LDA, size_t LDB, cudaStream_t stream)
{
    dim3 threads(16, 16);
    dim3 blocks(DIVUP(n, 16), DIVUP(m, 16));
    kernel<T><<<blocks, threads, 0, stream>>>(A, B, m, n, LDA, LDB);
    after_kernel_launch();
}

#define INST(T) \
template void transpose<T>(const T*, T*, size_t, size_t, size_t, size_t, \
        cudaStream_t);
#define cb(DType) INST(typename DTypeTrait<DType>::ctype)

MEGDNN_FOREACH_COMPUTING_DTYPE(cb)

#undef cb
#undef INST

} // namespace cuda
} // namespace megdnn

// vim: syntax=cpp.doxygen
