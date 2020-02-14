/**
 * \file dnn/src/cuda/relayout/kern_transpose.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/cuda/relayout/kern_transpose.cuh"

namespace {
template <typename T>
__global__ void kernel(const T* __restrict__ A, T* __restrict__ B,
                       uint32_t batch, uint32_t m, uint32_t n, uint32_t LDA,
                       uint32_t LDB, uint32_t stride_A, uint32_t stride_B) {
    const uint32_t batch_idx = blockIdx.z;
    A += batch_idx * stride_A;
    B += batch_idx * stride_B;

    // avoid shared memory bank conflict
    __shared__ T cache[16][16 + 1];
    {
        uint32_t y = threadIdx.y + blockIdx.y * 16;
        uint32_t x = threadIdx.x + blockIdx.x * 16;
        if (y < m && x < n)
            cache[threadIdx.y][threadIdx.x] = A[y * LDA + x];
    }
    __syncthreads();
    {
        // variable is idx wrt B rather than A (so x/y is swapped)
        uint32_t x = threadIdx.x + blockIdx.y * 16;
        uint32_t y = threadIdx.y + blockIdx.x * 16;
        if (y < n && x < m)
            B[y * LDB + x] = cache[threadIdx.x][threadIdx.y];
    }
}
}  // namespace

namespace megdnn {
namespace cuda {
template <typename T>
void copy_by_transpose(const T* A, T* B, size_t batch, size_t m, size_t n,
                       size_t lda, size_t ldb, size_t stride_A, size_t stride_B,
                       cudaStream_t stream) {
    dim3 threads(16, 16);
    dim3 blocks(DIVUP(n, 16), DIVUP(m, 16), batch);
    kernel<T><<<blocks, threads, 0, stream>>>(A, B, batch, m, n, lda, ldb,
                                              stride_A, stride_B);
    after_kernel_launch();
}

#define INST(T)                                                              \
    template void copy_by_transpose<T>(const T*, T*, size_t, size_t, size_t, \
                                       size_t, size_t, size_t,               \
                                       size_t, cudaStream_t);
#define cb(DType) INST(typename DTypeTrait<DType>::ctype)

MEGDNN_FOREACH_COMPUTING_DTYPE(cb)

#undef cb
#undef INST

}  // namespace cuda
}  // namespace megdnn


// vim: ft=cpp syntax=cuda.doxygen foldmethod=marker foldmarker=f{{{,f}}}

