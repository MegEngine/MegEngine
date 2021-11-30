/**
 * \file dnn/src/cuda/diag/diag.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "megdnn/dtype.h"
#include "src/cuda/diag/diag.cuh"
#include "src/cuda/utils.cuh"

namespace {

template <typename T>
__global__ void kernel_to_vector(
        T* src, T* dst, ptrdiff_t start, ptrdiff_t size, ptrdiff_t stride_sum,
        ptrdiff_t dst_stride) {
    ptrdiff_t i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < size) {
        dst[dst_stride * i] = src[start + stride_sum * i];
    }
}

template <typename T>
__global__ void kernel_to_matrix(
        T* src, T* dst, ptrdiff_t offset, ptrdiff_t n, ptrdiff_t k,
        ptrdiff_t dst_stride0, ptrdiff_t dst_stride1, ptrdiff_t src_stride) {
    ptrdiff_t i = threadIdx.x + blockIdx.x * blockDim.x;
    ptrdiff_t x = i % n;
    ptrdiff_t y = i / n;
    ptrdiff_t p = dst_stride0 * y + dst_stride1 * x;
    if (i < n * n) {
        if (y + k == x)
            dst[p] = src[src_stride * (y - offset)];
        else
            dst[p] = 0;
    }
}

}  // anonymous namespace

namespace megdnn {
namespace cuda {
namespace diag {

template <typename T>
void exec_internal_to_vector(
        T* src, T* dst, ptrdiff_t start, ptrdiff_t size, ptrdiff_t stride_sum,
        ptrdiff_t dst_stride, cudaStream_t stream) {
    kernel_to_vector<T><<<DIVUP(size, NR_THREADS), NR_THREADS, 0, stream>>>(
            src, dst, start, size, stride_sum, dst_stride);
    after_kernel_launch();
}

template <typename T>
void exec_internal_to_matrix(
        T* src, T* dst, ptrdiff_t offset, ptrdiff_t n, ptrdiff_t k,
        ptrdiff_t dst_stride0, ptrdiff_t dst_stride1, ptrdiff_t src_stride,
        cudaStream_t stream) {
    kernel_to_matrix<T><<<DIVUP(n * n, NR_THREADS), NR_THREADS, 0, stream>>>(
            src, dst, offset, n, k, dst_stride0, dst_stride1, src_stride);
    after_kernel_launch();
}

#define INST(T)                               \
    template void exec_internal_to_vector<T>( \
            T*, T*, ptrdiff_t, ptrdiff_t, ptrdiff_t, ptrdiff_t, cudaStream_t);
#define cb(DType) INST(typename DTypeTrait<DType>::ctype)
MEGDNN_FOREACH_COMPUTING_DTYPE(cb)
cb(::megdnn::dtype::Bool)
#undef INST
#undef cb

#define INST(T)                                                                       \
    template void exec_internal_to_matrix<T>(                                         \
            T*, T*, ptrdiff_t, ptrdiff_t, ptrdiff_t, ptrdiff_t, ptrdiff_t, ptrdiff_t, \
            cudaStream_t);
#define cb(DType) INST(typename DTypeTrait<DType>::ctype)
        MEGDNN_FOREACH_COMPUTING_DTYPE(cb) cb(::megdnn::dtype::Bool)

}  // namespace diag
}  // namespace cuda
}  // namespace megdnn
// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
