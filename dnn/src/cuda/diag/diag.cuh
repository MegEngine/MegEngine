/**
 * \file dnn/src/cuda/diag/diag.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include <cuda_runtime_api.h>
#include <stdint.h>

namespace megdnn {
namespace cuda {
namespace diag {

template <typename T>
void exec_internal_to_vector(
        T* src, T* dst, ptrdiff_t start, ptrdiff_t size, ptrdiff_t stride_sum,
        ptrdiff_t dst_stride, cudaStream_t stream);

template <typename T>
void exec_internal_to_matrix(
        T* src, T* dst, ptrdiff_t start, ptrdiff_t n, ptrdiff_t k,
        ptrdiff_t dst_stride0, ptrdiff_t dst_stride1, ptrdiff_t src_stride,
        cudaStream_t stream);

}  // namespace diag
}  // namespace cuda
}  // namespace megdnn
// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
