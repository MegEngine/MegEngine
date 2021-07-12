/**
 * \file dnn/src/cuda/matrix_mul/cutlass_matrix_mul_wrapper.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#pragma once
#include "cutlass/gemm/gemm.h"
#include "src/cuda/utils.cuh"

#if CUDA_VERSION >= 9020
namespace megdnn {
namespace cuda {
namespace cutlass_wrapper {

using GemmCoord = cutlass::gemm::GemmCoord;
using BatchedGemmCoord = cutlass::gemm::BatchedGemmCoord;

template <typename GemvKernel>
void cutlass_vector_matrix_mul_batched_strided_wrapper(
        BatchedGemmCoord const& problem_size,
        const typename GemvKernel::ElementA* d_A, size_t lda,
        size_t batch_stride_a, const typename GemvKernel::ElementB* d_B,
        size_t ldb, size_t batch_stride_b, typename GemvKernel::ElementCD* d_C,
        size_t ldc, size_t batch_stride_c, cudaStream_t stream);

void cutlass_matrix_mul_float32_simt_gemv_batched_strided(
        const float* d_A, size_t lda, size_t batch_stride_a, const float* d_B,
        size_t ldb, size_t batch_stride_b, float* d_C, size_t ldc,
        size_t batch_stride_c, BatchedGemmCoord const& problem_size,
        int threadblock_n, cudaStream_t stream);

}  // namespace cutlass_wrapper
}  // namespace cuda
}  // namespace megdnn
#endif

// vim: syntax=cuda.doxygen
