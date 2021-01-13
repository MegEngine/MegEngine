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

namespace megdnn {
namespace cuda {
namespace cutlass_wrapper {

using GemmCoord = cutlass::gemm::GemmCoord;

template <typename Gemm>
void cutlass_matrix_mul_wrapper(
        const typename Gemm::ElementA* d_A, size_t lda,
        const typename Gemm::ElementB* d_B, size_t ldb,
        typename Gemm::ElementC* d_C, size_t ldc, int* workspace,
        GemmCoord const& problem_size,
        typename Gemm::EpilogueOutputOp::Params const& epilogue,
        cudaStream_t stream, int split_k_slices = 1);

void cutlass_matrix_mul_float32_simt(
        const float* d_A, bool transpose_A, size_t lda, const float* d_B,
        bool transpose_B, size_t ldb, float* d_C, size_t ldc, int* workspace,
        GemmCoord const& problem_size, float alpha, float beta,
        const GemmCoord& threadblock_shape, const GemmCoord& warp_shape,
        cudaStream_t stream, int split_k_slices = 1);

size_t cutlass_matrix_mul_float32_simt_get_workspace_size(
        bool transpose_A, size_t lda, bool transpose_B, size_t ldb, size_t ldc,
        GemmCoord const& problem_size, float alpha, float beta,
        const GemmCoord& threadblock_shape, const GemmCoord& warp_shape, int split_k_slices = 1);

}  // namespace cutlass_wrapper
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cuda.doxygen
