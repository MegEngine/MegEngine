/**
 * \file dnn/src/cuda/matrix_mul/cutlass_matrix_mul_wrapper.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
// ignore warning of cutlass
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wstrict-aliasing"

#include "cuda.h"
#if __CUDACC_VER_MAJOR__ > 9 || \
        (__CUDACC_VER_MAJOR__ == 9 && __CUDACC_VER_MINOR__ >= 2)
#include "cutlass/gemm/device/gemm.h"
#endif
#include "src/common/opr_param_defs_enumv.cuh"
#include "src/cuda/matrix_mul/cutlass_matrix_mul_wrapper.cuh"
#pragma GCC diagnostic pop

using namespace megdnn;
using namespace cuda;
using namespace cutlass_wrapper;

/* ================= cutlass kernel wrapper for f32 matrix mul ================
 */
#define DISPATCH(cb)                                                         \
    cb(64, 256, 8, 32, 64, 8);                                               \
    cb(256, 64, 8, 64, 32, 8);                                               \
    cb(32, 256, 8, 16, 64, 8);                                               \
    cb(256, 32, 8, 64, 16, 8);                                               \
    cb(128, 128, 8, 32, 64, 8);                                              \
    cb(128, 64, 8, 64, 32, 8);                                               \
    cb(64, 128, 8, 32, 64, 8);                                               \
    cb(128, 32, 8, 64, 32, 8);                                               \
    cb(32, 128, 8, 32, 64, 8);                                               \
    cb(64, 64, 8, 32, 64, 8);                                                \
    cb(32, 64, 8, 32, 64, 8);                                                \
    cb(64, 32, 8, 64, 32, 8);                                                \
    cb(32, 32, 8, 32, 32, 8);                                                \
    cb(8, 32, 8, 8, 32, 8);                                                  \
    cb(16, 32, 8, 16, 32, 8);                                                \
    cb(16, 64, 8, 16, 64, 8);                                                \
    cb(16, 128, 8, 16, 64, 8);                                               \
    megdnn_assert(false,                                                     \
                  "unsupported threadblock shape (%dx%dx%d) and warp shape " \
                  "(%dx%dx%d)",                                              \
                  threadblock_shape.m(), threadblock_shape.n(),              \
                  threadblock_shape.k(), warp_shape.m(), warp_shape.n(),     \
                  warp_shape.k());

#if __CUDACC_VER_MAJOR__ < 9 || \
        (__CUDACC_VER_MAJOR__ == 9 && __CUDACC_VER_MINOR__ <= 2)
void megdnn::cuda::cutlass_wrapper::cutlass_matrix_mul_float32_simt(
        const float* /* d_A */, bool /* transpose_A */, size_t /* lda */,
        const float* /* d_B */, bool /* transpose_B */, size_t /* ldb */,
        float* /* d_C */, size_t /* ldc */, int* /* workspace */,
        GemmCoord const& /* problem_size */, float /* alpha */,
        float /* beta */, const GemmCoord& /* threadblock_shape */,
        const GemmCoord& /* warp_shape */, cudaStream_t /* stream */) {}
#else
void megdnn::cuda::cutlass_wrapper::cutlass_matrix_mul_float32_simt(
        const float* d_A, bool transpose_A, size_t lda, const float* d_B,
        bool transpose_B, size_t ldb, float* d_C, size_t ldc, int* workspace,
        GemmCoord const& problem_size, float alpha, float beta,
        const GemmCoord& threadblock_shape, const GemmCoord& warp_shape,
        cudaStream_t stream) {
#define cb(threadblock_m_, threadblock_n_, threadblock_k_, warp_m_, warp_n_,   \
           warp_k_)                                                            \
    if (threadblock_shape.m() == threadblock_m_ &&                             \
        threadblock_shape.n() == threadblock_n_ &&                             \
        threadblock_shape.k() == threadblock_k_ &&                             \
        warp_shape.m() == warp_m_ && warp_shape.n() == warp_n_ &&              \
        warp_shape.k() == warp_k_) {                                           \
        using ThreadBlockShape =                                               \
                cutlass::gemm::GemmShape<threadblock_m_, threadblock_n_,       \
                                         threadblock_k_>;                      \
        using WarpShape = cutlass::gemm::GemmShape<warp_m_, warp_n_, warp_k_>; \
        using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;            \
        using Gemm = cutlass::gemm::device::Gemm<                              \
                float, LayoutA, float, LayoutB, float,                         \
                cutlass::layout::RowMajor, float, cutlass::arch::OpClassSimt,  \
                cutlass::arch::Sm50, ThreadBlockShape, WarpShape,              \
                InstructionShape, EpilogueOp,                                  \
                cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,  \
                2>;                                                            \
        return cutlass_matrix_mul_wrapper<Gemm>(d_A, lda, d_B, ldb, d_C, ldc,  \
                                                workspace, problem_size,       \
                                                epilogue, stream);             \
    }
    static constexpr int kEpilogueElementsPerAccess = 1;
    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
            float, kEpilogueElementsPerAccess, float, float>;
    typename EpilogueOp::Params epilogue{alpha, beta};
    if (!transpose_A && !transpose_B) {
        using LayoutA = cutlass::layout::RowMajor;
        using LayoutB = cutlass::layout::RowMajor;
        DISPATCH(cb)
    } else if (!transpose_A && transpose_B) {
        using LayoutA = cutlass::layout::RowMajor;
        using LayoutB = cutlass::layout::ColumnMajor;
        DISPATCH(cb)
    } else if (transpose_A && !transpose_B) {
        using LayoutA = cutlass::layout::ColumnMajor;
        using LayoutB = cutlass::layout::RowMajor;
        DISPATCH(cb)
    } else {
        megdnn_assert(transpose_A && transpose_B);
        using LayoutA = cutlass::layout::ColumnMajor;
        using LayoutB = cutlass::layout::ColumnMajor;
        DISPATCH(cb)
    }
#undef cb
}
#endif

#if __CUDACC_VER_MAJOR__ < 9 || \
        (__CUDACC_VER_MAJOR__ == 9 && __CUDACC_VER_MINOR__ <= 2)
size_t megdnn::cuda::cutlass_wrapper::
        cutlass_matrix_mul_float32_simt_get_workspace_size(
                bool /* transpose_A */, size_t /* lda */,
                bool /* transpose_B */, size_t /* ldb */, size_t /* ldc */,
                GemmCoord const& /* problem_size */, float /* alpha */,
                float /* beta */, const GemmCoord& /* threadblock_shape */,
                const GemmCoord& /* warp_shape */) {
    return 0;
}
#else
size_t megdnn::cuda::cutlass_wrapper::
        cutlass_matrix_mul_float32_simt_get_workspace_size(
                bool transpose_A, size_t lda, bool transpose_B, size_t ldb,
                size_t ldc, GemmCoord const& problem_size, float alpha,
                float beta, const GemmCoord& threadblock_shape,
                const GemmCoord& warp_shape) {
#define cb(threadblock_m_, threadblock_n_, threadblock_k_, warp_m_, warp_n_,   \
           warp_k_)                                                            \
    if (threadblock_shape.m() == threadblock_m_ &&                             \
        threadblock_shape.n() == threadblock_n_ &&                             \
        threadblock_shape.k() == threadblock_k_ &&                             \
        warp_shape.m() == warp_m_ && warp_shape.n() == warp_n_ &&              \
        warp_shape.k() == warp_k_) {                                           \
        using ThreadBlockShape =                                               \
                cutlass::gemm::GemmShape<threadblock_m_, threadblock_n_,       \
                                         threadblock_k_>;                      \
        using WarpShape = cutlass::gemm::GemmShape<warp_m_, warp_n_, warp_k_>; \
        using InstructionShape = cutlass::gemm::GemmShape<1, 1, 1>;            \
        using Gemm = cutlass::gemm::device::Gemm<                              \
                float, LayoutA, float, LayoutB, float,                         \
                cutlass::layout::RowMajor, float, cutlass::arch::OpClassSimt,  \
                cutlass::arch::Sm50, ThreadBlockShape, WarpShape,              \
                InstructionShape, EpilogueOp,                                  \
                cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,  \
                2>;                                                            \
        typename Gemm::TensorRefA tensor_A{                                    \
                nullptr, Gemm::LayoutA{static_cast<int>(lda)}};                \
        typename Gemm::TensorRefB tensor_B{                                    \
                nullptr, Gemm::LayoutB{static_cast<int>(ldb)}};                \
        typename Gemm::TensorRefC tensor_C{                                    \
                nullptr, Gemm::LayoutC{static_cast<int>(ldc)}};                \
        typename Gemm::TensorRefD tensor_D{                                    \
                nullptr, Gemm::LayoutC{static_cast<int>(ldc)}};                \
        typename Gemm::Arguments arguments{problem_size,  tensor_A, tensor_B,  \
                                           tensor_C,      tensor_D, epilogue,  \
                                           split_k_slices};                    \
        return Gemm::get_workspace_size(arguments);                            \
    }
    static constexpr int kEpilogueElementsPerAccess = 1;
    static constexpr int split_k_slices = 1;
    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
            float, kEpilogueElementsPerAccess, float, float>;
    typename EpilogueOp::Params epilogue{alpha, beta};
    if (!transpose_A && !transpose_B) {
        using LayoutA = cutlass::layout::RowMajor;
        using LayoutB = cutlass::layout::RowMajor;
        DISPATCH(cb)
    } else if (!transpose_A && transpose_B) {
        using LayoutA = cutlass::layout::RowMajor;
        using LayoutB = cutlass::layout::ColumnMajor;
        DISPATCH(cb)
    } else if (transpose_A && !transpose_B) {
        using LayoutA = cutlass::layout::ColumnMajor;
        using LayoutB = cutlass::layout::RowMajor;
        DISPATCH(cb)
    } else {
        megdnn_assert(transpose_A && transpose_B);
        using LayoutA = cutlass::layout::ColumnMajor;
        using LayoutB = cutlass::layout::ColumnMajor;
        DISPATCH(cb)
    }
#undef cb
}
#endif

#undef DISPATCH
// vim: syntax=cuda.doxygen
