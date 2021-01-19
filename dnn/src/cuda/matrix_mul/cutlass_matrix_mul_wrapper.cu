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
#include "cuda.h"
#if __CUDACC_VER_MAJOR__ > 9 || \
        (__CUDACC_VER_MAJOR__ == 9 && __CUDACC_VER_MINOR__ >= 2)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wstrict-aliasing"

#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_splitk_parallel.h"
#include "cutlass/gemm/kernel/default_gemv.h"
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
void megdnn::cuda::cutlass_wrapper::cutlass_matrix_mul_float32_simt(
        const float* d_A, bool transpose_A, size_t lda, const float* d_B,
        bool transpose_B, size_t ldb, float* d_C, size_t ldc, int* workspace,
        GemmCoord const& problem_size, float alpha, float beta,
        const GemmCoord& threadblock_shape, const GemmCoord& warp_shape,
        cudaStream_t stream, int split_k_slices) {
    static constexpr int kEpilogueElementsPerAccess = 1;
    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
            float, kEpilogueElementsPerAccess, float, float>;
    typename EpilogueOp::Params epilogue{alpha, beta};
    if (split_k_slices == 1) {
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
    } else {
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
        using Gemm = cutlass::gemm::device::GemmSplitKParallel<                \
                float, LayoutA, float, LayoutB, float,                         \
                cutlass::layout::RowMajor, float, cutlass::arch::OpClassSimt,  \
                cutlass::arch::Sm50, ThreadBlockShape, WarpShape,              \
                InstructionShape, EpilogueOp>;                                 \
        return cutlass_matrix_mul_wrapper<Gemm>(                               \
                d_A, lda, d_B, ldb, d_C, ldc, workspace, problem_size,         \
                epilogue, stream, split_k_slices);                             \
    }
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
}

size_t megdnn::cuda::cutlass_wrapper::
        cutlass_matrix_mul_float32_simt_get_workspace_size(
                bool transpose_A, size_t lda, bool transpose_B, size_t ldb,
                size_t ldc, GemmCoord const& problem_size, float alpha,
                float beta, const GemmCoord& threadblock_shape,
                const GemmCoord& warp_shape, int split_k_slices) {
    static constexpr int kEpilogueElementsPerAccess = 1;
    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
            float, kEpilogueElementsPerAccess, float, float>;
    typename EpilogueOp::Params epilogue{alpha, beta};
    if (split_k_slices == 1) {
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
    } else {
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
        using Gemm = cutlass::gemm::device::GemmSplitKParallel<                \
                float, LayoutA, float, LayoutB, float,                         \
                cutlass::layout::RowMajor, float, cutlass::arch::OpClassSimt,  \
                cutlass::arch::Sm50, ThreadBlockShape, WarpShape,              \
                InstructionShape, EpilogueOp>;                                 \
        using TensorRefA = cutlass::TensorRef<typename Gemm::ElementA const,   \
                                              typename Gemm::LayoutA>;         \
        using TensorRefB = cutlass::TensorRef<typename Gemm::ElementB const,   \
                                              typename Gemm::LayoutB>;         \
        using TensorRefC = cutlass::TensorRef<typename Gemm::ElementC const,   \
                                              typename Gemm::LayoutC>;         \
        using TensorRefD = cutlass::TensorRef<typename Gemm::ElementC,         \
                                              typename Gemm::LayoutC>;         \
        TensorRefA tensor_A{nullptr, Gemm::LayoutA{static_cast<int>(lda)}};    \
        TensorRefB tensor_B{nullptr, Gemm::LayoutB{static_cast<int>(ldb)}};    \
        TensorRefC tensor_C{nullptr, Gemm::LayoutC{static_cast<int>(ldc)}};    \
        TensorRefD tensor_D{nullptr, Gemm::LayoutC{static_cast<int>(ldc)}};    \
        typename Gemm::Arguments arguments{problem_size,  tensor_A, tensor_B,  \
                                           tensor_C,      tensor_D, epilogue,  \
                                           split_k_slices};                    \
        return Gemm::get_workspace_size(arguments);                            \
    }
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
}
#undef DISPATCH

/* ============ cutlass kernel wrapper for f32 vector-matrix mul batched strided
 * ===========
 */
#define DISPATCH(cb)                                                         \
    cb(128, 4, 4);                                                           \
    cb(128, 4, 2);                                                           \
    cb(128, 4, 1);                                                           \
    cb(128, 2, 4);                                                           \
    cb(128, 1, 4);                                                           \
    cb(128, 2, 2);                                                           \
    cb(128, 1, 2);                                                           \
    cb(128, 2, 1);                                                           \
    cb(128, 1, 1);                                                           \
    cb(64, 4, 4);                                                            \
    cb(64, 4, 2);                                                            \
    cb(64, 4, 1);                                                            \
    cb(64, 2, 4);                                                            \
    cb(64, 1, 4);                                                            \
    cb(64, 2, 2);                                                            \
    cb(64, 1, 2);                                                            \
    cb(64, 2, 1);                                                            \
    cb(64, 1, 1);                                                            \
    cb(32, 4, 4);                                                            \
    cb(32, 4, 2);                                                            \
    cb(32, 4, 1);                                                            \
    cb(32, 2, 4);                                                            \
    cb(32, 1, 4);                                                            \
    cb(32, 2, 2);                                                            \
    cb(32, 1, 2);                                                            \
    cb(32, 2, 1);                                                            \
    cb(32, 1, 1);                                                            \
    megdnn_assert(false,                                                     \
                  "unsupported gemv batched strided A=%dX%dX%d, B=%dX%dX%d", \
                  problem_size.batch(), problem_size.m(), problem_size.k(),  \
                  problem_size.batch(), problem_size.k(), problem_size.n());

void megdnn::cuda::cutlass_wrapper::
        cutlass_matrix_mul_float32_simt_gemv_batched_strided(
                const float* d_A, size_t lda, size_t batch_stride_a,
                const float* d_B, size_t ldb, size_t batch_stride_b, float* d_C,
                size_t ldc, size_t batch_stride_c,
                BatchedGemmCoord const& problem_size, int threadblock_n,
                cudaStream_t stream) {
    int LDG_K, LDG_N;
    if (lda % 4 == 0)
        LDG_K = 4;
    else if (lda % 2 == 0)
        LDG_K = 2;
    else
        LDG_K = 1;

    if (ldb % 4 == 0)
        LDG_N = 4;
    else if (ldb % 2 == 0)
        LDG_N = 2;
    else
        LDG_N = 1;
#define cb(threadblock_n_, LDG_K_, LDG_N_)                                    \
    if (threadblock_n == threadblock_n_ && LDG_K == LDG_K_ &&                 \
        LDG_N == LDG_N_) {                                                    \
        using ThreadBlockShape =                                              \
                cutlass::gemm::GemmShape<1, threadblock_n_,                   \
                                         (256 * LDG_K_) /                     \
                                                 (threadblock_n_ / LDG_N_)>;  \
        using ThreadShape = cutlass::gemm::GemmShape<1, LDG_N_, LDG_K_>;      \
        using GemvKernel = cutlass::gemm::kernel::DefaultGemv<                \
                ThreadBlockShape, ThreadShape, float,                         \
                cutlass::layout::RowMajor, float, cutlass::layout::RowMajor,  \
                float, cutlass::layout::RowMajor>;                            \
        return cutlass_vector_matrix_mul_batched_strided_wrapper<GemvKernel>( \
                problem_size, d_A, lda, batch_stride_a, d_B, ldb,             \
                batch_stride_b, d_C, ldc, batch_stride_c, stream);            \
    }
    DISPATCH(cb)
#undef cb
}
#undef DISPATCH

#endif

// vim: syntax=cuda.doxygen
