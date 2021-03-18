/**
 * \file dnn/src/cuda/matrix_mul/cutlass_matrix_mul_wrapper_batched_gemv_strided.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
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
