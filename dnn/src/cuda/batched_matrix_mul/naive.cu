/**
 * \file dnn/src/cuda/batched_matrix_mul/naive.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include <cuda.h>
#include "src/cuda/matrix_mul/naive.cuh"
#include "src/cuda/utils.cuh"

namespace {

template <typename AType, typename BType, typename CType, typename CompType>
__global__ void do_exec(
        const AType* A, const BType* B, CType* C, size_t Batch, size_t M, size_t N,
        size_t K, size_t LDA, size_t LDB, size_t LDC, bool transA, bool transB) {
    for (int bid = blockIdx.x; bid < Batch; bid += gridDim.x) {
        const AType* A_r = A + (transA ? bid * K * LDA : bid * M * LDA);
        const BType* B_r = B + (transB ? bid * N * LDB : bid * K * LDB);
        CType* C_r = C + bid * M * LDC;

        for (size_t m = 0; m < M; ++m) {
            size_t n = threadIdx.x;
            for (; n < N; n += blockDim.x) {
                CompType res = static_cast<CompType>(0);
                for (size_t k = 0; k < K; ++k) {
                    AType av = transA ? A_r[k * LDA + m] : A_r[m * LDA + k];
                    BType bv = transB ? B_r[n * LDB + k] : B_r[k * LDB + n];
                    res += av * bv;
                }
                C_r[m * LDC + n] = res;
            }
        }
    }
}
}  // namespace

namespace megdnn {
namespace cuda {

template <typename AType, typename BType, typename CType, typename CompType>
void exec_bgemm_naive(
        const AType* A, const BType* B, CType* C, size_t Batch, size_t M, size_t N,
        size_t K, size_t LDA, size_t LDB, size_t LDC, bool transA, bool transB,
        cudaStream_t stream) {
    do_exec<AType, BType, CType, CompType><<<Batch, 128, 0, stream>>>(
            A, B, C, Batch, M, N, K, LDA, LDB, LDC, transA, transB);
}

#define INST(in_ct, out_ct, comp_ct)                                             \
    template void exec_bgemm_naive<                                              \
            typename in_ct, typename in_ct, typename out_ct, typename comp_ct>(  \
            const in_ct* A, const in_ct* B, out_ct* C, size_t Batch, size_t M,   \
            size_t N, size_t K, size_t LDA, size_t LDB, size_t LDC, bool transA, \
            bool transB, cudaStream_t stream);

INST(megdnn::dt_float32, megdnn::dt_float32, megdnn::dt_float32)
INST(megdnn::dt_float16, megdnn::dt_float16, megdnn::dt_float16)
INST(megdnn::dt_float16, megdnn::dt_float16, megdnn::dt_float32)

#undef cb
#undef INST

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
