/**
 * \file dnn/src/cuda/matrix_mul/naive.cu
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
__global__ void do_exec(const AType* A, const BType* B, CType* C, size_t M,
                        size_t N, size_t K, size_t LDA, size_t LDB, size_t LDC,
                        bool transA, bool transB) {
    size_t m = blockIdx.x;
    for (; m < M; m += gridDim.x) {
        size_t n = threadIdx.x;
        for (; n < N; n += blockDim.x) {
            CompType res = static_cast<CompType>(0);
            for (size_t k = 0; k < K; ++k) {
                AType av = transA ? A[k * LDA + m] : A[m * LDA + k],
                       bv = transB ? B[n * LDB + k] : B[k * LDB + n];
                res += av * bv;
            }
            C[m * LDC + n] = res;
        }
    }
}
}  // namespace

namespace megdnn {
namespace cuda {

template <typename AType, typename BType, typename CType, typename CompType>
void exec_gemm_naive(const AType* A, const BType* B, CType* C, size_t M,
                     size_t N, size_t K, size_t LDA, size_t LDB, size_t LDC,
                     bool transA, bool transB, cudaStream_t stream) {
    do_exec<AType, BType, CType, CompType><<<128, 128, 0, stream>>>(
            A, B, C, M, N, K, LDA, LDB, LDC, transA, transB);
}

#define INST(in_ct, out_ct, comp_ct)                                       \
    template void exec_gemm_naive<typename in_ct, typename in_ct,          \
                                  typename out_ct, typename comp_ct>(      \
            const in_ct* A, const in_ct* B, out_ct* C, size_t M, size_t N, \
            size_t K, size_t LDA, size_t LDB, size_t LDC, bool transA,     \
            bool transB, cudaStream_t stream);

INST(megdnn::dt_float32, megdnn::dt_float32, megdnn::dt_float32)
INST(megdnn::dt_float16, megdnn::dt_float16, megdnn::dt_float16)
INST(megdnn::dt_int8, megdnn::dt_int32, megdnn::dt_int32)
INST(megdnn::dt_float16, megdnn::dt_float16, megdnn::dt_float32)

#undef cb
#undef INST

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
