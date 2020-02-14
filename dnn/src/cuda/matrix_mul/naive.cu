/**
 * \file dnn/src/cuda/matrix_mul/naive.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include <cuda.h>
#include "src/cuda/matrix_mul/naive.cuh"
#include "src/cuda/utils.cuh"

namespace {
__global__ void do_exec(const int8_t* A, const int8_t* B, int32_t* C, size_t M,
                        size_t N, size_t K, size_t LDA, size_t LDB, size_t LDC,
                        bool transA, bool transB) {
    size_t m = blockIdx.x;
    for (; m < M; m += gridDim.x) {
        size_t n = threadIdx.x;
        for (; n < N; n += blockDim.x) {
            int32_t res = 0;
            for (size_t k = 0; k < K; ++k) {
                int8_t av = transA ? A[k * LDA + m] : A[m * LDA + k],
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

void exec_gemm_int8_naive(const int8_t* A, const int8_t* B, int32_t* C,
                          size_t M, size_t N, size_t K, size_t LDA, size_t LDB,
                          size_t LDC, bool transA, bool transB,
                          cudaStream_t stream) {
    do_exec<<<128, 128, 0, stream>>>(A, B, C, M, N, K, LDA, LDB, LDC, transA,
                                     transB);
}

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
