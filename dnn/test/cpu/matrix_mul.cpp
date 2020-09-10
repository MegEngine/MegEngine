/**
 * \file dnn/test/cpu/matrix_mul.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/cpu/fixture.h"

#include <chrono>
#include "test/common/benchmarker.h"
#include "test/common/checker.h"
#include "test/common/matrix_mul.h"

using namespace megdnn;
using namespace test;
#if MEGDNN_WITH_BENCHMARK
namespace {

void sgemm_sgemv_like(const float* __restrict A, const float* __restrict B,
                      float* __restrict C, size_t M, size_t N, size_t K,
                      size_t Astride, size_t Bstride, size_t Cstride) {
    for (size_t m = 0; m < M; ++m) {
        memset(C + m * Cstride, 0, sizeof(float) * N);
        for (size_t k = 0; k < K; ++k)
            for (size_t n = 0; n < N; ++n) {
                C[m * Cstride + n] += A[m * Astride + k] * B[k * Bstride + n];
            }
    }
}

float benchmark_sgemm_sgemv_like(size_t M, size_t N, size_t K) {
    float *A = (float*)malloc(sizeof(float) * M * K),
          *B = (float*)malloc(sizeof(float) * K * N),
          *C = (float*)malloc(sizeof(float) * M * N);
    for (size_t i = 0; i < M * K; ++i)
        A[i] = (float)rand() / RAND_MAX;
    for (size_t i = 0; i < K * N; ++i)
        B[i] = (float)rand() / RAND_MAX;
    sgemm_sgemv_like(A, B, C, M, N, K, K, N, N);
    auto start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < 100; ++i) {
        sgemm_sgemv_like(A, B, C, M, N, K, K, N, N);
    }
    auto stop = std::chrono::high_resolution_clock::now();
    free(A);
    free(B);
    free(C);
    return std::chrono::duration_cast<std::chrono::milliseconds>(stop - start)
            .count();
}

}  // namespace

TEST_F(CPU, BENCHMARK_MATRIX_MUL) {
    Benchmarker<MatrixMul> benchmarker(handle());
    benchmarker.set_times(100);
    benchmarker.set_display(false);
    auto run = [&](size_t M, size_t N, size_t K) {
        std::cout << M << "x" << N << "x" << K << " ";
        auto time_in_ms_megdnn = benchmarker.exec({{M, K}, {K, N}, {}});
        auto time_in_ms_our = benchmark_sgemm_sgemv_like(M, N, K);
        std::cout << "megdnn=" << (int)time_in_ms_megdnn
                  << " sgemv_like=" << time_in_ms_our << std::endl;
    };
    for (size_t m = 1; m <= 8; m *= 2)
        for (size_t nk = 128; nk <= 1024; nk *= 2) {
            run(m, nk, nk);
        }
}
#endif

TEST_F(CPU, MATRIX_MUL) {
    matrix_mul::check_matrix_mul(dtype::Float32{}, dtype::Float32{},
                                 dtype::Float32{}, handle());
}

// vim: syntax=cpp.doxygen
