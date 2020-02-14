/**
 * \file dnn/test/cuda/batched_matrix_mul.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/cuda/fixture.h"

#include "test/common/checker.h"
#include "test/common/matrix_mul.h"
#include "test/common/rng.h"
#include "test/cuda/benchmark.h"
#include "test/cuda/utils.h"

using namespace megdnn;
using namespace test;

#define F32_TEST_PART(x, algo)                                                 \
    matrix_mul::check_batched_matrix_mul(                                      \
            dtype::Float32{}, dtype::Float32{}, {}, handle_cuda(), algo, 1e-3, \
            matrix_mul::get_batched_matmul_args_mask(x))

TEST_F(CUDA, BATCHED_MATRIX_MUL_F32_PART1) {
    F32_TEST_PART(0, "CUBLAS");
}

TEST_F(CUDA, BATCHED_MATRIX_MUL_F32_PART2) {
    F32_TEST_PART(1, "CUBLAS");
}

TEST_F(CUDA, BATCHED_MATRIX_MUL_F32_PART3) {
    F32_TEST_PART(2, "CUBLAS");
}

TEST_F(CUDA, BATCHED_MATRIX_MUL_F32_PART4) {
    F32_TEST_PART(3, "CUBLAS");
}

TEST_F(CUDA, BATCHED_MATRIX_MUL_LT_F32_PART1) {
    require_compute_capability(7, 0);
    F32_TEST_PART(0, "CUBLAS_LT");
}

TEST_F(CUDA, BATCHED_MATRIX_MUL_LT_F32_PART2) {
    require_compute_capability(7, 0);
    F32_TEST_PART(1, "CUBLAS_LT");
}

TEST_F(CUDA, BATCHED_MATRIX_MUL_LT_F32_PART3) {
    require_compute_capability(7, 0);
    F32_TEST_PART(2, "CUBLAS_LT");
}

TEST_F(CUDA, BATCHED_MATRIX_MUL_LT_F32_PART4) {
    require_compute_capability(7, 0);
    F32_TEST_PART(3, "CUBLAS_LT");
}

#undef F32_TEST_PART

TEST_F(CUDA, BATCHED_MATRIX_MUL_F16_PART1) {
    require_compute_capability(6, 0);
    matrix_mul::check_batched_matrix_mul(
            dtype::Float16{}, dtype::Float16{}, {}, handle_cuda(), "CUBLAS",
            2e-2, matrix_mul::get_batched_matmul_args_mask(0));
}

TEST_F(CUDA, BATCHED_MATRIX_MUL_F16_PART2) {
    require_compute_capability(6, 0);
    matrix_mul::check_batched_matrix_mul(
            dtype::Float16{}, dtype::Float16{}, {}, handle_cuda(), "CUBLAS",
            2e-2, matrix_mul::get_batched_matmul_args_mask(1));
}

TEST_F(CUDA, BATCHED_MATRIX_MUL_F16_PART3) {
    require_compute_capability(6, 0);
    matrix_mul::check_batched_matrix_mul(
            dtype::Float16{}, dtype::Float16{}, {}, handle_cuda(), "CUBLAS",
            2e-2, matrix_mul::get_batched_matmul_args_mask(2));
}

TEST_F(CUDA, BATCHED_MATRIX_MUL_F16_PART4) {
    require_compute_capability(6, 0);
    matrix_mul::check_batched_matrix_mul(
            dtype::Float16{}, dtype::Float16{}, {}, handle_cuda(), "CUBLAS",
            2e-2, matrix_mul::get_batched_matmul_args_mask(3));
}

TEST_F(CUDA, BATCHED_MATRIX_MUL_CUBLASLT_F16_PART1) {
    require_compute_capability(7, 0);
    matrix_mul::check_batched_matrix_mul(
            dtype::Float16{}, dtype::Float16{}, {}, handle_cuda(), "CUBLAS_LT",
            2e-2, matrix_mul::get_batched_matmul_args_mask(0));
}

TEST_F(CUDA, BATCHED_MATRIX_MUL_CUBLASLT_F16_PART2) {
    require_compute_capability(7, 0);
    matrix_mul::check_batched_matrix_mul(
            dtype::Float16{}, dtype::Float16{}, {}, handle_cuda(), "CUBLAS_LT",
            2e-2, matrix_mul::get_batched_matmul_args_mask(1));
}

TEST_F(CUDA, BATCHED_MATRIX_MUL_CUBLASLT_F16_PART3) {
    require_compute_capability(7, 0);
    matrix_mul::check_batched_matrix_mul(
            dtype::Float16{}, dtype::Float16{}, {}, handle_cuda(), "CUBLAS_LT",
            2e-2, matrix_mul::get_batched_matmul_args_mask(2));
}

TEST_F(CUDA, BATCHED_MATRIX_MUL_CUBLASLT_F16_PART4) {
    require_compute_capability(7, 0);
    matrix_mul::check_batched_matrix_mul(
            dtype::Float16{}, dtype::Float16{}, {}, handle_cuda(), "CUBLAS_LT",
            2e-2, matrix_mul::get_batched_matmul_args_mask(3));
}

TEST_F(CUDA, BATCHED_MATRIX_MUL_CUBLASLT_INT8) {
    require_compute_capability(7, 5);
    matrix_mul::check_batched_matrix_mul(
            dtype::Int8{}, dtype::Int8{}, {}, handle_cuda(), "CUBLAS_LT", 1e-3,
            matrix_mul::get_batched_matmul_args_cublaslt());
}

TEST_F(CUDA, BATCHED_MATRIX_MUL_CUBLASLT_QS8) {
    require_compute_capability(7, 5);
    matrix_mul::check_batched_matrix_mul(
            dtype::QuantizedS8(1.2f), dtype::QuantizedS8(1.3f), {},
            handle_cuda(), "CUBLAS_LT", 1e-3,
            matrix_mul::get_batched_matmul_args_cublaslt());
}

TEST_F(CUDA, BATCHED_MATRIX_MUL_QS8) {
    matrix_mul::check_batched_matrix_mul(dtype::QuantizedS8(1.2f),
                                         dtype::QuantizedS8(1.3f), {},
                                         handle_cuda());
}

TEST_F(CUDA, BATCHED_MATRIX_MUL_INT8x8x32) {
    require_compute_capability(6, 1);
    matrix_mul::check_batched_matrix_mul(
            dtype::Int8{}, dtype::Int8{}, dtype::Int32{}, handle_cuda(),
            "INT8x8x32", 1e-2, matrix_mul::get_batched_matmul_args_int8x8x32());
}

#if MEGDNN_WITH_BENCHMARK
TEST_F(CUDA, BATCHED_MATMUL_8x8x32_BENCHMARK) {
    require_compute_capability(6, 1);
    auto run = [&](bool transA, bool transB, size_t m, size_t n, size_t k,
                   const char* algo1, const char* algo2, size_t b = 128) {
        size_t RUNS = 10;
        CUBenchmarker<BatchedMatrixMul> bencher1(handle_cuda());
        bencher1.set_display(false).set_times(RUNS);
        bencher1.set_before_exec_callback(AlgoChecker<BatchedMatrixMul>(algo1));
        CUBenchmarker<BatchedMatrixMul> bencher2(handle_cuda());
        bencher2.set_display(false).set_times(RUNS);
        bencher2.set_before_exec_callback(AlgoChecker<BatchedMatrixMul>(algo2));
        using Param = MatrixMul::Param;
        DType stype = dtype::Int8(), dtype = dtype::Int32();
        Param param;
        UniformIntRNG rng(-128, 127);
        param.transposeA = transA;
        param.transposeB = transB;
        TensorShape A, B;
        if (param.transposeA)
            A = TensorShape{b, k, m};
        else
            A = TensorShape{b, m, k};
        if (param.transposeB)
            B = TensorShape{b, n, k};
        else
            B = TensorShape{b, k, n};

        auto flo = (double)m * n * k * b * 2;
        bencher1.set_param(param)
                .set_dtype(0, stype)
                .set_dtype(1, stype)
                .set_dtype(2, dtype)
                .set_rng(0, &rng)
                .set_rng(1, &rng);
        auto time1 = bencher1.execs({A, B, {}}) / RUNS;
        auto flops1 = flo / time1 / 1e6;

        bencher2.set_param(param)
                .set_dtype(0, stype)
                .set_dtype(1, stype)
                .set_dtype(2, dtype)
                .set_rng(0, &rng)
                .set_rng(1, &rng);
        auto time2 = bencher2.execs({A, B, {}}) / RUNS;
        auto flops2 = flo / time2 / 1e6;

        printf("trA: %d, trB: %d, m: %ld, n: %ld, k: %ld, b: %ld, speedup: %s "
               "/ "
               "%s %.3f\n",
               transA, transB, m, n, k, b, algo1, algo2, flops1 / flops2);
    };

    for (bool transA : {0, 1})
        for (bool transB : {0, 1}) {
            run(transA, transB, 128, 576, 128, "INT8x8x32",
                "BRUTE_FORCE-CUBLAS");
            run(transA, transB, 256, 144, 256, "INT8x8x32",
                "BRUTE_FORCE-CUBLAS");
            run(transA, transB, 512, 36, 512, "INT8x8x32",
                "BRUTE_FORCE-CUBLAS");
            run(transA, transB, 1024, 8, 1024, "INT8x8x32",
                "BRUTE_FORCE-CUBLAS");
        }
}
#endif

// vim: syntax=cpp.doxygen
