/**
 * \file dnn/test/arm_common/matrix_mul.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/arm_common/fixture.h"

#include "test/common/benchmarker.h"
#include "test/common/checker.h"
#include "test/common/matrix_mul.h"
#include "test/common/rng.h"

using namespace megdnn;
using namespace test;

TEST_F(ARM_COMMON, MATRIX_MUL_INT8x8x32) {
    matrix_mul::check_matrix_mul(dtype::Int8{}, dtype::Int8{}, dtype::Int32{},
                                 handle());
}

TEST_F(ARM_COMMON, MATRIX_MUL_INT8x8x16) {
    matrix_mul::check_matrix_mul(dtype::Int8{}, dtype::Int8{}, dtype::Int16{},
                                 handle());
}

TEST_F(ARM_COMMON, MATRIX_MUL_QUINT8) {
    matrix_mul::check_matrix_mul(dtype::Quantized8Asymm(1.2f, (uint8_t)127),
                                 dtype::Quantized8Asymm(1.3f, (uint8_t)129),
                                 {},
                                 handle());
}

TEST_F(ARM_COMMON, MATRIX_MUL_FP32) {
    Checker<MatrixMul> checker(handle());
    using Param = MatrixMul::Param;

    auto run = [&](size_t M, size_t K, size_t N) {
        Param param;
        param.transposeA = false;
        param.transposeB = false;
        TensorShape A, B;
        A = TensorShape{M, K};
        B = TensorShape{K, N};
        checker.set_param(param)
                .set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .set_dtype(2, dtype::Float32())
                .execs({A, B, {}});
    };

    checker.set_before_exec_callback(
            AlgoChecker<MatrixMul>("ARM_COMMON_F32_GEMV"));
    // M < 8
    for (size_t M : {1, 2, 3, 4, 5, 6, 7})
        for (size_t K : {7, 1024, 2048})
            for (size_t N : {7, 1024, 2056})
                run(M, K, N);
    // M = 8,K = 1, 2
    for (size_t M : {8})
        for (size_t K : {1, 2})
            for (size_t N : {7, 1024, 2056})
                run(M, K, N);
    // N = 1
    for (size_t M : {1, 2, 3, 4, 5, 6, 7})
        for (size_t K : {7, 1024, 2048})
            for (size_t N : {1})
                run(M, K, N);
}
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

TEST_F(ARM_COMMON, MATRIX_MUL_FP16) {
    Checker<MatrixMul> checker(handle());
    checker.set_epsilon(1e-2);
    NormalRNG rng(2.f);
    checker.set_rng(0, &rng).set_rng(1, &rng);

    using Param = MatrixMul::Param;
    auto args = matrix_mul::get_matmul_args_no_mask();

    for (auto& arg : args) {
        size_t m = arg.m, n = arg.n, k = arg.k;
        Param param;
        param.transposeA = false;
        param.transposeB = false;
        TensorShape A, B;
        A = TensorShape{m, k};
        B = TensorShape{k, n};
        checker.set_param(param)
                .set_dtype(0, dtype::Float16())
                .set_dtype(1, dtype::Float16())
                .set_dtype(2, dtype::Float16())
                .execs({A, B, {}});
    }
}
TEST_F(ARM_COMMON, MATRIX_MUL_FP16_TEST) {
    Checker<MatrixMul> checker(handle());
    using Param = MatrixMul::Param;
    checker.set_epsilon(1e-2);
    NormalRNG rng(2.f);
    checker.set_rng(0, &rng).set_rng(1, &rng);

    auto run = [&](size_t M, size_t K, size_t N) {
        Param param;
        param.transposeA = false;
        param.transposeB = false;
        TensorShape A, B;
        A = TensorShape{M, K};
        B = TensorShape{K, N};
        checker.set_param(param)
                .set_dtype(0, dtype::Float16())
                .set_dtype(1, dtype::Float16())
                .set_dtype(2, dtype::Float16())
                .execs({A, B, {}});
    };
    checker.set_before_exec_callback(
            AlgoChecker<MatrixMul>("ARM_COMMON_F16_GEMV"));

    // M = 1, 2, 3, 4
    for (size_t M : {1, 2, 3, 4})
        for (size_t K : {7, 512, 1024})
            for (size_t N : {13, 1024, 2048})
                run(M, K, N);
    // N = 1
    for (size_t M : {1, 2, 3, 4})
        for (size_t K : {7, 512, 1024})
            for (size_t N : {1})
                run(M, K, N);
}
#endif


#if MEGDNN_WITH_BENCHMARK

TEST_F(ARM_COMMON, BENCHMARK_SGEMV) {
    int exec_times = 10;
    Benchmarker<MatrixMul> benchmarker(handle());
    benchmarker.set_times(exec_times);

    auto run = [&](size_t M, size_t K, size_t N) {
        std::cout << "SGEMV: (" << M << ", " << K << ", " << N << ")"
                  << std::endl;
        benchmarker.set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32());
        auto time = benchmarker.exec({{M, K}, {K, N}, {}}) / exec_times;
        auto computations = 2.f * M * K * N * 1e-6;
        auto perf = computations / time;
        std::cout << "gemv fp32, Performance is " << perf << " Gflops"
                  << std::endl;
    };

    std::cout << "warm up:\n";
    for (int i = 0; i < 50; i++) {
        benchmarker.set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .set_display(false)
                .exec({{2, 1024}, {1024, 512}, {}});
        benchmarker.set_display(true);
    }

    // run gemv
    for (size_t M : {1, 2, 4, 8})
        for (size_t K : {1024, 1536, 2048})
            for (size_t N : {512, 1024})
                run(M, K, N);
}
TEST_F(ARM_COMMON, BENCHMARK_SGEMV_FP16) {
    int exec_times = 50;
    Benchmarker<MatrixMul> benchmarker(handle());
    benchmarker.set_times(exec_times);
    benchmarker.set_before_exec_callback(
            AlgoChecker<MatrixMul>("ARM_COMMON_F16_GEMV"));

    auto run = [&](size_t M, size_t K, size_t N) {
        std::cout << "SGEMV: (" << M << ", " << K << ", " << N << ")"
                  << std::endl;
        benchmarker.set_dtype(0, dtype::Float16())
                .set_dtype(1, dtype::Float16())
                .set_dtype(2, dtype::Float16());
        auto time = benchmarker.exec({{M, K}, {K, N}, {}}) / exec_times;
        auto computations = 2 * M * K * N * 1e-6;
        auto perf = computations / time;
        std::cout << "gemv fp16, Performance is " << perf << " Gflops"
                  << std::endl;
    };

    std::cout << "warm up:\n";
    for (int i = 0; i < 50; i++) {
        benchmarker.set_dtype(0, dtype::Float16())
                .set_dtype(1, dtype::Float16())
                .set_dtype(2, dtype::Float16())
                .set_display(false)
                .exec({{2, 1024}, {1024, 512}, {}});
        benchmarker.set_display(true);
    }

    // run gemv
    for (size_t M : {1, 2, 3, 4})
        for (size_t K : {1024, 1536, 2048})
            for (size_t N : {512, 1024})
                run(M, K, N);
}
TEST_F(ARM_COMMON, BENCHMARK_SGEMM) {
    int exec_times = 10;
    Benchmarker<MatrixMul> benchmarker(handle());
    benchmarker.set_times(exec_times);

    float mod = 1000 * exec_times / 1e9;
    auto run = [&](size_t M, size_t K, size_t N) {
        float time = 1.f, perf = 1.f;
        std::cout << "SGEMM: (" << M << ", " << K << ", " << N << ")"
                  << std::endl;
        benchmarker.set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32());
        time = benchmarker.exec({{M, K}, {K, N}, {}});
        perf = 2.f * M * K * N / time * mod;
        std::cout << "gemm fp32, Performance is " << perf << " Gflops"
                  << std::endl;
    };

    std::cout << "warm up:\n";
    for (int i = 0; i < 50; i++) {
        benchmarker.set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .set_display(false)
                .exec({{2, 1024}, {1024, 512}, {}});
        benchmarker.set_display(true);
    }

    run(256, 12 * 24, 256);

    //////////////////////// gemv //////////////////////////
    for (size_t M : {8, 64, 112, 256}) {
        for (size_t K : {8, 64, 112, 256}) {
            run (M, 1, K);
        }
    }

    //////////////////////// gemm //////////////////////////
    for (size_t M : {8, 64, 112, 256}) {
        for (size_t K : {8, 16, 32, 64, 112, 256}) {
            for (size_t N : {8, 64, 112, 256}) {
                run(M, N, K);
            }
        }
    }

}


TEST_F(ARM_COMMON, BENCHMARK_MATRIX_MUL_INT8x8x32) {
    constexpr size_t RUNS = 50;
    param::MatrixMul param;
    Benchmarker<MatrixMul> benchmarker_int(handle());
    benchmarker_int.set_times(RUNS)
            .set_dtype(0, dtype::Int8{})
            .set_dtype(1, dtype::Int8{})
            .set_dtype(2, dtype::Int32{})
            .set_param(param).set_display(false);
    Benchmarker<MatrixMul> benchmarker_float(handle());
    benchmarker_float.set_display(false).set_times(RUNS);

    auto run = [&](size_t M, size_t N, size_t K) {
        auto int_used = benchmarker_int.exec({{M, K}, {K, N}, {}}) / RUNS;
        auto float_used = benchmarker_float.exec({{M, K}, {K, N}, {}}) / RUNS;
        float computations = 2.f * M * K * N * 1e-6;
        printf("run: {%zu{M} %zu{K} %zu{N}} float: %f ms %f Gflops int: %f ms "
               "%f Gflops speedup: %f\n",
               M, K, N, float_used, computations / float_used, int_used,
               computations / int_used, float_used / int_used);
    };

    run(256, 12 * 24, 256);

    //////////////////////// gemv //////////////////////////
    for (size_t M : {8, 64, 112, 256}) {
        for (size_t K : {8, 64, 112, 256}) {
            run (M, 1, K);
        }
    }

    //////////////////////// gemm //////////////////////////
    for (size_t M : {8, 64, 112, 256}) {
        for (size_t K : {8, 16, 32, 64, 112, 256}) {
            for (size_t N : {8, 64, 112, 256}) {
                run(M, N, K);
            }
        }
    }
}

TEST_F(ARM_COMMON, BENCHMARK_MATRIX_MUL_QUINT8) {
    constexpr size_t RUNS = 50;
    param::MatrixMul param;
    Benchmarker<MatrixMul> benchmarker_int(handle());
    benchmarker_int.set_times(RUNS)
            .set_dtype(0, dtype::Quantized8Asymm(1.2f, (uint8_t)127))
            .set_dtype(1, dtype::Quantized8Asymm(1.3f, (uint8_t)129))
            .set_dtype(2, {})
            .set_param(param)
            .set_display(false);
    Benchmarker<MatrixMul> benchmarker_float(handle());
    benchmarker_float.set_display(false).set_times(RUNS);

    auto run = [&](size_t M, size_t N, size_t K) {
        auto int_used = benchmarker_int.exec({{M, K}, {K, N}, {}}) / RUNS;
        auto float_used = benchmarker_float.exec({{M, K}, {K, N}, {}}) / RUNS;
        float computations = 2.f * M * K * N * 1e-6;
        printf("run: {%zu{M} %zu{K} %zu{N}} float: %f ms %f Gflops int: %f ms "
               "%f Gflops speedup: %f\n",
               M, K, N, float_used, computations / float_used, int_used,
               computations / int_used, float_used / int_used);
    };

    run(256, 12 * 24, 256);

    for (size_t M : {8, 64, 112, 256}) {
        for (size_t K : {8, 64, 112, 256}) {
            for (size_t N : {8, 64, 112, 256}) {
                run(M, N, K);
            }
        }
    }
}

TEST_F(ARM_COMMON, BENCHMARK_TRANSPOSED_MATRIX_MUL_QUINT8) {
    constexpr size_t RUNS = 50;
    param::MatrixMul param;
    param.transposeA = param.transposeB = true;
    Benchmarker<MatrixMul> benchmarker_int(handle());
    benchmarker_int.set_times(RUNS)
            .set_dtype(0, dtype::Quantized8Asymm(1.2f, (uint8_t)127))
            .set_dtype(1, dtype::Quantized8Asymm(1.3f, (uint8_t)129))
            .set_dtype(2, {})
            .set_param(param)
            .set_display(false);
    Benchmarker<MatrixMul> benchmarker_float(handle());
    benchmarker_float.set_param(param).set_display(false).set_times(RUNS);

    auto run = [&](size_t M, size_t N, size_t K) {
        auto int_used = benchmarker_int.exec({{K, M}, {N, K}, {}}) / RUNS;
        auto float_used = benchmarker_float.exec({{K, M}, {N, K}, {}}) / RUNS;
        float computations = 2.f * M * K * N * 1e-6;
        printf("run: {%zu{M} %zu{K} %zu{N}} float: %f ms %f Gflops int: %f ms "
               "%f Gflops speedup: %f\n",
               M, K, N, float_used, computations / float_used, int_used,
               computations / int_used, float_used / int_used);
    };

    run(256, 12 * 24, 256);

    for (size_t M : {8, 64, 112, 256}) {
        for (size_t K : {8, 64, 112, 256}) {
            for (size_t N : {8, 64, 112, 256}) {
                run(M, N, K);
            }
        }
    }
}

#endif


// vim: syntax=cpp.doxygen
