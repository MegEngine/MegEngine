/**
 * \file dnn/test/arm_common/matrix_mul.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
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
                                 dtype::Quantized8Asymm(1.3f, (uint8_t)129), {},
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

TEST_F(ARM_COMMON, QINT8x8x32_GEMV) {
    Checker<MatrixMul> checker(handle());
    using Param = MatrixMul::Param;

    checker.set_before_exec_callback(
            AlgoChecker<MatrixMul>("ARM_COMMON_INT8X8X32_GEMV"));

    std::unique_ptr<RNG> rng = std::make_unique<UniformIntRNG>(-127, 127);
    checker.set_rng(0, rng.get()).set_rng(1, rng.get());

    auto run = [&](size_t M, size_t K, size_t N) {
        Param param;
        param.transposeA = false;
        param.transposeB = false;
        TensorShape A, B;
        A = TensorShape{M, K};
        B = TensorShape{K, N};
        checker.set_param(param)
                .set_dtype(0, dtype::QuantizedS8(2.5f))
                .set_dtype(1, dtype::QuantizedS8(2.5f))
                .set_dtype(2, dtype::QuantizedS32(6.25f))
                .execs({A, B, {}});
    };

    // N = 1
    for (size_t M : {1, 10, 16, 33, 64})
        for (size_t K : {7, 512, 1024})
            for (size_t N : {1})
                run(M, K, N);
}

TEST_F(ARM_COMMON, QINT8x8x32_GEMV_MK4) {
    Checker<MatrixMul> checker(handle());
    using Param = MatrixMul::Param;

    checker.set_before_exec_callback(
            AlgoChecker<MatrixMul>("ARM_COMMON_INT8X8X32_GEMV_MK4"));

    std::unique_ptr<RNG> rng = std::make_unique<UniformIntRNG>(-127, 127);
    checker.set_rng(0, rng.get()).set_rng(1, rng.get());

    auto run = [&](size_t M, size_t K, size_t N) {
        MEGDNN_MARK_USED_VAR(N);
        Param param;
        param.format = param::MatrixMul::Format::MK4;
        param.transposeA = false;
        param.transposeB = false;
        TensorShape A, B;
        A = TensorShape{M / 4, K / 4, 4, 4};
        B = TensorShape{K / 4, 1, 4};
        checker.set_param(param)
                .set_dtype(0, dtype::QuantizedS8(2.5f))
                .set_dtype(1, dtype::QuantizedS8(2.5f))
                .set_dtype(2, dtype::QuantizedS32(6.25f))
                .execs({A, B, {}});
    };

    // N = 1
    for (size_t M : {4, 16, 128, 1024})
        for (size_t K : {4, 8, 12, 16, 20, 24, 256, 1024})
            run(M, K, 1);
}

#if __ARM_FEATURE_DOTPROD
TEST_F(ARM_COMMON, QINT8x8x32_GEMV_MK4_DOT) {
    Checker<MatrixMul> checker(handle());
    using Param = MatrixMul::Param;

    checker.set_before_exec_callback(
            AlgoChecker<MatrixMul>("ARM_COMMON_INT8X8X32_GEMV_MK4_DOT"));

    std::unique_ptr<RNG> rng = std::make_unique<UniformIntRNG>(-127, 127);
    checker.set_rng(0, rng.get()).set_rng(1, rng.get());

    auto run = [&](size_t M, size_t K, size_t N) {
        Param param;
        param.format = param::MatrixMul::Format::MK4_DOT;
        param.transposeA = false;
        param.transposeB = false;
        TensorShape A, B;
        A = TensorShape{M / 4, K / 4, 4, 4};
        B = TensorShape{K / 4, 1, 4};
        checker.set_param(param)
                .set_dtype(0, dtype::QuantizedS8(2.5f))
                .set_dtype(1, dtype::QuantizedS8(2.5f))
                .set_dtype(2, dtype::QuantizedS32(6.25f))
                .execs({A, B, {}});
    };

    // N = 1
    for (size_t M : {4, 16, 128, 1024})
        for (size_t K : {4, 8, 12, 16, 20, 24, 256, 1024})
            run(M, K, 1);
}
#endif

TEST_F(ARM_COMMON, QINT8x8x32_GEVM) {
    Checker<MatrixMul> checker(handle());
    using Param = MatrixMul::Param;

    checker.set_before_exec_callback(AlgoChecker<MatrixMul>("ARM_COMMON_GEVM"));

    std::unique_ptr<RNG> rng = std::make_unique<UniformIntRNG>(-127, 127);
    checker.set_rng(0, rng.get()).set_rng(1, rng.get());

    auto run = [&](size_t M, size_t K, size_t N) {
        Param param;
        param.transposeA = false;
        param.transposeB = true;
        TensorShape A, B;
        A = TensorShape{M, K};
        B = TensorShape{N, K};
        checker.set_param(param)
                .set_dtype(0, dtype::QuantizedS8(2.5f))
                .set_dtype(1, dtype::QuantizedS8(2.5f))
                .set_dtype(2, dtype::QuantizedS32(6.25f))
                .execs({A, B, {}});
    };

    // M = 1
    for (size_t N : {1, 10, 16, 33, 64})
        for (size_t K : {7, 512, 1024})
            for (size_t M : {1})
                run(M, K, N);
}

TEST_F(ARM_COMMON, FP32_GEVM) {
    Checker<MatrixMul> checker(handle());
    using Param = MatrixMul::Param;

    checker.set_before_exec_callback(AlgoChecker<MatrixMul>("ARM_COMMON_GEVM"));

    checker.set_epsilon(1e-2);
    auto run = [&](size_t M, size_t K, size_t N) {
        Param param;
        param.transposeA = false;
        param.transposeB = true;
        TensorShape A, B;
        A = TensorShape{M, K};
        B = TensorShape{N, K};
        checker.set_param(param).execs({A, B, {}});
    };

    // M = 1
    for (size_t M : {1})
        for (size_t K : {1000, 4096})
            for (size_t N : {1000, 4096})
                run(M, K, N);
}

TEST_F(ARM_COMMON, FP32_GEMV_MK4) {
    Checker<MatrixMul> checker(handle());
    using Param = MatrixMul::Param;

    checker.set_before_exec_callback(
            AlgoChecker<MatrixMul>("ARM_COMMON_F32_GEMV_MK4"));

    checker.set_epsilon(1e-2);
    auto run = [&](size_t M, size_t K) {
        Param param;
        param.format = param::MatrixMul::Format::MK4;
        param.transposeA = false;
        param.transposeB = false;
        TensorShape A, B;
        A = TensorShape{M / 4, K / 4, 4, 4};
        B = TensorShape{K / 4, 1, 4};
        checker.set_param(param).execs({A, B, {}});
    };

    // N = 1
    for (size_t M : {4, 16, 128, 1024})
        for (size_t K : {4, 8, 12, 128, 256, 4096})
            run(M, K);
}

#if MEGDNN_WITH_BENCHMARK

TEST_F(ARM_COMMON, BENCHMARK_SGEMV) {
    int exec_times = 10;
    Benchmarker<MatrixMul> benchmarker(handle());
    benchmarker.set_times(exec_times);

    auto run = [&](size_t M, size_t K, size_t N) {
        printf("SGEMV: (%zu, %zu, %zu)\n", M, K, N);
        benchmarker.set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32());
        auto time = benchmarker.exec({{M, K}, {K, N}, {}}) / exec_times;
        auto computations = 2.f * M * K * N * 1e-6;
        auto perf = computations / time;
        printf("gemv fp32, Performance is %f Gflops\n", perf);
    };

    printf("warm up:\n");
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

    for (size_t M : {4, 64, 1024, 4096})
        for (size_t K : {128, 256, 1024, 4096})
            run(M, K, 1);
}

TEST_F(ARM_COMMON, BENCHMARK_SGEMV_FP32) {
    int exec_times = 50;
    Benchmarker<MatrixMul> benchmarker(handle());
    benchmarker.set_times(exec_times);
    benchmarker.set_before_exec_callback(
            AlgoChecker<MatrixMul>("ARM_COMMON_F32_GEMV"));

    auto run = [&](size_t M, size_t K, size_t N) {
        printf("SGEMV: (%zu, %zu, %zu)\n", M, K, N);
        benchmarker.set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .set_dtype(2, dtype::Float32());
        auto time = benchmarker.exec({{M, K}, {K, N}, {}}) / exec_times;
        auto computations = 2 * M * K * N * 1e-6;
        auto perf = computations / time;
        printf("gemv fp32, Performance is %f Gflops\n", perf);
    };

    printf("warm up:\n");
    for (int i = 0; i < 50; i++) {
        benchmarker.set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .set_display(false)
                .exec({{2, 1024}, {1024, 512}, {}});
        benchmarker.set_display(true);
    }

    // run gemv
    run(12, 48, 1);
    run(48, 12, 1);
    run(32, 128, 1);
    run(128, 32, 1);
    run(64, 256, 1);
    run(256, 64, 1);
    run(128, 512, 1);
    run(512, 128, 1);
    run(256, 1024, 1);
    run(1024, 256, 1);
}

TEST_F(ARM_COMMON, BENCHMARK_SGEMV_MK4) {
    int exec_times = 10;
    using Param = MatrixMul::Param;
    Param param;
    param.format = param::MatrixMul::Format::MK4;
    param.transposeA = false;
    param.transposeB = false;
    Benchmarker<MatrixMul> benchmarker(handle());
    benchmarker.set_times(exec_times);
    benchmarker.set_dtype(0, dtype::Float32())
            .set_dtype(1, dtype::Float32())
            .set_param(param);

    auto run = [&](size_t M, size_t K) {
        printf("SGEMV_MK4: (%zu, %zu, 1)\n", M, K);
        TensorShape A, B;
        A = TensorShape{M / 4, K / 4, 4, 4};
        B = TensorShape{K / 4, 1, 4};
        auto time = benchmarker.exec({A, B, {}}) / exec_times;
        auto computations = 2.f * M * K * 1e-6;
        auto perf = computations / time;
        printf("gemv mk4 fp32, Performance is %f Gflops\n", perf);
    };

    printf("warm up:\n");
    for (int i = 0; i < 50; i++) {
        benchmarker.set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .set_dtype(2, dtype::Float32())
                .set_display(false)
                .exec({{4, 256, 4, 4}, {256, 1, 4}, {}});
    }

    // run gemv mk4
    for (size_t M : {4, 64, 1024, 4096})
        for (size_t K : {128, 1024, 4096})
            run(M, K);
}

TEST_F(ARM_COMMON, BENCHMARK_SGEMV_FP16) {
    int exec_times = 50;
    Benchmarker<MatrixMul> benchmarker(handle());
    benchmarker.set_times(exec_times);
    benchmarker.set_before_exec_callback(
            AlgoChecker<MatrixMul>("ARM_COMMON_F16_GEMV"));

    auto run = [&](size_t M, size_t K, size_t N) {
        printf("SGEMV_FP16: (%zu, %zu, %zu)\n", M, K, N);
        benchmarker.set_dtype(0, dtype::Float16())
                .set_dtype(1, dtype::Float16())
                .set_dtype(2, dtype::Float16());
        auto time = benchmarker.exec({{M, K}, {K, N}, {}}) / exec_times;
        auto computations = 2 * M * K * N * 1e-6;
        auto perf = computations / time;
        printf("gemv fp16, Performance is %f Gflops\n", perf);
    };

    printf("warm up:\n");
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
        printf("SGEMM: (%zu, %zu, %zu)\n", M, K, N);
        benchmarker.set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32());
        time = benchmarker.exec({{M, K}, {K, N}, {}});
        perf = 2.f * M * K * N / time * mod;
        printf("gemm, Performance is %f Gflops\n", perf);
    };

    printf("warm up:\n");
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
            run(M, 1, K);
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

    //////////////////////// gemv //////////////////////////
    for (size_t M : {8, 64, 112, 256}) {
        for (size_t K : {8, 64, 112, 256}) {
            run(M, 1, K);
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
