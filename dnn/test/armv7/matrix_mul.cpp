/**
 * \file dnn/test/armv7/matrix_mul.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#include "test/armv7/fixture.h"
#include "test/common/benchmarker.h"
#include "test/common/checker.h"
#include "test/common/matrix_mul.h"
#include "test/common/rng.h"

using namespace megdnn;
using namespace test;

TEST_F(ARMV7, MATRIX_MUL) {
    matrix_mul::check_matrix_mul(dtype::Float32{}, dtype::Float32{},
                                 dtype::Float32{}, handle(), "ARMV7_F32");
}

TEST_F(ARMV7, MATRIX_MUL_MK4) {
    matrix_mul::check_matrix_mul(
            dtype::Float32{}, dtype::Float32{}, dtype::Float32{}, handle(),
            "ARMV7_F32_MK4_4x8", param::MatrixMul::Format::MK4, 1);
}

TEST_F(ARMV7, MATRIX_MUL_PACK_MK4) {
    matrix_mul::check_matrix_mul(
            dtype::Float32{}, dtype::Float32{}, dtype::Float32{}, handle(),
            "ARMV7_F32_MK4_PACK_4X12", param::MatrixMul::Format::MK4, 1);
}

TEST_F(ARMV7, MATRIX_MUL_MK4_INT8) {
    std::vector<matrix_mul::TestArg> args;
    for (size_t m : {1, 2, 3, 4, 5, 7, 10, 11})
        for (size_t n : {1, 2, 3, 4, 5, 8, 16, 24, 25, 32})
            for (size_t k : {1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 33, 34})
                args.emplace_back(m, n, k, 0);
    matrix_mul::check_matrix_mul(dtype::Int8{}, dtype::Int8{}, dtype::Int32{},
                                 handle(), "ARMV7_INT8X8X32_MK4_4X2X16",
                                 param::MatrixMul::Format::MK4, 1, 1e-3,
                                 std::move(args));
}

TEST_F(ARMV7, MATRIX_MUL_INT8x8x16_K4x8x8) {
    matrix_mul::check_matrix_mul(dtype::Int8{}, dtype::Int8{}, dtype::Int16{},
                                 handle(), "ARMV7_INT8X8X16_K4X8X8");
}

TEST_F(ARMV7, MATRIX_MUL_INT8x8x16_MK4_K8x8x4) {
    matrix_mul::check_matrix_mul(dtype::Int8{}, dtype::Int8{}, dtype::Int16{},
                                 handle(), "ARMV7_INT8X8X16_MK4_K8X8X4",
                                 param::MatrixMul::Format::MK4, 1);
}

TEST_F(ARMV7, MATRIX_MUL_INT16x16x32) {
    matrix_mul::check_matrix_mul(dtype::Int16{}, dtype::Int16{}, dtype::Int32{},
                                 handle(), "ARMV7_INT16X16X32_K12X4X1");
}

TEST_F(ARMV7, MATRIX_MUL_INT16x16x32_MK8) {
    matrix_mul::check_matrix_mul(dtype::Int16{}, dtype::Int16{}, dtype::Int32{},
                                 handle(), "ARMV7_INT16X16X32_MK8_4X8",
                                 param::MatrixMul::Format::MK8, 1);
}

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_F(ARMV7, MATRIX_MUL_FP16) {
    matrix_mul::check_matrix_mul(dtype::Float16{}, dtype::Float16{},
                                 dtype::Float16{}, handle(),
                                 "AARCH32_F16_K4X16X1");
}
TEST_F(ARMV7, MATRIX_MUL_F16_MK8) {
    matrix_mul::check_matrix_mul(
            dtype::Float16{}, dtype::Float16{}, dtype::Float16{}, handle(),
            "AARCH32_F16_MK8_4X8", param::MatrixMul::Format::MK8, 1);
}
#endif

#if __ARM_FEATURE_DOTPROD
TEST_F(ARMV7, MATRIX_MUL_SDOT) {
    matrix_mul::check_matrix_mul(dtype::Int8(), dtype::Int8(), dtype::Int32(),
                                 handle(), "AARCH32_INT8_K6X8X4");
}

TEST_F(ARMV7, MATRIX_MUL_UDOT) {
    matrix_mul::check_matrix_mul(
            dtype::Quantized8Asymm(4.0f, static_cast<uint8_t>(10)),
            dtype::Quantized8Asymm(3.0f, static_cast<uint8_t>(54)),
            dtype::QuantizedS32(12.0f), handle(), "AARCH32_QUINT8_K4X8X4");
}

TEST_F(ARMV7, MATRIX_MUL_MK4_DOT_INT8) {
    std::vector<matrix_mul::TestArg> args;
    for (size_t m : {1, 2, 3, 4, 5, 7, 10, 11})
        for (size_t n : {1, 2, 3, 4, 5, 8, 16, 24, 25, 32})
            for (size_t k : {1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 33, 34})
                args.emplace_back(m, n, k, 0);
    matrix_mul::check_matrix_mul(dtype::Int8{}, dtype::Int8{}, dtype::Int32{},
                                 handle(), "AARCH32_INT8_MK4_8X4X4_DOTPROD",
                                 param::MatrixMul::Format::MK4_DOT, 1, 1e-3,
                                 std::move(args));
}
#endif

#if MEGDNN_WITH_BENCHMARK

namespace {
void run_8x8x16_benchmark(
        const char* algo, Handle* handle,
        MatrixMul::Param::Format format = MatrixMul::Param::Format::DEFAULT) {
    constexpr size_t RUNS = 50;
    param::MatrixMul param;
    Benchmarker<MatrixMul> benchmarker_int(handle);
    Benchmarker<MatrixMul> benchmarker_int_kern_4x2x16(handle);
    benchmarker_int.set_before_exec_callback(
            AlgoChecker<MatrixMul>("ARM_COMMON_INT8X8X16"));
    benchmarker_int.set_times(RUNS)
            .set_dtype(0, dtype::Int8{})
            .set_dtype(1, dtype::Int8{})
            .set_dtype(2, dtype::Int16{})
            .set_param(param)
            .set_display(false);
    param::MatrixMul target_param;
    target_param.format = format;
    benchmarker_int_kern_4x2x16.set_before_exec_callback(
            AlgoChecker<MatrixMul>(algo));
    benchmarker_int_kern_4x2x16.set_times(RUNS)
            .set_dtype(0, dtype::Int8{})
            .set_dtype(1, dtype::Int8{})
            .set_dtype(2, dtype::Int16{})
            .set_param(target_param)
            .set_display(false);
    Benchmarker<MatrixMul> benchmarker_float(handle);
    benchmarker_float.set_display(false).set_times(RUNS);

    auto run = [&](size_t M, size_t N, size_t K) {
        auto int_used = benchmarker_int.exec({{M, K}, {K, N}, {}}) / RUNS;
        auto int_kern_used = 1e10;
        if (format == MatrixMul::Param::Format::MK4) {
            int_kern_used = benchmarker_int_kern_4x2x16.exec(
                                    {{M / 4, K / 4, 4, 4}, {K / 4, N, 4}, {}}) /
                            RUNS;
        } else {
            int_kern_used =
                    benchmarker_int_kern_4x2x16.exec({{M, K}, {K, N}, {}}) /
                    RUNS;
        }
        auto float_used = benchmarker_float.exec({{M, K}, {K, N}, {}}) / RUNS;
        float computations = 2.f * M * K * N * 1e-6;
        printf("run: {%zu{M} %zu{K} %zu{N}} float: %f ms %f Gflops int: %f "
               "ms "
               "%f Gflops %s: %f ms %f Gflops "
               "speedup(%s/arm_common, %s/float): %f "
               "%f\n",
               M, K, N, float_used, computations / float_used, int_used,
               computations / int_used, algo, int_kern_used,
               computations / int_kern_used, algo, algo,
               int_used / int_kern_used, float_used / int_kern_used);
    };

    run(256, 12 * 24, 256);
    run(256, 256, 256);

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
void run_16x16x32_benchmark(const char* algo, Handle* handle) {
    constexpr size_t RUNS = 50;
    param::MatrixMul param;
    Benchmarker<MatrixMul> benchmarker_int(handle);
    benchmarker_int.set_before_exec_callback(
            AlgoChecker<MatrixMul>("ARMV7_INT16X16X32_K12X4X1"));
    benchmarker_int.set_times(RUNS)
            .set_dtype(0, dtype::Int16{})
            .set_dtype(1, dtype::Int16{})
            .set_dtype(2, dtype::Int32{})
            .set_param(param)
            .set_display(false);
    Benchmarker<MatrixMul> benchmarker_float(handle);
    benchmarker_float.set_display(false).set_times(RUNS);

    auto run = [&](size_t M, size_t N, size_t K) {
        auto int_used = benchmarker_int.exec({{M, K}, {K, N}, {}}) / RUNS;
        auto float_used = benchmarker_float.exec({{M, K}, {K, N}, {}}) / RUNS;
        float computations = 2.f * M * K * N * 1e-6;
        printf("run: {%zu{M} %zu{K} %zu{N}} float: %f ms %f Gflops \n"
               "int: %f ms %f Gflops %s: \n"
               "speedup(%s/arm_common, %s/float): %f\n",
               M, K, N, float_used, computations / float_used, int_used,
               computations / int_used, algo, algo, algo,
               float_used / int_used);
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
            for (size_t N :
                 {1, 2, 3, 4, 8, 64, 112, 113, 114, 115, 256, 257, 258, 259}) {
                run(M, N, K);
            }
        }
    }
}

#if __ARM_FEATURE_DOTPROD
void run_8x8x32_benchmark(const char* algo, Handle* handle) {
    constexpr size_t RUNS = 50;
    param::MatrixMul param;
    Benchmarker<MatrixMul> benchmarker_int8(handle);
    benchmarker_int8.set_before_exec_callback(AlgoChecker<MatrixMul>(algo));
    benchmarker_int8.set_times(RUNS)
            .set_dtype(0, dtype::Int8{})
            .set_dtype(1, dtype::Int8{})
            .set_dtype(2, dtype::Int32{})
            .set_param(param)
            .set_display(false);
    Benchmarker<MatrixMul> benchmarker_float(handle);
    benchmarker_float.set_display(false).set_times(RUNS);

    auto run = [&](size_t M, size_t N, size_t K) {
        auto int_used = benchmarker_int8.exec({{M, K}, {K, N}, {}}) / RUNS;
        auto float_used = benchmarker_float.exec({{M, K}, {K, N}, {}}) / RUNS;
        float computations = 2.f * M * K * N * 1e-6;
        printf("run: {%zu{M} %zu{K} %zu{N}} float: %f ms %f Gflops \n"
               "int: %f ms %f Gflops %s: \n"
               "speedup(%s/arm_common, %s/float): %f\n",
               M, K, N, float_used, computations / float_used, int_used,
               computations / int_used, algo, algo, algo,
               float_used / int_used);
    };

    run(256, 12 * 24, 256);
    //////////////////////// gemm //////////////////////////
    for (size_t M : {8, 64, 112, 256}) {
        for (size_t K : {8, 16, 32, 64, 112, 256}) {
            for (size_t N : {113, 114, 115, 256, 1024}) {
                run(M, N, K);
            }
        }
    }
}

void run_8x8x32_quint_benchmark(Handle* handle) {
    constexpr size_t RUNS = 50;
    param::MatrixMul param;
    Benchmarker<MatrixMul> benchmarker_quint8_dot(handle);
    benchmarker_quint8_dot.set_before_exec_callback(
            AlgoChecker<MatrixMul>("AARCH32_QUINT8_K4X8X4"));
    benchmarker_quint8_dot.set_times(RUNS)
            .set_dtype(0,
                       dtype::Quantized8Asymm(2.3f, static_cast<uint8_t>(20)))
            .set_dtype(1,
                       dtype::Quantized8Asymm(3.1f, static_cast<uint8_t>(30)))
            .set_dtype(2, dtype::QuantizedS32(2.3f * 3.1f))
            .set_param(param)
            .set_display(false);

    Benchmarker<MatrixMul> benchmarker_quint8(handle);
    benchmarker_quint8.set_before_exec_callback(
            AlgoChecker<MatrixMul>("ARMV7_QUINT8_K4X8X8"));
    benchmarker_quint8.set_times(RUNS)
            .set_dtype(0,
                       dtype::Quantized8Asymm(2.3f, static_cast<uint8_t>(20)))
            .set_dtype(1,
                       dtype::Quantized8Asymm(3.1f, static_cast<uint8_t>(30)))
            .set_dtype(2, dtype::QuantizedS32(2.3f * 3.1f))
            .set_param(param)
            .set_display(false);

    auto run = [&](size_t M, size_t N, size_t K) {
        auto dot_used =
                benchmarker_quint8_dot.exec({{M, K}, {K, N}, {}}) / RUNS;
        auto normal_used = benchmarker_quint8.exec({{M, K}, {K, N}, {}}) / RUNS;
        float computations = 2.f * M * K * N * 1e-6;
        printf("run: {%zu{M} %zu{K} %zu{N}} dot: %f ms %f Gflops \n"
               "normal: %f ms %f Gflops.speedup: %f\n",
               M, K, N, dot_used, computations / dot_used, normal_used,
               computations / normal_used, normal_used / dot_used);
    };

    run(256, 12 * 24, 256);
    //////////////////////// gemm //////////////////////////
    for (size_t M : {8, 64, 112, 256}) {
        for (size_t K : {8, 16, 32, 64, 112, 256}) {
            for (size_t N : {113, 114, 115, 256, 1024}) {
                run(M, N, K);
            }
        }
    }
}
#endif
}  // namespace

#if __ARM_FEATURE_DOTPROD
TEST_F(ARMV7, BENCHMARK_MATRIX_MUL_INT8x8x32_K6x8x4) {
    run_8x8x32_benchmark("AARCH32_INT8_K6X8X4", handle());
}
TEST_F(ARMV7, BENCHMARK_MATRIX_MUL_QUINT8x8x32_K4x8x4) {
    run_8x8x32_quint_benchmark(handle());
}

TEST_F(ARMV7, BENCHMARK_MATRIX_MUL_INT8x8x32_MK4_DOT) {
    constexpr size_t RUNS = 50;
    param::MatrixMul param;
    Benchmarker<MatrixMul> benchmarker_default(handle());
    benchmarker_default.set_times(RUNS)
            .set_dtype(0, dtype::Int8())
            .set_dtype(1, dtype::Int8())
            .set_dtype(2, dtype::Int32())
            .set_param(param)
            .set_display(false);
    benchmarker_default.set_before_exec_callback(
            AlgoChecker<MatrixMul>("AARCH32_INT8_K6X8X4"));

    param.format = MatrixMul::Param::Format::MK4_DOT;
    Benchmarker<MatrixMul> benchmarker_mk4_dot(handle());
    benchmarker_mk4_dot.set_before_exec_callback(
            AlgoChecker<MatrixMul>("AARCH32_INT8_MK4_8X4X4_DOTPROD"));
    benchmarker_mk4_dot.set_param(param)
            .set_dtype(0, dtype::Int8())
            .set_dtype(1, dtype::Int8())
            .set_dtype(2, dtype::Int32())
            .set_display(false)
            .set_times(RUNS);

    auto run = [&](size_t M, size_t N, size_t K) {
        auto default_used =
                benchmarker_default.exec({{M, K}, {K, N}, {}}) / RUNS;
        auto mk4_dot_used = benchmarker_mk4_dot.exec(
                                    {{M / 4, K / 4, 4, 4}, {K / 4, N, 4}, {}}) /
                            RUNS;
        float computations = 2.f * M * K * N * 1e-6;
        printf("run: {%zu{M} %zu{K} %zu{N}} default: %f ms %f Gflops mk4_dot: "
               "%f ms "
               "%f Gflops speedup: %f\n",
               M, K, N, default_used, computations / default_used, mk4_dot_used,
               computations / mk4_dot_used, default_used / mk4_dot_used);
    };

    for (size_t M = 4; M < 512; M *= 2) {
        for (size_t K = 4; K < 512; K *= 2) {
            for (size_t N : {4, 8, 33, 113, 128}) {
                run(M, N, K);
            }
        }
    }
}
#endif

TEST_F(ARMV7, BENCHMARK_MATRIX_MUL_INT8x8x16_K4x2x16) {
    run_8x8x16_benchmark("ARMV7_INT8X8X16_K4X2X16", handle());
}

TEST_F(ARMV7, BENCHMARK_MATRIX_MUL_INT8x8x16_K4x8x8) {
    run_8x8x16_benchmark("ARMV7_INT8X8X16_K4X8X8", handle());
}

TEST_F(ARMV7, BENCHMARK_MATRIX_MUL_INT8x8x16_MK4_K4x8x8) {
    run_8x8x16_benchmark("ARMV7_INT8X8X16_MK4_K8X8X4", handle(),
                         MatrixMul::Param::Format::MK4);
}

TEST_F(ARMV7, BENCHMARK_MATRIX_MUL_INT16x16x32_K12x4x1) {
    run_16x16x32_benchmark("ARMV7_INT16X16X32_K12X4X1", handle());
}

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_F(ARMV7, BENCHMARK_MATRIX_MUL_FP16) {
    constexpr size_t RUNS = 50;
    param::MatrixMul param;
    Benchmarker<MatrixMul> benchmarker_fp16(handle());
    benchmarker_fp16.set_times(RUNS)
            .set_dtype(0, dtype::Float16())
            .set_dtype(1, dtype::Float16())
            .set_dtype(2, dtype::Float16())
            .set_param(param)
            .set_display(false);
    Benchmarker<MatrixMul> benchmarker_float(handle());
    benchmarker_float.set_param(param).set_display(false).set_times(RUNS);

    auto run = [&](size_t M, size_t N, size_t K) {
        auto fp16_used = benchmarker_fp16.exec({{M, K}, {K, N}, {}}) / RUNS;
        auto float_used = benchmarker_float.exec({{M, K}, {K, N}, {}}) / RUNS;
        float computations = 2.f * M * K * N * 1e-6;
        printf("run: {%zu{M} %zu{K} %zu{N}} float: %f ms %f Gflops fp16: %f ms "
               "%f Gflops speedup: %f\n",
               M, K, N, float_used, computations / float_used, fp16_used,
               computations / fp16_used, float_used / fp16_used);
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

TEST_F(ARMV7, BENCHMARK_MATRIX_MUL_F16_MK8) {
    auto args = matrix_mul::get_benchmark_matmul_mk_packed_args(4);
    matrix_mul::benchmark_with_contrast(
            handle(), args, dtype::Float16{}, dtype::Float16{},
            dtype::Float16{}, "AARCH32_F16_MK8_4X8",
            param::MatrixMul::Format::MK8, dtype::Float16{}, dtype::Float16{},
            dtype::Float16{}, "AARCH32_F16_K4X16X1");
}
#endif

TEST_F(ARMV7, BENCHMARK_MATRIX_MUL_MK4) {
    auto args = matrix_mul::get_benchmark_matmul_mk_packed_args(8);
    matrix_mul::benchmark_with_contrast(
            handle(), args, dtype::Float32{}, dtype::Float32{},
            dtype::Float32{}, "ARMV7_F32_MK4_4x8",
            param::MatrixMul::Format::MK4, dtype::Float32{}, dtype::Float32{},
            dtype::Float32{});
}

TEST_F(ARMV7, BENCHMARK_MATRIX_MUL_PACK_MK4) {
    auto args = matrix_mul::get_benchmark_matmul_mk_packed_args(8);
    matrix_mul::benchmark_with_contrast(
            handle(), args, dtype::Float32{}, dtype::Float32{},
            dtype::Float32{}, "ARMV7_F32_MK4_PACK_4X12",
            param::MatrixMul::Format::MK4, dtype::Float32{}, dtype::Float32{},
            dtype::Float32{});
}

TEST_F(ARMV7, BENCHMARK_MATRIX_MUL_INT16x16x32_MK8) {
    auto args = matrix_mul::get_benchmark_matmul_mk_packed_args(4);
    matrix_mul::benchmark_with_contrast(
            handle(), args, dtype::Int16{}, dtype::Int16{}, dtype::Int32{},
            "ARMV7_INT16X16X32_MK8_4X8", param::MatrixMul::Format::MK8,
            dtype::Int16{}, dtype::Int16{}, dtype::Int32{});
}
TEST_F(ARMV7, BENCHMARK_MATRIX_MUL_INT32_MK_4X2X16) {
    constexpr size_t RUNS = 50;
    param::MatrixMul param;
    param.transposeA = false;
    param.transposeB = false;
    Benchmarker<MatrixMul> benchmarker(handle());
    Benchmarker<MatrixMul> benchmarker_mk4(handle());
    benchmarker.set_times(RUNS)
            .set_dtype(0, dtype::Int8{})
            .set_dtype(1, dtype::Int8{})
            .set_dtype(2, dtype::Int32{})
            .set_param(param)
            .set_display(false);
    benchmarker.set_before_exec_callback(
            AlgoChecker<MatrixMul>("ARMV7_INT8X8X32_K4X2X16"));

    param.format = MatrixMul::Param::Format::MK4;
    benchmarker_mk4.set_before_exec_callback(
            AlgoChecker<MatrixMul>("ARMV7_INT8X8X32_MK4_4X2X16"));
    benchmarker_mk4.set_times(RUNS)
            .set_dtype(0, dtype::Int8{})
            .set_dtype(1, dtype::Int8{})
            .set_dtype(2, dtype::Int32{})
            .set_param(param)
            .set_display(false);

    auto run = [&](size_t M, size_t N, size_t K) {
        auto mk_used = benchmarker_mk4.exec(
                               {{M / 4, K / 4, 4, 4}, {K / 4, N, 4}, {}}) /
                       RUNS;
        auto default_used = benchmarker.exec({{M, K}, {K, N}, {}}) / RUNS;
        float computations = 2.f * M * K * N * 1e-6;
        printf("run: {%zu{M} %zu{K} %zu{N}} normal: %f ms %f Gflops mk4: %f ms "
               "%f Gflops speedup_vs_normal: %f\n",
               M, K, N, default_used, computations / default_used, mk_used,
               computations / mk_used, default_used / mk_used);
    };

    run(256, 256, 128);
    for (size_t k = 4; k <= 512; k *= 2) {
        for (size_t m = 4; m <= 512; m *= 2) {
            for (size_t n = 4; n <= 512; n *= 2) {
                run(m, n, k);
            }
        }
        std::cout << std::endl;
    }
}

#endif

// vim: syntax=cpp.doxygen
