/**
 * \file dnn/test/aarch64/matrix_mul.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#include "test/aarch64/fixture.h"

#include "test/common/benchmarker.h"
#include "test/common/checker.h"

#include "test/common/matrix_mul.h"
#include "test/common/rng.h"

#include "test/arm_common/cpuinfo_help.h"
using namespace megdnn;
using namespace test;

TEST_F(AARCH64, MATRIX_MUL_FP32K8X12) {
    matrix_mul::check_matrix_mul(dtype::Float32{}, dtype::Float32{},
                                 dtype::Float32{}, handle(),
                                 "AARCH64_F32K8X12X1");
}
#if MGB_ENABLE_CPUINFO
TEST_F(AARCH64, MATRIX_MUL_FP32K8X12_A53) {
    CpuInfoTmpReplace cpu_replace_guard(cpuinfo_uarch_cortex_a53);
    matrix_mul::check_matrix_mul(dtype::Float32{}, dtype::Float32{},
                                 dtype::Float32{}, handle(),
                                 "AARCH64_F32K8X12X1");
}
TEST_F(AARCH64, MATRIX_MUL_FP32K8X12_A55) {
    CpuInfoTmpReplace cpu_replace_guard(cpuinfo_uarch_cortex_a55);
    matrix_mul::check_matrix_mul(dtype::Float32{}, dtype::Float32{},
                                 dtype::Float32{}, handle(),
                                 "AARCH64_F32K8X12X1");
}
#endif

TEST_F(AARCH64, MATRIX_MUL_FP32K4X16) {
    matrix_mul::check_matrix_mul(dtype::Float32{}, dtype::Float32{},
                                 dtype::Float32{}, handle(),
                                 "AARCH64_F32K4X16X1");
}

TEST_F(AARCH64, MATRIX_MUL_FP32_PACK_MK4) {
    matrix_mul::check_matrix_mul(
            dtype::Float32{}, dtype::Float32{}, dtype::Float32{}, handle(),
            "AARCH64_F32_MK4_K8X12X1", param::MatrixMul::Format::MK4, 1);
}
#if MGB_ENABLE_CPUINFO
TEST_F(AARCH64, MATRIX_MUL_FP32_PACK_MK4_A53) {
    CpuInfoTmpReplace cpu_replace_guard(cpuinfo_uarch_cortex_a53);
    matrix_mul::check_matrix_mul(
            dtype::Float32{}, dtype::Float32{}, dtype::Float32{}, handle(),
            "AARCH64_F32_MK4_K8X12X1", param::MatrixMul::Format::MK4, 1);
}
TEST_F(AARCH64, MATRIX_MUL_FP32_PACK_MK4_A55) {
    CpuInfoTmpReplace cpu_replace_guard(cpuinfo_uarch_cortex_a55);
    matrix_mul::check_matrix_mul(
            dtype::Float32{}, dtype::Float32{}, dtype::Float32{}, handle(),
            "AARCH64_F32_MK4_K8X12X1", param::MatrixMul::Format::MK4, 1);
}
#endif

TEST_F(AARCH64, MATRIX_MUL_FP32_MK4) {
    matrix_mul::check_matrix_mul(
            dtype::Float32{}, dtype::Float32{}, dtype::Float32{}, handle(),
            "AARCH64_F32_MK4_4x16", param::MatrixMul::Format::MK4, 1);
}

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_F(AARCH64, MATRIX_MUL_F16_K8X24X1) {
    matrix_mul::check_matrix_mul(dtype::Float16{}, dtype::Float16{},
                                 dtype::Float16{}, handle(),
                                 "AARCH64_F16_K8X24X1");
}

TEST_F(AARCH64, MATRIX_MUL_F16_MK8) {
    matrix_mul::check_matrix_mul(
            dtype::Float16{}, dtype::Float16{}, dtype::Float16{}, handle(),
            "AARCH64_F16_MK8_8X8", param::MatrixMul::Format::MK8, 1);
}
#endif

#if __ARM_FEATURE_DOTPROD
TEST_F(AARCH64, MATRIX_MUL_INT8X8X32_K8X12X4_DOTPROD) {
    matrix_mul::check_matrix_mul(dtype::Int8{}, dtype::Int8{}, dtype::Int32{},
                                 handle(), "AARCH64_INT8X8X32_K8X12X4_DOTPROD");
}

TEST_F(AARCH64, MATRIX_MUL_INT8X8X32_MK4_8X12X4_DOTPROD) {
    std::vector<matrix_mul::TestArg> args;
    for (size_t m : {1, 2, 3, 4, 5, 6, 7, 10, 11})
        for (size_t n : {2, 3, 4, 5, 8, 12, 13, 14, 15, 16, 31})
            for (size_t k : {1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 33, 34})
                args.emplace_back(m, n, k, 0);
    matrix_mul::check_matrix_mul(
            dtype::Int8{}, dtype::Int8{}, dtype::Int32{}, handle(),
            "AARCH64_INT8X8X32_MK4_8X12X4_DOTPROD",
            param::MatrixMul::Format::MK4_DOT, 1, 1e-3, std::move(args));
}
#else
TEST_F(AARCH64, MATRIX_MUL_INT8X8X32_K4X4X16) {
    matrix_mul::check_matrix_mul(dtype::Int8{}, dtype::Int8{}, dtype::Int32{},
                                 handle(), "AARCH64_INT8X8X32_K4X4X16");
}

TEST_F(AARCH64, MATRIX_MUL_INT8_MK4) {
    std::vector<matrix_mul::TestArg> args;
    for (size_t m : {1, 2, 3, 4, 5, 7, 10, 11})
        for (size_t n : {1, 2, 3, 4, 5, 8, 16, 24, 25, 32})
            for (size_t k : {1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 33, 34})
                args.emplace_back(m, n, k, 0);
    matrix_mul::check_matrix_mul(dtype::Int8{}, dtype::Int8{}, dtype::Int32{},
                                 handle(), "AARCH64_INT8X8X32_MK4_4X4X16",
                                 param::MatrixMul::Format::MK4, 1, 1e-3,
                                 std::move(args));
}

TEST_F(AARCH64, MATRIX_MUL_INT8x8x16_MK4) {
    std::vector<matrix_mul::TestArg> args;
    for (size_t m : {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17})
        for (size_t n :
             {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 24})
            for (size_t k :
                 {2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15,
                  16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29})
                args.emplace_back(m, n, k, 0);
    matrix_mul::check_matrix_mul(dtype::Int8{}, dtype::Int8{}, dtype::Int16{},
                                 handle(), "AARCH64_INT8X8X16_MK4_K8X8X8",
                                 param::MatrixMul::Format::MK4, 1, 1e-3,
                                 std::move(args));
}
TEST_F(AARCH64, MATRIX_MUL_MK4_8x8x16_4x4) {
    matrix_mul::check_matrix_mul(dtype::Int8{}, dtype::Int8{}, dtype::Int16{},
                                 handle(), "AARCH64_INT8X8X16_MK4_4X4X8",
                                 param::MatrixMul::Format::MK4, 1);
}

TEST_F(AARCH64, MATRIX_MUL_MK4_8x8x16) {
    matrix_mul::check_matrix_mul(dtype::Int8{}, dtype::Int8{}, dtype::Int16{},
                                 handle(), "AARCH64_INT8X8X16_MK4_16X12X4",
                                 param::MatrixMul::Format::MK4, 1);
}

TEST_F(AARCH64, MATRIX_MUL_INT8x8x32_K8x8x8) {
    matrix_mul::check_matrix_mul(dtype::Int8{}, dtype::Int8{}, dtype::Int32{},
                                 handle(), "AARCH64_INT8X8X32_K8X8X8");
}
#endif

TEST_F(AARCH64, MATRIX_MUL_INT8x8x16_K8x8x8) {
    matrix_mul::check_matrix_mul(dtype::Int8{}, dtype::Int8{}, dtype::Int16{},
                                 handle(), "AARCH64_INT8X8X16_K8X8X8");
}

TEST_F(AARCH64, MATRIX_MUL_INT8x8x16_K4x4x16) {
    matrix_mul::check_matrix_mul(dtype::Int8{}, dtype::Int8{}, dtype::Int16{},
                                 handle(), "AARCH64_INT8X8X16_K4X4X16");
}

TEST_F(AARCH64, MATRIX_MUL_INT4x4x16_K8x8x8_QUANTIZEDS4) {
    param::MatrixMul param;
    param.transposeA = false;
    param.transposeB = false;

    Checker<MatrixMul> checker(handle());
    checker.set_dtype(0, dtype::QuantizedS4{0.6})
            .set_dtype(1, dtype::QuantizedS4{0.5})
            .set_dtype(2, dtype::QuantizedS16{0.6 * 0.5})
            .set_param(param);
    checker.set_before_exec_callback(
            AlgoChecker<MatrixMul>("AARCH64_INT4X4X16_K8X8X8"));

    auto run = [&](size_t M, size_t N, size_t K) {
        printf("M N K %zu %zu %zu \n", M, N, K);
        TensorShape A, B;
        if (param.transposeA) {
            A = TensorShape{K, M};
        } else {
            A = TensorShape{M, K};
        }
        if (param.transposeB) {
            B = TensorShape{N, K};
        } else {
            B = TensorShape{K, N};
        }
        checker.exec({A, B, {}});
    };

    for (size_t m : {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 20})
        for (size_t n : {2, 4, 6, 8, 10, 12, 14, 16, 24})
            for (size_t k : {2, 4, 6, 8, 10, 12, 14, 16, 32})
                run(m, n, k);

    for (size_t k = 4; k <= 256; k *= 8) {
        for (size_t m = 4; m <= 256; m *= 4) {
            for (size_t n = 4; n <= 256; n *= 4) {
                run(m, n, k);
            }
        }
    }
    param.transposeA = true;
    run(8,8,8);
    run(16,8,16);
    param.transposeB = true;
    run(8,8,8);
    run(16,16,16);
}

TEST_F(AARCH64, MATRIX_MUL_INT16x16x32_K12X8X1) {
    matrix_mul::check_matrix_mul(dtype::Int16{}, dtype::Int16{}, dtype::Int32{},
                                 handle(), "AARCH64_INT16X16X32_K12X8X1");
}

TEST_F(AARCH64, MATRIX_MUL_INT16x16x32_MK8) {
    matrix_mul::check_matrix_mul(dtype::Int16{}, dtype::Int16{}, dtype::Int32{},
                                 handle(), "AARCH64_INT16X16X32_MK8_8X8",
                                 param::MatrixMul::Format::MK8, 1);
}

//! FIXME: need to add tests of GEMV and QUINT8

#if MEGDNN_WITH_BENCHMARK

TEST_F(AARCH64, BENCHMARK_MATRIX_MUL_FP32_K4X16) {
    constexpr size_t RUNS = 50;
    param::MatrixMul param;
    param.transposeA = false;
    param.transposeB = false;
    Benchmarker<MatrixMul> benchmarker_K4X16(handle());
    Benchmarker<MatrixMul> benchmarker_K12X8(handle());
    benchmarker_K4X16.set_times(RUNS)
            .set_dtype(0, dtype::Float32{})
            .set_dtype(1, dtype::Float32{})
            .set_dtype(2, dtype::Float32{})
            .set_param(param)
            .set_display(false);
    benchmarker_K4X16.set_before_exec_callback(
            AlgoChecker<MatrixMul>("AARCH64_F32K4X16X1"));

    benchmarker_K12X8.set_before_exec_callback(
            AlgoChecker<MatrixMul>("AARCH64_F32K8X12X1"));
    benchmarker_K12X8.set_times(RUNS)
            .set_dtype(0, dtype::Float32{})
            .set_dtype(1, dtype::Float32{})
            .set_dtype(2, dtype::Float32{})
            .set_param(param)
            .set_display(false);

    auto run = [&](size_t M, size_t N, size_t K) {
        TensorShape A, B;
        if (param.transposeA) {
            A = TensorShape{K, M};
        } else {
            A = TensorShape{M, K};
        }
        if (param.transposeB) {
            B = TensorShape{N, K};
        } else {
            B = TensorShape{K, N};
        }

        auto k4x16_used = benchmarker_K4X16.exec({A, B, {}}) / RUNS;
        auto k12x8_used = benchmarker_K12X8.exec({A, B, {}}) / RUNS;
        float computations = 2.f * M * K * N * 1e-6;
        printf("run: {%zu{M} %zu{K} %zu{N}} k4x16: %f ms %f Gflops k12x8: %f "
               "ms "
               "%f Gflops k4x16_vs_k12x8: %f\n",
               M, K, N, k4x16_used, computations / k4x16_used, k12x8_used,
               computations / k12x8_used, k12x8_used / k4x16_used);
    };

    run(256, 256, 128);
    run(384, 384, 384);

    for (size_t k = 4; k <= 256; k *= 8) {
        for (size_t m = 4; m <= 256; m *= 4) {
            for (size_t n = 4; n <= 256; n *= 4) {
                run(m, n, k);
            }
            printf("\n");
        }
        printf("\n");
    }
}

TEST_F(AARCH64, BENCHMARK_MATRIX_MUL_INT16_8X8X8) {
    constexpr size_t RUNS = 50;
    param::MatrixMul param;
    param.transposeA = false;
    param.transposeB = false;
    Benchmarker<MatrixMul> benchmarker_int(handle());
    Benchmarker<MatrixMul> benchmarker_int32(handle());
    benchmarker_int.set_times(RUNS)
            .set_dtype(0, dtype::Int8{})
            .set_dtype(1, dtype::Int8{})
            .set_dtype(2, dtype::Int16{})
            .set_param(param)
            .set_display(false);
    benchmarker_int.set_before_exec_callback(
            AlgoChecker<MatrixMul>("AARCH64_INT8X8X16_K8X8X8"));

    benchmarker_int32.set_before_exec_callback(
            AlgoChecker<MatrixMul>("AARCH64_INT8X8X32_K8X8X8"));
    benchmarker_int32.set_times(RUNS)
            .set_dtype(0, dtype::Int8{})
            .set_dtype(1, dtype::Int8{})
            .set_dtype(2, dtype::Int32{})
            .set_param(param)
            .set_display(false);
    Benchmarker<MatrixMul> benchmarker_float(handle());
    benchmarker_float.set_param(param).set_display(false).set_times(RUNS);

    auto run = [&](size_t M, size_t N, size_t K) {
        TensorShape A, B;
        if (param.transposeA) {
            A = TensorShape{K, M};
        } else {
            A = TensorShape{M, K};
        }
        if (param.transposeB) {
            B = TensorShape{N, K};
        } else {
            B = TensorShape{K, N};
        }

        auto int_used = benchmarker_int.exec({A, B, {}}) / RUNS;
        auto float_used = benchmarker_float.exec({A, B, {}}) / RUNS;
        auto int32_used = benchmarker_int32.exec({A, B, {}}) / RUNS;
        float computations = 2.f * M * K * N * 1e-6;
        printf("run: {%zu{M} %zu{K} %zu{N}} float: %f ms %f Gflops int: %f ms "
               "%f Gflops speedup_vs_fp32: %f, speedup_vs_int32: %f\n",
               M, K, N, float_used, computations / float_used, int_used,
               computations / int_used, float_used / int_used,
               int32_used / int_used);
    };

    run(256, 256, 256);

    for (size_t k = 4; k <= 256; k *= 8) {
        for (size_t m = 4; m <= 256; m *= 4) {
            for (size_t n = 4; n <= 256; n *= 4) {
                run(m, n, k);
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
}

TEST_F(AARCH64, BENCHMARK_MATRIX_MUL_INT32_MK_4X4X16) {
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
            AlgoChecker<MatrixMul>("AARCH64_INT8X8X32_K4X4X16"));

    param.format = MatrixMul::Param::Format::MK4;
    benchmarker_mk4.set_before_exec_callback(
            AlgoChecker<MatrixMul>("AARCH64_INT8X8X32_MK4_4X4X16"));
    benchmarker_mk4.set_times(RUNS)
            .set_dtype(0, dtype::Int8{})
            .set_dtype(1, dtype::Int8{})
            .set_dtype(2, dtype::Int32{})
            .set_param(param)
            .set_display(false);

    auto run = [&](size_t M, size_t N, size_t K) {
        auto default_used = benchmarker.exec({{M, K}, {K, N}, {}}) / RUNS;
        auto mk_used = benchmarker_mk4.exec(
                               {{M / 4, K / 4, 4, 4}, {K / 4, N, 4}, {}}) /
                       RUNS;
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

TEST_F(AARCH64, BENCHMARK_MATRIX_MUL_MK4_8x8x16) {
    constexpr size_t RUNS = 50;
    param::MatrixMul param;
    param.transposeA = false;
    param.transposeB = false;
    Benchmarker<MatrixMul> benchmarker(handle());
    Benchmarker<MatrixMul> benchmarker_mk4(handle());
    Benchmarker<MatrixMul> benchmarker_mk4_16x12(handle());
    benchmarker.set_times(RUNS)
            .set_dtype(0, dtype::Int8{})
            .set_dtype(1, dtype::Int8{})
            .set_dtype(2, dtype::Int16{})
            .set_param(param)
            .set_display(false);
    benchmarker.set_before_exec_callback(
            AlgoChecker<MatrixMul>("AARCH64_INT8X8X16_K4X4X16"));

    param.format = MatrixMul::Param::Format::MK4;
    benchmarker_mk4.set_before_exec_callback(
            AlgoChecker<MatrixMul>("AARCH64_INT8X8X16_MK4_4X4X8"));
    benchmarker_mk4.set_times(RUNS)
            .set_dtype(0, dtype::Int8{})
            .set_dtype(1, dtype::Int8{})
            .set_dtype(2, dtype::Int16{})
            .set_param(param)
            .set_display(false);

    benchmarker_mk4_16x12.set_before_exec_callback(
            AlgoChecker<MatrixMul>("AARCH64_INT8X8X16_MK4_16X12X4"));
    benchmarker_mk4_16x12.set_times(RUNS)
            .set_dtype(0, dtype::Int8{})
            .set_dtype(1, dtype::Int8{})
            .set_dtype(2, dtype::Int16{})
            .set_param(param)
            .set_display(false);

    auto run = [&](size_t M, size_t N, size_t K) {
        auto default_used = benchmarker.exec({{M, K}, {K, N}, {}}) / RUNS;
        auto mk_used = benchmarker_mk4.exec(
                               {{M / 4, K / 4, 4, 4}, {K / 4, N, 4}, {}}) /
                       RUNS;
        auto mk4_16x12_used =
                benchmarker_mk4_16x12.exec(
                        {{M / 4, K / 4, 4, 4}, {K / 4, N, 4}, {}}) /
                RUNS;
        float computations = 2.f * M * K * N * 1e-6;
        printf("run: {%zu{M} %zu{K} %zu{N}} normal: %f ms %f Gflops mk4: %f ms "
               "%f Gflops speedup: %f, mk4_16x12 %f Gflops speedup: %f\n",
               M, K, N, default_used, computations / default_used, mk_used,
               computations / mk_used, default_used / mk_used,
               computations / mk4_16x12_used, default_used / mk4_16x12_used);
    };

    run(384, 384, 384);
}

TEST_F(AARCH64, BENCHMARK_4x4x16_vs_8x8x16) {
    constexpr size_t RUNS = 50;
    param::MatrixMul param;
    param.transposeA = false;
    param.transposeB = false;
    Benchmarker<MatrixMul> benchmarker(handle());
    Benchmarker<MatrixMul> benchmarker_int4_4x4x16(handle());
    benchmarker_int4_4x4x16.set_times(RUNS)
            .set_dtype(0, dtype::QuantizedS4{0.3})
            .set_dtype(1, dtype::QuantizedS4{0.3})
            .set_dtype(2, dtype::QuantizedS16{0.09})
            .set_param(param)
            .set_display(false);
    benchmarker.set_times(RUNS)
            .set_dtype(0, dtype::Int8{})
            .set_dtype(1, dtype::Int8{})
            .set_dtype(2, dtype::Int16{})
            .set_param(param)
            .set_display(false);
    benchmarker.set_before_exec_callback(
            AlgoChecker<MatrixMul>("AARCH64_INT8X8X16_K4X4X16"));

    auto run = [&](size_t M, size_t N, size_t K) {
        auto default_used = benchmarker.exec({{M, K}, {K, N}, {}}) / RUNS;
        auto int4416_used =
                benchmarker_int4_4x4x16.exec({{M, K}, {K, N}, {}}) / RUNS;
        float computations = 2.f * M * K * N * 1e-6;
        printf("run: {%zu{M} %zu{K} %zu{N}} normal 8x8x16 used: %f ms %f "
               "Gflops int4416 used %f int4416_gflops %f speedup %f\n",
               M, K, N, default_used, computations / default_used, int4416_used,
               computations / int4416_used, default_used / int4416_used);
    };

    for (int m = 32; m <= 1024; m += 32)
        for (int n = 32; n <= 1024; n += 32)
            for (int k = 32; k <= 512; k += 32)
                run(m, n, k);

    run(32, 32, 32);
    run(32, 32, 8);
    run(32, 32, 16);
    run(32, 32, 24);
    run(32 * 2, 32 * 2, 32);
    run(32 * 4, 32 * 4, 32);
    run(32 * 6, 32 * 6, 32);
    run(32 * 8, 32 * 8, 32);
    run(32 * 2, 32 * 2, 32 * 2);
    run(32 * 4, 32 * 4, 32 * 3);
    run(32 * 6, 32 * 6, 32 * 4);
    run(32 * 8, 32 * 8, 32 * 5);
    run(32 * 10, 32 * 10, 32 * 10);
    run(384, 384, 384);
    run(256, 256, 384);
    run(512, 512, 384);
    run(1024, 1024, 384);
}

TEST_F(AARCH64, BENCHMARK_MATRIX_MUL_MK4_8x8x8_8x8x16_vs_4x4x16_8x8x16) {
    constexpr size_t RUNS = 50;
    param::MatrixMul param;
    param.transposeA = false;
    param.transposeB = false;
    Benchmarker<MatrixMul> benchmarker(handle());
    Benchmarker<MatrixMul> benchmarker_mk4(handle());
    Benchmarker<MatrixMul> benchmarker_mk4_4x4x8(handle());
    benchmarker.set_times(RUNS)
            .set_dtype(0, dtype::Int8{})
            .set_dtype(1, dtype::Int8{})
            .set_dtype(2, dtype::Int16{})
            .set_param(param)
            .set_display(false);
    benchmarker.set_before_exec_callback(
            AlgoChecker<MatrixMul>("AARCH64_INT8X8X16_K4X4X16"));

    param.format = MatrixMul::Param::Format::MK4;
    benchmarker_mk4.set_before_exec_callback(
            AlgoChecker<MatrixMul>(
                "AARCH64_INT8X8X16_MK4_K8X8X8"
                ));
    benchmarker_mk4.set_times(RUNS)
            .set_dtype(0, dtype::Int8{})
            .set_dtype(1, dtype::Int8{})
            .set_dtype(2, dtype::Int16{})
            .set_param(param)
            .set_display(false);

    benchmarker_mk4_4x4x8.set_before_exec_callback(
            AlgoChecker<MatrixMul>("AARCH64_INT8X8X16_MK4_4X4X8"));
    benchmarker_mk4_4x4x8.set_times(RUNS)
            .set_dtype(0, dtype::Int8{})
            .set_dtype(1, dtype::Int8{})
            .set_dtype(2, dtype::Int16{})
            .set_param(param)
            .set_display(false);

    auto run = [&](size_t M, size_t N, size_t K) {
        auto default_used = benchmarker.exec({{M, K}, {K, N}, {}}) / RUNS;
        auto mk_used = benchmarker_mk4.exec(
                               {{M / 4, K / 4, 4, 4}, {K / 4, N, 4}, {}}) /
                       RUNS;
        auto mk4_4x4x8_used =
                benchmarker_mk4_4x4x8.exec(
                        {{M / 4, K / 4, 4, 4}, {K / 4, N, 4}, {}}) /
                RUNS;
        float computations = 2.f * M * K * N * 1e-6;
        printf("run: {%zu{M} %zu{K} %zu{N}} normal: %f ms %f Gflops mk4: %f ms "
               "%f Gflops speedup: %f, mk4_4x4x8 %f Gflops %f ms speedup: %f\n",
               M, K, N, default_used, computations / default_used, mk_used,
               computations / mk_used, default_used / mk_used,
               computations / mk4_4x4x8_used, mk4_4x4x8_used , mk4_4x4x8_used/mk_used);
    };

    run(384, 384, 384);
    run(512, 512, 512);
    run(1024, 1024, 384);
    run(256, 256, 384);
    for(int m = 32; m <= 512;m*=2)
    for(int n = 32; n <= 512;n*=2)
    for(int k = 32; k < 512;k*=2){
        run(m,n,k);
    }
}
TEST_F(AARCH64, BENCHMARK_MATRIX_MUL_INT16_4X4X16) {
    constexpr size_t RUNS = 50;
    param::MatrixMul param;
    param.transposeA = false;
    param.transposeB = false;
    Benchmarker<MatrixMul> benchmarker_int(handle());
    Benchmarker<MatrixMul> benchmarker_int32(handle());
    benchmarker_int.set_times(RUNS)
            .set_dtype(0, dtype::Int8{})
            .set_dtype(1, dtype::Int8{})
            .set_dtype(2, dtype::Int16{})
            .set_param(param)
            .set_display(false);
    benchmarker_int.set_before_exec_callback(
            AlgoChecker<MatrixMul>("AARCH64_INT8X8X16_K4X4X16"));

    benchmarker_int32.set_before_exec_callback(
            AlgoChecker<MatrixMul>("AARCH64_INT8X8X32_K4X4X16"));
    benchmarker_int32.set_times(RUNS)
            .set_dtype(0, dtype::Int8{})
            .set_dtype(1, dtype::Int8{})
            .set_dtype(2, dtype::Int32{})
            .set_param(param)
            .set_display(false);
    Benchmarker<MatrixMul> benchmarker_float(handle());
    benchmarker_float.set_param(param).set_display(false).set_times(RUNS);

    auto run = [&](size_t M, size_t N, size_t K) {
        TensorShape A, B;
        if (param.transposeA) {
            A = TensorShape{K, M};
        } else {
            A = TensorShape{M, K};
        }
        if (param.transposeB) {
            B = TensorShape{N, K};
        } else {
            B = TensorShape{K, N};
        }

        auto int_used = benchmarker_int.exec({A, B, {}}) / RUNS;
        auto float_used = benchmarker_float.exec({A, B, {}}) / RUNS;
        auto int32_used = benchmarker_int32.exec({A, B, {}}) / RUNS;
        float computations = 2.f * M * K * N * 1e-6;
        printf("run: {%zu{M} %zu{K} %zu{N}} float: %f ms %f Gflops int: %f ms "
               "%f Gflops speedup_vs_fp32: %f, speedup_vs_int32: %f\n",
               M, K, N, float_used, computations / float_used, int_used,
               computations / int_used, float_used / int_used,
               int32_used / int_used);
    };

    run(256, 256, 128);

    run(256, 256, 256);

    for (size_t k = 4; k <= 256; k *= 4) {
        for (size_t m = 4; m <= 256; m *= 4) {
            for (size_t n = 4; n <= 256; n *= 4) {
                run(m, n, k);
            }
        }
        std::cout << std::endl;
    }
}

TEST_F(AARCH64, BENCHMARK_GEMV) {
    int exec_times = 10;
    Benchmarker<MatrixMul> benchmarker_gemm(handle());
    benchmarker_gemm.set_times(exec_times);

    float mod = 1000 * exec_times / 1e9;
    auto run = [&](size_t M, size_t K, size_t N) {
        float time = 1.f, perf = 1.f;

        std::cout << "GEMM: (" << M << ", " << K << ", " << N << ")"
                  << std::endl;
        benchmarker_gemm.set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32());
        time = benchmarker_gemm.exec({{M, K}, {K, N}, {}});
        perf = 2.f * M * K * N / time * mod;
        std::cout << "gemm fp32, Performance is " << perf << " Gflops"
                  << std::endl;

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        benchmarker_gemm.set_dtype(0, dtype::Float16())
                .set_dtype(1, dtype::Float16());
        time = benchmarker_gemm.exec({{M, K}, {K, N}, {}});
        perf = 2.f * M * K * N / time * mod;
        std::cout << "gemm fp16, Performance is " << perf << " Gflops"
                  << std::endl;
#endif
    };

    std::cout << "warm up:\n";
    for (int i = 0; i < 50; i++) {
        benchmarker_gemm.set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .set_display(false)
                .exec({{256, 256}, {256, 256}, {}});
        benchmarker_gemm.set_display(true);
    }

    // run gemv
    for (size_t M : {1, 2, 3, 4, 5, 6, 7, 8, 64, 256})
        for (size_t K : {1, 2, 3, 4, 5, 6, 7, 8, 64, 256})
            for (size_t N : {112})
                run(M, K, N);
}

#if __ARM_FEATURE_DOTPROD
TEST_F(AARCH64, BENCHMARK_TRANSPOSED_MATRIX_MUL_INT_8X8X32) {
    constexpr size_t RUNS = 50;
    param::MatrixMul param;
    param.transposeA = param.transposeB = true;
    Benchmarker<MatrixMul> benchmarker_int(handle());
    benchmarker_int.set_times(RUNS)
            .set_dtype(0, dtype::Int8{})
            .set_dtype(1, dtype::Int8{})
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

TEST_F(AARCH64, BENCHMARK_GEMV_INT_8X8X32) {
    constexpr size_t RUNS = 50;
    param::MatrixMul param;
    Benchmarker<MatrixMul> benchmarker_int(handle());
    benchmarker_int.set_times(RUNS)
            .set_dtype(0, dtype::Int8{})
            .set_dtype(1, dtype::Int8{})
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

    for (size_t M : {1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 256})
        for (size_t N : {1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 256})
            for (size_t K : {1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64, 256})
                run(M, N, K);
}

TEST_F(AARCH64, BENCHMARK_MATRIX_MUL_INT8X8X32_MK4_8X12X4) {
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
            AlgoChecker<MatrixMul>("AARCH64_INT8X8X32_K8X12X4"));

    param.format = MatrixMul::Param::Format::MK4_DOT;
    benchmarker_mk4.set_before_exec_callback(
            AlgoChecker<MatrixMul>("AARCH64_INT8X8X32_MK4_8X12X4_DOTPROD"));
    benchmarker_mk4.set_times(RUNS)
            .set_dtype(0, dtype::Int8{})
            .set_dtype(1, dtype::Int8{})
            .set_dtype(2, dtype::Int32{})
            .set_param(param)
            .set_display(false);

    auto run = [&](size_t M, size_t N, size_t K) {
        auto default_used = benchmarker.exec({{M, K}, {K, N}, {}}) / RUNS;
        auto mk_used = benchmarker_mk4.exec(
                               {{M / 4, K / 4, 4, 4}, {K / 4, N, 4}, {}}) /
                       RUNS;
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
#endif  // __ARM_FEATURE_DOTPROD

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_F(AARCH64, BENCHMARK_MATRIX_MUL_F16_MK8) {
    auto args = matrix_mul::get_benchmark_matmul_mk_packed_args(8);
    matrix_mul::benchmark_with_contrast(
            handle(), args, dtype::Float16{}, dtype::Float16{},
            dtype::Float16{}, "AARCH64_F16_MK8_8X8",
            param::MatrixMul::Format::MK8, dtype::Float16{}, dtype::Float16{},
            dtype::Float16{}, "AARCH64_F16_K8X24X1");
}
#endif

TEST_F(AARCH64, BENCHMARK_MATRIX_MUL_INT16x16x32) {
    constexpr size_t RUNS = 50;
    Benchmarker<MatrixMul> benchmarker_int(handle());
    benchmarker_int.set_times(RUNS)
            .set_dtype(0, dtype::Int16{})
            .set_dtype(1, dtype::Int16{})
            .set_dtype(2, dtype::Int32{})
            .set_display(false);
    Benchmarker<MatrixMul> benchmarker_float(handle());
    benchmarker_float.set_display(false).set_times(RUNS);

    auto run = [&](size_t M, size_t N, size_t K, int mask) {
        param::MatrixMul param;
        param.transposeA = mask & 0x1;
        param.transposeB = mask & 0x2;
        benchmarker_int.set_param(param);
        benchmarker_float.set_param(param);
        TensorShape A, B;
        if (param.transposeA) {
            A = TensorShape{K, M};
        } else {
            A = TensorShape{M, K};
        }
        if (param.transposeB) {
            B = TensorShape{N, K};
        } else {
            B = TensorShape{K, N};
        }
        auto int_used = benchmarker_int.exec({A, B, {}}) / RUNS;
        auto float_used = benchmarker_float.exec({A, B, {}}) / RUNS;
        float computations = 2.f * M * K * N * 1e-6;
        printf("run: {%zu{M} %zu{K} %zu{N} %d{TA} %d{TB}} "
               "float: %f ms %f Gflops int: %f ms "
               "%f Gflops speedup: %f\n",
               M, K, N, param.transposeA, param.transposeB, float_used,
               computations / float_used, int_used, computations / int_used,
               float_used / int_used);
    };

    constexpr int mask = 4;
    for (auto i = 0; i < mask; i++) {
        for (size_t M : {8, 64, 112, 256}) {
            for (size_t K : {8, 64, 112, 256}) {
                for (size_t N : {8, 64, 112, 256}) {
                    run(M, N, K, i);
                }
            }
        }
    }
}

TEST_F(AARCH64, BENCHMARK_MATRIX_MUL_FP32_MK4) {
    auto args = matrix_mul::get_benchmark_matmul_mk_packed_args(16);
    matrix_mul::benchmark_with_contrast(
            handle(), args, dtype::Float32{}, dtype::Float32{},
            dtype::Float32{}, "AARCH64_F32_MK4_4x16",
            param::MatrixMul::Format::MK4, dtype::Float32{}, dtype::Float32{},
            dtype::Float32{});
}

TEST_F(AARCH64, BENCHMARK_MATRIX_MUL_FP32_PACK_MK4) {
    auto args = matrix_mul::get_benchmark_matmul_mk_packed_args(16);
    matrix_mul::benchmark_with_contrast(
            handle(), args, dtype::Float32{}, dtype::Float32{},
            dtype::Float32{}, "AARCH64_F32_MK4_K8X12X1",
            param::MatrixMul::Format::MK4, dtype::Float32{}, dtype::Float32{},
            dtype::Float32{}, "AARCH64_F32K8X12X1");
}

TEST_F(AARCH64, BENCHMARK_MATRIX_MUL_INT16x16x32_MK8) {
    auto args = matrix_mul::get_benchmark_matmul_mk_packed_args(8);
    matrix_mul::benchmark_with_contrast(
            handle(), args, dtype::Int16{}, dtype::Int16{}, dtype::Int32{},
            "AARCH64_INT16X16X32_MK8_8X8", param::MatrixMul::Format::MK8,
            dtype::Int16{}, dtype::Int16{}, dtype::Int32{});
}

TEST_F(AARCH64, BENCHMARK_MATRIX_MUL_FP32_K8X12) {
    constexpr size_t RUNS = 50;
    param::MatrixMul param;
    param.transposeA = param.transposeB = true;
    Benchmarker<MatrixMul> benchmarker_k12x8(handle());
    Benchmarker<MatrixMul> benchmarker_k8x12(handle());
    benchmarker_k12x8.set_param(param).set_display(false).set_times(RUNS);
    benchmarker_k8x12.set_param(param).set_display(false).set_times(RUNS);
    benchmarker_k12x8.set_before_exec_callback(
            AlgoChecker<MatrixMul>("AARCH64_F32K4X16X1"));

    benchmarker_k8x12.set_before_exec_callback(
            AlgoChecker<MatrixMul>("AARCH64_F32K8X12X1"));

    auto run = [&](size_t M, size_t N, size_t K) {
        auto k12x8_used = benchmarker_k12x8.exec({{K, M}, {N, K}, {}}) / RUNS;
        auto k8x12_used = benchmarker_k8x12.exec({{K, M}, {N, K}, {}}) / RUNS;
        float computations = 2.f * M * K * N * 1e-6;
        printf("run: {%zu{M} %zu{K} %zu{N}} float k12x8: %f ms %f Gflops "
               "k8x12: %f ms "
               "%f Gflops speedup: %f\n",
               M, K, N, k12x8_used, computations / k12x8_used, k8x12_used,
               computations / k8x12_used, k12x8_used / k8x12_used);
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

TEST_F(AARCH64, BENCHMARK_MATRIX_MUL_FP32_K8X12_NO_TRANS) {
    constexpr size_t RUNS = 50;
    param::MatrixMul param;
    param.transposeA = param.transposeB = false;
    Benchmarker<MatrixMul> benchmarker_k12x8(handle());
    Benchmarker<MatrixMul> benchmarker_k8x12(handle());
    benchmarker_k12x8.set_param(param).set_display(false).set_times(RUNS);
    benchmarker_k8x12.set_param(param).set_display(false).set_times(RUNS);
    benchmarker_k12x8.set_before_exec_callback(
            AlgoChecker<MatrixMul>("AARCH64_F32K4X16X1"));

    benchmarker_k8x12.set_before_exec_callback(
            AlgoChecker<MatrixMul>("AARCH64_F32K8X12X1"));

    auto run = [&](size_t M, size_t N, size_t K) {
        auto k12x8_used = benchmarker_k12x8.exec({{M, K}, {K, N}, {}}) / RUNS;
        auto k8x12_used = benchmarker_k8x12.exec({{M, K}, {K, N}, {}}) / RUNS;
        float computations = 2.f * M * K * N * 1e-6;
        printf("run: {%zu{M} %zu{K} %zu{N}} float k12x8: %f ms %f Gflops "
               "k8x12: %f ms "
               "%f Gflops speedup: %f\n",
               M, K, N, k12x8_used, computations / k12x8_used, k8x12_used,
               computations / k8x12_used, k12x8_used / k8x12_used);
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

#endif  // MEGDNN_WITH_BENCHMARK

// vim: syntax=cpp.doxygen
