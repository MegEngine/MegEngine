/**
 * \file dnn/test/x86/matrix_mul.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#include "test/x86/fixture.h"

#include "src/x86/utils.h"
#include "test/common/benchmarker.h"
#include "test/common/checker.h"
#include "test/common/matrix_mul.h"
#include "test/common/rng.h"
using namespace megdnn;
using namespace test;
using namespace megdnn::x86;

#if MEGDNN_X86_WITH_VNNI
TEST_F(X86, MATRIX_MUL_VNNI_8X8X32) {
    matrix_mul::check_matrix_mul(dtype::Int8{}, dtype::Int8{}, dtype::Int32{},
                                 handle(), "X86_INT8X8X32_VNNI");
}
#endif

#if MEGDNN_X86_WITH_MKL_DNN
TEST_F(X86, MATRIX_MUL_MKLDNN_8X8X32) {
    if (is_supported(SIMDType::VNNI)) {
        matrix_mul::check_matrix_mul(dtype::Int8{}, dtype::Int8{},
                                     dtype::Int32{}, handle(),
                                     "X86_INT8X8X32_MKLDNN");
    } else {
        std::cout << "can not do mkldnn matmul check for no vnni support"
                  << std::endl;
        matrix_mul::check_matrix_mul(dtype::Int8{}, dtype::Int8{},
                                     dtype::Int32{}, handle());
    }
}
#endif
//! FIXME: need to add tests of GEMV and QUINT8
TEST_F(X86, MATRIX_MUL_AVX2_8X8X32) {
    matrix_mul::check_matrix_mul(dtype::Int8{}, dtype::Int8{}, dtype::Int32{},
                                 handle(), "X86_INT8X8X32_AVX2_2X4X16");
    matrix_mul::check_matrix_mul(dtype::Int8{}, dtype::Int8{}, dtype::Int32{},
                                 handle(), "X86_INT8X8X32_AVX2_4X16X2");
}
TEST_F(X86, MATRIX_MUL_AVX2_8X8X16) {
    matrix_mul::check_matrix_mul(dtype::Int8{}, dtype::Int8{}, dtype::Int16{},
                                 handle(), "X86_INT8X8X16_AVX2");
}
TEST_F(X86, MATRIX_MUL_SSE_8X8X16) {
    matrix_mul::check_matrix_mul(dtype::Int8{}, dtype::Int8{}, dtype::Int16{},
                                 handle(), "X86_INT8X8X16_SSE");
}
TEST_F(X86, MATRIX_MUL_SSE_8X8X32) {
    matrix_mul::check_matrix_mul(dtype::Int8{}, dtype::Int8{}, dtype::Int32{},
                                 handle(), "X86_INT8X8X32_SSE_4X8X2");
}

#if MEGDNN_X86_WITH_MKL && SUPPORT_MKL_PACKED_GEMM
TEST_F(X86, MATRIX_MUL_MKL_PACKA) {
    matrix_mul::check_matrix_mul(dtype::Float32{}, dtype::Float32{},
                                 dtype::Float32{}, handle(),
                                 "X86_F32_MKL_PACKA");
}
#endif

TEST_F(X86, MATRIX_MUL_AVX2_MK8_8X8) {
    matrix_mul::check_matrix_mul(dtype::Float32{}, dtype::Float32{},
                                 dtype::Float32{}, handle(), "X86_F32MK8_8X8",
                                 param::MatrixMul::Format::MK8, 1);
}

#if MEGDNN_WITH_BENCHMARK

TEST_F(X86, BENCHMARK_MATRIX_MUL_AVX2_MK8_8X8) {
    auto args = matrix_mul::get_benchmark_matmul_mk_packed_args(8);
    matrix_mul::benchmark_with_contrast(
            handle(), args, dtype::Float32{}, dtype::Float32{},
            dtype::Float32{}, "X86_F32MK8_8X8", param::MatrixMul::Format::MK8,
            dtype::Float32{}, dtype::Float32{}, dtype::Float32{},
            "X86_F32_BLAS");
}

TEST_F(X86, BENCHMARK_MATRIX_MUL_8X8X32) {
    constexpr size_t RUNS = 50;
    auto rng = std::make_unique<UniformIntRNG>(-127, 127);
#if MEGDNN_X86_WITH_VNNI
    Benchmarker<MatrixMul> benchmarker_vnni(handle());
    benchmarker_vnni.set_times(RUNS)
            .set_dtype(0, dtype::Int8{})
            .set_dtype(1, dtype::Int8{})
            .set_dtype(2, dtype::Int32{})
            .set_display(false)
            .set_rng(0, rng.get())
            .set_rng(1, rng.get());
    benchmarker_vnni.set_before_exec_callback(
            AlgoChecker<MatrixMul>("X86_INT8X8X32_VNNI"));
#endif

#if MEGDNN_X86_WITH_MKL_DNN
    Benchmarker<MatrixMul> benchmarker_mkldnn(handle());
    benchmarker_mkldnn.set_times(RUNS)
            .set_dtype(0, dtype::Int8{})
            .set_dtype(1, dtype::Int8{})
            .set_dtype(2, dtype::Int32{})
            .set_display(false)
            .set_rng(0, rng.get())
            .set_rng(1, rng.get());
    benchmarker_mkldnn.set_before_exec_callback(
            AlgoChecker<MatrixMul>("X86_INT8X8X32_MKLDNN"));
#endif
    Benchmarker<MatrixMul> benchmarker_avx2_4x16x2(handle());
    benchmarker_avx2_4x16x2.set_display(false)
            .set_times(RUNS)
            .set_dtype(0, dtype::Int8{})
            .set_dtype(1, dtype::Int8{})
            .set_dtype(2, dtype::Int32{})
            .set_rng(0, rng.get())
            .set_rng(1, rng.get());
    benchmarker_avx2_4x16x2.set_before_exec_callback(
            AlgoChecker<MatrixMul>("X86_INT8X8X32_AVX2_4X16X2"));

    Benchmarker<MatrixMul> benchmarker_avx2_4x16x2_8816(handle());
    benchmarker_avx2_4x16x2_8816.set_display(false)
            .set_times(RUNS)
            .set_dtype(0, dtype::Int8{})
            .set_dtype(1, dtype::Int8{})
            .set_dtype(2, dtype::Int16{})
            .set_rng(0, rng.get())
            .set_rng(1, rng.get());
    benchmarker_avx2_4x16x2_8816.set_before_exec_callback(
            AlgoChecker<MatrixMul>("X86_INT8X8X16_AVX2"));

    Benchmarker<MatrixMul> benchmarker_sse_4x8x2_8816(handle());
    benchmarker_sse_4x8x2_8816.set_display(false)
            .set_times(RUNS)
            .set_dtype(0, dtype::Int8{})
            .set_dtype(1, dtype::Int8{})
            .set_dtype(2, dtype::Int16{})
            .set_rng(0, rng.get())
            .set_rng(1, rng.get());
    benchmarker_sse_4x8x2_8816.set_before_exec_callback(
            AlgoChecker<MatrixMul>("X86_INT8X8X16_SSE"));

    Benchmarker<MatrixMul> benchmarker_avx2_2x4x16(handle());
    benchmarker_avx2_2x4x16.set_display(false)
            .set_times(RUNS)
            .set_dtype(0, dtype::Int8{})
            .set_dtype(1, dtype::Int8{})
            .set_dtype(2, dtype::Int32{})
            .set_rng(0, rng.get())
            .set_rng(1, rng.get());
    benchmarker_avx2_2x4x16.set_before_exec_callback(
            AlgoChecker<MatrixMul>("X86_INT8X8X32_AVX2_2X4X16"));

    Benchmarker<MatrixMul> benchmarker_sse_4x8x2(handle());
    benchmarker_sse_4x8x2.set_display(false)
            .set_times(RUNS)
            .set_dtype(0, dtype::Int8{})
            .set_dtype(1, dtype::Int8{})
            .set_dtype(2, dtype::Int32{})
            .set_rng(0, rng.get())
            .set_rng(1, rng.get());
    benchmarker_sse_4x8x2.set_before_exec_callback(
            AlgoChecker<MatrixMul>("X86_INT8X8X32_SSE_4X8X2"));

    Benchmarker<MatrixMul> benchmarker_float(handle());
    benchmarker_float.set_display(false)
            .set_times(RUNS)
            .set_rng(0, rng.get())
            .set_rng(1, rng.get());
    benchmarker_float.set_before_exec_callback(
            AlgoChecker<MatrixMul>("X86_F32_BLAS"));

    auto run = [&](size_t M, size_t N, size_t K) {
        const float computations = 2.f * M * K * N * 1e-6;
        std::cout << "run : {" << M << "," << N << "," << K << "} ";
        auto float_used = benchmarker_float.exec({{M, K}, {K, N}, {}}) / RUNS;
        std::cout << "float: " << float_used << " ms, "
                  << computations / float_used << " Gflops, ";

#if MEGDNN_X86_WITH_VNNI
        if (is_supported(SIMDType::VNNI)) {
            auto vnni_used = benchmarker_vnni.exec({{M, K}, {K, N}, {}}) / RUNS;
            std::cout << "vnni: " << vnni_used << " ms, "
                      << computations / vnni_used << " Gflops, "
                      << "speed_up " << float_used / vnni_used << ", ";
        }
#endif

#if MEGDNN_X86_WITH_MKL_DNN
        if (is_supported(SIMDType::VNNI)) {
            auto mkldnn_used =
                    benchmarker_mkldnn.exec({{M, K}, {K, N}, {}}) / RUNS;
            std::cout << "mkldnn: " << mkldnn_used << " ms, "
                      << computations / mkldnn_used << " Gflops, "
                      << "speed_up " << float_used / mkldnn_used << ", ";
        }

#endif
        if (is_supported(SIMDType::AVX2)) {
            auto avx2_used_4x16x2 =
                    benchmarker_avx2_4x16x2.exec({{M, K}, {K, N}, {}}) / RUNS;
            auto avx2_used_2x4x16 =
                    benchmarker_avx2_2x4x16.exec({{M, K}, {K, N}, {}}) / RUNS;
            std::cout << "avx2_k2: " << avx2_used_4x16x2
                      << " ms, k2 throughput "
                      << computations / avx2_used_4x16x2 << " Gflops, "
                      << "k2_speed_up " << float_used / avx2_used_4x16x2
                      << ", k16_speed_up " << float_used / avx2_used_2x4x16
                      << ",";
            auto avx2_used_4x16x2_8816 =
                    benchmarker_avx2_4x16x2_8816.exec({{M, K}, {K, N}, {}}) /
                    RUNS;
            std::cout << "avx2_8816: " << avx2_used_4x16x2_8816
                      << " ms, 8816 throughput "
                      << computations / avx2_used_4x16x2_8816 << " Gflops,";
        }
        if (is_supported(SIMDType::SSE4_1)) {
            auto sse_used =
                    benchmarker_sse_4x8x2.exec({{M, K}, {K, N}, {}}) / RUNS;
            std::cout << "sse: " << sse_used << " ms, "
                      << computations / sse_used << " Gflops, "
                      << "speed_up " << float_used / sse_used << ", ";
            auto sse_used_8816 =
                    benchmarker_sse_4x8x2_8816.exec({{M, K}, {K, N}, {}}) /
                    RUNS;
            std::cout << "sse_8816: " << sse_used_8816 << " ms, "
                      << computations / sse_used_8816 << " Gflops, ";
        }
        std::cout << std::endl;
    };
    run(256, 256, 256);

    for (size_t M : {8, 64, 112, 256, 512}) {
        for (size_t K : {8, 16, 32, 64, 112, 256, 512}) {
            for (size_t N : {8, 64, 112, 256, 512}) {
                run(M, N, K);
            }
        }
    }
}

#endif  // MEGDNN_WITH_BENCHMARK

// vim: syntax=cpp.doxygen
