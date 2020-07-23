/**
 * \file dnn/test/common/matrix_mul.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#include "src/common/utils.h"
#include "test/common/benchmarker.h"
#include "test/common/checker.h"
#include "test/common/matrix_mul.h"

using namespace megdnn;
using namespace test;

std::vector<matrix_mul::TestArg> matrix_mul::get_matmul_args_no_mask() {
    std::vector<TestArg> args;

    for (size_t m : {1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 15, 16, 32})
        for (size_t n : {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
                         14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 32})
            for (size_t k : {1, 2, 4, 8, 11, 12, 15, 16, 31, 32, 37})
                args.emplace_back(m, n, k, 0);

    for (size_t m : {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17})
        args.emplace_back(m, m + 1, m + 2, 0);
    for (size_t mbase : {11})
        for (size_t test_case_offset : {64, 256, 512}) {
            size_t mnk = mbase + test_case_offset;
            args.emplace_back(mnk, mnk, mnk, 0);
            return args;
        }
    return args;
}

std::vector<matrix_mul::TestArg> matrix_mul::get_matmul_mk_packed_args(
        size_t nbase) {
    std::vector<TestArg> args;
    for (size_t m : {1, 2, 3, 4, 5, 6, 7, 8, 11})
        for (size_t n : {1, 2, 3, 4, 5, 8, 12, 16, 24})
            for (size_t k : {1, 2, 3, 4, 5, 9, 10, 11})
                args.emplace_back(m, n * nbase, k, 0);
    return args;
}

std::vector<matrix_mul::TestArg>
matrix_mul::get_batched_matmul_args_cublaslt() {
    std::vector<TestArg> args;
    for (size_t m : {4, 6, 8, 16}) {
        for (size_t n : {4, 6, 8, 16}) {
            //[TODO]: the following test case are disabled due to the
            // cublasLt(version: 10020) produce wrong result when k in [65, 97],
            // so please uncomment it if the bug is fixed

            for (size_t k : {32, 64}) {
                args.emplace_back(m, n, k, 0, 0, 0, 0, 2);
            }
        }
    }
    return args;
}

std::vector<matrix_mul::TestArg>
matrix_mul::get_batched_matmul_args_int8x8x32() {
    std::vector<TestArg> args;
    for (size_t m : {1, 2, 3, 4, 5, 8, 64}) {
        for (size_t n : {1, 2, 3, 4, 5, 8, 64}) {
            for (size_t k : {1, 2, 3, 4, 5, 8, 64}) {
                args.emplace_back(m, n, k, 0, 0, 0, 0, 2);
            }
        }
    }
    return args;
}

std::vector<matrix_mul::TestArg> matrix_mul::get_matmul_args_mask(
        uint8_t mask) {
    std::vector<TestArg> args;

    std::vector<TestArg> args_temp = matrix_mul::get_matmul_args_no_mask();
    for (auto arg : args_temp) {
        arg.mask = mask;
        args.emplace_back(arg);
    }

    // non-contiguous case
    for (size_t m : {110})
        for (size_t n : {119})
            for (size_t k : {120}) {
                // A: (m, k)
                size_t Astride = mask & 1 ? m + 2 : k + 2;
                // B: (k, n)
                size_t Bstride = mask & 2 ? k + 2 : n + 2;
                size_t Cstride = n + 2;
                args.emplace_back(m, n, k, mask, Astride, Bstride, Cstride);
            }
    return args;
}

std::vector<matrix_mul::TestArg> matrix_mul::get_matmul_args() {
    std::vector<TestArg> args;
    for (size_t mask = 0; mask < 4; ++mask) {
        std::vector<TestArg> args_temp = matrix_mul::get_matmul_args_mask(mask);
        for (auto arg : args_temp)
            args.emplace_back(arg);
    }
    return args;
}

std::vector<matrix_mul::TestArg> matrix_mul::get_batched_matmul_args_mask(
        uint8_t mask) {
    std::vector<TestArg> args;
    for (size_t b : {1, 2, 3}) {
        std::vector<TestArg> args_temp =
                megdnn::test::matrix_mul::get_matmul_args_mask(mask);
        for (auto arg : args_temp) {
            arg.b = b;
            args.emplace_back(arg);
        }
    }
    return args;
}

std::vector<matrix_mul::TestArg> matrix_mul::get_batched_matmul_args() {
    std::vector<TestArg> args;
    for (size_t mask = 0; mask < 4; ++mask) {
        std::vector<TestArg> args_temp =
                matrix_mul::get_batched_matmul_args_mask(mask);
        for (auto arg : args_temp)
            args.emplace_back(arg);
    }
    return args;
}

template <typename Opr>
void matrix_mul::check_matrix_mul(DType A_dtype, DType B_dtype, DType C_dtype,
                                  Handle* handle, const char* algo,
                                  param::MatrixMul::Format format, size_t nbase,
                                  float eps, std::vector<TestArg>&& user_args) {
    megdnn_assert(A_dtype.enumv() == B_dtype.enumv());
    Checker<Opr> checker(handle);
    if (algo) {
        checker.set_before_exec_callback(AlgoChecker<Opr>(algo));
    }
    std::unique_ptr<RNG> rng;
    checker.set_epsilon(eps);
    if (A_dtype.enumv() == DTypeEnum::Int8 ||
        A_dtype.enumv() == DTypeEnum::QuantizedS8) {
        //! use larger rng to check the overflow
        rng = std::make_unique<UniformIntRNG>(-127, 127);
    } else if (A_dtype.enumv() == DTypeEnum::Uint8 ||
               A_dtype.enumv() == DTypeEnum::Quantized8Asymm) {
        rng = std::make_unique<NormalRNG>(128.f);
    } else if (A_dtype.enumv() == DTypeEnum::Int16) {
        rng = std::make_unique<UniformIntRNG>(-32767, 32767);
    } else if (A_dtype.enumv() == DTypeEnum::Float16) {
        rng = std::make_unique<NormalRNG>(2.f);
        //! if fp16 not set eps, default 1e-3, we just set it to 1e-2
        if (eps < 1e-2) {
            checker.set_epsilon(1e-2);
        }
    }

    if (rng) {
        checker.set_rng(0, rng.get()).set_rng(1, rng.get());
    }

    //! return expect if stride == 0, stride otherwise
    auto stride_val = [](size_t stride, size_t expect) -> size_t {
        if (stride == 0) {
            return expect;
        } else {
            return stride;
        }
    };

    constexpr static bool batched =
            std::is_same<Opr, megdnn::BatchedMatrixMul>::value;
    using Param = MatrixMul::Param;
    std::vector<TestArg> args;
    if (user_args.empty()) {
        if (format == param::MatrixMul::Format::DEFAULT) {
            if (batched) {
                args = matrix_mul::get_batched_matmul_args();
            } else {
                args = matrix_mul::get_matmul_args();
            }

        } else {
            megdnn_assert(!batched,
                          "BatchedMatrixMul does not support MK4/MK8");
            args = matrix_mul::get_matmul_mk_packed_args(nbase);
        }
    } else {
        args = user_args;
    }
    size_t pack_size = MatrixMulForward::pack_size(format);
    for (auto& arg : args) {
        size_t m = arg.m, n = arg.n, k = arg.k;

#if MEGDNN_WITH_CUDA
        //[NOTE]: cublas can only process 4B aligned 8-bit input matrix;
        bool is_dt_8bit = A_dtype.enumv() == DTypeEnum::Int8 ||
                          A_dtype.enumv() == DTypeEnum::QuantizedS8 ||
                          A_dtype.enumv() == DTypeEnum::Uint8 ||
                          A_dtype.enumv() == DTypeEnum::Quantized8Asymm;
        if (is_dt_8bit && ((m % 4 != 0) || (n % 4 != 0))) {
            continue;
        }
#endif

        Param param;
        param.transposeA = arg.mask & 0x1;
        param.transposeB = arg.mask & 0x2;
        param.format = format;
        checker.set_dtype(0, A_dtype)
                .set_dtype(1, B_dtype)
                .set_dtype(2, C_dtype);
        size_t A0 = m, A1 = k, B0 = k, B1 = n;
        TensorShape A, B;
        if (param.transposeA) {
            std::swap(A0, A1);
        }
        if (param.transposeB) {
            std::swap(B0, B1);
        }
        ptrdiff_t A_stride = arg.A_stride, B_stride = arg.B_stride,
                  C_stride = arg.C_stride, A_batch_stride = arg.A_batch_stride,
                  B_batch_stride = arg.B_batch_stride,
                  C_batch_stride = arg.C_batch_stride;
        A_stride = stride_val(A_stride, A1);
        B_stride = stride_val(B_stride, B1);
        C_stride = stride_val(C_stride, n);
        A_batch_stride = stride_val(A_batch_stride, A0 * A_stride);
        B_batch_stride = stride_val(B_batch_stride, B0 * B_stride);
        C_batch_stride = stride_val(C_batch_stride, m * C_stride);

        checker.set_param(param);
        if (format == param::MatrixMul::Format::DEFAULT) {
            if (batched) {
                checker.execl({TensorLayout{{arg.b, A0, A1},
                                            {A_batch_stride, A_stride, 1},
                                            A_dtype},
                               TensorLayout{{arg.b, B0, B1},
                                            {B_batch_stride, B_stride, 1},
                                            B_dtype},
                               TensorLayout{{arg.b, m, n},
                                            {C_batch_stride, C_stride, 1},
                                            C_dtype}});
            } else {
                checker.execl({TensorLayout{{A0, A1}, {A_stride, 1}, A_dtype},
                               TensorLayout{{B0, B1}, {B_stride, 1}, B_dtype},
                               TensorLayout{{m, n}, {C_stride, 1}, C_dtype}});
            }
        } else {
            //! ignore non-contiguous, only DEFAULT format support
            //! non-contiguous input
            checker.execs(
                    {{A0, A1, pack_size, pack_size}, {B0, B1, pack_size}, {}});
        }
    }
}

void matrix_mul::check_batched_matrix_mul(DType A_dtype, DType B_dtype,
                                          DType C_dtype, Handle* handle,
                                          const char* algo, float eps,
                                          std::vector<TestArg>&& args) {
    check_matrix_mul<megdnn::BatchedMatrixMul>(
            A_dtype, B_dtype, C_dtype, handle, algo,
            param::MatrixMul::Format::DEFAULT, 8, eps,
            std::forward<decltype(args)>(args));
}

void matrix_mul::check_matrix_mul(DType A_dtype, DType B_dtype, DType C_dtype,
                                  Handle* handle, const char* algo,
                                  param::MatrixMul::Format format, size_t nbase,
                                  float eps) {
    check_matrix_mul<megdnn::MatrixMul>(A_dtype, B_dtype, C_dtype, handle, algo,
                                        format, nbase, eps);
}

#if MEGDNN_WITH_BENCHMARK
std::vector<matrix_mul::TestArg> matrix_mul::get_benchmark_matmul_args() {
    std::vector<matrix_mul::TestArg> args;
    args.emplace_back(256, 12 * 24, 256, 0);

    //////////////////////// gemv //////////////////////////
    for (size_t M : {8, 64, 112, 256}) {
        for (size_t K : {8, 64, 112, 256}) {
            args.emplace_back(M, 1, K, 0);
        }
    }

    //////////////////////// gemm //////////////////////////
    for (size_t M : {8, 64, 112, 256}) {
        for (size_t K : {8, 16, 32, 64, 112, 256}) {
            for (size_t N : {8, 64, 112, 256}) {
                args.emplace_back(M, N, K, 0);
            }
        }
    }
    return args;
}

std::vector<matrix_mul::TestArg>
matrix_mul::get_benchmark_matmul_mk_packed_args(size_t nbase) {
    std::vector<TestArg> args;
    for (size_t m : {2, 4, 8, 16, 24, 32, 64})
        for (size_t n : {1, 2, 3, 4, 8, 16, 32, 64})
            for (size_t k : {2, 4, 8, 16, 24, 32, 64})
                args.emplace_back(m, n * nbase, k, 0);
    return args;
}

void matrix_mul::benchmark_with_contrast(
        Handle* handle, const std::vector<TestArg>& args, DType A_dtype,
        DType B_dtype, DType C_dtype, const char* algo,
        param::MatrixMul::Format format, DType contrast_A_dtype,
        DType contrast_B_dtype, DType contrast_C_dtype,
        const char* contrast_algo, param::MatrixMul::Format contrast_format) {
    using Param = MatrixMul::Param;

    megdnn_assert(A_dtype.enumv() == B_dtype.enumv());
    megdnn_assert(contrast_A_dtype.enumv() == contrast_B_dtype.enumv());
    Benchmarker<MatrixMul> benchmark_contrast(handle);
    Benchmarker<MatrixMul> benchmark(handle);
    constexpr size_t RUNS = 50;
    if (algo) {
        benchmark.set_before_exec_callback(AlgoChecker<MatrixMul>(algo));
    }
    if (contrast_algo) {
        benchmark_contrast.set_before_exec_callback(
                AlgoChecker<MatrixMul>(contrast_algo));
    }
    benchmark.set_dtype(0, A_dtype).set_dtype(1, B_dtype).set_dtype(2, C_dtype);
    benchmark.set_times(RUNS);
    benchmark_contrast.set_dtype(0, contrast_A_dtype)
            .set_dtype(1, contrast_B_dtype)
            .set_dtype(2, contrast_C_dtype);
    benchmark_contrast.set_times(RUNS);

    auto bench = [](Benchmarker<MatrixMul>& benchmark, Param param,
                    param::MatrixMul::Format format, size_t m, size_t n,
                    size_t k, size_t pack_size) -> float {
        param.format = format;
        benchmark.set_param(param);
        float used_algo = 1.0;
        if (format == param::MatrixMul::Format::DEFAULT) {
            size_t A0 = m * pack_size, A1 = k * pack_size, B0 = k * pack_size,
                   B1 = n;
            TensorShape A, B;
            if (param.transposeA) {
                std::swap(A0, A1);
            }
            if (param.transposeB) {
                std::swap(B0, B1);
            }
            used_algo = benchmark.execs({{A0, A1}, {B0, B1}, {}}) / RUNS;
        } else {
            size_t A0 = m, A1 = k, B0 = k, B1 = n;
            if (param.transposeA) {
                std::swap(A0, A1);
            }
            if (param.transposeB) {
                std::swap(B0, B1);
            }

            used_algo = benchmark.execs({{A0, A1, pack_size, pack_size},
                                         {B0, B1, pack_size},
                                         {}}) /
                        RUNS;
        }
        return used_algo;
    };

    size_t mk_size = MatrixMulForward::pack_size(format);
    size_t mk_size_contrast = MatrixMulForward::pack_size(contrast_format);
    size_t pack_size = std::max(mk_size, mk_size_contrast);
    for (auto& arg : args) {
        Param param;
        param.transposeA = arg.mask & 0x1;
        param.transposeB = arg.mask & 0x2;

        auto used_contrast = bench(benchmark_contrast, param, contrast_format,
                                   arg.m, arg.n, arg.k, pack_size);
        auto used_algo =
                bench(benchmark, param, format, arg.m, arg.n, arg.k, pack_size);

        float computations =
                2.f * arg.m * pack_size * arg.k * pack_size * arg.n * 1e-6;
        printf("run: {(%zu, %zu) x (%zu, %zu)} contrast: %f ms %f Gflops %s: "
               "%f "
               "ms "
               "%f Gflops "
               "speedup: %f \n",
               arg.m * pack_size, arg.k * pack_size, arg.k * pack_size, arg.n,
               used_contrast, computations / used_contrast, algo, used_algo,
               computations / used_algo, used_contrast / used_algo);
    }
}

#endif

// vim: syntax=cpp.doxygen
