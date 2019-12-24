/**
 * \file dnn/test/cuda/matrix_mul.cpp
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
#include "test/common/benchmarker.h"

#include "src/cuda/utils.h"
#if defined(cuda_check)
#undef cuda_check
#endif
#include "test/cuda/utils.h"

#include <cuda.h>

namespace megdnn {
namespace test {

#if CUDA_VERSION >= 10000
TEST_F(CUDA, MATRIX_MUL_QUANTIZED4x4x32_EXCEPTION) {
    if (cuda::current_device_prop().major > 7 ||
        (cuda::current_device_prop().major == 7 &&
         cuda::current_device_prop().minor >= 5)) {
        printf("Skip CUDA.MATRIX_MUL_QUANTIZED4x4x32_EXCEPTION test as current "
               "device support wmma intrinsics\n");
        return;
    }

    Checker<MatrixMul> checker(handle_cuda(), false);
    using Param = MatrixMul::Param;
    Param param;
    param.transposeB = true;
    checker.set_param(param);
    checker.set_dtype(0, dtype::Quantized4Asymm(1.3f, (uint8_t)3));
    checker.set_dtype(1, dtype::Quantized4Asymm(1.3f, (uint8_t)3));
    checker.set_dtype(2, dtype::QuantizedS32(1.3f * 1.3f));
    ASSERT_THROW(checker.exec({{256, 256}, {256, 256}, {256, 256}}),
                 MegDNNError);
}

TEST_F(CUDA, MATRIX_MUL_QUANTIZED4x4x32) {
    if (cuda::current_device_prop().major < 7 ||
        (cuda::current_device_prop().major == 7 &&
         cuda::current_device_prop().minor < 5)) {
        printf("Skip CUDA.MATRIX_MUL_QUANTIZED4x4x32 test as current device doesn't support\n");
        return;
    }
    Checker<MatrixMul> checker(handle_cuda(), false);
    using Param = MatrixMul::Param;
    Param param;
    param.transposeB = true;
    checker.set_param(param);
    checker.set_dtype(0, dtype::Quantized4Asymm(1.3f, (uint8_t)3));
    checker.set_dtype(1, dtype::Quantized4Asymm(1.3f, (uint8_t)3));
    checker.set_dtype(2, dtype::QuantizedS32(1.3f*1.3f));
    checker.exec({{256, 256}, {256, 256}, {256, 256}});
    auto args = matrix_mul::get_matmul_args();
    for (auto arg : args) {
        size_t m = DIVUP(arg.m, 8) * 8, n = DIVUP(arg.n, 8) * 8,
               k = DIVUP(arg.k, 32) * 32;
        checker.exec({{m, k}, {n, k}, {m, n}});
    }
}

#if MEGDNN_WITH_BENCHMARK
TEST_F(CUDA, BENCHMARK_MATRIX_MUL_QUANTIZED4x4x32) {
    if (cuda::current_device_prop().major < 7 ||
        (cuda::current_device_prop().major == 7 &&
         cuda::current_device_prop().minor < 5)) {
        printf("Skip CUDA.BENCHMARK_MATRIX_MUL_QUANTIZED4x4x32 test as current "
               "device doesn't support\n");
        return;
    }
    Benchmarker<MatrixMul> bencher(handle_cuda());
    using Param = MatrixMul::Param;
    Param param;
    param.transposeB = true;
    bencher.set_param(param);
    bencher.set_dtype(0, dtype::Quantized4Asymm(1.0f, (uint8_t)3));
    bencher.set_dtype(1, dtype::Quantized4Asymm(1.0f, (uint8_t)3));
    bencher.set_dtype(2, dtype::QuantizedS32(1.0f));
    for (size_t m : {256, 1024, 4096, 10240, 40960}) {
        for (size_t n : {256, 1024, 4096}) {
            for (size_t k :{512, 1024, 2048}) {
                bencher.set_times(400);
                auto time_in_ms = bencher.exec({{m, k}, {n, k}, {m, n}}) / 400;
                auto gflps = 2.0 * m * k * n / (time_in_ms * 1e-3) * 1e-12;
                printf("m=%zu, k=%zu, n=%zu, time: %fms, perf: %f TFlops\n",
                        m, k, n, time_in_ms, gflps);
            }
        }
    }
}

TEST_F(CUDA, PEAK_BENCHMARK_MATRIX_MUL_QUANTIZED4x4x32) {
    if (cuda::current_device_prop().major < 7 ||
        (cuda::current_device_prop().major == 7 &&
         cuda::current_device_prop().minor < 5)) {
        printf("Skip CUDA.PEAK_BENCHMARK_MATRIX_MUL_QUANTIZED4x4x32 test as "
               "current "
               "device doesn't support\n");
        return;
    }
    Benchmarker<MatrixMul> bencher(handle_cuda());
    using Param = MatrixMul::Param;
    Param param;
    param.transposeB = true;
    bencher.set_param(param);
    bencher.set_dtype(0, dtype::Quantized4Asymm(1.0f, (uint8_t)3));
    bencher.set_dtype(1, dtype::Quantized4Asymm(1.0f, (uint8_t)3));
    bencher.set_dtype(2, dtype::QuantizedS32(1.0f));
    bencher.set_times(400);
    size_t m = 4096, n = 4096, k = 81920;
    auto time_in_ms = bencher.exec({{m, k}, {n, k}, {m, n}}) / 400;
    auto tflps = 2.0 * m * k * n / (time_in_ms * 1e-3) * 1e-12;
    printf("m=%zu, k=%zu, n=%zu, time: %fms, perf: %f TFlops\n", m, k, n,
           time_in_ms, tflps);
}
#endif
#endif

TEST_F(CUDA, MATRIX_MUL_INT8x8x32_WITH_SPETIAL_STRIDES) {
    if (!cuda::is_compute_capability_required(6, 1)) {
        printf("Skip CUDA.MATRIX_MUL test as current device doesn't support\n");
        return;
    }
    Checker<MatrixMul> checker(handle_cuda());
    using Param = MatrixMul::Param;
    Param param;
    DType stype = dtype::Int8();
    checker.set_param(param)
            .set_dtype(0, stype)
            .set_dtype(1, stype)
            .set_dtype(2, dtype::Int32())
            .set_epsilon(5e-3);
    size_t m = 1024, n = 1024, k = 1024;
    {
        TensorLayout A{{m, k}, {2048, 1}, dtype::Int8()},
                B{{k, n}, {2048, 1}, dtype::Int8()}, C{{m, n}, dtype::Int32()};
        checker.execl({A, B, {}});
    }
}

TEST_F(CUDA, MATRIX_MUL_INT8x8x32_NAIVE) {
    if (!cuda::is_compute_capability_required(6, 1)) {
        printf("Skip CUDA.MATRIX_MUL test as current device doesn't support\n");
        return;
    }

    using Param = MatrixMul::Param;
    UniformIntRNG rng{-128, 127};
    Checker<MatrixMul> checker(handle_cuda());
    checker.set_rng(0, &rng).set_rng(1, &rng);

    size_t m = 1007, n = 1003, k = 129;
    for (unsigned mask = 0; mask < 4; ++mask) {
        Param param;
        param.transposeA = mask & 1;
        param.transposeB = mask & 2;
        TensorShape A, B;
        if (param.transposeA)
            A = TensorShape{k, m};
        else
            A = TensorShape{m, k};
        if (param.transposeB)
            B = TensorShape{n, k};
        else
            B = TensorShape{k, n};
        checker.set_param(param)
                .set_dtype(0, dtype::Int8())
                .set_dtype(1, dtype::Int8())
                .set_dtype(2, dtype::Int32())
                .set_epsilon(0)
                .execs({A, B, {}});
    }
}

TEST_F(CUDA, MATRIX_MUL) {
    if (cuda::current_device_prop().major < 6) {
        printf("Skip CUDA.MATRIX_MUL test as current device doesn't support\n");
        return;
    }
    Checker<MatrixMul> checker(handle_cuda());
    using Param = MatrixMul::Param;
    size_t m = 12, n = 16, k = 20;

    bool is_int_available = cuda::is_compute_capability_required(6, 1);
    std::vector<DType> dtype_array;
    dtype_array.push_back(dtype::Float32());
    dtype_array.push_back(dtype::Float16());
    dtype_array.push_back(dtype::BFloat16());
    if (is_int_available)
        dtype_array.push_back(dtype::Int32());

    for (DType dtype : dtype_array) {
        for (unsigned mask = 0; mask < 4; ++mask) {
            Param param;
            param.transposeA = mask & 1;
            param.transposeB = mask & 2;
            DType stype = dtype == dtype::Int32() ? dtype::Int8() : dtype;
            TensorShape A, B;
            if (param.transposeA)
                A = TensorShape{k, m};
            else
                A = TensorShape{m, k};
            if (param.transposeB)
                B = TensorShape{n, k};
            else
                B = TensorShape{k, n};
            if (dtype == dtype::BFloat16()) {
                param.compute_mode = param::MatrixMul::ComputeMode::FLOAT32;
            }
            checker.set_param(param)
                    .set_dtype(0, stype)
                    .set_dtype(1, stype)
                    .set_dtype(2, dtype)
                    .set_epsilon(dtype == dtype::Float16() ||
                                                 dtype == dtype::BFloat16()
                                         ? 5e-2
                                         : 5e-3)
                    .execs({A, B, {}});
        }
    }

    // general tests
    auto args = matrix_mul::get_matmul_args();
    for (auto arg: args) {
        auto m = arg.m, n = arg.n, k = arg.k;
        auto mask = arg.mask;
        Param param;
        param.transposeA = mask & 1;
        param.transposeB = mask & 2;
        TensorShape AS, BS, CS;
        if (param.transposeA)
            AS = TensorShape{k, m};
        else
            AS = TensorShape{m, k};
        if (param.transposeB)
            BS = TensorShape{n, k};
        else
            BS = TensorShape{k, n};
        CS = TensorShape{m, n};
        TensorLayout AL, BL, CL;
        if (arg.A_stride == 0) {
            AL = TensorLayout(AS, dtype::Float32());
        } else {
            AL = TensorLayout(AS, {ptrdiff_t(arg.A_stride), 1},
                              dtype::Float32());
        }
        if (arg.B_stride == 0) {
            BL = TensorLayout(BS, dtype::Float32());
        } else {
            BL = TensorLayout(BS, {ptrdiff_t(arg.B_stride), 1},
                              dtype::Float32());
        }
        if (arg.C_stride == 0) {
            CL = TensorLayout(CS, dtype::Float32());
        } else {
            CL = TensorLayout(CS, {ptrdiff_t(arg.C_stride), 1},
                              dtype::Float32());
        }
        checker.set_param(param).execl({AL, BL, CL});
    }
}

TEST_F(CUDA, MATRIX_MUL_CUBLASLT)
{
    require_compute_capability(7, 5);
    NormalRNG normal_rng;
    Checker<MatrixMul> checker(handle_cuda());
    checker.set_rng(0, &normal_rng)
           .set_rng(1, &normal_rng)
           .set_before_exec_callback(AlgoChecker<MatrixMulForward>("CUBLAS_LT"));
    using Param = MatrixMul::Param;
    size_t m = 32, n = 32, k = 32;
    // test Int8 matmul
    {
        DType dtype=dtype::Int32();
        Param param;
        param.transposeA = false;
        param.transposeB = false;
        DType stype = dtype == dtype::Int32() ? dtype::Int8() : dtype;
        TensorShape A, B;
        A = TensorShape{m, k};
        B = TensorShape{k, n};
        checker.set_param(param).
            set_dtype(0, stype).
            set_dtype(1, stype).
            set_dtype(2, dtype).
            set_epsilon(dtype == dtype::Float16() ? 5e-2 : 5e-3).
            execs({A, B, {}});
    }
    // test float-point matmul
    for (DType dtype: std::array<DType, 2>{
            {dtype::Float32(), dtype::Float16()}}) {
        for (unsigned mask = 0; mask < 4; ++mask) {
            Param param;
            param.transposeA = mask & 1;
            param.transposeB = mask & 2;
            DType stype = dtype == dtype::Int32() ? dtype::Int8() : dtype;
            TensorShape A, B;
            if (param.transposeA)
                A = TensorShape{k, m};
            else
                A = TensorShape{m, k};
            if (param.transposeB)
                B = TensorShape{n, k};
            else
                B = TensorShape{k, n};
            checker.set_param(param).
                set_dtype(0, stype).
                set_dtype(1, stype).
                set_dtype(2, dtype).
                set_epsilon(dtype == dtype::Float16() ? 5e-2 : 8e-3).
                execs({A, B, {}});
        }
    }
    // general tests
    auto args = matrix_mul::get_matmul_args();
    for (auto arg: args) {
        auto m = arg.m, n = arg.n, k = arg.k;
        auto mask = arg.mask;
        Param param;
        param.transposeA = mask & 1;
        param.transposeB = mask & 2;
        TensorShape AS, BS, CS;
        if (param.transposeA)
            AS = TensorShape{k, m};
        else
            AS = TensorShape{m, k};
        if (param.transposeB)
            BS = TensorShape{n, k};
        else
            BS = TensorShape{k, n};
        CS = TensorShape{m, n};
        TensorLayout AL, BL, CL;
        if (arg.A_stride == 0) {
            AL = TensorLayout(AS, dtype::Float32());
        } else {
            AL = TensorLayout(AS, {ptrdiff_t(arg.A_stride), 1},
                              dtype::Float32());
        }
        if (arg.B_stride == 0) {
            BL = TensorLayout(BS, dtype::Float32());
        } else {
            BL = TensorLayout(BS, {ptrdiff_t(arg.B_stride), 1},
                              dtype::Float32());
        }
        if (arg.C_stride == 0) {
            CL = TensorLayout(CS, dtype::Float32());
        } else {
            CL = TensorLayout(CS, {ptrdiff_t(arg.C_stride), 1},
                              dtype::Float32());
        }
        checker.set_param(param).execl({AL, BL, CL});
    }
}
TEST_F(CUDA, MATRIX_MUL_CUBLASLT_SPECIAL_CASE) {
    require_compute_capability(7, 5);
    size_t m = 12, n = 16, k = 20;
    Checker<MatrixMul> checker(handle_cuda());
    checker.set_before_exec_callback(
        AlgoChecker<MatrixMulForward>("CUBLAS_LT"));

    using Param = MatrixMul::Param;

    Param param;
    DType stype = dtype::Float32();
    DType dtype = dtype::Float32();
    TensorShape A, B;
    param.transposeA=param.transposeB=1;
    if (param.transposeA)
        A = TensorShape{k, m};
    else
        A = TensorShape{m, k};
    if (param.transposeB)
        B = TensorShape{n, k};
    else
        B = TensorShape{k, n};
    checker.set_param(param).
        set_dtype(0, stype).
        set_dtype(1, stype).
        set_dtype(2, dtype).
        set_epsilon(dtype == dtype::Float16() ? 5e-1 : 5e-2).
        execs({A, B, {}});
}
TEST_F(CUDA, MATRIX_MUL_CUBLASLT_INT8) {
    require_compute_capability(7, 5);
    NormalRNG normal_rng;
    Checker<MatrixMul> checker(handle_cuda());
    checker.set_rng(0, &normal_rng)
           .set_rng(1, &normal_rng)
           .set_before_exec_callback(AlgoChecker<MatrixMulForward>("CUBLAS_LT"));
    using Param = MatrixMul::Param;

    //size_t m = 32, n = 32, k = 32;
    // test Int8 matmul
    for (size_t m=8; m<=64; m+=4)
    for (size_t n=8; n<=64; n+=4)
    for (size_t k=8; k<=64; k+=4)
    {
        DType dtype=dtype::Int32();
        Param param;
        param.transposeA = false;
        param.transposeB = false;
        DType stype = dtype == dtype::Int32() ? dtype::Int8() : dtype;
        TensorShape A, B;
        A = TensorShape{m, k};
        B = TensorShape{k, n};
        checker.set_param(param).
            set_dtype(0, stype).
            set_dtype(1, stype).
            set_dtype(2, dtype).
            set_epsilon(dtype == dtype::Float16() ? 5e-2 : 5e-3).
            execs({A, B, {}});
    }
}

} // namespace test
} // namespace megdnn
// vim: syntax=cpp.doxygen
