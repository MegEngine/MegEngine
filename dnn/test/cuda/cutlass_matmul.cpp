/**
 * \file dnn/test/cuda/cutlass_matmul.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#include <cuda.h>
#include "megdnn/oprs/linalg.h"

#include "src/common/utils.h"
#include "test/common/checker.h"
#include "test/common/matrix_mul.h"
#include "test/common/tensor.h"
#include "test/common/workspace_wrapper.h"
#include "test/cuda/benchmark.h"
#include "test/cuda/fixture.h"
#include "test/cuda/utils.h"

#if CUDA_VERSION >= 9020
namespace megdnn {
namespace test {
namespace {
void test_multibatchsize(
        Handle* handle_cuda, DType A_dtype, DType B_dtype, DType C_dtype,
        const char* algo, const std::vector<matrix_mul::TestArg>& args,
        param::MatrixMul::Format format = param::MatrixMul::Format::DEFAULT,
        const std::function<bool(const matrix_mul::TestArg&)>& filter = {}) {
    Checker<MatrixMulForward> checker(handle_cuda, false);
    if (algo) {
        checker.set_before_exec_callback(AlgoChecker<MatrixMulForward>(algo));
    }
    std::unique_ptr<RNG> rng;
    if (A_dtype.enumv() == DTypeEnum::Float32) {
        rng = std::make_unique<UniformFloatRNG>(-1, 1);
        megdnn_assert(B_dtype.enumv() == DTypeEnum::Float32 &&
                      C_dtype.enumv() == DTypeEnum::Float32);
    }
    megdnn_assert(rng != nullptr);

    struct Compare {
        bool is_same(dt_float32 expected, dt_float32 actual) const {
            return expected == actual;
        }
    };

    // copy rhs->lhs, lhs is 8 times of rhs
    auto copy = [](SyncedTensor<dt_float32, Compare>& lhs,
                   SyncedTensor<dt_float32, Compare>& rhs) {
        size_t chunk = rhs.layout().span().dist_byte();
        size_t tot = lhs.layout().span().dist_byte();
        megdnn_assert(tot % chunk == 0);
        char* pointer_lhs = reinterpret_cast<char*>(lhs.ptr_mutable_host());
        const char* pointer_rhs = reinterpret_cast<const char*>(rhs.ptr_host());
        for (size_t i = 0; i < tot; i += chunk) {
            std::memcpy(pointer_lhs + i, pointer_rhs, chunk);
        }
    };
    using Param = param::MatrixMul;
    megdnn_assert(format == Param::Format::DEFAULT);
    for (auto&& arg : args) {
        megdnn_assert(arg.mask == 0x0);
        // make m, n, k big enough
        size_t m = arg.m, n = (arg.n << 3), k = (arg.k << 3);
        size_t m_prime = (m << 3);
        if (filter && filter(arg))
            continue;
        TensorShape A{m, k}, B{k, n}, C{m, n};
        TensorShape A_prime{m_prime, k}, C_prime{m_prime, n};
        SyncedTensor<dt_float32, Compare> A_tensor{handle_cuda, {A, A_dtype}},
                B_tensor{handle_cuda, {B, B_dtype}},
                C_tensor{handle_cuda, {C, C_dtype}},
                A_tensor_prime{handle_cuda, {A_prime, A_dtype}},
                C_tensor_prime{handle_cuda, {C_prime, C_dtype}},
                C_tensor_batch{handle_cuda, {C_prime, C_dtype}};
        rng->gen(A_tensor.tensornd_host());
        rng->gen(B_tensor.tensornd_host());
        copy(A_tensor_prime, A_tensor);

        auto opr_reference = handle_cuda->create_operator<MatrixMulForward>();
        {
            opr_reference->execution_policy().algo.reset();
            for (auto i : opr_reference->get_all_algorithms_info(
                         A_tensor.layout(), B_tensor.layout(),
                         C_tensor.layout())) {
                if (std::regex_match(
                            i.desc.name.c_str(),
                            std::regex("(" + std::string(algo) + ")(.*)"))) {
                    opr_reference->execution_policy().algo = i.desc;
                    break;
                }
            }
            megdnn_assert(opr_reference->execution_policy().algo.valid());
            size_t ws_size = opr_reference->get_workspace_in_bytes(
                    A_tensor.layout(), B_tensor.layout(), C_tensor.layout());
            WorkspaceWrapper ws_reference(handle_cuda, ws_size);
            opr_reference->exec(
                    A_tensor.tensornd_dev(), B_tensor.tensornd_dev(),
                    C_tensor.tensornd_dev(), ws_reference.workspace());
        }
        copy(C_tensor_prime, C_tensor);
        checker.set_dtype(0, A_dtype)
                .set_dtype(1, B_dtype)
                .set_dtype(2, C_dtype)
                .set_epsilon(1e-6)
                .exect({A_tensor_prime.tensornd_host(),
                        B_tensor.tensornd_host(),
                        {}},
                       {{}, {}, C_tensor_prime.tensornd_host()});
        {
            opr_reference->execution_policy().algo.reset();
            for (auto i : opr_reference->get_all_algorithms_info(
                         A_tensor_prime.layout(), B_tensor.layout(),
                         C_tensor_batch.layout())) {
                if (std::regex_match(
                            i.desc.name.c_str(),
                            std::regex("(" + std::string(algo) + ")(.*)"))) {
                    opr_reference->execution_policy().algo = i.desc;
                    break;
                }
            }
            megdnn_assert(opr_reference->execution_policy().algo.valid());
            size_t ws_size = opr_reference->get_workspace_in_bytes(
                    A_tensor_prime.layout(), B_tensor.layout(),
                    C_tensor_batch.layout());
            WorkspaceWrapper ws_reference(handle_cuda, ws_size);
            opr_reference->exec(
                    A_tensor_prime.tensornd_dev(), B_tensor.tensornd_dev(),
                    C_tensor_batch.tensornd_dev(), ws_reference.workspace());
        }
        C_tensor_batch.check_with(C_tensor_prime);
    }
}

#if MEGDNN_WITH_BENCHMARK
struct BenchArgs {
    size_t m, n, k, mask = 0x0;
};

std::vector<BenchArgs> get_square_matmul_args() {
    std::vector<BenchArgs> args;
    args.emplace_back(BenchArgs{128, 128, 128});
    args.emplace_back(BenchArgs{256, 256, 256});
    args.emplace_back(BenchArgs{512, 512, 512});
    args.emplace_back(BenchArgs{1024, 1024, 1024});
    args.emplace_back(BenchArgs{2048, 2048, 2048});
    args.emplace_back(BenchArgs{4096, 4096, 4096});

    return args;
}

std::vector<BenchArgs> get_feat_model_args() {
    std::vector<BenchArgs> args;

    args.emplace_back(BenchArgs{2, 4096, 4096});
    args.emplace_back(BenchArgs{2, 1024, 6912});
    args.emplace_back(BenchArgs{2, 3456, 3456});
    args.emplace_back(BenchArgs{2, 2304, 2304});
    args.emplace_back(BenchArgs{1, 256, 8192});
    args.emplace_back(BenchArgs{2, 864, 864});
    args.emplace_back(BenchArgs{2, 9, 64});

    args.emplace_back(BenchArgs{4, 4096, 4096});
    args.emplace_back(BenchArgs{4, 1024, 6912});
    args.emplace_back(BenchArgs{4, 3456, 3456});
    args.emplace_back(BenchArgs{4, 2304, 2304});
    args.emplace_back(BenchArgs{2, 256, 8192});
    args.emplace_back(BenchArgs{4, 864, 864});
    args.emplace_back(BenchArgs{4, 9, 64});

    args.emplace_back(BenchArgs{8, 4096, 4096});
    args.emplace_back(BenchArgs{8, 1024, 6912});
    args.emplace_back(BenchArgs{8, 3456, 3456});
    args.emplace_back(BenchArgs{8, 2304, 2304});
    args.emplace_back(BenchArgs{4, 256, 8192});
    args.emplace_back(BenchArgs{8, 864, 864});
    args.emplace_back(BenchArgs{4, 9, 64});

    args.emplace_back(BenchArgs{16, 4096, 4096});
    args.emplace_back(BenchArgs{16, 1024, 6912});
    args.emplace_back(BenchArgs{16, 3456, 3456});
    args.emplace_back(BenchArgs{16, 2304, 2304});
    args.emplace_back(BenchArgs{8, 256, 8192});
    args.emplace_back(BenchArgs{16, 864, 864});
    args.emplace_back(BenchArgs{8, 9, 64});

    args.emplace_back(BenchArgs{32, 4096, 4096});
    args.emplace_back(BenchArgs{32, 1024, 6912});
    args.emplace_back(BenchArgs{32, 3456, 3456});
    args.emplace_back(BenchArgs{32, 2304, 2304});
    args.emplace_back(BenchArgs{16, 256, 8192});
    args.emplace_back(BenchArgs{32, 864, 864});
    args.emplace_back(BenchArgs{32, 9, 64});

    args.emplace_back(BenchArgs{64, 4096, 4096});
    args.emplace_back(BenchArgs{64, 1024, 6912});
    args.emplace_back(BenchArgs{64, 3456, 3456});
    args.emplace_back(BenchArgs{64, 2304, 2304});
    args.emplace_back(BenchArgs{32, 256, 8192});
    args.emplace_back(BenchArgs{64, 864, 864});
    args.emplace_back(BenchArgs{64, 9, 64});

    args.emplace_back(BenchArgs{128, 4096, 4096});
    args.emplace_back(BenchArgs{128, 1024, 6912});
    args.emplace_back(BenchArgs{128, 3456, 3456});
    args.emplace_back(BenchArgs{128, 2304, 2304});
    args.emplace_back(BenchArgs{64, 256, 8192});
    args.emplace_back(BenchArgs{128, 864, 864});
    args.emplace_back(BenchArgs{128, 9, 64});

    return args;
}

void benchmark_matrix_mul(
        Handle* handle, const std::vector<BenchArgs>& args, DType A_dtype,
        DType B_dtype, DType C_dtype, const char* algo = nullptr,
        param::MatrixMul::Format format = param::MatrixMul::Format::DEFAULT) {
    megdnn_assert(A_dtype.enumv() == B_dtype.enumv());
    CUBenchmarker<MatrixMulForward> benchmarker(handle);
    CUBenchmarker<MatrixMulForward> benchmarker_cublas(handle);
    size_t RUNS = 1000;
    benchmarker.set_display(false).set_times(RUNS);
    benchmarker_cublas.set_display(false).set_times(RUNS);
    benchmarker_cublas.set_before_exec_callback(
            AlgoChecker<MatrixMulForward>("CUBLAS"));
    benchmarker.set_dtype(0, A_dtype)
            .set_dtype(1, B_dtype)
            .set_dtype(2, C_dtype);
    benchmarker_cublas.set_dtype(0, A_dtype)
            .set_dtype(1, B_dtype)
            .set_dtype(2, C_dtype);
    using Param = MatrixMul::Param;
    for (auto&& arg : args) {
        size_t m = arg.m, n = arg.n, k = arg.k;
        Param param;
        param.transposeA = arg.mask & 0x1;
        param.transposeB = arg.mask & 0x2;
        param.format = format;
        size_t A0 = m, A1 = k, B0 = k, B1 = n;
        if (param.transposeA) {
            std::swap(A0, A1);
        }
        if (param.transposeB) {
            std::swap(B0, B1);
        }

        benchmarker.set_param(param);
        TensorShape A{A0, A1}, B{B0, B1}, C{m, n};
        float time_in_ms = 0.f;
        if (algo) {
            time_in_ms =
                    algo_benchmark<MatrixMulForward, OprProxy<MatrixMulForward>,
                                   CUTimer>(benchmarker, {A, B, C}, algo) /
                    RUNS;
        } else {
            time_in_ms = benchmarker.execs({A, B, C}) / RUNS;
        }
        benchmarker_cublas.set_param(param);
        auto time_in_ms_cublas = benchmarker_cublas.execs({A, B, C}) / RUNS;
        float flo = 2.0 * m * n * k / (1e12);
        printf("A=%s, B=%s, C=%s, time(algo=%s)=%.2f %.2fTops, "
               "time(cublas)=%.2f %.2fTops, "
               "perf(algo=%s)/perf(cublas)=%.2f\n",
               A.to_string().c_str(), B.to_string().c_str(),
               C.to_string().c_str(), algo, time_in_ms,
               (flo / (time_in_ms * 1e-3)), time_in_ms_cublas,
               (flo / (time_in_ms_cublas * 1e-3)), algo,
               time_in_ms_cublas / time_in_ms);
    }
}
#endif
}  // namespace

TEST_F(CUDA, CUTLASS_GEMM_MULTI_BATCHSIZE) {
    auto args = matrix_mul::get_matmul_args_no_mask();
    test_multibatchsize(handle_cuda(), dtype::Float32(), dtype::Float32(),
                        dtype::Float32(),
                        "CUTLASS_FLOAT32_SIMT_128X128X8_32X64X8", args,
                        param::MatrixMul::Format::DEFAULT);
}

TEST_F(CUDA, CUTLASS_GEMM_SPLIT_K_MULTI_BATCHSIZE) {
    auto args = matrix_mul::get_matmul_args_no_mask();
    test_multibatchsize(
            handle_cuda(), dtype::Float32(), dtype::Float32(), dtype::Float32(),
            "CUTLASS_FLOAT32_SIMT_SPLIT_K_128X128X8_32X64X8", args,
            param::MatrixMul::Format::DEFAULT,
            [](const matrix_mul::TestArg& arg) { return arg.k <= arg.n; });
}

TEST_F(CUDA, CUTLASS_GEMV_BATCHED_STRIDED_128_MULTI_BATCHSIZE) {
    auto args = matrix_mul::get_matmul_args_no_mask();
    test_multibatchsize(handle_cuda(), dtype::Float32(), dtype::Float32(),
                        dtype::Float32(),
                        "CUTLASS_FLOAT32_SIMT_GEMV_BATCHED_STRIDED_128", args,
                        param::MatrixMul::Format::DEFAULT);
}

TEST_F(CUDA, CUTLASS_GEMV_BATCHED_STRIDED_64_MULTI_BATCHSIZE) {
    auto args = matrix_mul::get_matmul_args_no_mask();
    test_multibatchsize(handle_cuda(), dtype::Float32(), dtype::Float32(),
                        dtype::Float32(),
                        "CUTLASS_FLOAT32_SIMT_GEMV_BATCHED_STRIDED_64", args,
                        param::MatrixMul::Format::DEFAULT);
}

TEST_F(CUDA, CUTLASS_GEMV_BATCHED_STRIDED_32_MULTI_BATCHSIZE) {
    auto args = matrix_mul::get_matmul_args_no_mask();
    test_multibatchsize(handle_cuda(), dtype::Float32(), dtype::Float32(),
                        dtype::Float32(),
                        "CUTLASS_FLOAT32_SIMT_GEMV_BATCHED_STRIDED_32", args,
                        param::MatrixMul::Format::DEFAULT);
}

#define MEGDNN_FOREACH_CUTLASS_KERNEL(cb) \
    cb(1, 64, 256, 8, 32, 64, 8);         \
    cb(2, 256, 64, 8, 64, 32, 8);         \
    cb(3, 32, 256, 8, 16, 64, 8);         \
    cb(4, 256, 32, 8, 64, 16, 8);         \
    cb(5, 128, 128, 8, 32, 64, 8);        \
    cb(6, 128, 64, 8, 64, 32, 8);         \
    cb(7, 64, 128, 8, 32, 64, 8);         \
    cb(8, 128, 32, 8, 64, 32, 8);         \
    cb(9, 32, 128, 8, 32, 64, 8);         \
    cb(10, 64, 64, 8, 32, 64, 8);         \
    cb(11, 32, 64, 8, 32, 64, 8);         \
    cb(12, 64, 32, 8, 64, 32, 8);         \
    cb(13, 32, 32, 8, 32, 32, 8);         \
    cb(14, 8, 32, 8, 8, 32, 8);           \
    cb(15, 16, 32, 8, 16, 32, 8);         \
    cb(16, 16, 64, 8, 16, 64, 8);         \
    cb(17, 16, 128, 8, 16, 64, 8);

#define cb(name, tbm, tbn, tbk, wm, wn, wk)                                    \
    TEST_F(CUDA, CUTLASS_GEMM_##name) {                                        \
        matrix_mul::check_matrix_mul<MatrixMulForward>(                        \
                dtype::Float32(), dtype::Float32(), dtype::Float32(),          \
                handle_cuda(),                                                 \
                "CUTLASS_FLOAT32_SIMT_" #tbm "X" #tbn "X" #tbk "_" #wm "X" #wn \
                "X" #wk);                                                      \
    }

MEGDNN_FOREACH_CUTLASS_KERNEL(cb)

#undef cb

#define cb(name, tbm, tbn, tbk, wm, wn, wk)                                    \
    TEST_F(CUDA, CUTLASS_GEMM_SPLIT_K_##name) {                                \
        matrix_mul::check_matrix_mul<MatrixMulForward>(                        \
                dtype::Float32(), dtype::Float32(), dtype::Float32(),          \
                handle_cuda(),                                                 \
                "CUTLASS_FLOAT32_SIMT_SPLIT_K_" #tbm "X" #tbn "X" #tbk "_" #wm \
                "X" #wn "X" #wk,                                               \
                param::MatrixMul::Format::DEFAULT, 8, 1e-3,                    \
                matrix_mul::get_matmul_args_split_k());                        \
    }

MEGDNN_FOREACH_CUTLASS_KERNEL(cb)

#undef cb
#undef MEGDNN_FOREACH_CUTLASS_KERNEL

#if MEGDNN_WITH_BENCHMARK
TEST_F(CUDA, BENCHMARK_CUTLASS_MATMUL) {
    benchmark_matrix_mul(handle_cuda(), get_square_matmul_args(),
                         dtype::Float32(), dtype::Float32(), dtype::Float32(),
                         "CUTLASS_FLOAT32_SIMT");
}

TEST_F(CUDA, BENCHMARK_CUTLASS_MATMUL_FEAT) {
    benchmark_matrix_mul(handle_cuda(), get_feat_model_args(), dtype::Float32(),
                         dtype::Float32(), dtype::Float32(),
                         "CUTLASS_FLOAT32_SIMT");
}
#endif
}  // namespace test
}  // namespace megdnn
#endif

// vim: syntax=cpp.doxygen
