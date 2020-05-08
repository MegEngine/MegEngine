/**
 * \file dnn/test/aarch64/batched_matrix_mul.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "test/common/benchmarker.h"
#include "test/common/checker.h"
#include "test/common/rng.h"
#include "test/common/matrix_mul.h"

#include "test/aarch64/fixture.h"

namespace megdnn {
namespace test {

TEST_F(AARCH64, BATCHED_MATRIX_MUL) {
    Checker<BatchedMatrixMul> checker(handle());
    checker.set_epsilon(1e-2);
    using Param = MatrixMul::Param;
    // auto args = get_batch_matmul_args();
    auto args = matrix_mul::get_batched_matmul_args();

    for (DType dtype : std::vector<DType>{dtype::Float32()}) {
        for (unsigned mask = 0; mask < 4; ++mask) {
            for (auto& arg : args) {
                size_t b = arg.b, m = arg.m, n = arg.n, k = arg.k;
                //! if test all batch sizes, the test case will time out.
                if (b != 2) {
                    continue;
                }
                Param param;
                param.transposeA = mask & 1;
                param.transposeB = mask & 2;
                TensorShape A, B;
                if (param.transposeA)
                    A = TensorShape{b, k, m};
                else
                    A = TensorShape{b, m, k};
                if (param.transposeB)
                    B = TensorShape{b, n, k};
                else
                    B = TensorShape{b, k, n};
                checker.set_param(param)
                        .set_dtype(0, dtype)
                        .set_dtype(1, dtype)
                        .execs({A, B, {}});
            }
        }
    }
}

#ifdef __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_F(AARCH64, BATCHED_MATRIX_MUL_FP16) {
    Checker<BatchedMatrixMul> checker(handle());
    using Param = MatrixMul::Param;
    auto args = matrix_mul::get_batched_matmul_args();

    NormalRNG rng(1.f);
    checker.set_rng(0, &rng).set_rng(1, &rng).set_epsilon(1e-2);
    for (DType dtype : std::vector<DType>{dtype::Float16()}) {
        for (unsigned mask = 0; mask < 4; ++mask) {
            for (auto& arg : args) {
                size_t b = arg.b, m = arg.m, n = arg.n, k = arg.k;
                //! if test all batch sizes, the test case will time out on
                //! sdm855
                if (b != 1) {
                    continue;
                }
                Param param;
                param.transposeA = mask & 1;
                param.transposeB = mask & 2;
                TensorShape A, B;
                if (param.transposeA)
                    A = TensorShape{b, k, m};
                else
                    A = TensorShape{b, m, k};
                if (param.transposeB)
                    B = TensorShape{b, n, k};
                else
                    B = TensorShape{b, k, n};
                checker.set_param(param)
                        .set_dtype(0, dtype)
                        .set_dtype(1, dtype)
                        .set_dtype(2, dtype)
                        .execs({A, B, {}});
            }
        }
    }
}

#if MEGDNN_WITH_BENCHMARK
TEST_F(AARCH64, BENCHMARK_TRANSPOSED_MATRIX_MUL_QUICK_FP16) {
    int exec_times = 10;
    Benchmarker<MatrixMul> benchmarker_gemm(handle());
    benchmarker_gemm.set_times(exec_times);

    float mod = 1000 * exec_times / 1e9;
    using Param = MatrixMul::Param;
    auto run = [&](size_t M, size_t K, size_t N) {
        float time = 1.f, perf = 1.f;

        std::cout << "GEMM: (" << M << ", " << K << ", " << N << ")"
                  << std::endl;
        Param param;
        param.transposeA = true;
        param.transposeB = true;
        benchmarker_gemm.set_param(param)
                .set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32());
        time = benchmarker_gemm.exec({{M, K}, {K, N}, {}});
        perf = 2.f * M * K * N / time * mod;
        std::cout << "gemm fp32, Performance is " << perf << " Gflops"
                  << std::endl;
        benchmarker_gemm.set_param(param)
                .set_dtype(0, dtype::Float16())
                .set_dtype(1, dtype::Float16());
        time = benchmarker_gemm.exec({{M, K}, {K, N}, {}});
        perf = 2.f * M * K * N / time * mod;
        std::cout << "gemm fp16, Performance is " << perf << " Gflops"
                  << std::endl;

    };

    // run M = K = N
    run(32, 32, 32);
    run(64, 64, 64);
    run(128, 128, 128);
    run(256, 256, 256);
    run(512, 512, 512);
    run(1024, 1024, 1024);
    run(2048, 2048, 2048);
}

TEST_F(AARCH64, BENCHMARK_TRANSPOSED_MATRIX_MUL_ALL_SIZES_FP16) {
    int exec_times = 50;
    Benchmarker<MatrixMul> benchmarker_gemm(handle());
    benchmarker_gemm.set_times(exec_times);

    float mod = 1000 * exec_times / 1e9;
    using Param = MatrixMul::Param;
    auto run = [&](size_t M, size_t K, size_t N) {
        float time = 1.f, perf = 1.f;

        std::cout << "GEMM: (" << M << ", " << K << ", " << N << ")"
                  << std::endl;
        Param param;
        param.transposeA = param.transposeB = true;
        benchmarker_gemm.set_param(param)
                .set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32());
        time = benchmarker_gemm.exec({{K, M}, {N, K}, {}});
        perf = 2.f * M * K * N / time * mod;
        std::cout << "gemm fp32, Performance is " << perf << " Gflops"
                  << std::endl;
        benchmarker_gemm.set_param(param)
                .set_dtype(0, dtype::Float16())
                .set_dtype(1, dtype::Float16());
        time = benchmarker_gemm.exec({{K, M}, {N, K}, {}});
        perf = 2.f * M * K * N / time * mod;
        std::cout << "gemm fp16, Performance is " << perf << " Gflops"
                  << std::endl;

    };

    std::cout << "warm up:\n";
    for (int i = 0; i < 50; i++) {
        benchmarker_gemm.set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .set_display(false)
                .exec({{256, 256}, {256, 256}, {}});
        benchmarker_gemm.set_display(true);
    }

    // run M = K = N
    run(8, 8, 8);
    run(16, 16, 16);
    run(32, 32, 32);
    run(64, 64, 64);
    run(128, 128, 128);
    run(256, 256, 256);
    run(512, 512, 512);
    run(1024, 1024, 1024);
    run(2048, 2048, 2048);

    // run sgmev like
    run(32, 32, 1);
    run(64, 64, 1);
    run(128, 128, 1);
    run(256, 256, 1);
    run(512, 512, 1);

    // run M, N >> K
    run(32, 16, 32);
    run(64, 16, 64);
    run(128, 16, 128);
    run(256, 16, 256);
    run(512, 16, 512);

    // run N, K >> M
    run(16, 32, 32);
    run(16, 64, 64);
    run(16, 128, 128);
    run(16, 256, 256);
    run(16, 512, 512);

    // run M >> K, N
    run(32, 16, 16);
    run(64, 16, 16);
    run(128, 16, 16);
    run(256, 16, 16);
    run(512, 16, 16);

    // run K >> M, N
    run(16, 32, 16);
    run(16, 64, 16);
    run(16, 128, 16);
    run(16, 256, 16);
    run(16, 512, 16);

    // run N >> M, K
    run(16, 16, 32);
    run(16, 16, 64);
    run(16, 16, 128);
    run(16, 16, 256);
    run(16, 16, 512);

    // run VGG
    // conv 1.1
    run(64, 3 * 3 * 3, 224 * 224);
    // conv 1.2
    run(128, 64 * 3 * 3, 112 * 112);
    // conv 2.1
    run(128, 128 * 3 * 3, 112 * 112);
    // conv 2.2
    run(128, 128 * 3 * 3, 56 * 56);
    // conv 3.1
    run(256, 128 * 3 * 3, 56 * 56);
    // conv 3.2
    run(256, 256 * 3 * 3, 28 * 28);
    // conv 4.1
    run(512, 256 * 3 * 3, 28 * 28);
    // conv 4.2
    run(512, 512 * 3 * 3, 14 * 14);
}

#endif
#endif

}  // namespace test
}  // namespace megdnn
// vim: syntax=cpp.doxygen
