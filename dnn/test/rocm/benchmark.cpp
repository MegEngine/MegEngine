/**
 * \file dnn/test/rocm/benchmark.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "hcc_detail/hcc_defs_prologue.h"
#include "test/rocm/fixture.h"

#include "test/common/tensor.h"
#include "test/common/timer.h"
#include "megdnn/oprs.h"
#include "test/common/workspace_wrapper.h"
#include "test/common/benchmarker.h"
#include "src/rocm/utils.h"
#include "test/rocm/benchmarker.h"

namespace megdnn {
namespace test {

#if MEGDNN_WITH_BENCHMARK

TEST_F(ROCM, REDUCE_BENCHMARK) {
    megdnn::rocm::enable_miopen_algo_search(handle_rocm(), true);
    auto benchmarker =
            ROCMBenchmarker<ReduceForward>(handle_rocm(), handle_naive(false));
    auto run = [&](size_t A, size_t B, size_t C) {
        auto dtype = dtype::Float32();
        benchmarker.set_dtype(0, dtype).set_dtype(1, dtype);
        benchmarker.set_display(true);
        ReduceForward::Param param;
        param.axis = 1;
        benchmarker.set_param(param);
        // warm up
        benchmarker.execs({{A, B, C}, {}});
        // do actual benchmark
        auto time_ms = benchmarker.execs({{A, B, C}, {}});
        time_ms = benchmarker.execs({{A, B, C}, {}});
        auto io = (double)(A * B * C + A * C) * dtype.size();
        auto gbps = io / (time_ms * 1e6);
        printf("io %.2fGB, flops %.3fGB/s\n", io / 1e9, gbps);

    };
    run(65536, 64, 1);
    run(1, 268435455, 1);
    run(256, 1048575, 1);
    run(1, 1048575, 256);
    run(256, 4095, 256);
}

TEST_F(ROCM, BATCHED_MATRIX_MUL_BENCHMARK) {
    megdnn::rocm::enable_miopen_algo_search(handle_rocm(), true);
    auto benchmarker = ROCMBenchmarker<BatchedMatrixMulForward>(
            handle_rocm(), handle_naive(false));
    auto run = [&](size_t b, size_t m, size_t n, size_t k) {
        auto dtype = dtype::Float32();
        benchmarker.set_dtype(0, dtype).set_dtype(1, dtype);
        benchmarker.set_display(true);
        // warm up
        benchmarker.execs({{b, m, k}, {b, k, n}, {}});
        // do actual benchmark
        auto time_ms = benchmarker.execs({{b, m, k}, {b, k, n}, {}});
        time_ms = benchmarker.execs({{b, m, k}, {b, k, n}, {}});
        double flo = 2.0 * b * m * n * k;
        double flops = flo / (time_ms * 1e9);
        printf("mxnxk=%zux%zux%zu flo %.2fGB, flops %.3fTFLOPS\n", m, n, k,
               flo / 1e9, flops);

    };
    run(32, 128, 128, 128);
    run(32, 256, 256, 256);
    run(32, 512, 512, 512);
    run(32, 1024, 1024, 1024);
    run(32, 4096, 4096, 4096);
    //! resnet50 fwd
    run(32, 12544, 1024, 256);
    run(32, 12544, 1024, 512);
    run(32, 12544, 256, 1024);
    run(32, 12544, 256, 512);
    run(32, 12544, 64, 147);
    run(32, 196, 256, 2304);
    run(32, 3025, 64, 576);
    run(32, 3136, 2048, 1024);
    run(32, 3136, 2048, 512);
    run(32, 3136, 512, 1024);
    run(32, 3136, 512, 2048);
    run(32, 3136, 64, 576);
    run(32, 49, 512, 4608);
    run(32, 50176, 128, 256);
    run(32, 50176, 512, 256);
    run(32, 784, 128, 1152);
    //! resnet50 bwdwrw
    run(32, 147, 64, 12544);
}

TEST_F(ROCM, MATRIX_MUL_BENCHMARK) {
    megdnn::rocm::enable_miopen_algo_search(handle_rocm(), true);
    auto benchmarker = ROCMBenchmarker<MatrixMulForward>(handle_rocm(),
                                                         handle_naive(false));
    auto run = [&](size_t m, size_t n, size_t k) {
        auto dtype = dtype::Float32();
        benchmarker.set_dtype(0, dtype).set_dtype(1, dtype);
        benchmarker.set_display(true);
        // warm up
        benchmarker.execs({{m, k}, {k, n}, {}});
        // do actual benchmark
        auto time_ms = benchmarker.execs({{m, k}, {k, n}, {}});
        time_ms = benchmarker.execs({{m, k}, {k, n}, {}});
        double flo = 2.0 * m * n * k;
        double flops = flo / (time_ms * 1e9);
        printf("mxnxk=%zux%zux%zu flo %.2fGB, flops %.3fTFLOPS\n", m, n, k,
               flo / 1e9, flops);

    };
    run(128, 128, 128);
    run(256, 256, 256);
    run(512, 512, 512);
    run(1024, 1024, 1024);
    run(4096, 4096, 4096);
    //! resnet50 fwd
    run(12544, 1024, 256);
    run(12544, 1024, 512);
    run(12544, 256, 1024);
    run(12544, 256, 512);
    run(12544, 64, 147);
    run(196, 256, 2304);
    run(3025, 64, 576);
    run(3136, 2048, 1024);
    run(3136, 2048, 512);
    run(3136, 512, 1024);
    run(3136, 512, 2048);
    run(3136, 64, 576);
    run(49, 512, 4608);
    run(50176, 128, 256);
    run(50176, 512, 256);
    run(784, 128, 1152);
    //! resnet50 bwdwrw
    run(147, 64, 12544);
}

#endif

} // namespace test
} // namespace megdnn

// vim: syntax=cpp.doxygen
