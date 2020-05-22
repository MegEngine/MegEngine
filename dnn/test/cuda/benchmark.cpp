/**
 * \file dnn/test/cuda/benchmark.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/cuda/fixture.h"

#include "test/common/tensor.h"
#include "test/common/timer.h"
#include "megdnn/oprs.h"
#include "test/common/workspace_wrapper.h"
#include "test/common/benchmarker.h"
#include "src/cuda/utils.h"

namespace megdnn {
namespace test {

#if MEGDNN_WITH_BENCHMARK

TEST_F(CUDA, BENCHMARK_CONVOLUTION_8X8X32)
{
    if (!cuda::is_compute_capability_required(6, 1)) {
        printf("Skip CUDA.BENCHMARK_CONVOLUTION_8X8X32 test as current device"
               "doesn't support\n");
        return;
    }
    using Param = param::Convolution;
    auto run_1x1 = [&](size_t N, size_t OC, size_t IC, size_t H, size_t W) {
        Benchmarker<Convolution> benchmarker(handle_cuda());
        Param param_base;
        Param param_float = param_base, param_int = param_base;
        param_int.format = Param::Format::NHWC;
        TensorShape src_float{N, IC, H, W}, filter_float{OC, IC, 1, 1};
        TensorShape src_int{N, H, W, IC}, filter_int{OC, 1, 1, IC};
        benchmarker.set_display(false);
        auto time_in_ms_float = benchmarker.set_param(param_float)
            .set_dtype(0, dtype::Float32())
            .set_dtype(1, dtype::Float32())
            .execs({src_float, filter_float, {}});
        auto time_in_ms_int = benchmarker.set_param(param_int)
            .set_dtype(0, dtype::Int8())
            .set_dtype(1, dtype::Int8())
            .execs({src_int, filter_int, {}});
        std::cout << "1x1: N=" << N << " OC=" << OC << " IC=" << IC
            << " H=" << H << " W=" << W
            << " time_float=" << time_in_ms_float << "ms"
            << " time_int=" << time_in_ms_int << "ms" << std::endl;
    };
    auto run_chanwise = [&](size_t N, size_t C, size_t H, size_t W,
            size_t F) {
        size_t P = F/2;
        Benchmarker<Convolution> benchmarker(handle_cuda());
        Param param_base;
        param_base.pad_h = param_base.pad_w = P;
        param_base.sparse = Param::Sparse::GROUP;
        Param param_float = param_base;
        Param param_int = param_base;
        param_int.format = Param::Format::NHWC;
        TensorShape src_float{N, C, H, W}, filter_float{C, 1, 1, F, F};
        TensorShape src_int{N, H, W, C}, filter_int{C, 1, F, F, 1};
        benchmarker.set_display(false);
        auto time_in_ms_float = benchmarker.set_param(param_float)
            .set_dtype(0, dtype::Float32())
            .set_dtype(1, dtype::Float32())
            .execs({src_float, filter_float, {}});
        auto time_in_ms_int = benchmarker.set_param(param_int)
            .set_dtype(0, dtype::Int8())
            .set_dtype(1, dtype::Int8())
            .execs({src_int, filter_int, {}});
        std::cout << "chanwise: N=" << N << " C=" << C
            << " H=" << H << " W=" << W << " F=" << F
            << " time_float=" << time_in_ms_float << "ms"
            << " time_int=" << time_in_ms_int << "ms" << std::endl;
    };
    run_chanwise(1, 384, 56, 56, 3);
    run_1x1(1, 32, 32, 56, 56);
    run_1x1(1, 256, 256, 7, 7);
}

TEST_F(CUDA, BENCHMARK_REDUCE)
{
    auto run = [&](size_t A, size_t B, size_t C) {
        Tensor<> src(handle_cuda(), TensorLayout({A, B, C}, dtype::Float32())),
                 dst(handle_cuda(), TensorLayout({A, 1, C}, dtype::Float32()));
        auto opr = handle_cuda()->create_operator<Reduce>();
        opr->param().axis = 1;
        WorkspaceWrapper workspace(handle_cuda(), opr->get_workspace_in_bytes(
                    src.layout(), dst.layout()));
        opr->exec(src.tensornd(), dst.tensornd(), workspace.workspace());
        Timer timer;
        megcoreSynchronize(handle_cuda()->megcore_computing_handle());
        timer.start();
        for (size_t i = 0; i < 10; ++i)
            opr->exec(src.tensornd(), dst.tensornd(), workspace.workspace());
        megcoreSynchronize(handle_cuda()->megcore_computing_handle());
        timer.stop();
        float time_in_us = timer.get_time_in_us();
        std::cout << "src = " << A << "x" << B << "x" << C << std::endl
            << "time = " << time_in_us / 1e3 << "ms" << std::endl;
    };
    run(65536, 64, 1);
    run(1, 268435455, 1);
    run(256, 1048575, 1);
    run(1, 1048575, 256);
    run(256, 4095, 256);
}

TEST_F(CUDA, BENCHMARK_BATCHED_MATRIX_MUL)
{
    auto run = [&](size_t b, size_t m, size_t n, size_t k) {
        Tensor<> A(handle_cuda(), TensorLayout({b, m, k}, dtype::Float32()));
        Tensor<> B(handle_cuda(), TensorLayout({b, k, n}, dtype::Float32()));
        Tensor<> C(handle_cuda(), TensorLayout({b, m, n}, dtype::Float32()));
        auto opr = handle_cuda()->create_operator<BatchedMatrixMul>();
        WorkspaceWrapper workspace(handle_cuda(), opr->get_workspace_in_bytes(
                    A.layout(), B.layout(), C.layout()));
        opr->exec(A.tensornd(), B.tensornd(), C.tensornd(),
                workspace.workspace());
        Timer timer;
        megcoreSynchronize(handle_cuda()->megcore_computing_handle());
        timer.start();
        opr->exec(A.tensornd(), B.tensornd(), C.tensornd(),
                workspace.workspace());
        megcoreSynchronize(handle_cuda()->megcore_computing_handle());
        timer.stop();
        float time_in_s = timer.get_time_in_us() / 1e6;
        float flo = b*m*n*k*2;
        float gflops = flo / time_in_s / 1e9;
        std::cout << "time_in_s = " << time_in_s << '\n'
            << "flo = " << flo << '\n'
            << "gflops = " << gflops << std::endl;
    };
    run(256, 256, 256, 256);
}

TEST_F(CUDA, BENCHMARK_MATRIX_MUL)
{
    auto run = [&](size_t m, size_t n, size_t k) {
        Tensor<> A(handle_cuda(), TensorLayout({m, k}, dtype::Float32()));
        Tensor<> B(handle_cuda(), TensorLayout({k, n}, dtype::Float32()));
        Tensor<> C(handle_cuda(), TensorLayout({m, n}, dtype::Float32()));
        auto opr = handle_cuda()->create_operator<MatrixMul>();
        WorkspaceWrapper workspace(handle_cuda(), opr->get_workspace_in_bytes(
                    A.layout(), B.layout(), C.layout()));
        opr->exec(A.tensornd(), B.tensornd(), C.tensornd(),
                workspace.workspace());
        Timer timer;
        megcoreSynchronize(handle_cuda()->megcore_computing_handle());
        timer.start();
        opr->exec(A.tensornd(), B.tensornd(), C.tensornd(),
                workspace.workspace());
        megcoreSynchronize(handle_cuda()->megcore_computing_handle());
        timer.stop();
        float time_in_s = timer.get_time_in_us() / 1e6;
        float flo = m*n*k*2;
        float gflops = flo / time_in_s / 1e9;
        std::cout << "time_in_s = " << time_in_s << '\n'
            << "flo = " << flo << '\n'
            << "gflops = " << gflops << std::endl;
    };
    run(4096, 4096, 4096);
}

TEST_F(CUDA, BENCHMARK_LOCAL)
{
    auto run = [&](size_t N, size_t IC, size_t IH, size_t IW,
            size_t OC, size_t OH, size_t OW, size_t FH, size_t FW) {
        Tensor<> src(handle_cuda(), TensorLayout({N, IC, IH, IW},
                    dtype::Float32()));
        Tensor<> filter(handle_cuda(), TensorLayout({OH, OW, IC, FH, FW, OC},
                    dtype::Float32()));
        Tensor<> dst(handle_cuda(), TensorLayout({N, OC, OH, OW},
                    dtype::Float32()));
        auto opr = handle_cuda()->create_operator<Local>();
        WorkspaceWrapper workspace(handle_cuda(), opr->get_workspace_in_bytes(
                    src.layout(), filter.layout(), dst.layout()));
        opr->exec(src.tensornd(), filter.tensornd(), dst.tensornd(),
                workspace.workspace());
        Timer timer;
        megcoreSynchronize(handle_cuda()->megcore_computing_handle());
        timer.start();
        opr->exec(src.tensornd(), filter.tensornd(), dst.tensornd(),
                workspace.workspace());
        megcoreSynchronize(handle_cuda()->megcore_computing_handle());
        timer.stop();
        float time_in_us = timer.get_time_in_us();
        std::cout << "time = " << time_in_us << "us" << std::endl;
    };
    run(32, 64, 7, 7, 64, 5, 5, 3, 3);
}
#endif

} // namespace test
} // namespace megdnn

// vim: syntax=cpp.doxygen
