/**
 * \file dnn/test/aarch64/convolution.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/aarch64/fixture.h"

#include "test/common/benchmarker.h"
#include "test/common/checker.h"
#include "test/common/convolution.h"

#include "test/common/rng.h"

using namespace megdnn;
using namespace test;
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_F(AARCH64, CONVOLUTION_BACKWARD_DATA_FP16) {
    Checker<ConvolutionBackwardData> checker(handle());
    using Param = ConvolutionBackwardData::Param;
    Param param;
    auto run = [&](size_t n, size_t ic, size_t oh, size_t ow, size_t oc,
                   size_t fh, size_t fw, size_t stride, size_t padding,
                   size_t group = 1) {
        param.pad_h = param.pad_w = padding;
        param.stride_h = param.stride_w = stride;

        TensorLayout diff =
                TensorLayout{{n, oc * group, oh, ow}, dtype::Float16()};
        TensorLayout grad;
        TensorLayout filter;
        if (group == 1) {
            param.sparse = Param::Sparse::DENSE;
            filter = {{oc, ic, fh, fw}, dtype::Float16()};
        } else {
            param.sparse = Param::Sparse::GROUP;
            filter = {{group, oc, ic, fh, fw}, dtype::Float16()};
        }
        // TensorLayout grad;
        {
            auto opr = handle()->create_operator<ConvolutionBackwardData>();
            opr->param() = param;
            opr->deduce_layout(filter, diff, grad);
        }
        NormalRNG rng(10.f);
        checker.set_param(param)
                .set_dtype(0, dtype::Float16())
                .set_dtype(1, dtype::Float16())
                .set_dtype(2, dtype::Float16())
                .set_rng(0, &rng).set_rng(1, &rng)
                .set_epsilon(1e-2)
                .set_before_exec_callback(
                AlgoChecker<ConvolutionBackwardData>("DeconvMatmul"));
        checker.exec(TensorLayoutArray{filter, diff, grad});
    };

    for (auto mode :
         {Param::Mode::CONVOLUTION, Param::Mode::CROSS_CORRELATION}) {
        param.mode = mode;
        run(4, 3, 10, 13, 5, 1, 1, 1, 0, 1);
        run(4, 3, 10, 45, 2, 1, 1, 1, 0, 4);
        run(2, 3, 9, 12, 2, 4, 6, 1, 0, 1);
        run(3, 4, 17, 32, 2, 3, 2, 5, 4, 4);
        run(5, 5, 24, 43, 11, 9, 3, 3, 12, 2);
        run(2, 3, 20, 33, 3, 5, 7, 4, 15, 2);
        run(4, 4, 6, 7, 9, 3, 2, 2, 1, 3);
    }
}


#if MEGDNN_WITH_BENCHMARK
TEST_F(AARCH64, BENCHMARK_MATRIX_MUL_QUICK_FP16) {
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
        benchmarker_gemm.set_dtype(0, dtype::Float16())
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

TEST_F(AARCH64, BENCHMARK_MATRIX_MUL_ALL_SIZES_FP16) {
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
        benchmarker_gemm.set_dtype(0, dtype::Float16())
                .set_dtype(1, dtype::Float16());
        time = benchmarker_gemm.exec({{M, K}, {K, N}, {}});
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

#if MEGDNN_WITH_BENCHMARK
TEST_F(AARCH64, BENCHMARK_CONVOLUTION_STRIDE2) {
    using Param = param::Convolution;
    auto run = [&](const TensorShapeArray& shapes, Param param) {
        Benchmarker<Convolution> benchmarker_float(handle());
        size_t RUN = 50;
        auto tfloat =
                benchmarker_float.set_display(false)
                        .set_dtype(0, dtype::Float32{})
                        .set_dtype(1, dtype::Float32{})
                        .set_before_exec_callback(AlgoChecker<Convolution>(
                                "CONVOLUTION_DEFAULT_ARMV8F32STRD2_LARGE_"
                                "GROUP"))
                        .set_times(RUN)
                        .set_param(param)
                        .exec(shapes);
        size_t IC = shapes[1][1];
        size_t FH = shapes[1][2];
        size_t FW = shapes[1][3];
        TensorLayout dst_layout;
        auto opr = handle()->create_operator<Convolution>();
        opr->param() = param;
        opr->deduce_layout({shapes[0], dtype::Float32()},
                           {shapes[1], dtype::Float32()}, dst_layout);
        printf("fp32 flops: %.3f mflops\n",
               (IC * dst_layout.total_nr_elems() * FH * FW * 2) /
                       (tfloat / RUN * 1000));
    };
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    auto run1 = [&](const TensorShapeArray& shapes, Param param) {
        Benchmarker<Convolution> benchmarker_float(handle());
        size_t RUN = 50;
        auto tfloat =
                benchmarker_float.set_display(false)
                        .set_dtype(0, dtype::Float16())
                        .set_dtype(1, dtype::Float16())
                        .set_before_exec_callback(AlgoChecker<Convolution>(
                                "CONVOLUTION_DEFAULT_ARMV8F16STRD2_LARGE_"
                                "GROUP"))
                        .set_times(RUN)
                        .set_param(param)
                        .exec(shapes);
        size_t IC = shapes[1][1];
        size_t FH = shapes[1][2];
        size_t FW = shapes[1][3];
        TensorLayout dst_layout;
        auto opr = handle()->create_operator<Convolution>();
        opr->param() = param;
        opr->deduce_layout({shapes[0], dtype::Float16()},
                           {shapes[1], dtype::Float16()}, dst_layout);
        printf("fp16 flops: %.3f mflops\n",
               (IC * dst_layout.total_nr_elems() * FH * FW * 2) /
                       (tfloat / RUN * 1000));
    };
#endif
    auto profile = [&](size_t oc, size_t ic, size_t w, size_t h, size_t kernel,
                       size_t stride) {
        Param param;
        param.stride_h = stride;
        param.stride_w = stride;
        param.pad_h = kernel / 2;
        param.pad_w = kernel / 2;
        printf("oc: %zd ic: %zd w: %zd h: %zd stride: %zd kernel_size: %zd\n",
               oc, ic, w, h, stride, kernel);

        run({{1, ic, h, w}, {oc, ic, kernel, kernel}, {}}, param);
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        run1({{1, ic, h, w}, {oc, ic, kernel, kernel}, {}}, param);
#endif

    };

    for (size_t kernel : {2, 3, 5, 7}) {
        for (size_t ic : {3, 6, 12, 24}) {
            for (size_t oc : {3, 6, 12, 24}) {
                for (size_t size : {4, 7, 8, 14, 16, 17, 28, 32, 34, 64, 112}) {
                    profile(oc, ic, size, size, kernel, 2);
                }
            }
        }
    }
}
#endif


// vim: syntax=cpp.doxygen

