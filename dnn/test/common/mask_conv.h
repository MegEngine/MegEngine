/**
 * \file dnn/test/common/mask_conv.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "megdnn/oprs.h"
#include "test/common/benchmarker.h"
#include "test/common/checker.h"
#include "test/common/rng.h"

#pragma once

namespace {

using namespace megdnn;
using namespace test;

std::vector<std::vector<int>> get_args() {
    std::vector<std::vector<int>> args;
    args.push_back({2, 1, 1, 5, 5, 3, 3, 1, 1, 0, 0, 1, 1});
    args.push_back({1, 2, 3, 24, 24, 3, 5, 1, 1, 0, 0, 1, 1});
    args.push_back({20, 3, 4, 24, 21, 5, 3, 1, 1, 0, 0, 1, 1});
    args.push_back({20, 3, 4, 24, 21, 5, 3, 2, 2, 0, 0, 1, 1});
    args.push_back({20, 3, 4, 24, 21, 5, 3, 2, 2, 2, 2, 1, 1});
    args.push_back({20, 3, 4, 24, 21, 5, 3, 2, 2, 1, 2, 1, 1});
    args.push_back({20, 3, 4, 24, 21, 5, 3, 2, 2, 1, 2, 2, 3});
    args.push_back({20, 3, 4, 24, 21, 5, 3, 2, 2, 1, 2, 3, 2});

    args.push_back({2, 108, 108, 14, 14, 3, 3, 1, 1, 0, 0, 1, 1});
    args.push_back({2, 108, 108, 14, 14, 3, 3, 1, 1, 2, 2, 1, 1});
    args.push_back({2, 108, 108, 14, 14, 3, 3, 2, 2, 2, 2, 1, 1});
    args.push_back({2, 108, 108, 14, 14, 3, 3, 2, 2, 0, 0, 1, 1});

    args.push_back({2, 3, 3, 224, 224, 3, 3, 1, 1, 0, 0, 1, 1});
    args.push_back({2, 3, 3, 224, 224, 3, 3, 2, 2, 0, 0, 1, 1});
    return args;
}

void mask_conv_test(Handle* handle) {
    auto run = [&](size_t N, size_t IC, size_t OC, size_t IH, size_t IW,
                   size_t FH, size_t FW, size_t SH, size_t SW, size_t PH,
                   size_t PW, size_t DH, size_t DW) {
        size_t OH = (IH + 2 * PH - ((FH - 1) * DH + 1)) / SH + 1;
        size_t OW = (IW + 2 * PW - ((FW - 1) * DW + 1)) / SW + 1;
        Checker<MaskConvolution> checker(handle);
        using Param = param::Convolution;
        Param param(Param::Mode::CROSS_CORRELATION,
                    // pad
                    PH, PW,
                    // stride
                    SH, SW,
                    // dilate
                    DH, DW, Param::Sparse::DENSE, Param::Format::NCHW);
        TensorShape src_shape({N, IC, IH, IW}), filter_shape({OC, IC, FH, FW}),
                mask({OH, OW}), dst({});
        auto rng = std::make_unique<BernoulliRNG>(0.5);
        checker.set_param(param);

        checker.set_dtype(2, dtype::Int8())
                .execs({src_shape, filter_shape, mask, dst});
        checker.set_dtype(2, dtype::Int16())
                .execs({src_shape, filter_shape, mask, dst});
        checker.set_dtype(2, dtype::Int32())
                .execs({src_shape, filter_shape, mask, dst});
    };
    auto test_args = get_args();
    for (auto&& arg : test_args) {
        run(arg[0], arg[1], arg[2], arg[3], arg[4], arg[5], arg[6], arg[7],
            arg[8], arg[9], arg[10], arg[11], arg[12]);
    }
}
#if MEGDNN_WITH_BENCHMARK
void mask_conv_benchmark(Handle* handle) {
    auto benchmark = [&](size_t N, size_t IC, size_t OC, size_t IH, size_t IW,
                         size_t FH, size_t FW, size_t SH, size_t SW, size_t PH,
                         size_t PW, size_t DH, size_t DW) {
        size_t OH = (IH + 2 * PH - ((FH - 1) * DH + 1)) / SH + 1;
        size_t OW = (IW + 2 * PW - ((FW - 1) * DW + 1)) / SW + 1;
        Benchmarker<MaskConvolution> benchmark_fallback(handle);
        Benchmarker<Convolution> benchmark_naive(handle);
        using Param = param::Convolution;
        Param param(Param::Mode::CROSS_CORRELATION,
                    // pad
                    PH, PW,
                    // stride
                    SH, SW,
                    // dilate
                    DH, DW, Param::Sparse::DENSE, Param::Format::NCHW);
        TensorShape src_shape({N, IC, IH, IW}), filter_shape({OC, IC, FH, FW}),
                mask({OH, OW}), dst({});
        benchmark_fallback.set_param(param)
                .set_dtype(2, dtype::Int32())
                .set_times(20);
        printf("Execing mask conv: \n");
#define test(p)                                        \
    benchmark_fallback.set_rng(2, new BernoulliRNG(p)) \
            .execs({src_shape, filter_shape, mask, dst})
        for (auto p : {0.1, 0.2, 0.3, 0.4, 0.5, 0.99})
            test(p);
        printf("Execing normal conv: \n");
        benchmark_naive.set_param(param).set_times(20).execs(
                {src_shape, filter_shape, dst});
#undef test
    };
    auto test_args = get_args();
    for (auto&& arg : test_args) {
        benchmark(arg[0], arg[1], arg[2], arg[3], arg[4], arg[5], arg[6],
                  arg[7], arg[8], arg[9], arg[10], arg[11], arg[12]);
    }
}
#endif

}  // namespace
