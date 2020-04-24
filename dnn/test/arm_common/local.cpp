/**
 * \file dnn/test/arm_common/local.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/arm_common/fixture.h"

#include "test/common/benchmarker.h"
#include "test/common/checker.h"
#include "test/common/local.h"
#include "test/common/timer.h"

namespace megdnn {
namespace test {
using Param = param::Convolution;

TEST_F(ARM_COMMON, LOCAL_FORWARD) {
    auto args = local::get_args();
    Checker<LocalForward> checker(handle());
    for (auto&& arg : args) {
        checker.set_param(arg.param).execs(
                {arg.sshape(), arg.fshape(), arg.dshape()});
    }

    NormalRNG rng(10.f);
    checker.set_rng(0, &rng).set_rng(1, &rng);
    args = local::get_args_for_fp16();

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    for (auto&& arg : args) {
        checker.set_dtype(0, dtype::Float16())
                .set_dtype(1, dtype::Float16())
                .set_dtype(2, dtype::Float16());
        checker.set_epsilon(1e-2);
        checker.set_param(arg.param).execs(
                {arg.sshape(), arg.fshape(), arg.dshape()});
    }
#endif
}

#if MEGDNN_WITH_BENCHMARK
TEST_F(ARM_COMMON, BENCHMARK_LOCAL_FORWARD) {
    auto run = [&](const TensorShapeArray& shapes, Param param) {
        Benchmarker<LocalForward> benchmarker(handle());
        size_t RUN = 50;
        benchmarker.set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .set_dtype(2, dtype::Float32());
        auto tfloat32 = benchmarker.set_display(true)
                                .set_times(RUN)
                                .set_param(param)
                                .exec(shapes);
        int N = shapes[0][0];
        int IC = shapes[0][1];
        int IH = shapes[0][2];
        int IW = shapes[0][3];
        int OH = shapes[1][0];
        int OW = shapes[1][1];
        int FH = shapes[1][3];
        int FW = shapes[1][4];
        int OC = shapes[1][5];
        std::cout << "LOCAL FORWARD, src: {" << N << ", " << IC << ", " << IH
                  << ", " << IW << "}" << std::endl;
        std::cout << "LOCAL FORWARD, filter: {" << OH << ", " << OW << ", "
                  << IC << ", " << FH << ", " << FW << ", " << OC << "}"
                  << std::endl;
        std::cout << "LOCAL FORWARD (f32), bandwidth: "
                  << (1.f * N * OC * OH * OW * FH * FW * IC +
                      1.f * N * IC * IH * IW) *
                             sizeof(float) * 1e-9 / (tfloat32 / RUN * 1e-3)
                  << "GBPS" << std::endl;

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        benchmarker.set_dtype(0, dtype::Float16())
                .set_dtype(1, dtype::Float16())
                .set_dtype(2, dtype::Float16());
        auto tfloat16 = benchmarker.set_display(true)
                                .set_times(RUN)
                                .set_param(param)
                                .exec(shapes);
        std::cout << "LOCAL FORWARD (f16), bandwidth: "
                  << (1.f * N * OC * OH * OW * FH * FW * IC +
                      1.f * N * IC * IH * IW) *
                             sizeof(dt_float16) * 1e-9 / (tfloat16 / RUN * 1e-3)
                  << "GBPS" << std::endl;
#endif
    };

    Param param;
    param.mode = param::Convolution::Mode::CONVOLUTION;
    param.pad_h = param.pad_w = 1;
    param.stride_h = param.stride_w = 1;
    run({{1, 4, 320, 256}, {320, 256, 4, 3, 3, 24}, {}}, param);
    param.stride_h = param.stride_w = 2;
    run({{1, 4, 320, 256}, {160, 128, 4, 3, 3, 24}, {}}, param);

    param.pad_h = param.pad_w = 2;
    param.stride_h = param.stride_w = 1;
    run({{1, 4, 64, 64}, {64, 64, 4, 5, 5, 24}, {}}, param);
    param.stride_h = param.stride_w = 2;
    run({{1, 4, 64, 64}, {32, 32, 4, 5, 5, 24}, {}}, param);

    param.pad_h = param.pad_w = 3;
    param.stride_h = param.stride_w = 1;
    run({{1, 4, 64, 64}, {64, 64, 4, 7, 7, 24}, {}}, param);
    param.stride_h = param.stride_w = 2;
    run({{1, 4, 64, 64}, {32, 32, 4, 7, 7, 24}, {}}, param);

    param.pad_h = param.pad_w = 1;
    param.stride_h = param.stride_w = 1;
    run({{2, 128, 8, 8}, {8, 8, 128, 3, 3, 128}, {}}, param);
    run({{1, 16, 64, 64}, {64, 64, 16, 3, 3, 16}, {}}, param);
}
#endif

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
