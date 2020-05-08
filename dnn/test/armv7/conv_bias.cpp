/**
 * \file dnn/test/armv7/conv_bias.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/armv7/fixture.h"

#include "test/common/convolution.h"
#include "test/common/conv_bias.h"
#include "test/common/checker.h"
#include "test/common/benchmarker.h"

#include "test/common/rng.h"

using namespace megdnn;
using namespace test;

namespace{

TEST_F(ARMV7, CONV_BIAS_MATMUL_QU8) {
    using namespace conv_bias;
    std::vector<TestArg> args = get_quantized_args();
    Checker<ConvBiasForward> checker(handle());
    checker.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBias>("QU8MATMUL"));

    UniformIntRNG rng{0, 127};
    for (auto&& arg : args) {
        if (arg.bias.ndim == 4 && arg.bias[2] != 1 && arg.bias[3] != 1)
            continue;
        checker.set_dtype(0, dtype::Quantized8Asymm(2.5f,
                                                    static_cast<uint8_t>(127)))
                .set_dtype(1, dtype::Quantized8Asymm(2.7f,
                                                     static_cast<uint8_t>(129)))
                .set_dtype(2, dtype::QuantizedS32(6.75f))
                .set_dtype(4, dtype::Quantized8Asymm(60.25f,
                                                     static_cast<uint8_t>(125)))
                .set_rng(0, &rng)
                .set_rng(1, &rng)
                .set_rng(2, &rng)
                .set_param(arg.param)
                .execs({arg.src, arg.filter, arg.bias, {}, {}});
    }
}

TEST_F(ARMV7, CONV_BIAS_MATMUL_QS8) {
    using namespace conv_bias;
    std::vector<TestArg> args = get_quantized_args();
    Checker<ConvBiasForward> checker(handle());
    checker.set_before_exec_callback(
            conv_bias::ConvBiasAlgoChecker<ConvBias>("S8MATMUL"));

    UniformIntRNG rng{0, 127};
    for (auto&& arg : args) {
        if (arg.bias.ndim == 4 && arg.bias[2] != 1 && arg.bias[3] != 1)
            continue;
        checker.set_dtype(0, dtype::QuantizedS8(2.5f))
                .set_dtype(1, dtype::QuantizedS8(2.7f))
                .set_dtype(2, dtype::QuantizedS32(6.75f))
                .set_dtype(4, dtype::QuantizedS8(60.25f))
                .set_rng(0, &rng)
                .set_rng(1, &rng)
                .set_rng(2, &rng)
                .set_param(arg.param)
                .set_epsilon(1.0f)
                .execs({arg.src, arg.filter, arg.bias, {}, {}});
    }
}
}
// vim: syntax=cpp.doxygen
