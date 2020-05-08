/**
 * \file dnn/test/arm_common/winograd_filter_preprocess.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/common/checker.h"
#include "test/common/benchmarker.h"
#include "test/common/winograd_filter_preprocess.h"

#include "test/arm_common/fixture.h"

using namespace megdnn;
using namespace test;

TEST_F(ARM_COMMON, WinogradFilterPreprocessF32) {
    using namespace winograd_filter_preprocess;
    Checker<WinogradFilterPreprocess> checker(handle());
    // default
    std::vector<TestArg> args = get_args(6, 3);
    std::vector<TestArg> args54 = get_args(5, 4);
    std::vector<TestArg> args45 = get_args(4, 5);

    // mk4
    std::vector<TestArg> args_mk4_out2 =
            get_mk_packed_args(2, param::Winograd::Format::MK4, 4);
    std::vector<TestArg> args_mk4_out6 =
            get_mk_packed_args(6, param::Winograd::Format::MK4, 4);

    args.insert(args.end(), args54.begin(), args54.end());
    args.insert(args.end(), args45.begin(), args45.end());
    args.insert(args.end(), args_mk4_out2.begin(), args_mk4_out2.end());
    args.insert(args.end(), args_mk4_out6.begin(), args_mk4_out6.end());
    for (auto&& arg : args) {
        checker.set_param(arg.param)
                .set_dtype(0, dtype::Float32{})
                .set_dtype(1, dtype::Float32{})
                .execs({arg.src, {}});
    }
}

TEST_F(ARM_COMMON, WinogradFilterPreprocessQs8) {
    using namespace winograd_filter_preprocess;
    std::vector<TestArg> args =
            get_mk_packed_args(2, param::Winograd::Format::MK8, 8);
    Checker<WinogradFilterPreprocess> checker(handle());
    UniformIntRNG rng{-50, 50};
    checker.set_rng(0, &rng).set_rng(1, &rng).set_rng(2, &rng);
    for (auto&& arg : args) {
        checker.set_param(arg.param)
                .set_dtype(0, dtype::QuantizedS8(2.5f))
                .set_dtype(1, dtype::QuantizedS16(2.5f))
                .execs({arg.src, {}});
    }
}

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
TEST_F(ARM_COMMON, WinogradFilterPreprocessF16) {
    using namespace winograd_filter_preprocess;
    Checker<WinogradFilterPreprocess> checker(handle());
    // default
    std::vector<TestArg> args = get_args(6, 3);
    std::vector<TestArg> args_23 =
            get_mk_packed_args(2, param::Winograd::Format::DEFAULT, 4);
    std::vector<TestArg> args45 = get_args(4, 5);

    // mk8
    std::vector<TestArg> args_mk8_out2 =
            get_mk_packed_args(2, param::Winograd::Format::MK8, 8);

    args.insert(args.end(), args_23.begin(), args_23.end());
    args.insert(args.end(), args45.begin(), args45.end());
    args.insert(args.end(), args_mk8_out2.begin(), args_mk8_out2.end());

    Float16PeriodicalRNG* rng = new Float16PeriodicalRNG(0x3c00);
    for (auto&& arg : args) {
        checker.set_param(arg.param)
                .set_rng(0, rng)
                .set_dtype(0, dtype::Float16{})
                .set_dtype(1, dtype::Float16{})
                .execs({arg.src, {}});
    }
}

#endif

// vim: syntax=cpp.doxygen
