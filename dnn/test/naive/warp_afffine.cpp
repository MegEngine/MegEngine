/**
 * \file dnn/test/naive/warp_afffine.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/naive/fixture.h"

#include "megdnn/oprs/cv.h"
#include "test/common/checker.h"
#include "test/common/warp_affine.h"

using namespace megdnn;
using namespace test;

TEST_F(NAIVE, WARP_AFFINE) {
    Checker<WarpAffine> checker(handle(), false);
    WarpAffine::Param param;
    param.border_mode = WarpAffine::Param::BorderMode::BORDER_REFLECT;
    param.imode = WarpAffine::Param::InterpolationMode::LINEAR;
    param.format = WarpAffine::Param::Format::NCHW;


    checker.set_param(param).exect(
            Testcase{TensorValue({1, 1, 3, 3}, dtype::Uint8{},
                                 {131, 255, 180, 245, 8, 0, 10, 3, 178}),

                     TensorValue({1, 2, 3}, dtype::Float32{},
                                 {1.2f, 1.2f, 0.6f, -1.05f, -2.0f, -0.7f}),
                     {}},
            Testcase{{},
                     {},
                     TensorValue({1, 1, 2, 2}, dtype::Uint8{},
                                 {205, 50, 101, 178})});

    checker.set_param(param).exect(
            Testcase{TensorValue({1, 1, 3, 3},
                                 dtype::Quantized8Asymm{
                                         1.4f, static_cast<uint8_t>(127)},
                                 {131, 255, 180, 245, 8, 0, 10, 3, 178}),

                     TensorValue({1, 2, 3}, dtype::Float32{},
                                 {1.2f, 1.2f, 0.6f, -1.05f, -2.0f, -0.7f}),
                     {}},
            Testcase{{},
                     {},
                     TensorValue({1, 1, 2, 2},
                                 dtype::Quantized8Asymm{
                                         1.4f, static_cast<uint8_t>(127)},
                                 {205, 50, 101, 178})});
}

TEST_F(NAIVE_MULTI_THREADS, WARP_AFFINE_CV) {
    using namespace warp_affine;
    std::vector<TestArg> args = get_cv_args();
    Checker<WarpAffine> checker(handle());

    for (auto &&arg: args) {
        checker.set_param(arg.param)
            .set_dtype(0, dtype::Uint8())
            .set_dtype(1, dtype::Float32())
            .set_dtype(2, dtype::Uint8())
            .execs({arg.src, arg.trans, arg.dst});
    }

    for (auto &&arg: args) {
        checker.set_param(arg.param)
            .set_dtype(0, dtype::Float32())
            .set_dtype(1, dtype::Float32())
            .set_dtype(2, dtype::Float32())
            .execs({arg.src, arg.trans, arg.dst});
    }

}

// vim: syntax=cpp.doxygen
