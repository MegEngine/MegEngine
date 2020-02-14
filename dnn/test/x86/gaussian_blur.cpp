/**
 * \file dnn/test/x86/gaussian_blur.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/x86/fixture.h"
#include "test/common/gaussian_blur.h"
#include "test/common/checker.h"

namespace megdnn {
namespace test {

TEST_F(X86, GAUSSIAN_BLUR)
{
    using namespace gaussian_blur;
    std::vector<TestArg> args = get_args();
    Checker<GaussianBlur> checker(handle());

    for (auto &&arg: args) {
        checker.set_param(arg.param)
            .set_dtype(0, dtype::Float32())
            .set_dtype(1, dtype::Float32())
            .execs({arg.src, {}});
    }

    for (auto &&arg: args) {
        checker.set_param(arg.param)
            .set_epsilon(1+1e-3)
            .set_dtype(0, dtype::Uint8())
            .set_dtype(1, dtype::Uint8())
            .execs({arg.src, {}});
    }
}

} // namespace test
} // namespace megdnn
// vim: syntax=cpp.doxygen
