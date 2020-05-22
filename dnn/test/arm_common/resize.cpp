/**
 * \file dnn/test/arm_common/resize.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/arm_common/fixture.h"
#include "test/common/resize.h"
#include "test/common/checker.h"

namespace megdnn {
namespace test {

TEST_F(ARM_COMMON, RESIZE_CV)
{
    using namespace resize;
    std::vector<TestArg> args = get_cv_args();
    Checker<Resize> checker(handle());

    for (auto &&arg: args) {
        checker.set_param(arg.param)
            .set_epsilon(1 + 1e-3)
            .set_dtype(0, dtype::Uint8())
            .set_dtype(1, dtype::Uint8())
            .execs({arg.src, arg.dst});
    }

    for (auto &&arg: args) {
        checker.set_param(arg.param)
            .set_dtype(0, dtype::Float32())
            .set_dtype(1, dtype::Float32())
            .execs({arg.src, arg.dst});
    }

}

} // namespace test
} // namespace megdnn

// vim: syntax=cpp.doxygen

