/**
 * \file dnn/test/arm_common/separable_filter.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/arm_common/fixture.h"
#include "test/common/separable_filter.h"
#include "test/common/checker.h"
#include "test/common/rng.h"

namespace megdnn {
namespace test {

TEST_F(ARM_COMMON, SEPARABLE_FILTER)
{
    using namespace separable_filter;
    std::vector<TestArg> args = get_args();
    Checker<SeparableFilter> checker(handle());
    for (auto &&arg: args) {
        checker.set_param(arg.param).execs({arg.src, arg.filter_x, arg.filter_y, {}});
    }

    checker.set_dtype(0, dtype::Uint8())
            .set_dtype(1, dtype::Float32())
            .set_dtype(2, dtype::Float32())
            .set_dtype(3, dtype::Uint8())
            .set_epsilon(1+1e-3);
    for (auto&& arg : args) {
        checker.set_param(arg.param).execs(
                {arg.src, arg.filter_x, arg.filter_y, {}});
    }
}

} // namespace test
} // namespace megdnn
// vim: syntax=cpp.doxygen
