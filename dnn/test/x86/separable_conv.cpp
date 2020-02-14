/**
 * \file dnn/test/x86/separable_conv.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/x86/fixture.h"
#include "test/common/separable_conv.h"
#include "test/common/checker.h"

namespace megdnn {
namespace test {

TEST_F(X86, SEPARABLE_CONV)
{
    using namespace separable_conv;
    std::vector<TestArg> args = get_args();
    Checker<SeparableConvForward> checker(handle());

    for (auto &&arg: args) {
        checker.set_param(arg.param).execs({arg.src, arg.filter_x, arg.filter_y, {}});
    }
}

} // namespace test
} // namespace megdnn
// vim: syntax=cpp.doxygen
