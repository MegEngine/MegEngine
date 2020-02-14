/**
 * \file dnn/test/fallback/repeat.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/fallback/fixture.h"

#include "test/common/checker.h"
#include "test/common/tile_repeat.h"

namespace megdnn {
namespace test {

TEST_F(FALLBACK, REPEAT)
{
    Checker<RepeatForward> checker(handle());
    auto args = tile_repeat::get_args();
    for (auto &&arg: args) {
        checker.set_param(arg.param()).execs({arg.src, {}});
    }
}

} // namespace test
} // namespace megdnn

// vim: syntax=cpp.doxygen


