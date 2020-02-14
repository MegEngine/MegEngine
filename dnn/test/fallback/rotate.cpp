/**
 * \file dnn/test/fallback/rotate.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include <gtest/gtest.h>

#include "megdnn.h"
#include "megdnn/oprs.h"
#include "test/common/tensor.h"
#include "test/common/rotate.h"
#include "test/common/checker.h"
#include "test/fallback/fixture.h"

namespace megdnn {
namespace test {

TEST_F(FALLBACK, ROTATE) {
    using namespace rotate;
    std::vector<TestArg> args = get_args();
    Checker<Rotate> checker(handle());
    checker.set_dtype(0, dtype::Int32());
    checker.set_dtype(1, dtype::Int32());

    for (auto&& arg : args) {
        checker.set_dtype(0, arg.dtype)
            .set_dtype(1, arg.dtype)
            .execs({arg.src, {}});
    }
}

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
