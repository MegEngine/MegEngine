/**
 * \file dnn/test/fallback/flip.cpp
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
#include "test/common/flip.h"
#include "test/common/checker.h"
#include "test/fallback/fixture.h"

namespace megdnn {
namespace test {

TEST_F(FALLBACK, FLIP) {
    using namespace flip;
    std::vector<TestArg> args = get_args();
    Checker<Flip> checker(handle());
    checker.set_dtype(0, dtype::Int32());
    checker.set_dtype(1, dtype::Int32());

    for (auto &&arg : args) {
        checker.execs({arg.src, {}});
    }

}

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
