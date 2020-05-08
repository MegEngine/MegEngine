/**
 * \file dnn/test/armv7/pooling.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/armv7/fixture.h"

#include "test/common/pooling.h"
#include "test/common/checker.h"

namespace megdnn {
namespace test {

TEST_F(ARMV7, POOLING)
{
    auto args = pooling::get_args();
    for (auto &&arg: args) {
        Checker<Pooling> checker(handle());
        checker.set_param(arg.param).exec(TensorShapeArray{
                arg.ishape, {}});
    }
}

} // namespace test
} // namespace megdnn
// vim: syntax=cpp.doxygen


