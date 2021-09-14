/**
 * \file test/naive/check_non_finite.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "test/naive/fixture.h"

#include "megdnn/oprs.h"
#include "test/common/checker.h"

namespace megdnn {
namespace test {

TEST_F(NAIVE, CHECK_NON_FINITE_BASIC) {
    Checker<CheckNonFinite> checker(handle(), false);
    checker.exect(Testcase{TensorValue({4}, dtype::Float32(),
                                       {1.1, 2.2, 3.3, 4.3}),
                           {}},
                  Testcase{{}, TensorValue({1}, dtype::Int32(), {0})});
    checker.exect(
            Testcase{TensorValue({4}, dtype::Float32(),
                                 {1.1f, 2.2f, 3.3f,
                                  std::numeric_limits<float>::infinity()}),
                     {}},
            Testcase{{}, TensorValue({1}, dtype::Int32(), {1})});
    checker.exect(
            Testcase{TensorValue({4}, dtype::Float32(),
                                 {1.1f, 2.2f, 3.3f,
                                  std::numeric_limits<float>::quiet_NaN()}),
                     {}},
            Testcase{{}, TensorValue({1}, dtype::Int32(), {1})});
}

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
