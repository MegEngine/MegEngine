/**
 * \file dnn/test/cuda/check_non_finite.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "megdnn/oprs.h"
#include "test/common/checker.h"
#include "test/cuda/fixture.h"

namespace megdnn {
namespace test {

TEST_F(CUDA, CHECK_NON_FINITE_BASIC) {
    Checker<CheckNonFinite> checker(handle_cuda());
    checker.set_allow_invalid_check(true);
    const auto inf = std::numeric_limits<float>::infinity();
    const auto nan = std::numeric_limits<float>::quiet_NaN();
    UniformFloatWithValueRNG rng(-1.0f, 1.0f, 0.1f, inf);
    checker.set_rng(0, &rng);
    checker.execs({{512 * 4}, {4}, {1}});
    rng = UniformFloatWithValueRNG(-1.0f, 1.0f, 1.f, inf);
    checker.set_rng(0, &rng);
    checker.execs({{4}, {512 * 4}, {1}});
    rng = UniformFloatWithValueRNG(-1.0f, 1.0f, 1.f, nan);
    checker.set_rng(0, &rng);
    checker.execs({{32}, {256}, {1}});
    rng = UniformFloatWithValueRNG(-1.0f, 1.0f, 0.f, nan);
    checker.set_rng(0, &rng);
    checker.execs({{16}, {16}, {2}, {1}});
}

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
