/**
 * \file dnn/test/cuda/repeat.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/cuda/fixture.h"

#include "test/common/checker.h"
#include "test/common/tile_repeat.h"

namespace megdnn {
namespace test {

TEST_F(CUDA, REPEAT_FORWARD)
{
    Checker<RepeatForward> checker(handle_cuda());
    auto args = tile_repeat::get_args();
    for (auto &&arg: args) {
        checker.set_dtype(0, dtype::Float32()).
            set_param(arg.param()).execs({arg.src, {}});
        checker.set_dtype(0, dtype::Float16()).
            set_param(arg.param()).execs({arg.src, {}});
    }
}

TEST_F(CUDA, REPEAT_BACKWARD)
{
    Checker<RepeatBackward> checker(handle_cuda());
    UniformFloatRNG rng(1, 2);
    checker.set_epsilon(1e-2).set_rng(0, &rng);;
    auto args = tile_repeat::get_args();
    for (auto &&arg: args) {
        checker.set_dtype(0, dtype::Float32()).set_dtype(1, dtype::Float32()).
            set_param(arg.param()).execs({arg.dst, arg.src});
        checker.set_dtype(0, dtype::Float16()).set_dtype(1, dtype::Float16()).
            set_param(arg.param()).execs({arg.dst, arg.src});
    }
}

} // namespace test
} // namespace megdnn

// vim: syntax=cpp.doxygen

