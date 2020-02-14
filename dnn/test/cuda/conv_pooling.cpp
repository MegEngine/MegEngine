/**
 * \file dnn/test/cuda/conv_pooling.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/cuda/fixture.h"

#include "test/common/tensor.h"
#include "megdnn/oprs.h"
#include "test/common/workspace_wrapper.h"
#include "megdnn/opr_param_defs.h"
#include "test/common/checker.h"
#include "test/common/conv_pooling.h"
#include "test/common/rng.h"

namespace megdnn {
namespace test {

#if 0
TEST_F(CUDA, CONV_POOLING_FORWARD)
{
    using namespace conv_pooling;
    std::vector<TestArg> args = get_args();
    Checker<ConvPoolingForward> checker(handle_cuda());
    NormalRNG default_rng;
    ConstValue const_val;
    for (auto &&arg: args) {
        float scale = 1.0f / sqrt(arg.filter[1] * arg.filter[2] * arg.filter[3]);
        UniformFloatRNG rng(scale, 2 * scale);
        checker.
            set_dtype(0, dtype::Float32()).
            set_dtype(1, dtype::Float32()).
            set_dtype(2, dtype::Float32()).
            set_rng(0, &default_rng).
            set_rng(1, &default_rng).
            set_rng(2, &default_rng).
            set_epsilon(1e-3).
            set_param(arg.param).
            execs({arg.src, arg.filter, arg.bias, {}});
        /*checker.
            set_dtype(0, dtype::Float16()).
            set_dtype(1, dtype::Float16()).
            set_rng(0, &rng).
            set_rng(1, &rng).
            set_epsilon(1e-1).
            set_param(arg.param).
            execs({arg.src, arg.filter, {}});
        */
    }
}
#endif


} // namespace test
} // namespace megdnn

// vim: syntax=cpp.doxygen
