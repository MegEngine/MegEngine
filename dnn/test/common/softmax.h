/**
 * \file dnn/test/common/softmax.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include <cstddef>
#include "megdnn/basic_types.h"
#include "megdnn/opr_param_defs.h"

namespace megdnn {
namespace test {
namespace softmax {

struct TestArg {
    param::Softmax param;
    TensorShape ishape;
    TestArg(param::Softmax param, TensorShape ishape) : param(param), ishape(ishape) {}
};

inline std::vector<TestArg> get_args() {
    std::vector<TestArg> args;
    using Param = param::Softmax;

    for (int32_t axis = 0; axis < 5; axis++) {
        args.emplace_back(Param{axis}, TensorShape{2, 23, 32, 30, 17});
    }

    return args;
}

}  // namespace softmax
}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen