/**
 * \file dnn/test/common/separable_conv.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "megdnn/opr_param_defs.h"
#include "megdnn/basic_types.h"

namespace megdnn {
namespace test {
namespace separable_conv {

struct TestArg {
    param::SeparableConv param;
    TensorShape src, filter_x, filter_y;
    TestArg(param::SeparableConv param, TensorShape src, TensorShape filter_x,
            TensorShape filter_y)
            : param(param), src(src), filter_x(filter_x), filter_y(filter_y) {}
};

std::vector<TestArg> get_args();

}  // namespace separable_conv
}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
