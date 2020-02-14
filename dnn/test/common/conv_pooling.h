/**
 * \file dnn/test/common/conv_pooling.h
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
namespace conv_pooling {

struct TestArg {
    param::ConvPooling param;
    TensorShape src, filter, bias;
    TestArg(param::ConvPooling param, TensorShape src, TensorShape filter,
            TensorShape bias)
            : param(param), src(src), filter(filter), bias(bias) {}
};

std::vector<TestArg> get_args();

}  // namespace conv_pooling
}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
