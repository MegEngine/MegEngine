/**
 * \file dnn/test/common/bn.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "megdnn/basic_types.h"
#include "megdnn/opr_param_defs.h"

namespace megdnn {
namespace test {
namespace batch_normalization {

struct TestArg {
    param::BN param;
    TensorShape src, param_shape;
    DType dtype;
    TestArg(param::BN param, TensorShape src, TensorShape param_shape,
            DType dtype)
            : param(param), src(src), param_shape(param_shape), dtype(dtype) {}
};

std::vector<TestArg> get_args() {
    std::vector<TestArg> args;
    // Case 1
    // ParamDim: 1 x 1 x H x W
    // N = 3, C = 3
    for (size_t i = 4; i < 257; i *= 4) {
        param::BN param;
        param.fwd_mode = param::BN::FwdMode::TRAINING;
        param.param_dim = param::BN::ParamDim::DIM_11HW;
        param.avg_factor = 1.f;
        args.emplace_back(param, TensorShape{2, 3, i, i},
                          TensorShape{1, 1, i, i}, dtype::Float32());
        args.emplace_back(param, TensorShape{2, 3, i, i},
                          TensorShape{1, 1, i, i}, dtype::Float16());
    }

    // case 2: 1 x C x 1 x 1

    for (size_t i = 4; i < 257; i *= 4) {
        param::BN param;
        param.fwd_mode = param::BN::FwdMode::TRAINING;
        param.param_dim = param::BN::ParamDim::DIM_1C11;
        args.emplace_back(param, TensorShape{3, 3, i, i},
                          TensorShape{1, 3, 1, 1}, dtype::Float32());
        args.emplace_back(param, TensorShape{3, 3, i, i},
                          TensorShape{1, 3, 1, 1}, dtype::Float16());
    }

    return args;
}

}  // namespace batch_normalization
}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen