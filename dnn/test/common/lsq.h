/**
 * \file dnn/test/common/lsq.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#pragma once
#include "megdnn/basic_types.h"
#include "megdnn/opr_param_defs.h"

namespace megdnn {
namespace test {
namespace lsq {

struct TestArg {
    param::LSQ param;
    TensorShape ishape;
    TensorShape scale_shape;
    TensorShape zeropoint_shape;
    TensorShape gradscale_shape;
    TestArg(param::LSQ param, TensorShape ishape, TensorShape scale_shape,
            TensorShape zeropoint_shape, TensorShape gradscale_shape)
            : param(param),
              ishape(ishape),
              scale_shape(scale_shape),
              zeropoint_shape(zeropoint_shape),
              gradscale_shape(gradscale_shape) {}
};

inline std::vector<TestArg> get_args() {
    std::vector<TestArg> args;
    param::LSQ cur_param;

    cur_param.qmin = -127;
    cur_param.qmax = 127;

    for (size_t i = 10; i < 30; i += 2) {
        args.emplace_back(
                cur_param, TensorShape{10, 64, i, i}, TensorShape{1}, TensorShape{1},
                TensorShape{1});
    }

    return args;
}

}  // namespace lsq
}  // namespace test
}  // namespace megdnn