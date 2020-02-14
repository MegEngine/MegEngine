/**
 * \file dnn/test/common/rotate.h
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
namespace rotate {

struct TestArg {
    param::Rotate param;
    TensorShape src;
    DType dtype;
    TestArg(param::Rotate param, TensorShape src, DType dtype)
            : param(param), src(src), dtype(dtype) {}
};

static inline std::vector<TestArg> get_args() {
    std::vector<TestArg> args;

    param::Rotate cur_param;
    for (size_t i = 8; i < 129; i *= 4) {
        cur_param.clockwise = true;
        args.emplace_back(cur_param, TensorShape{1, i, i, 1}, dtype::Uint8());
        args.emplace_back(cur_param, TensorShape{1, i, i, 3}, dtype::Uint8());
        args.emplace_back(cur_param, TensorShape{2, i, i, 3}, dtype::Uint8());
        args.emplace_back(cur_param, TensorShape{2, i, i, 3}, dtype::Float32());

        cur_param.clockwise = false;
        args.emplace_back(cur_param, TensorShape{2, i, i, 3}, dtype::Uint8());
        args.emplace_back(cur_param, TensorShape{2, i, i, 3}, dtype::Float32());
    }

    std::vector<std::pair<size_t, size_t>> test_cases = {
            {23, 28}, {17, 3}, {3, 83}};
    for (auto&& item : test_cases) {
        for (auto&& CH : {1U, 3U}) {
            for (bool clockwise : {false, true}) {
                cur_param.clockwise = clockwise;
                args.emplace_back(cur_param,
                                  TensorShape{1, item.first, item.second, CH},
                                  dtype::Uint8());
                args.emplace_back(cur_param,
                                  TensorShape{1, item.first, item.second, CH},
                                  dtype::Float32());
            }
        }
    }
    return args;
}

}  // namespace rotate
}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
