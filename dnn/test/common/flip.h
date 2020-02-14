/**
 * \file dnn/test/common/flip.h
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
namespace flip {

struct TestArg {
    param::Flip param;
    TensorShape src;
    TestArg(param::Flip param_, TensorShape src_) : param(param_), src(src_) {}
};

static inline std::vector<TestArg> get_args() {
    std::vector<TestArg> args;
    param::Flip cur_param;

    for (size_t h : {4, 5}) {
        for (size_t w : {3, 4}) {
            for (size_t c : {1, 3}) {
                for (bool vertical : {false, true}) {
                    for (bool horizontal : {false, true}) {
                        cur_param.horizontal = horizontal;
                        cur_param.vertical = vertical;
                        args.emplace_back(cur_param, TensorShape{2, h, w, c});
                    }
                }
            }
        }
    }

    return args;
}

}  // namespace flip
}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
