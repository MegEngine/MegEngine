/**
 * \file dnn/test/common/adaptive_pooling.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#pragma once
#include <cstddef>
#include "megdnn/basic_types.h"
#include "megdnn/opr_param_defs.h"

namespace megdnn {
namespace test {
namespace adaptive_pooling {

struct TestArg {
    param::AdaptivePooling param;
    TensorShape ishape;
    TensorShape oshape;
    TestArg(param::AdaptivePooling param, TensorShape ishape,
            TensorShape oshape)
            : param(param), ishape(ishape), oshape(oshape) {}
};

inline std::vector<TestArg> get_args() {
    std::vector<TestArg> args;
    using Param = param::AdaptivePooling;
    using Mode = param::AdaptivePooling::Mode;

    for (size_t i = 36; i < 40; ++i) {
        args.emplace_back(Param{Mode::AVERAGE}, TensorShape{2, 3, i, i + 1},
                          TensorShape{2, 3, i - 4, i - 2});
        args.emplace_back(Param{Mode::MAX}, TensorShape{2, 3, i, i + 1},
                          TensorShape{2, 3, i - 4, i - 2});
    }

    for (size_t i = 5; i < 10; ++i) {
        args.emplace_back(Param{Mode::AVERAGE}, TensorShape{2, 3, i, i + 1},
                          TensorShape{2, 3, i - 3, i - 2});
        args.emplace_back(Param{Mode::MAX}, TensorShape{2, 3, i, i + 1},
                          TensorShape{2, 3, i - 3, i - 2});
    }
    return args;
}

}  // namespace adaptive_pooling
}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
