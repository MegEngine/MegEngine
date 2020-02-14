/**
 * \file dnn/test/common/pooling.h
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
#include <cstddef>

namespace megdnn {
namespace test {
namespace pooling {

struct TestArg {
    param::Pooling param;
    TensorShape ishape;
    TestArg(param::Pooling param, TensorShape ishape)
            : param(param), ishape(ishape) {}
};

inline std::vector<TestArg> get_args() {
    std::vector<TestArg> args;
    using Param = param::Pooling;
    using Mode = param::Pooling::Mode;
    // ppssww
    for (size_t i = 32; i < 40; ++i) {
        args.emplace_back(Param{Mode::AVERAGE, 1, 1, 2, 2, 2, 2},
                          TensorShape{2, 3, i, i + 1});
        /* reserved for future test */
        /*
        args.emplace_back(Param{Mode::AVERAGE_COUNT_EXCLUDE_PADDING, 1, 1, 2, 2,
        2, 2}, TensorShape{2, 3, i, i+1});
        */
        args.emplace_back(Param{Mode::MAX, 1, 1, 2, 2, 2, 2},
                          TensorShape{2, 3, i, i + 1});
    }
    for (size_t i = 32; i < 40; ++i) {
        args.emplace_back(Param{Mode::MAX, 1, 1, 2, 2, 3, 3},
                          TensorShape{2, 3, i, i + 1});
    }
    for (uint32_t ph : {0, 1, 2})
        for (uint32_t pw : {0, 1, 2}) {
            args.emplace_back(Param{Mode::MAX, ph, pw, 1, 1, 3, 3},
                              TensorShape{2, 3, 20, 22});
        }
    // small shape for float16
    for (size_t i = 5; i < 10; ++i) {
        args.emplace_back(Param{Mode::AVERAGE, 1, 1, 2, 2, 2, 2},
                          TensorShape{2, 3, i, i + 1});
        /* reserved for future test */
        /*
        args.emplace_back(Param{Mode::AVERAGE_COUNT_EXCLUDE_PADDING, 1, 1, 2, 2,
        2, 2}, TensorShape{2, 3, i, i+1});
        */
    }
    for (uint32_t ph : {0, 1, 2})
        for (uint32_t pw : {0, 1, 2}) {
            args.emplace_back(Param{Mode::MAX, ph, pw, 1, 1, 3, 3},
                              TensorShape{1, 2, 10, 11});
        }
    return args;
}

}  // namespace pooling
}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
