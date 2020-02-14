/**
 * \file dnn/test/common/images2neibs.h
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
namespace images2neibs {

struct TestArg {
    param::Images2Neibs param;
    TensorShape ishape;
    TestArg(param::Images2Neibs param, TensorShape ishape)
            : param(param), ishape(ishape) {}
};

inline std::vector<TestArg> get_args() {
    std::vector<TestArg> args;
    // clang-format off
    for (uint32_t ph : {0, 1})
    for (uint32_t pw : {0, 1})
    for (uint32_t sh : {1, 2})
    for (uint32_t sw : {1, 2})
    for (uint32_t wh : {3, 4})
    for (uint32_t ww : {3, 4}) {
        args.emplace_back(param::Images2Neibs{ph, pw, sh, sw, wh, ww},
                          TensorShape{2, 3, 5, 6});
    }
    // clang-format on
    // large window case
    args.emplace_back(param::Images2Neibs{0, 0, 1, 1, 32, 64},
                      TensorShape{2, 3, 96, 128});
    // large size
    args.emplace_back(param::Images2Neibs{0, 0, 1, 1, 1, 1},
                      TensorShape{128, 128, 28, 24});

    return args;
}

inline std::vector<TestArg> get_benchmark_args() {
    std::vector<TestArg> args;
    // clang-format off
    for (uint32_t ph : {0, 1})
    for (uint32_t pw : {0, 1})
    for (uint32_t sh : {1, 2})
    for (uint32_t sw : {1, 2})
    for (uint32_t wh : {3, 4})
    for (uint32_t ww : {3, 4})
    for (uint32_t b : {1, 64})
    for (uint32_t c : {64, 128})
    for (uint32_t hw : {64, 128}) {
        args.emplace_back(param::Images2Neibs{ph, pw, sh, sw, wh, ww},
                          TensorShape{b, c, hw, hw});
    }
    // clang-format on
    // large size
    args.emplace_back(param::Images2Neibs{0, 0, 1, 1, 1, 1},
                      TensorShape{1024, 128, 28, 24});

    return args;
}

}  // namespace images2neibs
}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
