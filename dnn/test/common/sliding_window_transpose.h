/**
 * \file dnn/test/common/sliding_window_transpose.h
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
namespace sliding_window_transpose {

struct TestArg {
    param::SlidingWindowTranspose param;
    TensorShape ishape;
    TestArg(param::SlidingWindowTranspose param, TensorShape ishape)
            : param(param), ishape(ishape) {}
};

inline std::vector<TestArg> get_args() {
    std::vector<TestArg> args;
    // clang-format off
    for (uint32_t ih : {25, 96})
    for (uint32_t iw : {26, 128})
    for (uint32_t ph : {0, 1})
    for (uint32_t pw : {0, 1})
    for (uint32_t sh : {1, 2})
    for (uint32_t sw : {1, 2})
    for (uint32_t dh : {1, 2})
    for (uint32_t dw : {1, 2})
    for (uint32_t wh : {3, 4})
    for (uint32_t ww : {3, 4}) {
        unsigned long int oh = (ih + 2 * ph - dh * (wh-1)-1) / sh + 1;
        unsigned long int ow = (iw + 2 * pw - dw * (ww-1)-1) / sw + 1;
        args.emplace_back(param::SlidingWindowTranspose{ih, iw, ph, pw, sh, sw, dh, dw, wh, ww},
                          TensorShape{2, 3, oh, ow, wh, ww});
    }
    // clang-format on
    // large window case
    args.emplace_back(
            param::SlidingWindowTranspose{96, 128, 0, 0, 1, 1, 1, 1, 32, 64},
            TensorShape{2, 3, 65, 65, 32, 64});
    // // large size
    args.emplace_back(
            param::SlidingWindowTranspose{28, 24, 0, 0, 1, 1, 1, 1, 1, 1},
            TensorShape{128, 128, 28, 24, 1, 1});

    return args;
}

inline std::vector<TestArg> get_benchmark_args() {
    std::vector<TestArg> args;
    // clang-format off
    for (uint32_t ph : {0, 1})
    for (uint32_t pw : {0, 1})
    for (uint32_t sh : {1, 2})
    for (uint32_t sw : {1, 2})
    for (uint32_t dh : {1, 2})
    for (uint32_t dw : {1, 2})
    for (uint32_t wh : {3, 4})
    for (uint32_t ww : {3, 4})
    for (uint32_t b : {1, 64})
    for (uint32_t c : {64, 128})
    for (uint32_t hw : {64, 128}) {
        unsigned long int o_hw = (hw + 2 * ph - dh * (wh-1)-1) / sh + 1;
        args.emplace_back(param::SlidingWindowTranspose{hw, hw, ph, pw, sh, sw, dh, dw, wh, ww},
                          TensorShape{b, c, o_hw, o_hw, wh, ww});
    }
    // clang-format on
    // large size
    args.emplace_back(
            param::SlidingWindowTranspose{28, 24, 0, 0, 1, 1, 1, 1, 1, 1},
            TensorShape{1024, 128, 28, 24, 1, 1});

    return args;
}

}  // namespace sliding_window_transpose
}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
