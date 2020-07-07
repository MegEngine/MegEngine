/**
 * \file dnn/test/common/remap.h
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
#include <iostream>
#include "megdnn/basic_types.h"
#include "megdnn/opr_param_defs.h"

#include "./rng.h"
namespace megdnn {
namespace test {
namespace remap {

struct TestArg {
    param::Remap param;
    TensorShape src;
    TensorShape map_xy;
    TensorShape dst;
    TestArg(param::Remap param_, TensorShape src_, TensorShape map_xy_,
            TensorShape dst_)
            : param(param_), src(src_), map_xy(map_xy_), dst(dst_) {}
};

static inline std::vector<TestArg> get_nchw_args() {
    std::vector<TestArg> args;

    param::Remap param;
    std::vector<param::Remap::Format> format_vec = {param::Remap::Format::NCHW};
    std::vector<param::Remap::BorderMode> border_mode_vec = {
            param::Remap::BorderMode::CONSTANT,
            param::Remap::BorderMode::REFLECT_101,
            param::Remap::BorderMode::REFLECT,
            param::Remap::BorderMode::WRAP,
            param::Remap::BorderMode::REPLICATE};
    // current do not test this.
    std::vector<float> scalar;
    for (auto fmt : format_vec) {
        for (auto border_type : border_mode_vec) {
            param.format = fmt;
            param.border_type = border_type;
            args.emplace_back(param, TensorShape{70000, 1, 2, 2},
                              TensorShape{70000, 2, 2, 2}, TensorShape{70000, 1, 2, 2});

            args.emplace_back(param, TensorShape{1, 1, 2, 2},
                              TensorShape{1, 2, 2, 2}, TensorShape{1, 1, 2, 2});

            args.emplace_back(param, TensorShape{1, 3, 2, 2},
                              TensorShape{1, 2, 2, 2}, TensorShape{1, 3, 2, 2});

            args.emplace_back(param, TensorShape{1, 10, 100, 100},
                              TensorShape{1, 100, 100, 2},
                              TensorShape{1, 10, 100, 100});

            args.emplace_back(param, TensorShape{2, 4, 100, 200},
                              TensorShape{2, 100, 200, 2},
                              TensorShape{2, 4, 100, 200});

            args.emplace_back(param, TensorShape{2, 4, 100, 200},
                              TensorShape{2, 20, 30, 2},
                              TensorShape{2, 4, 20, 30});

            args.emplace_back(param, TensorShape{2, 4, 10, 10},
                              TensorShape{2, 20, 30, 2},
                              TensorShape{2, 4, 20, 30});
        }
    }
    return args;
}

static inline std::vector<TestArg> get_nhwc_args() {
    std::vector<TestArg> args;

    param::Remap param;
    std::vector<param::Remap::Format> format_vec = {param::Remap::Format::NHWC};
    std::vector<param::Remap::BorderMode> border_mode_vec = {
            param::Remap::BorderMode::CONSTANT,
            param::Remap::BorderMode::REFLECT_101,
            param::Remap::BorderMode::REFLECT,
            param::Remap::BorderMode::WRAP,
            param::Remap::BorderMode::REPLICATE};
    // current do not test this.
    std::vector<float> scalar;
    for (auto fmt : format_vec) {
        for (auto border_type : border_mode_vec) {
            param.format = fmt;
            param.border_type = border_type;
            param.scalar = 12.f;
            args.emplace_back(param, TensorShape{70000, 2, 2, 1},
                              TensorShape{70000, 2, 2, 2}, TensorShape{70000, 2, 2, 1});

            args.emplace_back(param, TensorShape{1, 2, 2, 1},
                              TensorShape{1, 2, 2, 2}, TensorShape{1, 2, 2, 1});

            args.emplace_back(param, TensorShape{1, 2, 2, 3},
                              TensorShape{1, 2, 2, 2}, TensorShape{1, 2, 2, 3});

            args.emplace_back(param, TensorShape{1, 2, 2, 66},
                              TensorShape{1, 2, 2, 2},
                              TensorShape{1, 2, 2, 66});

            args.emplace_back(param, TensorShape{1, 100, 100, 10},
                              TensorShape{1, 100, 100, 2},
                              TensorShape{1, 100, 100, 10});

            args.emplace_back(param, TensorShape{2, 100, 200, 4},
                              TensorShape{2, 100, 200, 2},
                              TensorShape{2, 100, 200, 4});

            args.emplace_back(param, TensorShape{2, 100, 200, 4},
                              TensorShape{2, 20, 30, 2},
                              TensorShape{2, 20, 30, 4});

            args.emplace_back(param, TensorShape{2, 10, 10, 4},
                              TensorShape{2, 20, 30, 2},
                              TensorShape{2, 20, 30, 4});
        }
    }
    return args;
}

}  // namespace remap
}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
