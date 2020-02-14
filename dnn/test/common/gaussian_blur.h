/**
 * \file dnn/test/common/gaussian_blur.h
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
namespace gaussian_blur {

struct TestArg {
    param::GaussianBlur param;
    TensorShape src;
    TestArg(param::GaussianBlur param, TensorShape src)
            : param(param), src(src) {}
};

inline static std::vector<TestArg> get_args() {
    std::vector<TestArg> args;

    param::GaussianBlur cur_param;
    for (size_t i : {8, 11}) {
        for (size_t j : {8, 23}) {
            for (size_t kh = 3; kh <= 9; kh += 2) {
                for (size_t kw = 3; kw <= 9; kw += 2) {
                    cur_param.kernel_height = kh;
                    cur_param.kernel_width = kw;
                    cur_param.border_mode =
                            param::GaussianBlur::BorderMode::BORDER_REPLICATE;
                    args.emplace_back(cur_param, TensorShape{1, i, j, 1});
                    args.emplace_back(cur_param, TensorShape{3, i, j, 3});

                    cur_param.border_mode =
                            param::GaussianBlur::BorderMode::BORDER_REFLECT;
                    args.emplace_back(cur_param, TensorShape{3, i, j, 3});
                    args.emplace_back(cur_param, TensorShape{3, i, j, 1});

                    cur_param.border_mode =
                            param::GaussianBlur::BorderMode::BORDER_REFLECT_101;
                    args.emplace_back(cur_param, TensorShape{3, i, j, 3});
                    args.emplace_back(cur_param, TensorShape{3, i, j, 1});

                    cur_param.border_mode =
                            param::GaussianBlur::BorderMode::BORDER_CONSTANT;
                    args.emplace_back(cur_param, TensorShape{3, i, j, 3});
                    args.emplace_back(cur_param, TensorShape{3, i, j, 1});
                }
            }
        }
    }
    cur_param.kernel_height = 0;
    cur_param.kernel_width = 0;
    cur_param.sigma_x = 0.8;
    cur_param.sigma_y = 0.9;
    args.emplace_back(cur_param, TensorShape{1, 8, 9, 3});
    args.emplace_back(cur_param, TensorShape{1, 8, 9, 1});

    return args;
}

}  // namespace gaussian_blur
}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
