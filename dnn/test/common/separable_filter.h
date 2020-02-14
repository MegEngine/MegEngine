/**
 * \file dnn/test/common/separable_filter.h
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
namespace separable_filter {

struct TestArg {
    param::SeparableFilter param;
    TensorShape src, filter_x, filter_y;
    TestArg(param::SeparableFilter param, TensorShape src, TensorShape filter_x,
            TensorShape filter_y)
            : param(param), src(src), filter_x(filter_x), filter_y(filter_y) {}
};

std::vector<TestArg> get_args() {
    std::vector<TestArg> args;

    param::SeparableFilter cur_param;
    cur_param.format = param::SeparableFilter::Format::NHWC;
    cur_param.is_symm_kernel = false;
    for (size_t i : {8, 11}) {
        for (size_t j : {8, 23}) {
            for (size_t kh = 3; kh < 9; kh += 2) {
                for (size_t kw = 3; kw < 9; kw += 2) {
                    cur_param.ksize_h = kh;
                    cur_param.ksize_w = kw;
                    cur_param.borderMode = param::SeparableFilter::BorderMode::
                            BORDER_REPLICATE;
                    args.emplace_back(cur_param, TensorShape{1, i, j, 1},
                                      TensorShape{1, 1, 1, (size_t)kh},
                                      TensorShape{1, 1, 1, (size_t)kw});
                    args.emplace_back(cur_param, TensorShape{3, i, j, 3},
                                      TensorShape{1, 1, 1, (size_t)kh},
                                      TensorShape{1, 1, 1, (size_t)kw});

                    cur_param.borderMode =
                            param::SeparableFilter::BorderMode::BORDER_REFLECT;
                    args.emplace_back(cur_param, TensorShape{1, i, j, 1},
                                      TensorShape{1, 1, 1, (size_t)kh},
                                      TensorShape{1, 1, 1, (size_t)kw});
                    args.emplace_back(cur_param, TensorShape{3, i, j, 3},
                                      TensorShape{1, 1, 1, (size_t)kh},
                                      TensorShape{1, 1, 1, (size_t)kw});

                    cur_param.borderMode = param::SeparableFilter::BorderMode::
                            BORDER_REFLECT_101;
                    args.emplace_back(cur_param, TensorShape{1, i, j, 1},
                                      TensorShape{1, 1, 1, (size_t)kh},
                                      TensorShape{1, 1, 1, (size_t)kw});
                    args.emplace_back(cur_param, TensorShape{3, i, j, 3},
                                      TensorShape{1, 1, 1, (size_t)kh},
                                      TensorShape{1, 1, 1, (size_t)kw});

                    cur_param.borderMode =
                            param::SeparableFilter::BorderMode::BORDER_CONSTANT;
                    args.emplace_back(cur_param, TensorShape{1, i, j, 1},
                                      TensorShape{1, 1, 1, (size_t)kh},
                                      TensorShape{1, 1, 1, (size_t)kw});
                    args.emplace_back(cur_param, TensorShape{3, i, j, 3},
                                      TensorShape{1, 1, 1, (size_t)kh},
                                      TensorShape{1, 1, 1, (size_t)kw});
                }
            }
        }
    }

    return args;
}

}  // namespace separable_filter
}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
