/**
 * \file dnn/test/common/resize.h
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
#include <iostream>

#include "./rng.h"
namespace megdnn {
namespace test {
namespace resize {

struct TestArg {
    param::Resize param;
    TensorShape src;
    TensorShape dst;
    TestArg(param::Resize param_, TensorShape src_, TensorShape dst_)
            : param(param_), src(src_), dst(dst_) {}
};

// Get the args for linear test
static void set_linear_args(std::vector<TestArg>& args) {
    // test src_rows == dst_rows * 2 && src_cols == dst_cols * 2
    param::Resize cur_param;
    cur_param.format = param::Resize::Format::NHWC;
    cur_param.imode = param::Resize::InterpolationMode::INTER_LINEAR;

    args.emplace_back(cur_param, TensorShape{1, 6, 6, 1},
                      TensorShape{1, 3, 3, 1});

    // test resize_linear_Restric_kernel
    // CH == 3 && dst_rows < src_rows && dst_cols < src_cols
    args.emplace_back(cur_param, TensorShape{1, 4, 4, 3},
                      TensorShape{1, 3, 3, 3});

    // test else
    args.emplace_back(cur_param, TensorShape{1, 4, 4, 1},
                      TensorShape{1, 3, 3, 1});

    args.emplace_back(cur_param, TensorShape{1, 4, 6, 1},
                      TensorShape{1, 10, 9, 1});

    args.emplace_back(cur_param, TensorShape{1, 4, 6, 3},
                      TensorShape{1, 10, 9, 3});
}

static void set_nchw_args(std::vector<TestArg>& args) {
    param::Resize param;
    param.format = param::Resize::Format::NCHW;
    param.imode = param::Resize::InterpolationMode::LINEAR;

    args.emplace_back(param, TensorShape{2, 2, 3, 4}, TensorShape{2, 2, 6, 8});
    args.emplace_back(param, TensorShape{1, 2, 2, 2}, TensorShape{1, 2, 4, 3});
    args.emplace_back(param, TensorShape{1, 2, 6, 8}, TensorShape{1, 2, 3, 4});
}

static inline std::vector<TestArg> get_args() {
    std::vector<TestArg> args;
    set_nchw_args(args);

    //! test NHWC with ch != 1 or ch != 3
    param::Resize param;
    param.format = param::Resize::Format::NHWC;
    param.imode = param::Resize::InterpolationMode::LINEAR;
    args.emplace_back(param, TensorShape{2, 2, 3, 4}, TensorShape{2, 4, 6, 4});
    args.emplace_back(param, TensorShape{2, 4, 6, 4}, TensorShape{2, 2, 3, 4});

    return args;
}

static inline std::vector<TestArg> get_nhwcd4_args() {
    std::vector<TestArg> args;

    param::Resize param;
    param.format = param::Resize::Format::NHWCD4;
    param.imode = param::Resize::InterpolationMode::LINEAR;
    args.emplace_back(param, TensorShape{2, 2, 1, 3, 4},
                      TensorShape{2, 4, 1, 6, 4});
    args.emplace_back(param, TensorShape{2, 4, 1, 6, 4},
                      TensorShape{2, 2, 1, 3, 4});

    return args;
}

static inline std::vector<TestArg> get_nchw4_args() {
    std::vector<TestArg> args;

    param::Resize param;
    param.format = param::Resize::Format::NCHW4;
    param.imode = param::Resize::InterpolationMode::LINEAR;
    args.emplace_back(param, TensorShape{1, 1, 2, 3, 4},
                      TensorShape{1, 1, 2, 6, 4});
    args.emplace_back(param, TensorShape{2, 2, 2, 2, 4},
                      TensorShape{2, 2, 2, 4, 4});
    args.emplace_back(param, TensorShape{2, 4, 6, 8, 4},
                      TensorShape{2, 4, 3, 4, 4});
    return args;
}

static inline std::vector<TestArg> get_cv_args() {
    std::vector<TestArg> args;

    set_linear_args(args);

    param::Resize cur_param;
    cur_param.format = param::Resize::Format::NHWC;
    for (size_t i = 8; i < 129; i *= 4) {
        cur_param.imode = param::Resize::InterpolationMode::INTER_NEAREST;

        args.emplace_back(cur_param, TensorShape{1, i, i, 3},
                          TensorShape{1, i / 2, i / 2, 3});

        args.emplace_back(cur_param, TensorShape{1, i, i, 1},
                          TensorShape{1, 8, 8, 1});

        cur_param.imode = param::Resize::InterpolationMode::INTER_AREA;
        args.emplace_back(cur_param, TensorShape{1, i, i, 3},
                          TensorShape{1, 8, 8, 3});
        cur_param.imode = param::Resize::InterpolationMode::INTER_CUBIC;
        args.emplace_back(cur_param, TensorShape{1, i, i, 3},
                          TensorShape{1, 8, 8, 3});
        cur_param.imode = param::Resize::InterpolationMode::INTER_LANCZOS4;
        args.emplace_back(cur_param, TensorShape{1, i, i, 3},
                          TensorShape{1, 8, 8, 3});
    }

    //! cuda not use vector
    //! enlarge==true && dst_area_size > 500 * 500
    cur_param.imode = param::Resize::InterpolationMode::INTER_CUBIC;
    args.emplace_back(cur_param, TensorShape{1, 3, 3, 1},
                      TensorShape{1, 500, 600, 1});
    cur_param.imode = param::Resize::InterpolationMode::INTER_LANCZOS4;
    args.emplace_back(cur_param, TensorShape{1, 3, 3, 1},
                      TensorShape{1, 500, 600, 1});
    return args;
}

}  // namespace resize
}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
