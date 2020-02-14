/**
 * \file dnn/test/common/roi_copy.h
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
namespace roi_copy {

struct TestArg {
    param::ROICopy param;
    TensorShape src;
    TestArg(param::ROICopy param_, TensorShape src_)
            : param(param_), src(src_) {}
};

static inline std::vector<TestArg> get_args() {
    std::vector<TestArg> args;
    param::ROICopy cur_param;

    cur_param.row_from = 2;
    cur_param.row_to = 5;
    cur_param.col_from = 3;
    cur_param.col_to = 5;
    //! Inner region
    args.emplace_back(cur_param, TensorShape{2, 7, 8, 1});
    args.emplace_back(cur_param, TensorShape{2, 7, 8, 3});

    //! row start from begin
    cur_param.row_from = 0;
    args.emplace_back(cur_param, TensorShape{2, 7, 8, 1});
    args.emplace_back(cur_param, TensorShape{2, 7, 8, 3});

    //! col start from begin
    cur_param.col_from = 0;
    args.emplace_back(cur_param, TensorShape{2, 7, 8, 1});
    args.emplace_back(cur_param, TensorShape{2, 7, 8, 3});

    //! col end to the end
    cur_param.col_to = 8;
    args.emplace_back(cur_param, TensorShape{2, 7, 8, 1});
    args.emplace_back(cur_param, TensorShape{2, 7, 8, 3});

    //! row end to the end
    cur_param.row_to = 7;
    args.emplace_back(cur_param, TensorShape{2, 7, 8, 1});
    args.emplace_back(cur_param, TensorShape{2, 7, 8, 3});
    return args;
}

}  // namespace roi_copy
}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
