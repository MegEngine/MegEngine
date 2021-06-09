/**
 * \file dnn/test/common/padding.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#pragma once
#include <cstddef>
#include <iostream>
#include "megdnn/basic_types.h"
#include "megdnn/opr_param_defs.h"

namespace megdnn {
namespace test {
namespace padding {

struct TestArg {
    param::Padding param;
    TensorShape src;
    TensorShape dst;
    TestArg(param::Padding _param, TensorShape _src, TensorShape _dst)
            : param(_param), src(_src), dst(_dst) {}
};

inline std::vector<TestArg> get_args() {
    size_t src_shape_dim0 = 5;
    size_t src_shape_dim1 = 5;
    size_t src_shape_dim2 = 5;
    size_t src_shape_dim3 = 5;
    size_t src_shape_dim4 = 5;
    size_t src_shape_dim5 = 5;
    size_t src_shape_dim6 = 5;
    
    size_t dst_shape_dim0 = 8;
    size_t dst_shape_dim1 = 8;
    size_t dst_shape_dim2 = 8;
    size_t dst_shape_dim3 = 8;
    size_t dst_shape_dim4 = 8;
    size_t dst_shape_dim5 = 8;
    size_t dst_shape_dim6 = 8;

    std::vector<TestArg> args;

    param::Padding cur_param;
    
    cur_param.front_offset_dim0 = 0;
    cur_param.front_offset_dim1 = 0;
    cur_param.front_offset_dim2 = 0;
    cur_param.front_offset_dim3 = 0;
    cur_param.front_offset_dim4 = 0;
    cur_param.front_offset_dim5 = 0;
    cur_param.front_offset_dim6 = 0;
    cur_param.back_offset_dim0 = 0;
    cur_param.back_offset_dim1 = 0;
    cur_param.back_offset_dim2 = 0;
    cur_param.back_offset_dim3 = 0;
    cur_param.back_offset_dim4 = 0;
    cur_param.back_offset_dim5 = 0;
    cur_param.back_offset_dim6 = 0;

    cur_param.padding_val = 2;

    cur_param.front_offset_dim0 = 1;
    cur_param.back_offset_dim0 = 2;

    cur_param.padding_mode = param::Padding::PaddingMode::CONSTANT;
    args.emplace_back(cur_param, TensorShape{src_shape_dim0},
                      TensorShape{dst_shape_dim0});

    cur_param.padding_mode = param::Padding::PaddingMode::REPLICATE;
    args.emplace_back(cur_param, TensorShape{src_shape_dim0},
                      TensorShape{dst_shape_dim0});

    cur_param.padding_mode = param::Padding::PaddingMode::REFLECT;
    args.emplace_back(cur_param, TensorShape{src_shape_dim0},
                      TensorShape{dst_shape_dim0});

    cur_param.front_offset_dim1 = 2;
    cur_param.back_offset_dim1 = 1;

    cur_param.padding_mode = param::Padding::PaddingMode::CONSTANT;
    args.emplace_back(cur_param, TensorShape{src_shape_dim0, src_shape_dim1},
                      TensorShape{dst_shape_dim0, dst_shape_dim1});

    cur_param.padding_mode = param::Padding::PaddingMode::REPLICATE;
    args.emplace_back(cur_param, TensorShape{src_shape_dim0, src_shape_dim1},
                      TensorShape{dst_shape_dim0, dst_shape_dim1});

    cur_param.padding_mode = param::Padding::PaddingMode::REFLECT;
    args.emplace_back(cur_param, TensorShape{src_shape_dim0, src_shape_dim1},
                      TensorShape{dst_shape_dim0, dst_shape_dim1});

    cur_param.front_offset_dim2 = 1;
    cur_param.back_offset_dim2 = 2;

    cur_param.padding_mode = param::Padding::PaddingMode::CONSTANT;
    args.emplace_back(
            cur_param,
            TensorShape{src_shape_dim0, src_shape_dim1, src_shape_dim2},
            TensorShape{dst_shape_dim0, dst_shape_dim1, dst_shape_dim2});

    cur_param.padding_mode = param::Padding::PaddingMode::REPLICATE;
    args.emplace_back(
            cur_param,
            TensorShape{src_shape_dim0, src_shape_dim1, src_shape_dim2},
            TensorShape{dst_shape_dim0, dst_shape_dim1, dst_shape_dim2});

    cur_param.padding_mode = param::Padding::PaddingMode::REFLECT;
    args.emplace_back(
            cur_param,
            TensorShape{src_shape_dim0, src_shape_dim1, src_shape_dim2},
            TensorShape{dst_shape_dim0, dst_shape_dim1, dst_shape_dim2});

    cur_param.front_offset_dim3 = 0;
    cur_param.back_offset_dim3 = 3;

    cur_param.padding_mode = param::Padding::PaddingMode::CONSTANT;
    args.emplace_back(cur_param,
                      TensorShape{src_shape_dim0, src_shape_dim1,
                                  src_shape_dim2, src_shape_dim3},
                      TensorShape{dst_shape_dim0, dst_shape_dim1,
                                  dst_shape_dim2, dst_shape_dim3});

    cur_param.padding_mode = param::Padding::PaddingMode::REPLICATE;
    args.emplace_back(cur_param,
                      TensorShape{src_shape_dim0, src_shape_dim1,
                                  src_shape_dim2, src_shape_dim3},
                      TensorShape{dst_shape_dim0, dst_shape_dim1,
                                  dst_shape_dim2, dst_shape_dim3});

    cur_param.padding_mode = param::Padding::PaddingMode::REFLECT;
    args.emplace_back(cur_param,
                      TensorShape{src_shape_dim0, src_shape_dim1,
                                  src_shape_dim2, src_shape_dim3},
                      TensorShape{dst_shape_dim0, dst_shape_dim1,
                                  dst_shape_dim2, dst_shape_dim3});

    cur_param.front_offset_dim4 = 3;
    cur_param.back_offset_dim4 = 0;

    cur_param.padding_mode = param::Padding::PaddingMode::CONSTANT;
    args.emplace_back(
            cur_param,
            TensorShape{src_shape_dim0, src_shape_dim1, src_shape_dim2,
                        src_shape_dim3, src_shape_dim4},
            TensorShape{dst_shape_dim0, dst_shape_dim1, dst_shape_dim2,
                        dst_shape_dim3, dst_shape_dim4});

    cur_param.padding_mode = param::Padding::PaddingMode::REPLICATE;
    args.emplace_back(
            cur_param,
            TensorShape{src_shape_dim0, src_shape_dim1, src_shape_dim2,
                        src_shape_dim3, src_shape_dim4},
            TensorShape{dst_shape_dim0, dst_shape_dim1, dst_shape_dim2,
                        dst_shape_dim3, dst_shape_dim4});

    cur_param.padding_mode = param::Padding::PaddingMode::REFLECT;
    args.emplace_back(
            cur_param,
            TensorShape{src_shape_dim0, src_shape_dim1, src_shape_dim2,
                        src_shape_dim3, src_shape_dim4},
            TensorShape{dst_shape_dim0, dst_shape_dim1, dst_shape_dim2,
                        dst_shape_dim3, dst_shape_dim4});

    cur_param.front_offset_dim5 = 1;
    cur_param.back_offset_dim5 = 2;

    cur_param.padding_mode = param::Padding::PaddingMode::CONSTANT;
    args.emplace_back(
            cur_param,
            TensorShape{src_shape_dim0, src_shape_dim1, src_shape_dim2,
                        src_shape_dim3, src_shape_dim4, src_shape_dim5},
            TensorShape{dst_shape_dim0, dst_shape_dim1, dst_shape_dim2,
                        dst_shape_dim3, dst_shape_dim4, dst_shape_dim5});

    cur_param.padding_mode = param::Padding::PaddingMode::REPLICATE;
    args.emplace_back(
            cur_param,
            TensorShape{src_shape_dim0, src_shape_dim1, src_shape_dim2,
                        src_shape_dim3, src_shape_dim4, src_shape_dim5},
            TensorShape{dst_shape_dim0, dst_shape_dim1, dst_shape_dim2,
                        dst_shape_dim3, dst_shape_dim4, dst_shape_dim5});

    cur_param.padding_mode = param::Padding::PaddingMode::REFLECT;
    args.emplace_back(
            cur_param,
            TensorShape{src_shape_dim0, src_shape_dim1, src_shape_dim2,
                        src_shape_dim3, src_shape_dim4, src_shape_dim5},
            TensorShape{dst_shape_dim0, dst_shape_dim1, dst_shape_dim2,
                        dst_shape_dim3, dst_shape_dim4, dst_shape_dim5});

    cur_param.front_offset_dim6 = 0;
    cur_param.front_offset_dim6 = 3;

    cur_param.padding_mode = param::Padding::PaddingMode::CONSTANT;
    args.emplace_back(
            cur_param,
            TensorShape{src_shape_dim0, src_shape_dim1, src_shape_dim2,
                        src_shape_dim3, src_shape_dim4, src_shape_dim5,
                        src_shape_dim6},
            TensorShape{dst_shape_dim0, dst_shape_dim1, dst_shape_dim2,
                        dst_shape_dim3, dst_shape_dim4, dst_shape_dim5,
                        dst_shape_dim6});

    cur_param.padding_mode = param::Padding::PaddingMode::REPLICATE;
    args.emplace_back(
            cur_param,
            TensorShape{src_shape_dim0, src_shape_dim1, src_shape_dim2,
                        src_shape_dim3, src_shape_dim4, src_shape_dim5,
                        src_shape_dim6},
            TensorShape{dst_shape_dim0, dst_shape_dim1, dst_shape_dim2,
                        dst_shape_dim3, dst_shape_dim4, dst_shape_dim5,
                        dst_shape_dim6});

    cur_param.padding_mode = param::Padding::PaddingMode::REFLECT;
    args.emplace_back(
            cur_param,
            TensorShape{src_shape_dim0, src_shape_dim1, src_shape_dim2,
                        src_shape_dim3, src_shape_dim4, src_shape_dim5,
                        src_shape_dim6},
            TensorShape{dst_shape_dim0, dst_shape_dim1, dst_shape_dim2,
                        dst_shape_dim3, dst_shape_dim4, dst_shape_dim5,
                        dst_shape_dim6});

    return args;

}

inline std::vector<TestArg> get_args_backward() {
    size_t src_shape_dim0 = 8;
    size_t src_shape_dim1 = 8;
    size_t src_shape_dim2 = 8;
    size_t src_shape_dim3 = 8;
    size_t src_shape_dim4 = 8;
    size_t src_shape_dim5 = 8;
    size_t src_shape_dim6 = 8;
    
    size_t dst_shape_dim0 = 5;
    size_t dst_shape_dim1 = 5;
    size_t dst_shape_dim2 = 5;
    size_t dst_shape_dim3 = 5;
    size_t dst_shape_dim4 = 5;
    size_t dst_shape_dim5 = 5;
    size_t dst_shape_dim6 = 5;

    std::vector<TestArg> args;

    param::Padding cur_param;
    
    cur_param.front_offset_dim0 = 0;
    cur_param.front_offset_dim1 = 0;
    cur_param.front_offset_dim2 = 0;
    cur_param.front_offset_dim3 = 0;
    cur_param.front_offset_dim4 = 0;
    cur_param.front_offset_dim5 = 0;
    cur_param.front_offset_dim6 = 0;
    cur_param.back_offset_dim0 = 0;
    cur_param.back_offset_dim1 = 0;
    cur_param.back_offset_dim2 = 0;
    cur_param.back_offset_dim3 = 0;
    cur_param.back_offset_dim4 = 0;
    cur_param.back_offset_dim5 = 0;
    cur_param.back_offset_dim6 = 0;

    cur_param.padding_val = 2;

    cur_param.front_offset_dim0 = 1;
    cur_param.back_offset_dim0 = 2;

    cur_param.padding_mode = param::Padding::PaddingMode::CONSTANT;
    args.emplace_back(cur_param, TensorShape{src_shape_dim0},
                      TensorShape{dst_shape_dim0});

    cur_param.padding_mode = param::Padding::PaddingMode::REPLICATE;
    args.emplace_back(cur_param, TensorShape{src_shape_dim0},
                      TensorShape{dst_shape_dim0});

    cur_param.padding_mode = param::Padding::PaddingMode::REFLECT;
    args.emplace_back(cur_param, TensorShape{src_shape_dim0},
                      TensorShape{dst_shape_dim0});
                 

    cur_param.front_offset_dim1 = 2;
    cur_param.back_offset_dim1 = 1;

    cur_param.padding_mode = param::Padding::PaddingMode::CONSTANT;
    args.emplace_back(cur_param, TensorShape{src_shape_dim0, src_shape_dim1},
                      TensorShape{dst_shape_dim0, dst_shape_dim1});

    cur_param.padding_mode = param::Padding::PaddingMode::REPLICATE;
    args.emplace_back(cur_param, TensorShape{src_shape_dim0, src_shape_dim1},
                      TensorShape{dst_shape_dim0, dst_shape_dim1});

    cur_param.padding_mode = param::Padding::PaddingMode::REFLECT;
    args.emplace_back(cur_param, TensorShape{src_shape_dim0, src_shape_dim1},
                      TensorShape{dst_shape_dim0, dst_shape_dim1});

    cur_param.front_offset_dim2 = 1;
    cur_param.back_offset_dim2 = 2;

    cur_param.padding_mode = param::Padding::PaddingMode::CONSTANT;
    args.emplace_back(
            cur_param,
            TensorShape{src_shape_dim0, src_shape_dim1, src_shape_dim2},
            TensorShape{dst_shape_dim0, dst_shape_dim1, dst_shape_dim2});

    cur_param.padding_mode = param::Padding::PaddingMode::REPLICATE;
    args.emplace_back(
            cur_param,
            TensorShape{src_shape_dim0, src_shape_dim1, src_shape_dim2},
            TensorShape{dst_shape_dim0, dst_shape_dim1, dst_shape_dim2});

    cur_param.padding_mode = param::Padding::PaddingMode::REFLECT;
    args.emplace_back(
            cur_param,
            TensorShape{src_shape_dim0, src_shape_dim1, src_shape_dim2},
            TensorShape{dst_shape_dim0, dst_shape_dim1, dst_shape_dim2});

    cur_param.front_offset_dim3 = 0;
    cur_param.back_offset_dim3 = 3;

    cur_param.padding_mode = param::Padding::PaddingMode::CONSTANT;
    args.emplace_back(cur_param,
                      TensorShape{src_shape_dim0, src_shape_dim1,
                                  src_shape_dim2, src_shape_dim3},
                      TensorShape{dst_shape_dim0, dst_shape_dim1,
                                  dst_shape_dim2, dst_shape_dim3});

    cur_param.padding_mode = param::Padding::PaddingMode::REPLICATE;
    args.emplace_back(cur_param,
                      TensorShape{src_shape_dim0, src_shape_dim1,
                                  src_shape_dim2, src_shape_dim3},
                      TensorShape{dst_shape_dim0, dst_shape_dim1,
                                  dst_shape_dim2, dst_shape_dim3});

    cur_param.padding_mode = param::Padding::PaddingMode::REFLECT;
    args.emplace_back(cur_param,
                      TensorShape{src_shape_dim0, src_shape_dim1,
                                  src_shape_dim2, src_shape_dim3},
                      TensorShape{dst_shape_dim0, dst_shape_dim1,
                                  dst_shape_dim2, dst_shape_dim3});

    cur_param.front_offset_dim4 = 3;
    cur_param.back_offset_dim4 =0;

    cur_param.padding_mode = param::Padding::PaddingMode::CONSTANT;
    args.emplace_back(
            cur_param,
            TensorShape{src_shape_dim0, src_shape_dim1, src_shape_dim2,
                        src_shape_dim3, src_shape_dim4},
            TensorShape{dst_shape_dim0, dst_shape_dim1, dst_shape_dim2,
                        dst_shape_dim3, dst_shape_dim4});

    cur_param.padding_mode = param::Padding::PaddingMode::REPLICATE;
    args.emplace_back(
            cur_param,
            TensorShape{src_shape_dim0, src_shape_dim1, src_shape_dim2,
                        src_shape_dim3, src_shape_dim4},
            TensorShape{dst_shape_dim0, dst_shape_dim1, dst_shape_dim2,
                        dst_shape_dim3, dst_shape_dim4});

    cur_param.padding_mode = param::Padding::PaddingMode::REFLECT;
    args.emplace_back(
            cur_param,
            TensorShape{src_shape_dim0, src_shape_dim1, src_shape_dim2,
                        src_shape_dim3, src_shape_dim4},
            TensorShape{dst_shape_dim0, dst_shape_dim1, dst_shape_dim2,
                        dst_shape_dim3, dst_shape_dim4});

    cur_param.front_offset_dim5 = 1;
    cur_param.back_offset_dim5 = 2;

    cur_param.padding_mode = param::Padding::PaddingMode::CONSTANT;
    args.emplace_back(
            cur_param,
            TensorShape{src_shape_dim0, src_shape_dim1, src_shape_dim2,
                        src_shape_dim3, src_shape_dim4, src_shape_dim5},
            TensorShape{dst_shape_dim0, dst_shape_dim1, dst_shape_dim2,
                        dst_shape_dim3, dst_shape_dim4, dst_shape_dim5});

    cur_param.padding_mode = param::Padding::PaddingMode::REPLICATE;
    args.emplace_back(
            cur_param,
            TensorShape{src_shape_dim0, src_shape_dim1, src_shape_dim2,
                        src_shape_dim3, src_shape_dim4, src_shape_dim5},
            TensorShape{dst_shape_dim0, dst_shape_dim1, dst_shape_dim2,
                        dst_shape_dim3, dst_shape_dim4, dst_shape_dim5});

    cur_param.padding_mode = param::Padding::PaddingMode::REFLECT;
    args.emplace_back(
            cur_param,
            TensorShape{src_shape_dim0, src_shape_dim1, src_shape_dim2,
                        src_shape_dim3, src_shape_dim4, src_shape_dim5},
            TensorShape{dst_shape_dim0, dst_shape_dim1, dst_shape_dim2,
                        dst_shape_dim3, dst_shape_dim4, dst_shape_dim5});

    cur_param.front_offset_dim6 = 0;
    cur_param.back_offset_dim6 = 3;

    cur_param.padding_mode = param::Padding::PaddingMode::CONSTANT;
    args.emplace_back(
            cur_param,
            TensorShape{src_shape_dim0, src_shape_dim1, src_shape_dim2,
                        src_shape_dim3, src_shape_dim4, src_shape_dim5,
                        src_shape_dim6},
            TensorShape{dst_shape_dim0, dst_shape_dim1, dst_shape_dim2,
                        dst_shape_dim3, dst_shape_dim4, dst_shape_dim5,
                        dst_shape_dim6});

    cur_param.padding_mode = param::Padding::PaddingMode::REPLICATE;
    args.emplace_back(
            cur_param,
            TensorShape{src_shape_dim0, src_shape_dim1, src_shape_dim2,
                        src_shape_dim3, src_shape_dim4, src_shape_dim5,
                        src_shape_dim6},
            TensorShape{dst_shape_dim0, dst_shape_dim1, dst_shape_dim2,
                        dst_shape_dim3, dst_shape_dim4, dst_shape_dim5,
                        dst_shape_dim6});

    cur_param.padding_mode = param::Padding::PaddingMode::REFLECT;
    args.emplace_back(
            cur_param,
            TensorShape{src_shape_dim0, src_shape_dim1, src_shape_dim2,
                        src_shape_dim3, src_shape_dim4, src_shape_dim5,
                        src_shape_dim6},
            TensorShape{dst_shape_dim0, dst_shape_dim1, dst_shape_dim2,
                        dst_shape_dim3, dst_shape_dim4, dst_shape_dim5,
                        dst_shape_dim6});

    return args;

}

}  // namespace padding
}  // namespace test
}  // namespace megdnn