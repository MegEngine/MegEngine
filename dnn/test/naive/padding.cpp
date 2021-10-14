/**
 * \file dnn/test/naive/padding.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/common/padding.h"
#include "megdnn/dtype.h"
#include "megdnn/oprs.h"
#include "test/common/checker.h"
#include "test/naive/fixture.h"

namespace megdnn {
namespace test {

TEST_F(NAIVE, PADDING) {
    std::vector<padding::TestArg> args = padding::get_args();
    Checker<Padding> checker(handle());
    for (auto&& arg : args) {
        checker.set_param(arg.param)
                .set_dtype(0, dtype::Float32())
                .set_dtype(1, dtype::Float32())
                .execs({arg.src, arg.dst});
    }
}

TEST_F(NAIVE, PADDING_CONSTANT) {
    Checker<Padding> checker(handle(), false);
    param::Padding param;
    param.padding_val = 10;
    param.padding_mode = param::Padding::PaddingMode::CONSTANT;
    param.front_offset_dim0 = 2;
    param.front_offset_dim1 = 1;
    param.front_offset_dim2 = 0;
    param.front_offset_dim3 = 0;
    param.front_offset_dim4 = 0;
    param.front_offset_dim5 = 0;
    param.front_offset_dim6 = 0;
    param.back_offset_dim0 = 2;
    param.back_offset_dim1 = 3;
    param.back_offset_dim2 = 0;
    param.back_offset_dim3 = 0;
    param.back_offset_dim4 = 0;
    param.back_offset_dim5 = 0;
    param.back_offset_dim6 = 0;
    checker.set_param(param).exect(
            Testcase{TensorValue({1, 1}, dtype::Float32(), {1}), {}},
            Testcase{{}, TensorValue({5, 5}, dtype::Float32(), {10, 10, 10, 10, 10,
                                                                10, 10, 10, 10, 10,
                                                                10, 1,  10, 10, 10,
                                                                10, 10, 10, 10, 10,
                                                                10, 10, 10, 10, 10})});
}

TEST_F(NAIVE, PADDING_REFLECT) {
    Checker<Padding> checker(handle(), false);
    param::Padding param;
    param.padding_val = 10;
    param.padding_mode = param::Padding::PaddingMode::REFLECT;
    param.front_offset_dim0 = 2;
    param.front_offset_dim1 = 0;
    param.front_offset_dim2 = 0;
    param.front_offset_dim3 = 0;
    param.front_offset_dim4 = 0;
    param.front_offset_dim5 = 0;
    param.front_offset_dim6 = 0;
    param.back_offset_dim0 = 3;
    param.back_offset_dim1 = 0;
    param.back_offset_dim2 = 0;
    param.back_offset_dim3 = 0;
    param.back_offset_dim4 = 0;
    param.back_offset_dim5 = 0;
    param.back_offset_dim6 = 0;
    checker.set_param(param).exect(
            Testcase{TensorValue({5}, dtype::Float32(), {1, 2, 3, 4, 5}), {}},
            Testcase{
                    {},
                    TensorValue(
                            {10}, dtype::Float32(), {3, 2, 1, 2, 3, 4, 5, 4, 3, 2})});
}

TEST_F(NAIVE, PADDING_REPLICATE) {
    Checker<Padding> checker(handle(), false);
    param::Padding param;
    param.padding_val = 10;
    param.padding_mode = param::Padding::PaddingMode::REPLICATE;
    param.front_offset_dim0 = 1;
    param.front_offset_dim1 = 0;
    param.front_offset_dim2 = 0;
    param.front_offset_dim3 = 0;
    param.front_offset_dim4 = 0;
    param.front_offset_dim5 = 0;
    param.front_offset_dim6 = 0;
    param.back_offset_dim0 = 2;
    param.back_offset_dim1 = 0;
    param.back_offset_dim2 = 0;
    param.back_offset_dim3 = 0;
    param.back_offset_dim4 = 0;
    param.back_offset_dim5 = 0;
    param.back_offset_dim6 = 0;
    checker.set_param(param).exect(
            Testcase{
                    TensorValue({9}, dtype::Float32(), {1, 2, 3, 4, 5, 6, 7, 8, 9}),
                    {}},
            Testcase{
                    {},
                    TensorValue(
                            {12}, dtype::Float32(),
                            {1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9})});
}

TEST_F(NAIVE, PADDING_REPLICATE2) {
    Checker<Padding> checker(handle(), false);
    param::Padding param;
    param.padding_val = 10;
    param.padding_mode = param::Padding::PaddingMode::REPLICATE;
    param.front_offset_dim0 = 2;
    param.front_offset_dim1 = 1;
    param.front_offset_dim2 = 0;
    param.front_offset_dim3 = 0;
    param.front_offset_dim4 = 0;
    param.front_offset_dim5 = 0;
    param.front_offset_dim6 = 0;
    param.back_offset_dim0 = 0;
    param.back_offset_dim1 = 3;
    param.back_offset_dim2 = 0;
    param.back_offset_dim3 = 0;
    param.back_offset_dim4 = 0;
    param.back_offset_dim5 = 0;
    param.back_offset_dim6 = 0;
    checker.set_param(param).exect(
            Testcase{
                    TensorValue({3, 3}, dtype::Float32(), {1, 2, 3, 4, 5, 6, 7, 8, 9}),
                    {}},
            Testcase{{}, TensorValue({5, 7}, dtype::Float32(), {1, 1, 2, 3, 3, 3, 3,
                                                                1, 1, 2, 3, 3, 3, 3,
                                                                1, 1, 2, 3, 3, 3, 3,
                                                                4, 4, 5, 6, 6, 6, 6,
                                                                7, 7, 8, 9, 9, 9, 9})});
}

}  // namespace test
}  // namespace megdnn