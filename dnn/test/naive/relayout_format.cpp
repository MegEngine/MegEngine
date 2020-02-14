/**
 * \file dnn/test/naive/relayout_format.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/naive/fixture.h"

#include "megdnn/oprs/nn.h"
#include "test/common/checker.h"
#include "test/common/random_state.h"

using namespace megdnn;
using namespace test;

TEST_F(NAIVE, RELAYOUT_FORMAT_NCHW88) {
    Checker<RelayoutFormat> checker(handle(), /* check_dispatch */ false);

    {
        auto tensor_nchw = TensorValue(
                {1, 8, 1, 2}, dtype::Float32(),
                {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
        auto tensor_nchw88 = TensorValue(
                {1, 1, 1, 2, 8}, dtype::Float32(),
                {1, 3, 5, 7, 9, 11, 13, 15, 2, 4, 6, 8, 10, 12, 14, 16});
        RelayoutFormat::Param param{RelayoutFormat::Param::Mode::NCHW_NCHW88};

        checker.set_param(param).exect(Testcase{tensor_nchw, {}},
                                       Testcase{{}, tensor_nchw88});
    }

    {
        auto tensor_nchw = TensorValue(
                {2, 8, 1, 2}, dtype::Float32(),
                {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
        auto tensor_nchw88 = TensorValue(
                {2, 1, 1, 2, 8}, dtype::Float32(),
                {1, 3, 5, 7, 9, 11, 13, 15, 2, 4, 6, 8, 10, 12, 14, 16,
                 1, 3, 5, 7, 9, 11, 13, 15, 2, 4, 6, 8, 10, 12, 14, 16});
        RelayoutFormat::Param param{RelayoutFormat::Param::Mode::NCHW_NCHW88};

        checker.set_param(param).exect(Testcase{tensor_nchw, {}},
                                       Testcase{{}, tensor_nchw88});
    }

    {
        auto tensor_nchw =
                TensorValue({2, 4, 1, 2}, dtype::Float32(),
                            {1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8});
        auto tensor_nchw88 =
                TensorValue({2, 1, 1, 2, 8}, dtype::Float32(),
                            {1, 3, 5, 7, 0, 0, 0, 0, 2, 4, 6, 8, 0, 0, 0, 0,
                             1, 3, 5, 7, 0, 0, 0, 0, 2, 4, 6, 8, 0, 0, 0, 0});
        RelayoutFormat::Param param{RelayoutFormat::Param::Mode::NCHW_NCHW88};

        checker.set_param(param).exect(Testcase{tensor_nchw, {}},
                                       Testcase{{}, tensor_nchw88});

        checker.set_param(param).exec({TensorShape{1, 3, 64, 64}, {}});
    }

    {
        auto tensor_nchw = TensorValue(
                {1, 8, 1, 2}, dtype::Float32(),
                {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
        auto tensor_nchw88 = TensorValue(
                {1, 1, 1, 2, 8}, dtype::Float32(),
                {1, 3, 5, 7, 9, 11, 13, 15, 2, 4, 6, 8, 10, 12, 14, 16});
        RelayoutFormat::Param param{RelayoutFormat::Param::Mode::NCHW88_NCHW};
        checker.set_param(param).exect(Testcase{tensor_nchw88, {}},
                                       Testcase{{}, tensor_nchw});
    }
}
TEST_F(NAIVE, RELAYOUT_FORMAT_NCHW88_DENSE) {
    Checker<RelayoutFormat> checker(handle(), /* check_dispatch */ false);
    {
        auto tensor_oihw =
                TensorValue({8, 8, 1, 1}, dtype::Float32(),
                            {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
                             14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                             27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                             40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
                             53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64});
        auto tensor_oihw8i8o = TensorValue(
                {1, 1, 1, 1, 8, 8}, dtype::Float32(),
                {
                        1,  9,  17, 25, 33, 41, 49, 57, 2,  10, 18, 26, 34,
                        42, 50, 58, 3,  11, 19, 27, 35, 43, 51, 59, 4,  12,
                        20, 28, 36, 44, 52, 60, 5,  13, 21, 29, 37, 45, 53,
                        61, 6,  14, 22, 30, 38, 46, 54, 62, 7,  15, 23, 31,
                        39, 47, 55, 63, 8,  16, 24, 32, 40, 48, 56, 64,
                });

        RelayoutFormat::Param param{
                RelayoutFormat::Param::Mode::NCHW_NCHW88_CONV_DENSE_WEIGHT};
        checker.set_param(param).exect(Testcase{tensor_oihw, {}},
                                       Testcase{{}, tensor_oihw8i8o});
    }

    {
        auto tensor_oihw = TensorValue(
                {8, 2, 1, 1}, dtype::Float32(),
                {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
        auto tensor_oihw8i8o = TensorValue(
                {1, 1, 1, 1, 8, 8}, dtype::Float32(),
                {
                        1, 3, 5, 7, 9, 11, 13, 15, 2, 4, 6, 8, 10, 12, 14, 16,
                        0, 0, 0, 0, 0, 0,  0,  0,  0, 0, 0, 0, 0,  0,  0,  0,
                        0, 0, 0, 0, 0, 0,  0,  0,  0, 0, 0, 0, 0,  0,  0,  0,
                        0, 0, 0, 0, 0, 0,  0,  0,  0, 0, 0, 0, 0,  0,  0,  0,
                });

        RelayoutFormat::Param param{
                RelayoutFormat::Param::Mode::NCHW_NCHW88_CONV_DENSE_WEIGHT};
        checker.set_param(param).exect(Testcase{tensor_oihw, {}},
                                       Testcase{{}, tensor_oihw8i8o});
    }
}

TEST_F(NAIVE, RELAYOUT_FORMAT_NCHW88_CHAIN) {
    Checker<RelayoutFormat> checker(handle(), /* check_dispatch */ false);
    {
        auto tensor_goihw = TensorValue(
                {8, 1, 1, 1, 2}, dtype::Float32(),
                {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
        auto tensor_goihw8g = TensorValue(
                {1, 1, 1, 1, 2, 8}, dtype::Float32(),
                {1, 3, 5, 7, 9, 11, 13, 15, 2, 4, 6, 8, 10, 12, 14, 16});

        RelayoutFormat::Param param{
                RelayoutFormat::Param::Mode::NCHW_NCHW88_CONV_CHAN_WEIGHT};
        checker.set_param(param).exect(Testcase{tensor_goihw, {}},
                                       Testcase{{}, tensor_goihw8g});
    }

    {
        auto tensor_goihw =
                TensorValue({2, 1, 1, 1, 2}, dtype::Float32(), {1, 2, 3, 4});
        auto tensor_goihw8g =
                TensorValue({1, 1, 1, 1, 2, 8}, dtype::Float32(),
                            {1, 3, 0, 0, 0, 0, 0, 0, 2, 4, 0, 0, 0, 0, 0, 0});

        RelayoutFormat::Param param{
                RelayoutFormat::Param::Mode::NCHW_NCHW88_CONV_CHAN_WEIGHT};
        checker.set_param(param).exect(Testcase{tensor_goihw, {}},
                                       Testcase{{}, tensor_goihw8g});
    }
}
TEST_F(NAIVE, RELAYOUT_FORMAT_NCHW88_GROUP) {
    Checker<RelayoutFormat> checker(handle(), /* check_dispatch */ false);
    {
        auto tensor_goihw =
                TensorValue({1, 8, 8, 1, 1}, dtype::Float32(),
                            {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13,
                             14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26,
                             27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39,
                             40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
                             53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64});
        auto tensor_goihw8i8o = TensorValue(
                {1, 1, 1, 1, 1, 8, 8}, dtype::Float32(),
                {
                        1,  9,  17, 25, 33, 41, 49, 57, 2,  10, 18, 26, 34,
                        42, 50, 58, 3,  11, 19, 27, 35, 43, 51, 59, 4,  12,
                        20, 28, 36, 44, 52, 60, 5,  13, 21, 29, 37, 45, 53,
                        61, 6,  14, 22, 30, 38, 46, 54, 62, 7,  15, 23, 31,
                        39, 47, 55, 63, 8,  16, 24, 32, 40, 48, 56, 64,
                });

        RelayoutFormat::Param param{
                RelayoutFormat::Param::Mode::NCHW_NCHW88_CONV_GROUP_WEIGHT};
        checker.set_param(param).exect(Testcase{tensor_goihw, {}},
                                       Testcase{{}, tensor_goihw8i8o});
    }
    {
        auto tensor_goihw = TensorValue(
                {1, 8, 2, 1, 1}, dtype::Float32(),
                {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16});
        auto tensor_goihw8i8o = TensorValue(
                {1, 1, 1, 1, 1, 8, 8}, dtype::Float32(),
                {
                        1, 3, 5, 7, 9, 11, 13, 15, 2, 4, 6, 8, 10, 12, 14, 16,
                        0, 0, 0, 0, 0, 0,  0,  0,  0, 0, 0, 0, 0,  0,  0,  0,
                        0, 0, 0, 0, 0, 0,  0,  0,  0, 0, 0, 0, 0,  0,  0,  0,
                        0, 0, 0, 0, 0, 0,  0,  0,  0, 0, 0, 0, 0,  0,  0,  0,
                });

        RelayoutFormat::Param param{
                RelayoutFormat::Param::Mode::NCHW_NCHW88_CONV_GROUP_WEIGHT};
        checker.set_param(param).exect(Testcase{tensor_goihw, {}},
                                       Testcase{{}, tensor_goihw8i8o});
    }

    {
        RelayoutFormat::Param param{RelayoutFormat::Param::Mode::NCHW88_NCHW};
        checker.set_param(param).exec({TensorShape{1, 8, 64, 64, 8}, {}});
    }
}