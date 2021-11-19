/**
 * \file dnn/test/naive/pooling.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
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

TEST_F(NAIVE, POOLING_QUANTIZED) {
    using Mode = Pooling::Param::Mode;

    Checker<Pooling> checker(handle(), /* check_dispatch */ false);
    Pooling::Param param{Mode::MAX, 1, 1, 2, 2, 2, 2};
    auto dt = dtype::Quantized8Asymm(0.1f, (uint8_t)128);
    Testcase input{
            TensorValue({1, 1, 3, 3}, dt, {90, 136, 85, 48, 9, 226, 118, 109, 87}), {}};
    checker.set_param(param).exect(
            input, Testcase{{}, TensorValue({1, 1, 2, 2}, dt, {90, 136, 118, 226})});
    param = {Mode::AVERAGE, 1, 1, 2, 2, 2, 2};
    checker.set_param(param).exect(
            input, Testcase{{}, TensorValue({1, 1, 2, 2}, dt, {119, 119, 106, 108})});
    param = {Mode::AVERAGE_COUNT_EXCLUDE_PADDING, 1, 1, 2, 2, 2, 2};
    checker.set_param(param).exect(
            input, Testcase{{}, TensorValue({1, 1, 2, 2}, dt, {90, 111, 83, 108})});

    auto dt32 = dtype::QuantizedS32(0.233f);
    Testcase input32{
            TensorValue(
                    {1, 1, 3, 3}, dt32,
                    {12315, 10086, 10010, 12306, 23333, 19191, 9987, 12450, 12345}),
            {}};
    param = {Mode::MAX, 1, 1, 2, 2, 2, 2};
    checker.set_param(param).exect(
            input32,
            Testcase{
                    {}, TensorValue({1, 1, 2, 2}, dt32, {12315, 10086, 12306, 23333})});
}

TEST_F(NAIVE, POOLING_QUANTIZED_Q4) {
    using Mode = Pooling::Param::Mode;

    Checker<Pooling> checker(handle(), /* check_dispatch */ false);

    {
        auto q4_dt = dtype::QuantizedS4(1.f);
        std::vector<int> i8_src_vec{1, 2, 3, 4, 5, 6, 7, -1, -2};
        std::vector<int> i8_max_dst_vec{1, 3, 7, 6};

        std::vector<int> i8_avg_dst_vec{0, 1, 3, 2};
        std::vector<int> i8_avg_exclu_dst_vec{1, 3, 6, 2};
        Pooling::Param param{Mode::MAX, 1, 1, 2, 2, 2, 2};
        Testcase input{TensorValueLowbit4({1, 1, 3, 3}, q4_dt, i8_src_vec), {}};

        checker.set_param(param).exect(
                input,
                Testcase{{}, TensorValueLowbit4({1, 1, 2, 2}, q4_dt, i8_max_dst_vec)});
        param = {Mode::AVERAGE, 1, 1, 2, 2, 2, 2};
        checker.set_param(param).exect(
                input,
                Testcase{{}, TensorValueLowbit4({1, 1, 2, 2}, q4_dt, i8_avg_dst_vec)});
        param = {Mode::AVERAGE_COUNT_EXCLUDE_PADDING, 1, 1, 2, 2, 2, 2};
        checker.set_param(param).exect(
                input,
                Testcase{
                        {},
                        TensorValueLowbit4({1, 1, 2, 2}, q4_dt, i8_avg_exclu_dst_vec)});
    }

    {
        auto u4_dt = dtype::Quantized4Asymm(0.1f, 3);
        std::vector<int> u8_src_vec{1, 2, 3, 4, 5, 6, 7, 8, 9};
        std::vector<int> u8_max_dst_vec{1, 3, 7, 9};
        std::vector<int> u8_avg_dst_vec{3, 3, 4, 7};
        std::vector<int> u8_avg_exclu_dst_vec{1, 3, 6, 7};
        Pooling::Param param{Mode::MAX, 1, 1, 2, 2, 2, 2};
        Testcase input{TensorValueLowbit4({1, 1, 3, 3}, u4_dt, u8_src_vec), {}};
        checker.set_param(param).exect(
                input,
                Testcase{{}, TensorValueLowbit4({1, 1, 2, 2}, u4_dt, u8_max_dst_vec)});
        param = {Mode::AVERAGE, 1, 1, 2, 2, 2, 2};
        checker.set_param(param).exect(
                input,
                Testcase{{}, TensorValueLowbit4({1, 1, 2, 2}, u4_dt, u8_avg_dst_vec)});
        param = {Mode::AVERAGE_COUNT_EXCLUDE_PADDING, 1, 1, 2, 2, 2, 2};
        checker.set_param(param).exect(
                input,
                Testcase{
                        {},
                        TensorValueLowbit4({1, 1, 2, 2}, u4_dt, u8_avg_exclu_dst_vec)});
    }
}

TEST_F(NAIVE, POOLING_INT_AVERAGE) {
    using Mode = Pooling::Param::Mode;

    Checker<Pooling> checker(handle(), /* check_dispatch */ false);
    auto dt = dtype::Int8();
    Pooling::Param param = {Mode::AVERAGE, 0, 0, 1, 1, 2, 2};
    Testcase input_positive{
            TensorValue(
                    {1, 1, 3, 3}, dt, {127, 127, 127, 127, 127, 127, 127, 127, 127}),
            {}};
    Testcase input_negative{
            TensorValue(
                    {1, 1, 3, 3}, dt,
                    {-127, -127, -127, -127, -127, -127, -127, -127, -127}),
            {}};
    checker.set_param(param).exect(
            input_positive,
            Testcase{{}, TensorValue({1, 1, 2, 2}, dt, {127, 127, 127, 127})});
    checker.set_param(param).exect(
            input_negative,
            Testcase{{}, TensorValue({1, 1, 2, 2}, dt, {-127, -127, -127, -127})});
    param = {Mode::AVERAGE_COUNT_EXCLUDE_PADDING, 0, 0, 1, 1, 2, 2};
    checker.set_param(param).exect(
            input_positive,
            Testcase{{}, TensorValue({1, 1, 2, 2}, dt, {127, 127, 127, 127})});
    checker.set_param(param).exect(
            input_negative,
            Testcase{{}, TensorValue({1, 1, 2, 2}, dt, {-127, -127, -127, -127})});
}

// vim: syntax=cpp.doxygen
