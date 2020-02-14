/**
 * \file dnn/test/naive/pooling.cpp
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

TEST_F(NAIVE, POOLING_QUANTIZED) {
    using Mode = Pooling::Param::Mode;

    Checker<Pooling> checker(handle(), /* check_dispatch */false);
    Pooling::Param param{Mode::MAX, 1, 1, 2, 2, 2, 2};
    auto dt = dtype::Quantized8Asymm(0.1f, (uint8_t)128);
    Testcase input{TensorValue({1, 1, 3, 3}, dt,
                               {90, 136, 85,
                                48, 9, 226,
                                118, 109, 87}), {}};
    checker.set_param(param).exect(input, Testcase{{},
          TensorValue({1, 1, 2, 2}, dt, {90, 136,
                                         118, 226})});
    param = {Mode::AVERAGE, 1, 1, 2, 2, 2, 2};
    checker.set_param(param).exect(input, Testcase{{},
          TensorValue({1, 1, 2, 2}, dt, {119, 119,
                                         106, 108})});
    param = {Mode::AVERAGE_COUNT_EXCLUDE_PADDING, 1, 1, 2, 2, 2, 2};
    checker.set_param(param).exect(input, Testcase{{},
          TensorValue({1, 1, 2, 2}, dt, {90, 111,
                                         83, 108})});

    auto dt32 = dtype::QuantizedS32(0.233f);
    Testcase input32{TensorValue({1, 1, 3, 3}, dt32,
                                 {12315, 10086, 10010,
                                  12306, 23333, 19191,
                                  9987,  12450, 12345}), {}};
    param = {Mode::MAX, 1, 1, 2, 2, 2, 2};
    checker.set_param(param).exect(input32, Testcase{{},
          TensorValue({1, 1, 2, 2}, dt32, {12315, 10086,
                                           12306, 23333})});
}

// vim: syntax=cpp.doxygen
