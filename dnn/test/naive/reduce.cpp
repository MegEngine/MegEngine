/**
 * \file dnn/test/naive/reduce.cpp
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

TEST_F(NAIVE, REDUCE_QUANTIZED) {
    using Mode = Reduce::Param::Mode;

    Checker<Reduce> checker(handle(), /* check_dispatch */false);

    Reduce::Param param;
    param.mode = Mode::SUM;
    param.data_type = param::Reduce::DataType::QUINT_I8xO32;
    param.axis = 0;
    checker.set_param(param).exect(
            Testcase{TensorValue({3, 4},
                                 dtype::Quantized8Asymm(0.1f, (uint8_t)128),
                                 {6, 97, 210, 47, 213, 246, 92, 121, 132, 133,
                                  222, 166}),
                     {}},
            Testcase{{},
                     TensorValue({1, 4}, dtype::QuantizedS32(0.1f),
                                 {-33, 92, 140, -50})});

    param.data_type = param::Reduce::DataType::DEFAULT;
    param.mode = Mode::MEAN;
    checker.set_param(param).exect(
            Testcase{TensorValue({3, 4},
                                 dtype::Quantized8Asymm(1.f, (uint8_t)128),
                                 {6, 97, 210, 47, 213, 246, 92, 121, 132, 133,
                                  222, 166}),
                     {}},
            Testcase{{},
                     TensorValue({1, 4},
                                 dtype::Quantized8Asymm(1.f, (uint8_t)128),
                                 {117, 159, 175, 111})});
    checker.exect(
            Testcase{TensorValue({3, 4},
                                 dtype::Quantized8Asymm(0.00233f, (uint8_t)128),
                                 {6, 97, 210, 47, 213, 246, 92, 121, 132, 133,
                                  222, 166}),
                     {}},
            Testcase{{},
                     TensorValue({1, 4},
                                 dtype::Quantized8Asymm(0.00233f, (uint8_t)128),
                                 {117, 159, 175, 111})});
    checker.exect(
            Testcase{TensorValue({3, 4},
                                 dtype::Quantized8Asymm(7e-10f, (uint8_t)45),
                                 {6, 97, 210, 47, 213, 246, 92, 121, 132, 133,
                                  222, 166}),
                     {}},
            Testcase{{},
                     TensorValue({1, 4},
                                 dtype::Quantized8Asymm(7e-10f, (uint8_t)45),
                                 {117, 159, 175, 111})});
}

// vim: syntax=cpp.doxygen
