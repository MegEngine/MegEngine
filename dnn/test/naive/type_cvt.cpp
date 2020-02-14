/**
 * \file dnn/test/naive/type_cvt.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/common/checker.h"
#include "test/common/convolution.h"
#include "test/common/random_state.h"
#include "test/naive/fixture.h"

#include "megdnn/oprs.h"

using namespace megdnn;
using namespace test;

TEST_F(NAIVE, TYPECVT_QUINT4) {
    Checker<TypeCvt> checker(handle(), false);

    checker.exect(
            Testcase{TensorValueLowbit4({1, 1, 4, 4},
                                       dtype::Quantized4Asymm(0.1f, (uint8_t)8),
                                       std::vector<uint8_t>(
                                               {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                                11, 12, 13, 14, 15, 0})),
                     {}},
            Testcase{{},
                     TensorValue({1, 1, 4, 4}, dtype::Float32(),
                                 {-0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.,
                                  0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, -0.8})}

    );

    checker.exect(
            Testcase{TensorValueLowbit4({1, 1, 4, 4},
                                       dtype::Quantized4Asymm(0.1f, (uint8_t)8),
                                       std::vector<uint8_t>(
                                               {1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                                11, 12, 13, 14, 15, 0})),
                     {}},
            Testcase{
                    {},
                    TensorValue({1, 1, 4, 4},
                                dtype::Quantized8Asymm(0.1f, (uint8_t)8),
                                {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                                 15, 0}),
            });

    checker.exect(
            Testcase{TensorValue({1, 1, 4, 4},
                                 dtype::Quantized8Asymm(0.1f, (uint8_t)8),
                                 {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14,
                                  15, 0}),
                     {}},
            Testcase{
                    {},
                    TensorValueLowbit4(
                            {1, 1, 4, 4},
                            dtype::Quantized4Asymm(0.1f, (uint8_t)8),
                            std::vector<uint8_t>({1, 2, 3, 4, 5, 6, 7, 8, 9,
                                                   10, 11, 12, 13, 14, 15, 0})),
            });

    // test overflow
    checker.exect(Testcase{TensorValue({6}, dtype::Float32(),
                                       {-1.2, -0.8, 0.0, 0.7, 0.8, 1.2}),
                           {}},
                  Testcase{
                          {},
                          TensorValueLowbit4(
                                  {6}, dtype::Quantized4Asymm(0.1f, (uint8_t)8),
                                  std::vector<uint8_t>({0, 0, 8, 15, 15, 15})),
                  });
}

TEST_F(NAIVE, TYPECVT_QINT4) {
    Checker<TypeCvt> checker(handle(), false);

    checker.exect(
            Testcase{TensorValueLowbit4(
                             {1, 1, 4, 4}, dtype::QuantizedS4(0.1f),
                             std::vector<int8_t>({-8, -7, -6, -5, -4, -3, -2,
                                                  -1, 0, 1, 2, 3, 4, 5, 6, 7})),
                     {}},
            Testcase{{},
                     TensorValue({1, 1, 4, 4}, dtype::Float32(),
                                 {-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2,
                                  -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7})}

    );
    checker.exect(
            Testcase{TensorValue({1, 1, 4, 4},
                                 dtype::Quantized8Asymm(0.1f, (uint8_t)8),
                                 {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
                                  14, 15}),
                     {}},
            Testcase{
                    {},
                    TensorValueLowbit4(
                            {1, 1, 4, 4}, dtype::QuantizedS4(0.1f),
                            std::vector<int8_t>({-8, -7, -6, -5, -4, -3, -2, -1,
                                                 0, 1, 2, 3, 4, 5, 6, 7})),
            });
    // test overflow
    checker.exect(Testcase{TensorValue({6}, dtype::Float32(),
                                       {-0.9, -0.8, 0.0, 0.7, 0.8, 1.0}),
                           {}},
                  Testcase{
                          {},
                          TensorValueLowbit4(
                                  {6}, dtype::QuantizedS4(0.1f),
                                  std::vector<int8_t>({-8, -8, 0, 7, 7, 7})),
                  });
}

// vim: syntax=cpp.doxygen
