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

TEST_F(NAIVE, TYPECVT_BFLOAT16) {
    Checker<TypeCvt> checker(handle(), false);

    checker.exect(
            Testcase{TensorValue({1, 1, 2, 4}, dtype::Float32(),
                                 {
                                         0.19921875,          // 0x3E4C0000
                                         0.19970703125,       // 0x3E4C8000
                                         0.1997108459472656,  // 0x3E4C8100
                                         0.1997032165527344,  // 0x3E4C7F00
                                         0.2001953125,        // 0x3E4D0000
                                         0.20068359375,       // 0x3E4D8000
                                         0.2006874084472656,  // 0x3E4D8100
                                         0.2006797790527344   // 0x3E4D7F00
                                 }),
                     {}},
            Testcase{{},
                     TensorValue({1, 1, 2, 4}, dtype::BFloat16(),
                                 {
                                         0.19921875,    // 0x3E4C
                                         0.19921875,    // 0x3E4C
                                         0.2001953125,  // 0x3E4D
                                         0.19921875,    // 0x3E4C
                                         0.2001953125,  // 0x3E4D
                                         0.201171875,   // 0x3E4E
                                         0.201171875,   // 0x3E4E
                                         0.2001953125   // 0x3E4D
                                 })}

    );
    checker.exect(Testcase{TensorValue({1, 1, 2, 2}, dtype::Float32(),
                                       {
                                               -123456.f,  // C7F12000
                                               -123648.f,  // C7F18000
                                               -123136.f,  // C7F08000
                                               -124160.f   // C7F28000
                                       }),
                           {}},
                  Testcase{{},
                           TensorValue({1, 1, 2, 2}, dtype::BFloat16(),
                                       {
                                               -123392.f,  // C7F1
                                               -123904.f,  // C7F2
                                               -122880.f,  // C7F0
                                               -123904.f   // C7F2
                                       })}

    );
}

// vim: syntax=cpp.doxygen
