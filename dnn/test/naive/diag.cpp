/**
 * \file dnn/test/naive/diag.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "megdnn/dtype.h"
#include "megdnn/oprs.h"
#include "test/common/checker.h"
#include "test/naive/fixture.h"

namespace megdnn {
namespace test {

TEST_F(NAIVE, DiagVector2Matrix) {
    Checker<Diag> checker(handle(), false);
    Diag::Param param;
    param.k = 0;
    checker.set_param(param).exect(
            Testcase{TensorValue({3}, dtype::Float32(), {1, 2, 3}), {}},
            Testcase{
                    {},
                    // clang-format off
                     TensorValue({3, 3}, dtype::Float32(), {1, 0, 0, 
                                                            0, 2, 0, 
                                                            0, 0, 3})});
    // clang-format on
}

TEST_F(NAIVE, DiagVector2Matrix_PositiveK) {
    Checker<Diag> checker(handle(), false);
    Diag::Param param;
    param.k = 1;
    checker.set_param(param).exect(
            Testcase{TensorValue({3}, dtype::Float32(), {1, 2, 3}), {}},
            Testcase{
                    {},
                    // clang-format off
                     TensorValue({4, 4}, dtype::Float32(), {0, 1, 0, 0, 
                                                            0, 0, 2, 0,
                                                            0, 0, 0, 3,
                                                            0, 0, 0, 0,})});
    // clang-format on
}

TEST_F(NAIVE, DiagVector2Matrix_NegativeK) {
    Checker<Diag> checker(handle(), false);
    Diag::Param param;
    param.k = -1;
    checker.set_param(param).exect(
            Testcase{TensorValue({3}, dtype::Float32(), {1, 2, 3}), {}},
            Testcase{
                    {},
                    // clang-format off
                     TensorValue({4, 4}, dtype::Float32(), {0, 0, 0, 0, 
                                                            1, 0, 0, 0,
                                                            0, 2, 0, 0,
                                                            0, 0, 3, 0,})});
    // clang-format on
}

TEST_F(NAIVE, DiagMatrix2Vector) {
    Checker<Diag> checker(handle(), false);
    Diag::Param param;
    param.k = 0;
    checker.set_param(param).exect(
            // clang-format off
            Testcase{TensorValue({3, 3}, dtype::Float32(), {1, 2, 3,
                                                            4, 5, 6, 
                                                            7, 8, 9}),
                    // clang-format on
                    {}},
            Testcase{{}, TensorValue({3}, dtype::Float32(), {1, 5, 9})});
}

TEST_F(NAIVE, DiagMatrix2Vector_PositiveK) {
    Checker<Diag> checker(handle(), false);
    Diag::Param param;
    param.k = 1;
    checker.set_param(param).exect(
            // clang-format off
            Testcase{TensorValue({3, 3}, dtype::Float32(), {1, 2, 3,
                                                            4, 5, 6, 
                                                            7, 8, 9}),
                    // clang-format on
                    {}},
            Testcase{{}, TensorValue({2}, dtype::Float32(), {2, 6})});
}

TEST_F(NAIVE, DiagMatrix2Vector_NegativeK) {
    Checker<Diag> checker(handle(), false);
    Diag::Param param;
    param.k = -1;
    checker.set_param(param).exect(
            // clang-format off
            Testcase{TensorValue({3, 3}, dtype::Float32(), {1, 2, 3,
                                                            4, 5, 6, 
                                                            7, 8, 9}),
                    // clang-format on
                    {}},
            Testcase{{}, TensorValue({2}, dtype::Float32(), {4, 8})});
}

}  // namespace test
}  // namespace megdnn
