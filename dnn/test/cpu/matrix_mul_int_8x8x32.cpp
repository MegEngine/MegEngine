/**
 * \file dnn/test/cpu/matrix_mul_int_8x8x32.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/cpu/fixture.h"

#include "test/common/convolution.h"
#include "test/common/checker.h"

namespace megdnn {
namespace test {

TEST_F(CPU, MATRIX_MUL_INT_8_8_32)
{
    Checker<MatrixMul> checker(handle());
    param::MatrixMul param;
    checker.set_dtype(0, dtype::Int8());
    checker.set_dtype(1, dtype::Int8());
    checker.set_dtype(2, dtype::Int32());
    checker.set_param(param);
    for (size_t b: {1, 2, 3})
    for (size_t i: {10, 20})
    for (size_t o: {11, 22})
    {
        checker.exec({{b, i}, {i, o}, {}});
    }
}

} // namespace test
} // namespace megdnn
// vim: syntax=cpp.doxygen


