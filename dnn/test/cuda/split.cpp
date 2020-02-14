/**
 * \file dnn/test/cuda/split.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/cuda/fixture.h"

#include "test/common/checker.h"

namespace megdnn {
namespace test {

TEST_F(CUDA, SPLIT)
{
    Checker<Split> checker(handle_cuda());
    using Param = Split::Param;
    for (auto dtype: std::vector<DType>{dtype::Float32(), dtype::Float16()})
    for (size_t axis = 0; axis < 4; ++axis) {
        Param param;
        param.axis = axis;
        TensorShapeArray shapes(5, TensorShape({2, 3, 4, 5}));
        shapes[0].shape[axis] = 10;
        for (size_t i = 1; i < 5; ++i) {
            shapes[i].shape[axis] = i;
        }
        for (size_t i = 0; i < shapes.size(); ++i) checker.set_dtype(i, dtype);
        checker.set_param(param).exec(shapes);
    }
}

} // namespace test
} // namespace megdnn

// vim: syntax=cpp.doxygen


