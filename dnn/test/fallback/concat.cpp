/**
 * \file dnn/test/fallback/concat.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/fallback/fixture.h"

#include "test/common/checker.h"
#include "test/common/task_record_check.h"
namespace megdnn {
namespace test {

TEST_F(FALLBACK, CONCAT) {
    Checker<Concat> checker(handle());
    using Param = Concat::Param;
    for (auto dtype : std::vector<DType>{
                 dtype::Float32(), dtype::Int32(), dtype::Int16(), dtype::Float16(),
                 dtype::Int8(), dtype::Uint8()}) {
        for (size_t axis = 0; axis < 4; ++axis) {
            Param param;
            param.axis = axis;
            TensorShapeArray shapes(4, TensorShape({12, 13, 14, 15}));
            for (size_t i = 0; i < 4; ++i) {
                shapes[i].shape[axis] = i + 1;
            }
            shapes.emplace_back();
            for (size_t i = 0; i < shapes.size(); ++i)
                checker.set_dtype(i, dtype);
            checker.set_param(param).exec(shapes);
        }
    }
}
TEST_F(FALLBACK, CONCAT_RECORD) {
    TaskRecordChecker<Concat> checker(1);
    using Param = Concat::Param;

    Param param;
    param.axis = 0;
    TensorShapeArray shapes(4, TensorShape({12, 13, 14, 15}));
    for (size_t i = 0; i < 4; ++i) {
        shapes[i].shape[0] = i + 1;
    }
    shapes.emplace_back();
    for (size_t i = 0; i < shapes.size(); ++i)
        checker.set_dtype(i, dtype::Float32());
    checker.set_param(param).exec(shapes);
}
}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
