/**
 * \file dnn/test/cuda/lrn.cpp
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
#include "test/common/local.h"

namespace megdnn {
namespace test {

TEST_F(CUDA, LRN_FORWARD)
{
    Checker<LRNForward> checker(handle_cuda());
    for (auto dtype: std::vector<DType>{dtype::Float32(), dtype::Float16()}) {
        checker.set_dtype(0, dtype);
        checker.execs({{2, 11, 12, 13}, {}});
        for (size_t w = 10; w <= 50; ++w) {
            checker.execs({{2, w, 12, 13}, {}});
        }
    }
}

TEST_F(CUDA, LRN_BACKWARD)
{
    Checker<LRNBackward> checker(handle_cuda());
    auto shape = TensorShape{2, 11, 12, 13};
    checker.set_dtype(0, dtype::Float32());
    checker.exec(TensorShapeArray{shape, shape, shape, shape});
    checker.set_dtype(1, dtype::Float32());
    checker.exec(TensorShapeArray{shape, shape, shape, shape});
}

} // namespace test
} // namespace megdnn

// vim: syntax=cpp.doxygen
