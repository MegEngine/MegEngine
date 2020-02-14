/**
 * \file dnn/test/cuda/transpose.cpp
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

TEST_F(CUDA, TRANSPOSE)
{
    Checker<Transpose> checker(handle_cuda());
    checker.execs({{17, 40}, {40, 17}});
    checker.exec(TensorLayoutArray{
            TensorLayout({17, 40}, {50, 1}, dtype::Float32()),
            TensorLayout({40, 17}, {50, 1}, dtype::Float32())
            });
    checker.exec(TensorLayoutArray{
            TensorLayout({17, 40}, {50, 1}, dtype::Float16()),
            TensorLayout({40, 17}, {50, 1}, dtype::Float16())
            });
    checker.exec(TensorLayoutArray{
            TensorLayout({40, 17}, {50, 1}, dtype::Float16()),
            TensorLayout({17, 40}, {50, 1}, dtype::Float16())
            });
}

} // namespace test
} // namespace megdnn
// vim: syntax=cpp.doxygen
