/**
 * \file dnn/test/cuda/dot.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "test/cuda/fixture.h"
#include "megdnn/oprs.h"
#include "test/common/checker.h"

using namespace megdnn;
using namespace test;

TEST_F(CUDA, DOT) {
    Checker<Dot> checker(handle_cuda());
    checker.set_epsilon(1e-2);
    // basic
    checker.execs({{23}, {23}, {1}});
    // non-contiguous
    checker.exec(TensorLayoutArray{
            TensorLayout({23}, {2}, dtype::Float32()),
            TensorLayout({23}, {3}, dtype::Float32()),
            TensorLayout({1}, {1}, dtype::Float32())
            });
    // fp16
    checker.exec(TensorLayoutArray{
            TensorLayout({23}, dtype::Float16()),
            TensorLayout({23}, dtype::Float16()),
            TensorLayout({1}, dtype::Float16())
            });
    // fp16 non-contiguous
    checker.exec(TensorLayoutArray{
            TensorLayout({23}, {2}, dtype::Float16()),
            TensorLayout({23}, {3}, dtype::Float16()),
            TensorLayout({1}, {1}, dtype::Float16())
            });
}

// vim: syntax=cpp.doxygen
