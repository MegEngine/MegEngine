/**
 * \file dnn/test/cpu/batched_matrix_mul.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/cpu/fixture.h"

#include <chrono>
#include "test/common/checker.h"
#include "test/common/matrix_mul.h"

using namespace megdnn;
using namespace test;

//! check batch=1 and batch_stride is arbitrarily
TEST_F(CPU, BATCHED_MATRIX_MUL_BATCH_1) {
    matrix_mul::check_batched_matrix_mul(
            dtype::Float32{}, dtype::Float32{}, dtype::Float32{}, handle(), "", 1e-3,
            std::vector<matrix_mul::TestArg>{{5, 5, 5, 0, 5, 5, 5, 1, 5, 5, 5}});
}

// vim: syntax=cpp.doxygen
