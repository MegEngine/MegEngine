/**
 * \file dnn/test/cuda/matrix_inverse.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/cuda/fixture.h"

#include "megdnn/oprs/linalg.h"
#include "test/common/checker.h"

using namespace megdnn;
using namespace test;

TEST_F(CUDA, MATRIX_INVERSE) {
    InvertibleMatrixRNG rng;
    Checker<MatrixInverse>{handle_cuda()}
            .set_rng(0, &rng)
            .execs({{4, 5, 5}, {}})
            .execs({{100, 3, 3}, {}});
}

// vim: syntax=cpp.doxygen
