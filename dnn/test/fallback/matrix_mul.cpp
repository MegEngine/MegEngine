/**
 * \file dnn/test/fallback/matrix_mul.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/fallback/fixture.h"
#include "test/common/rng.h"
#include "test/common/checker.h"
#include "test/common/matrix_mul.h"

namespace megdnn {
namespace test {

TEST_F(FALLBACK, MATRIX_MUL) {
    Checker<MatrixMul> checker(handle());
    using Param = MatrixMul::Param;
    auto args = matrix_mul::get_matmul_args();
    for (auto arg : args) {
        auto m = arg.m, n = arg.n, k = arg.k;
        auto mask = arg.mask;
        Param param;
        param.transposeA = mask & 1;
        param.transposeB = mask & 2;
        TensorShape AS, BS, CS;

        if (param.transposeA)
            AS = TensorShape{k, m};
        else
            AS = TensorShape{m, k};
        if (param.transposeB)
            BS = TensorShape{n, k};
        else
            BS = TensorShape{k, n};
        CS = TensorShape{m, n};
        TensorLayout AL, BL, CL;
        AL = TensorLayout(AS, dtype::Float32());
        BL = TensorLayout(BS, dtype::Float32());
        CL = TensorLayout(CS, dtype::Float32());
        checker.set_param(param);
        checker.execl({AL, BL, CL});
    }
}

TEST_F(FALLBACK, BATCHED_MATRIX_MUL) {

    Checker<BatchedMatrixMul> checker(handle());
    using Param = MatrixMul::Param;
    auto args = matrix_mul::get_batched_matmul_args();
    for (auto arg : args) {
        auto b = arg.b, m = arg.m, n = arg.n, k = arg.k;
        auto mask = arg.mask;
        Param param;
        param.transposeA = mask & 1;
        param.transposeB = mask & 2;
        TensorShape AS, BS, CS;

        if (param.transposeA)
            AS = TensorShape{b, k, m};
        else
            AS = TensorShape{b, m, k};
        if (param.transposeB)
            BS = TensorShape{b, n, k};
        else
            BS = TensorShape{b, k, n};
        TensorLayout AL, BL;
        AL = TensorLayout(AS, dtype::Float32());
        BL = TensorLayout(BS, dtype::Float32());
        checker.set_param(param);
        checker.execs({AL, BL, {}});
    }
}
}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
