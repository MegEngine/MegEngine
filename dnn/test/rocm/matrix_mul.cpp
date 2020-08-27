/**
 * \file dnn/test/rocm/matrix_mul.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "hcc_detail/hcc_defs_prologue.h"
#include "test/rocm/fixture.h"

#include "test/common/checker.h"
#include "test/common/matrix_mul.h"

#include "src/rocm/utils.h"

namespace megdnn {
namespace test {

TEST_F(ROCM, MATRIX_MUL) {
    Checker<MatrixMul> checker(handle_rocm());
    using Param = MatrixMul::Param;
    size_t m = 12, n = 16, k = 20;
    //! result error for Int8x8x32, not test correctness
    std::vector<DType> dtypes{MEGDNN_INC_FLOAT16(dtype::Float16() MEGDNN_COMMA)
                                      dtype::Float32()/*, dtype::Int32()*/};
    for (auto dtype : dtypes) {
        for (unsigned mask = 0; mask < 4; ++mask) {
            Param param;
            param.transposeA = mask & 1;
            param.transposeB = mask & 2;
            DType stype = dtype == dtype::Int32() ? dtype::Int8() : dtype;
            TensorShape A, B;
            if (param.transposeA)
                A = TensorShape{k, m};
            else
                A = TensorShape{m, k};
            if (param.transposeB)
                B = TensorShape{n, k};
            else
                B = TensorShape{k, n};
            checker.set_param(param)
                    .set_dtype(0, stype)
                    .set_dtype(1, stype)
                    .set_dtype(2, dtype)
                    .set_epsilon(MEGDNN_FLOAT16_SELECT(
                                         dtype == dtype::Float16(), false)
                                         ? 5e-2
                                         : 5e-3)
                    .execs({A, B, {}});
        }
    }
    // general tests
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
        if (arg.A_stride == 0) {
            AL = TensorLayout(AS, dtype::Float32());
        } else {
            AL = TensorLayout(AS, {ptrdiff_t(arg.A_stride), 1},
                              dtype::Float32());
        }
        if (arg.B_stride == 0) {
            BL = TensorLayout(BS, dtype::Float32());
        } else {
            BL = TensorLayout(BS, {ptrdiff_t(arg.B_stride), 1},
                              dtype::Float32());
        }
        if (arg.C_stride == 0) {
            CL = TensorLayout(CS, dtype::Float32());
        } else {
            CL = TensorLayout(CS, {ptrdiff_t(arg.C_stride), 1},
                              dtype::Float32());
        }
        checker.set_param(param).execl({AL, BL, CL});
    }
}

} // namespace test
} // namespace megdnn
   // vim: syntax=cpp.doxygen
