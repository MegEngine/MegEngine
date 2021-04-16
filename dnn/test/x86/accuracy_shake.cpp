/**
 * \file dnn/test/x86/accuracy_shake.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#include "test/x86/fixture.h"

#include "megdnn/opr_param_defs.h"
#include "megdnn/oprs.h"
#include "test/common/accuracy_shake_checker.h"
#include "test/common/convolution.h"
#include "test/common/rng.h"
#include "test/common/tensor.h"
#include "test/common/workspace_wrapper.h"

namespace megdnn {
namespace test {

TEST_F(X86, SHAKE_CONV_BIAS_FORWARD) {
    AccuracyShakeChecker<ConvBiasForward> checker(handle());
    NormalRNG default_rng;
    checker.set_dtype(0, dtype::Float32())
            .set_dtype(1, dtype::Float32())
            .set_dtype(2, dtype::Float32())
            .set_rng(0, &default_rng)
            .set_rng(1, &default_rng);
    checker.set_before_exec_callback(AlgoGenerator<ConvBiasForward>("X86"));
    // convolution
    checker.exec({{6, 16, 32, 32}, {64, 16, 3, 3}, {}, {}, {}});
    // convbias without z
    checker.exec({{6, 16, 32, 32}, {64, 16, 3, 3}, {1, 64, 1, 1}, {}, {}});
    // convbias with z
    checker.exec({{6, 16, 32, 32},
                  {64, 16, 3, 3},
                  {1, 64, 1, 1},
                  {6, 64, 30, 30},
                  {}});
    // group
    ConvBias::Param param;
    param.sparse = ConvBias::Param::Sparse::GROUP;
    checker.set_param(param);
    checker.exec({{6, 16, 32, 32}, {2, 32, 8, 3, 3}, {}, {}, {}});
    checker.exec({{6, 16, 32, 32}, {2, 32, 8, 3, 3}, {1, 64, 1, 1}, {}, {}});
    checker.exec({{6, 16, 32, 32},
                  {2, 32, 8, 3, 3},
                  {1, 64, 1, 1},
                  {6, 64, 30, 30},
                  {}});
}

TEST_F(X86, SHAKE_CONV_BIAS_FORWARD_INT8) {
    AccuracyShakeChecker<ConvBiasForward> checker(handle());
    UniformIntRNG rng{-50, 50};
    checker.set_dtype(0, dtype::QuantizedS8(2.5f))
            .set_dtype(1, dtype::QuantizedS8(2.5f))
            .set_dtype(2, dtype::QuantizedS32(6.25f))
            .set_dtype(3, dtype::QuantizedS32(6.25f))
            .set_dtype(4, {})
            .set_rng(0, &rng)
            .set_rng(1, &rng)
            .set_rng(2, &rng);
    checker.set_before_exec_callback(AlgoGenerator<ConvBiasForward>("X86"));
    // convolution
    checker.exec({{6, 16, 32, 32}, {64, 16, 3, 3}, {}, {}, {}});
    // convbias without z
    checker.exec({{6, 16, 32, 32}, {64, 16, 3, 3}, {1, 64, 1, 1}, {}, {}});
    // convbias with z
    checker.exec({{6, 16, 32, 32},
                  {64, 16, 3, 3},
                  {1, 64, 1, 1},
                  {6, 64, 30, 30},
                  {}});
    // group
    ConvBias::Param param;
    param.sparse = ConvBias::Param::Sparse::GROUP;
    checker.set_param(param);
    checker.exec({{6, 16, 32, 32}, {2, 32, 8, 3, 3}, {}, {}, {}});
    checker.exec({{6, 16, 32, 32}, {2, 32, 8, 3, 3}, {1, 64, 1, 1}, {}, {}});
    checker.exec({{6, 16, 32, 32},
                  {2, 32, 8, 3, 3},
                  {1, 64, 1, 1},
                  {6, 64, 30, 30},
                  {}});
}

TEST_F(X86, SHAKE_MATRIX_MUL_FORWARD) {
    AccuracyShakeChecker<MatrixMul> checker(handle());

    checker.set_dtype(0, dtype::Float32())
            .set_dtype(1, dtype::Float32())
            .set_dtype(2, dtype::Float32())
            .exec({{20, 100}, {100, 60}, {}});
}

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
