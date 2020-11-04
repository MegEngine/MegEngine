/**
 * \file dnn/test/rocm/bn.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#include "test/rocm/fixture.h"

#include "megdnn/opr_param_defs.h"
#include "megdnn/oprs.h"
#include "test/common/bn.h"
#include "test/common/checker.h"
#include "test/common/rng.h"
#include "test/common/tensor.h"
#include "test/common/workspace_wrapper.h"

namespace megdnn {
namespace test {

TEST_F(ROCM, BN_FORWARD) {
    using namespace batch_normalization;
    std::vector<TestArg> args = get_args();
    Checker<BNForward> checker(handle_rocm());
    for (auto&& arg : args) {
        for (int i = 0; i < 8; ++i) {
            checker.set_dtype(i, dtype::Float32());
        }
        checker.set_dtype(0, arg.dtype);
        checker.set_epsilon(1e-3).set_param(arg.param);
        for (bool need_statistic : {false, true})
            checker.exec({
                    arg.src,
                    arg.param_shape,  // bn_scale
                    arg.param_shape,  // bn_bias
                    need_statistic ? arg.param_shape
                                   : TensorShape({0}),  // mean
                    need_statistic ? arg.param_shape
                                   : TensorShape({0}),  // variance
                    arg.param_shape,                    // batch_mean
                    arg.param_shape,                    // batch_inv_variance
                    {}                                  // dst
            });
    }
}

TEST_F(ROCM, BN_BACKWARD) {
    using namespace batch_normalization;
    std::vector<TestArg> args = get_args();
    Checker<BNBackward> checker(handle_rocm());
    for (auto&& arg : args) {
        for (int i = 0; i < 8; ++i) {
            checker.set_dtype(i, dtype::Float32());
        }
        checker.set_dtype(0, arg.dtype)    // x
                .set_dtype(1, arg.dtype)   // dy
                .set_dtype(7, arg.dtype);  // dx
        checker.set_epsilon(1e-3).set_param(arg.param).exec(
                {arg.src, arg.src, arg.param_shape, arg.param_shape,
                 arg.param_shape, arg.param_shape, arg.param_shape, arg.src});
    }
}

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
