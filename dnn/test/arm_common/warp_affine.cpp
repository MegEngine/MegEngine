/**
 * \file dnn/test/arm_common/warp_affine.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/arm_common/fixture.h"
#include "test/common/warp_affine.h"
#include "test/common/checker.h"

namespace megdnn {
namespace test {

TEST_F(ARM_COMMON_MULTI_THREADS, WARP_AFFINE_CV) {
    using namespace warp_affine;
    std::vector<TestArg> args = get_cv_args();
    Checker<WarpAffine> checker(handle());

    for (auto &&arg : args) {
        checker.set_param(arg.param)
            .set_epsilon(1 + 1e-3)
            .set_dtype(0, dtype::Uint8())
            .set_dtype(1, dtype::Float32())
            .set_dtype(2, dtype::Uint8())
            .execs({arg.src, arg.trans, arg.dst});
    }

    for (auto &&arg: args) {
        checker.set_param(arg.param)
            .set_dtype(0, dtype::Float32())
            .set_dtype(1, dtype::Float32())
            .set_dtype(2, dtype::Float32())
            .execs({arg.src, arg.trans, arg.dst});
    }

}

} // namespace test
} // namespace megdnn
// vim: syntax=cpp.doxygen
