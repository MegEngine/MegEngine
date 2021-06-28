/**
 * \file dnn/test/cuda/lsq.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "test/common/lsq.h"
#include "megdnn/oprs.h"
#include "test/common/checker.h"
#include "test/cuda/fixture.h"

namespace megdnn {
namespace test {

using namespace lsq;

TEST_F(CUDA, LSQ) {
    std::vector<TestArg> args = get_args();
    auto dtype = dtype::Float32();

    for (auto&& arg : args) {
        auto param = arg.param;
        auto ishape = arg.ishape;
        auto scale_shape = arg.scale_shape;
        auto zeropoint_shape = arg.zeropoint_shape;
        auto gradscale_shape = arg.gradscale_shape;
        Checker<LSQForward> checker(handle_cuda());
        checker.set_param(param)
                .set_dtype(0, dtype)
                .set_dtype(1, dtype)
                .set_dtype(2, dtype)
                .set_dtype(3, dtype)
                .set_dtype(4, dtype)
                .execs({ishape, scale_shape, zeropoint_shape, gradscale_shape,
                        ishape});
    }
    // test noncontiguous layout
    for (auto&& arg : args) {
        auto param = arg.param;
        auto ishape = arg.ishape;
        auto sshape = arg.scale_shape;
        auto zeropoint_shape = arg.zeropoint_shape;
        auto gradscale_shape = arg.gradscale_shape;
        Checker<LSQForward> checker(handle_cuda());
        TensorLayout ilayout(
                ishape,
                {(long int)(ishape[1] * ishape[2] * ishape[3] * 2),
                 (long int)(ishape[2] * ishape[3]), (long int)ishape[3], 1},
                dtype::Float32());
        checker.set_param(param).execl({ilayout,
                                        {sshape, dtype::Float32()},
                                        {zeropoint_shape, dtype::Float32()},
                                        {gradscale_shape, dtype::Float32()},
                                        ilayout});
    }
}

TEST_F(CUDA, LSQ_BACKWARD) {
    std::vector<TestArg> args = get_args();
    auto dtype = dtype::Float32();

    for (auto&& arg : args) {
        auto param = arg.param;
        auto ishape = arg.ishape;
        auto scale_shape = arg.scale_shape;
        auto zeropoint_shape = arg.zeropoint_shape;
        auto gradscale_shape = arg.gradscale_shape;
        Checker<LSQBackward> checker(handle_cuda());
        checker.set_param(param)
                .set_dtype(0, dtype)
                .set_dtype(1, dtype)
                .set_dtype(2, dtype)
                .set_dtype(3, dtype)
                .set_dtype(4, dtype)
                .set_dtype(5, dtype)
                .set_dtype(6, dtype)
                .execs({ishape, ishape, scale_shape, zeropoint_shape,
                        gradscale_shape, ishape, ishape});
    }
    // test noncontiguous layout
    for (auto&& arg : args) {
        auto param = arg.param;
        auto ishape = arg.ishape;
        auto sshape = arg.scale_shape;
        auto zeropoint_shape = arg.zeropoint_shape;
        auto gradscale_shape = arg.gradscale_shape;
        Checker<LSQBackward> checker(handle_cuda());
        TensorLayout ilayout(
                ishape,
                {(long int)(ishape[1] * ishape[2] * ishape[3] * 2),
                 (long int)(ishape[2] * ishape[3]), (long int)ishape[3], 1},
                dtype::Float32());
        checker.set_param(param).execl({ilayout,
                                        ilayout,
                                        {sshape, dtype::Float32()},
                                        {zeropoint_shape, dtype::Float32()},
                                        {gradscale_shape, dtype::Float32()},
                                        ilayout,
                                        ilayout});
    }
}

}  // namespace test
}  // namespace megdnn