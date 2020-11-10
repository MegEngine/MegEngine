/**
 * \file dnn/test/cuda/fake_quant.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "test/common/fake_quant.h"
#include "megdnn/oprs.h"
#include "test/common/checker.h"
#include "test/cuda/fixture.h"

namespace megdnn {
namespace test {
using namespace fake_quant;
TEST_F(CUDA, FAKE_QUANT) {
    std::vector<TestArg> args = get_args();
    auto dtype = dtype::Float32();
    std::unique_ptr<RNG> rng;

    for (auto&& arg : args) {
        auto param = arg.param;
        auto ishape = arg.ishape;
        auto scale_shape = arg.scale_shape;
        auto zeropoint_shape = arg.zeropoint_shape;
        Checker<FakeQuantForward> checker(handle_cuda());
        checker.set_param(param)
                .set_dtype(0, dtype)
                .set_dtype(1, dtype)
                .set_dtype(2, dtype)
                .set_dtype(3, dtype)
                .execs(TensorShapeArray{ishape, scale_shape, zeropoint_shape,
                                        ishape});
    }
    // test noncontiguous layout
    for (auto&& arg : args) {
        auto param = arg.param;
        auto ishape = arg.ishape;
        auto scale_shape = arg.scale_shape;
        auto zeropoint_shape = arg.zeropoint_shape;
        Checker<FakeQuantForward> checker(handle_cuda());
        TensorLayout ilayout(
                ishape,
                {(long int)(ishape[1] * ishape[2] * ishape[3] * 2),
                 (long int)(ishape[2] * ishape[3]), (long int)ishape[3], 1},
                dtype::Float32());
        checker.set_param(param).execl({ilayout,
                                        {scale_shape, dtype::Float32()},
                                        {zeropoint_shape, dtype::Float32()},
                                        ilayout});
    }
}

TEST_F(CUDA, FAKE_QUANT_BACKWARD) {
    std::vector<TestArg> args = get_args();
    auto dtype = dtype::Float32();

    for (auto&& arg : args) {
        auto param = arg.param;
        auto ishape = arg.ishape;
        auto scale_shape = arg.scale_shape;
        auto zeropoint_shape = arg.zeropoint_shape;
        Checker<FakeQuantBackward> checker(handle_cuda());
        checker.set_param(param)
                .set_dtype(0, dtype)
                .set_dtype(1, dtype)
                .set_dtype(2, dtype)
                .set_dtype(3, dtype)
                .set_dtype(4, dtype)
                .execs(TensorShapeArray{ishape, ishape, scale_shape,
                                        zeropoint_shape, ishape});
    }
    // test noncontiguous layout
    for (auto&& arg : args) {
        auto param = arg.param;
        auto ishape = arg.ishape;
        auto scale_shape = arg.scale_shape;
        auto zeropoint_shape = arg.zeropoint_shape;
        Checker<FakeQuantBackward> checker(handle_cuda());
        TensorLayout ilayout(
                ishape,
                {(long int)(ishape[1] * ishape[2] * ishape[3] * 2),
                 (long int)(ishape[2] * ishape[3]), (long int)ishape[3], 1},
                dtype::Float32());
        checker.set_param(param).execl({ilayout,
                                        ilayout,
                                        {scale_shape, dtype::Float32()},
                                        {zeropoint_shape, dtype::Float32()},
                                        ilayout});
    }
}

}  // namespace test
}  // namespace megdnn