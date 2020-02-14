/**
 * \file dnn/test/cpu/resize.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/cpu/fixture.h"
#include "test/common/resize.h"
#include "test/common/checker.h"

namespace megdnn {
namespace test {

TEST_F(CPU, RESIZE_CV)
{
    using namespace resize;
    std::vector<TestArg> args = get_cv_args();
    Checker<Resize> checker(handle());

    for (auto &&arg: args) {
        checker.set_param(arg.param)
            .set_dtype(0, dtype::Uint8())
            .set_dtype(1, dtype::Uint8())
            .set_epsilon(1+1e-3)
            .execs({arg.src, arg.dst});
    }

    for (auto &&arg: args) {
        checker.set_param(arg.param)
            .set_dtype(0, dtype::Float32())
            .set_dtype(1, dtype::Float32())
            .execs({arg.src, arg.dst});
    }

}

TEST_F(CPU, RESIZE)
{
    using namespace resize;
    std::vector<TestArg> args = get_args();
    Checker<Resize> checker(handle());

    for (auto &&arg: args) {
        checker.set_param(arg.param)
            .set_dtype(0, dtype::Uint8())
            .set_dtype(1, dtype::Uint8())
            .set_epsilon(1+1e-3)
            .execs({arg.src, arg.dst});
    }

    for (auto &&arg: args) {
        checker.set_param(arg.param)
            .set_dtype(0, dtype::Float32())
            .set_dtype(1, dtype::Float32())
            .execs({arg.src, arg.dst});
    }

}

TEST_F(CPU, RESIZE_NCHW_WITH_STRIDE) {
    param::Resize param;
    param.format = param::Resize::Format::NCHW;
    param.imode = param::Resize::InterpolationMode::LINEAR;
    Checker<Resize> checker(handle());
    checker.set_epsilon(1 + 1e-3)
           .set_param(param);

    auto run = [&](TensorShape src_shape, std::vector<ptrdiff_t> src_layout,
                   TensorShape dst_shape, DType dtype) {
        checker.set_dtype(0, dtype)
               .set_dtype(1, dtype)
               .execl({{src_shape, src_layout, dtype}, {dst_shape, dtype}});
    };

    for (DType& dtype : std::vector<DType>{dtype::Float32(), dtype::Uint8()}) {
        run({2, 3, 4, 4}, {256, 32, 8, 1}, {2, 3, 3, 3}, dtype);
        run({1, 3, 4, 3}, {105, 35, 7, 2}, {1, 3, 5, 5}, dtype);
        run({2, 3, 4, 4}, {-256, 32, -8, 1}, {2, 3, 3, 3}, dtype);
        run({2, 3, 4, 4}, {256, -32, 8, -1}, {2, 3, 3, 3}, dtype);
        run({2, 3, 4, 4}, {-256, -32, -8, -1}, {2, 3, 3, 3}, dtype);
    }
}

TEST_F(CPU, RESIZE_NCHW4) {
    using namespace resize;
    auto args = get_nchw4_args();
    Checker<Resize> checker(handle());

    for (auto &&arg: args) {
        checker.set_param(arg.param)
            .set_dtype(0, dtype::QuantizedS8(1.0f))
            .set_dtype(1, dtype::QuantizedS8(1.0f))
            .set_epsilon(1+1e-3)
            .execs({arg.src, arg.dst});
    }

}

} // namespace test
} // namespace megdnn
// vim: syntax=cpp.doxygen

