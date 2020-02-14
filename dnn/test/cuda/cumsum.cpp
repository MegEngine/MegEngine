/**
 * \file dnn/test/cuda/cumsum.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/cuda/fixture.h"

#include "megdnn/oprs.h"
#include "test/common/checker.h"

namespace megdnn {
namespace test {

TEST_F(CUDA, CUMSUM)
{
    Checker<Cumsum> checker(handle_cuda());
    struct TestArg {
        param::Cumsum param;
        TensorShape shape;
        TestArg(param::Cumsum param, TensorShape shape):
            param(param), shape(shape)
        {}
    };
    std::vector<TestArg> args, args_int32;
    for (auto shape: TensorShapeArray{{10000}, {33000, 33},
            {100, 100, 100}, {30, 30, 30, 30}}) {
        for (size_t axis = 0; axis < shape.ndim; ++axis) {
            args.emplace_back(param::Cumsum(axis, true, true), shape);
            args.emplace_back(param::Cumsum(axis, true, false), shape);
            args.emplace_back(param::Cumsum(axis, false, true), shape);
            args.emplace_back(param::Cumsum(axis, false, false), shape);
        }
    }
    for (auto shape: TensorShapeArray{{1}, {10}, {100}, {1000}, {10000},
            {100000}})
    {
        args.emplace_back(param::Cumsum(0, true, true), shape);
        args.emplace_back(param::Cumsum(0, true, false), shape);
        args.emplace_back(param::Cumsum(0, false, true), shape);
        args.emplace_back(param::Cumsum(0, false, false), shape);
    }
    for (auto shape: TensorShapeArray{{1}, {10}, {100}, {1000}, {10000},
            {100000}, {1000000}, {1050000}, {2100000}})
    {
        args_int32.emplace_back(param::Cumsum(0, true, true), shape);
        args_int32.emplace_back(param::Cumsum(0, true, false), shape);
        args_int32.emplace_back(param::Cumsum(0, false, true), shape);
        args_int32.emplace_back(param::Cumsum(0, false, false), shape);
    }
    for (auto arg: args) {
        checker.set_param(arg.param);
        checker.set_epsilon(1e-2);
        checker.set_dtype(0, dtype::Float32()).execs({{arg.shape}, {}});
        checker.set_dtype(0, dtype::Int16()).execs({{arg.shape}, {}});
        checker.set_dtype(0, dtype::Int32()).execs({{arg.shape}, {}});
    }
    for (auto arg: args_int32) {
        checker.set_param(arg.param);
        checker.set_epsilon(1e-2);
        checker.set_dtype(0, dtype::Int32()).execs({{arg.shape}, {}});
    }
}

} // namespace test
} // namespace megdnn
// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
