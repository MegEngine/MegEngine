/**
 * \file dnn/test/cuda/images2neibs.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/cuda/fixture.h"

#include "test/common/checker.h"
#include "test/common/images2neibs.h"
#include "test/common/rng.h"
#include "test/cuda/benchmark.h"

namespace megdnn {
namespace test {

TEST_F(CUDA, IMAGES2NEIBS_FORWARD)
{
    auto args = images2neibs::get_args();
    for (auto &&arg: args) {
        Checker<Images2NeibsForward> checker(handle_cuda());
        checker.set_param(arg.param).set_dtype(0, dtype::Float32()).
            exec(TensorShapeArray{
                arg.ishape, {}});
        checker.set_param(arg.param).set_dtype(0, dtype::Float16()).
            exec(TensorShapeArray{
                arg.ishape, {}});
    }
}

#if MEGDNN_WITH_BENCHMARK
TEST_F(CUDA, BENCHMARK_IMAGES2NEIBS_FORWARD)
{
    auto args = images2neibs::get_benchmark_args();
    for (auto &&arg: args) {
        CUBenchmarker<Images2NeibsForward> bencher(handle_cuda());
        bencher.set_param(arg.param).set_dtype(0, dtype::Float32()).
            exec(TensorShapeArray{
                arg.ishape, {}});
    }
}
#endif

TEST_F(CUDA, IMAGES2NEIBS_BACKWARD)
{
    UniformFloatRNG rng(0, 1);
    auto args = images2neibs::get_args();
    for (auto &&arg: args) {
        Checker<Images2NeibsBackward> checker(handle_cuda());
        checker.set_epsilon(1e-2);
        checker.set_rng(0, &rng);
        TensorLayout ilayout = TensorLayout(arg.ishape, dtype::Float32());
        TensorLayout olayout;
        {
            auto opr = handle_cuda()->create_operator<Images2Neibs>();
            opr->param() = arg.param;
            opr->deduce_layout(ilayout, olayout);
        }
        auto set_dtype = [&checker](DType dtype)
        {
            checker.set_dtype(0, dtype).
                set_dtype(1, dtype);
        };
        set_dtype(dtype::Float32());
        checker.set_param(arg.param).exec(TensorShapeArray{
                olayout, ilayout});
        set_dtype(dtype::Float16());
        checker.set_param(arg.param).exec(TensorShapeArray{
                olayout, ilayout});
    }
}

} // namespace test
} // namespace megdnn

// vim: syntax=cpp.doxygen

