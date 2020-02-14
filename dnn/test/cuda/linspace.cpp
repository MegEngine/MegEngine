/**
 * \file dnn/test/cuda/linspace.cpp
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

TEST_F(CUDA, LINSPACE)
{
    Checker<Linspace> checker(handle_cuda());
    Linspace::Param param;
    param.start = 0.5;
    param.stop = 1.5;
    param.endpoint = true;
    for (DType dtype: std::vector<DType>{
            dtype::Float16(), dtype::Int32(), dtype::Float32()}) {
        checker.set_dtype(0, dtype).set_param(param).exec(
                TensorShapeArray{{11}});
    }
    param.endpoint = false;
    for (DType dtype: std::vector<DType>{
            dtype::Float16(), dtype::Int32(), dtype::Float32()}) {
        checker.set_dtype(0, dtype).set_param(param).exec(
                TensorShapeArray{{11}});
    }

}

} // namespace test
} // namespace megdnn
// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
