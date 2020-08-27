/**
 * \file dnn/test/rocm/linspace.cpp
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

#include "megdnn/oprs.h"
#include "test/common/checker.h"
#include "test/rocm/benchmarker.h"

namespace megdnn {
namespace test {

TEST_F(ROCM, LINSPACE)
{
    Checker<Linspace> checker(handle_rocm());
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

TEST_F(ROCM, LINSPACE_BENCHMARK)
{
    ROCMBenchmarker<Linspace> benchmarker(handle_rocm(), handle_naive(false));
    benchmarker.set_display(true);
    Linspace::Param param{0.1, 9999.9, true};
    size_t sz = 50000;
    auto time_ms = benchmarker.set_dtype(0, dtype::Float32())
       .set_param(param).execs({{sz}});
    double bytes = sz * dtype::Float32().size();
    printf("vec size = %ld, bandwidth = %.2f GB/s\n", sz, (float)(bytes / (time_ms * 1e6)));
}

} // namespace test
} // namespace megdnn
// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
