/**
 * \file dnn/test/fallback/roi_copy.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include <gtest/gtest.h>

#include "megdnn.h"
#include "megdnn/oprs.h"
#include "test/common/tensor.h"
#include "test/common/roi_copy.h"
#include "test/common/checker.h"
#include "test/common/benchmarker.h"
#include "test/fallback/fixture.h"

namespace megdnn {
namespace test {

TEST_F(FALLBACK, ROICOPY) {
    using namespace roi_copy;
    std::vector<TestArg> args = get_args();
    Checker<ROICopy> checker(handle());
    checker.set_dtype(0, dtype::Int32());
    checker.set_dtype(1, dtype::Int32());

    for (auto&& arg : args) {
        checker.set_param(arg.param).execs({arg.src, {}});
    }

}
#if MEGDNN_WITH_BENCHMARK
TEST_F(FALLBACK, BENCHMARK_ROICOPY) {
    auto run = [&](const TensorShapeArray& shapes) {
        Benchmarker<ROICopy> benchmarker(handle());

        benchmarker.set_dtype(0, dtype::Int32());
        benchmarker.set_dtype(1, dtype::Int32());

        benchmarker.set_times(5);
        ROICopy::Param param;

        for (auto&& shape : shapes) {
            param.row_from = shape[1]/10;
            param.row_to = shape[1]/2;
            param.col_from = shape[2]/9;
            param.col_to = shape[2]/3 * 2;
            benchmarker.set_param(param).execs({shape, {}});
        }
#undef BENCHMARK_ROICopy
    };

    TensorShapeArray shapes = {
        {3, 1021, 980, 1},
        {3, 1021, 980, 3}
    };

    run(shapes);
}
#endif


}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
