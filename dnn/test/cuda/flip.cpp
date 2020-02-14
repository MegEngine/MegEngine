/**
 * \file dnn/test/cuda/flip.cpp
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
#include "test/common/flip.h"
#include "test/common/checker.h"
#include "test/common/benchmarker.h"
#include "test/cuda/fixture.h"

namespace megdnn {
namespace test {

TEST_F(CUDA, FLIP) {
    using namespace flip;
    std::vector<TestArg> args = get_args();
    Checker<Flip> checker(handle_cuda());
    checker.set_dtype(0, dtype::Int32());
    checker.set_dtype(1, dtype::Int32());

    //! test for batch size exceed CUDNN_MAX_BATCH_X_CHANNEL_SIZE
    Flip::Param cur_param;
    for (bool vertical : {false, true}) {
        for (bool horizontal : {false, true}) {
            cur_param.horizontal = horizontal;
            cur_param.vertical = vertical;
            args.emplace_back(cur_param, TensorShape{65535, 3, 4, 1});
            args.emplace_back(cur_param, TensorShape{65540, 3, 4, 3});
        }
    }
    for (auto &&arg : args) {
        checker.execs({arg.src, {}});
    }

}

TEST_F(CUDA, FLIP_BENCHMARK) {
    auto run = [&](const TensorShapeArray& shapes) {
        Benchmarker<Flip> benchmarker(handle_cuda());

        benchmarker.set_dtype(0, dtype::Int32());
        benchmarker.set_dtype(1, dtype::Int32());

        benchmarker.set_times(5);
        Flip::Param param;

#define BENCHMARK_FLIP(is_vertical, is_horizontal)                            \
    param.vertical = is_vertical;                                             \
    param.horizontal = is_horizontal;                                         \
    benchmarker.set_param(param);                                             \
    printf("src:%s vertical==%d horizontal==%d\n", shape.to_string().c_str(), \
           is_vertical, is_horizontal);                                       \
    benchmarker.execs({shape, {}});

        for (auto&& shape : shapes) {
            BENCHMARK_FLIP(false, false);
            BENCHMARK_FLIP(false, true);
            BENCHMARK_FLIP(true, false);
            BENCHMARK_FLIP(true, true);
        }
#undef BENCHMARK_FLIP
    };

    TensorShapeArray shapes = {
        {3, 101, 98, 1},
        {3, 101, 98, 3}
    };

    run(shapes);
}

}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
