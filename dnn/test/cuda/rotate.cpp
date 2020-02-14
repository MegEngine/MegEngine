/**
 * \file dnn/test/cuda/rotate.cpp
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
#include "test/common/benchmarker.h"
#include "test/common/checker.h"
#include "test/common/rotate.h"
#include "test/common/tensor.h"
#include "test/cuda/fixture.h"

namespace megdnn {
namespace test {
namespace rotate {

TEST_F(CUDA, ROTATE) {
    using namespace rotate;
    std::vector<TestArg> args = get_args();
    Checker<Rotate> checker(handle_cuda());

    //! test for batch size exceed CUDNN_MAX_BATCH_X_CHANNEL_SIZE
    Rotate::Param cur_param;
    for (bool clockwise : {false, true}) {
        cur_param.clockwise = clockwise;
        args.emplace_back(cur_param, TensorShape{65535, 3, 4, 1},
                          dtype::Int32());
        args.emplace_back(cur_param, TensorShape{65540, 3, 4, 3},
                          dtype::Int32());
    }
    for (auto&& arg : args) {
        checker.set_dtype(0, arg.dtype)
            .set_dtype(1, arg.dtype)
            .execs({arg.src, {}});
    }
}

TEST_F(CUDA, BENCHMARK_ROTATE) {
    auto run = [&](const TensorShapeArray& shapes) {
        Benchmarker<Rotate> benchmarker(handle_cuda());
        Benchmarker<Rotate> benchmarker_cpu(handle_naive());

        benchmarker.set_dtype(0, dtype::Int32());
        benchmarker.set_dtype(1, dtype::Int32());
        benchmarker_cpu.set_dtype(0, dtype::Int32());
        benchmarker_cpu.set_dtype(1, dtype::Int32());

        benchmarker.set_times(5);
        benchmarker_cpu.set_times(5);
        Rotate::Param param;

#define BENCHMARK_rotate(is_clockwise)                                         \
    param.clockwise = is_clockwise;                                            \
    benchmarker.set_param(param);                                              \
    benchmarker_cpu.set_param(param);                                          \
    printf("src:%s clockwise==%d  cuda vs naive\n", shape.to_string().c_str(), \
           is_clockwise);                                                      \
    benchmarker.execs({shape, {}});                                            \
    benchmarker_cpu.execs({shape, {}});

        for (auto&& shape : shapes) {
            BENCHMARK_rotate(false);
            BENCHMARK_rotate(true);
        }
#undef BENCHMARK_rotate
    };

    TensorShapeArray shapes = {{3, 1001, 978, 1}, {3, 1001, 978, 3}};

    run(shapes);
}

}  // namespace rotate
}  // namespace test
}  // namespace megdnn

// vim: syntax=cpp.doxygen
