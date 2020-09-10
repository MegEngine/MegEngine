/**
 * \file dnn/test/cpu/relayout.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "test/common/checker.h"
#include "test/common/benchmarker.h"
#include "test/common/tensor.h"
#include "test/common/relayout.h"
#include "test/cpu/fixture.h"

#include "megdnn/basic_types.h"

using namespace megdnn;
using namespace test;


namespace {
template<typename tag>
class CPU_RELAYOUT: public CPU {
};
TYPED_TEST_CASE(CPU_RELAYOUT, relayout::test_types);
TYPED_TEST(CPU_RELAYOUT, run) {
    relayout::run_test<TypeParam>(this->handle());
}
}

#if MEGDNN_WITH_BENCHMARK
TEST_F(CPU, BENCHMARK_RELAYOUT_CV) {
    relayout::run_cv_benchmark(handle());
}

TEST_F(CPU, BENCHMARK_RELAYOUT) {
    // Check if invoke fallback if it's not satisfied cv.
    using namespace relayout;
    std::vector<TestArg> args;
    args.emplace_back(TensorLayout({1, 8, 3, 64, 64},
                                   {64 * 64 * 3, 64 * 8, 64 * 64, 64, 1},
                                   dtype::Float32()),
                      TensorLayout({1, 8, 3, 64, 64}, dtype::Float32()));
    auto handle_naive = create_cpu_handle(2);
    Benchmarker<Relayout> benchmarker(handle());
    Benchmarker<Relayout> benchmarker_naive(handle_naive.get());

    benchmarker_naive.set_times(1);
    benchmarker.set_times(1);
    for (auto &&arg : args) {
        float cpu_time = benchmarker.execl({arg.src, arg.dst});
        float naive_time = benchmarker_naive.execl({arg.src, arg.dst});
        ASSERT_LE(cpu_time * 5, naive_time);
    }
}
#endif

// vim: syntax=cpp.doxygen
