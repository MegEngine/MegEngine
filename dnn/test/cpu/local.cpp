/**
 * \file dnn/test/cpu/local.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/cpu/fixture.h"

#include "test/common/checker.h"
#include "test/common/local.h"
#include "test/common/benchmarker.h"
#include "test/common/timer.h"

namespace megdnn {
namespace test {

TEST_F(CPU, LOCAL)
{
    auto args = local::get_args();
    for (auto &&arg: args) {
        Checker<Local> checker(handle());
        checker.set_param(arg.param).exec(TensorShapeArray{
                arg.sshape(), arg.fshape(), arg.dshape()});
    }
}

#if MEGDNN_WITH_BENCHMARK
TEST_F(CPU, BENCHMARK_LOCAL)
{
    size_t T = 10;
    float memcpy_bandwidth, local_bandwidth;
    {
        std::vector<float> src(1000000), dst(1000000);
        auto total_mem = (src.size() + dst.size()) * sizeof(float) * T;
        Timer timer;
        timer.start();
        for (size_t t = 0; t < T; ++t) {
            std::memcpy(dst.data(), src.data(), sizeof(float) * src.size());
            // to prevent compiler optimizing out memcpy above.
            asm volatile ("");
        }
        timer.stop();
        auto time_in_ms = timer.get_time_in_us() / 1e3;
        auto bandwidth = total_mem / (time_in_ms/1000.0f);
        std::cout << "Copy from src(" << src.data()
            << ") to dst(" << dst.data()
            << ")" << std::endl;
        std::cout << "Memcpy bandwidth is " << bandwidth / 1e9 << "GB/s" << std::endl;
        memcpy_bandwidth = bandwidth;
    }
    {
        Benchmarker<Local> benchmarker(handle());
        TensorShape src{2, 64, 7, 7},
                    filter{5, 5, 64, 3, 3, 64},
                    dst{2, 64, 5, 5};
        Local::Param param;
        param.pad_h = param.pad_w = 0;
        auto time_in_ms = benchmarker.set_times(T).
            set_param(param).
            set_display(false).
            exec({src, filter, dst});
        auto total_mem = (src.total_nr_elems() +
                filter.total_nr_elems() +
                dst.total_nr_elems()) * sizeof(float)*T;
        auto bandwidth = total_mem / (time_in_ms/1000.0f);
        std::cout << "Bandwidth is " << bandwidth / 1e9 << "GB/s" << std::endl;
        local_bandwidth = bandwidth;
    }
    float ratio = local_bandwidth / memcpy_bandwidth;
    ASSERT_GE(ratio, 0.05);
}
#endif

} // namespace test
} // namespace megdnn

// vim: syntax=cpp.doxygen
