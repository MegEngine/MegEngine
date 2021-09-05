/**
 * \file dnn/test/fallback/local.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/fallback/fixture.h"

#include "test/common/checker.h"
#include "test/common/local.h"
#include "test/common/benchmarker.h"
#include "test/common/timer.h"

namespace megdnn {
namespace test {

TEST_F(FALLBACK, LOCAL)
{
    auto args = local::get_args();
    for (auto &&arg: args) {
        Checker<Local> checker(handle());
        checker.set_param(arg.param).exec(TensorShapeArray{
                arg.sshape(), arg.fshape(), arg.dshape()});
    }
}

#if MEGDNN_WITH_BENCHMARK
TEST_F(FALLBACK, BENCHMARK_LOCAL)
{
    size_t T = 1;
    std::vector<float> src_mem(1000000), dst_mem(1000000);
    auto total_mem = (src_mem.size() + dst_mem.size()) * sizeof(float) * T;
    Timer timer;
    timer.start();
    for (size_t t = 0; t < T; ++t) {
        std::memcpy(dst_mem.data(), src_mem.data(), sizeof(float) * src_mem.size());
        // to prevent compiler optimizing out memcpy above.
        asm volatile ("");
    }
    timer.stop();
    auto time_in_ms = timer.get_time_in_us() / 1e3;
    auto bandwidth = total_mem / (time_in_ms/1000.0f);
    printf("Memcpy bandwidth is %f GB/s\n",bandwidth / 1e9);

        size_t N = 16,IC = 16,OC = 16,IH = 300,IW = 300,FW = 2,FH = 2;
        Local::Param param;
        param.pad_h = param.pad_w = 0;
        size_t OH = (IH + 2 * param.pad_h - FH) / param.stride_h + 1;
        size_t OW = (IW + 2 * param.pad_w - FW) / param.stride_w + 1;
        Benchmarker<Local> benchmarker(handle());
        TensorShape src{N, IC, IH, IW},
                    filter{OH, OW, IC, FH, FW, OC},
                    dst{N, OC, OH, OW};
    time_in_ms = benchmarker.set_times(T).
            set_param(param).
            set_display(false).
            exec({src, filter, dst});
    total_mem = (src.total_nr_elems() +
            filter.total_nr_elems() +
            dst.total_nr_elems()) * sizeof(float)*T;
    bandwidth = total_mem / (time_in_ms/1000.0f);
    printf("Bandwidth is %f GB/s\n",bandwidth / 1e9);
}
#endif
} // namespace test
} // namespace megdnn

// vim: syntax=cpp.doxygen
