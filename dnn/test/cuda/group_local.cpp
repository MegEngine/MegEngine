/**
 * \file dnn/test/cuda/group_local.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "megdnn/oprs/nn.h"

#include "test/cuda/fixture.h"
#include "test/common/checker.h"

#if MEGDNN_WITH_BENCHMARK
#include "test/common/benchmarker.h"
#endif

namespace megdnn {
namespace test {

TEST_F(CUDA, GROUP_LOCAL_FORWARD)
{
    auto run = [&](size_t N, size_t IC, size_t IH, size_t IW,
            size_t FH, size_t FW,
            size_t OC, size_t OH, size_t OW,
            size_t PH, size_t PW,
            size_t SH, size_t SW,
            size_t group)
    {
        Checker<GroupLocal> checker(handle_cuda());
        GroupLocal::Param param;
        param.pad_h = PH;
        param.pad_w = PW;
        param.stride_h = SH;
        param.stride_w = SW;
        auto ICg = IC / group;
        auto OCg = OC / group;
        checker.set_param(param).exec({{N, IC, IH, IW},
                {group, OH, OW, ICg, FH, FW, OCg},
                {}});
    };
    for (size_t IC = 1; IC <= 16; ++IC)
    for (size_t N = 1; N <= 8; ++N)
    {
        size_t group = 5;
        size_t H = 7, W = 7;
        size_t OC = 7;
        run(N, IC*group, H, W, 3, 3, OC*group, H, W, 1, 1, 1, 1, group);
    }
    for (size_t N: {2, 64}) {
        // normal case
        run(N, 64, 7, 7,
                3, 3,
                32, 5, 5,
                0, 0,
                1, 1,
                2);
        // padded case
        run(N, 32, 7, 7,
                3, 3,
                64, 7, 7,
                1, 1,
                1, 1,
                2);
        // strided case
        run(N, 64, 7, 7,
                3, 3,
                64, 3, 3,
                0, 0,
                2, 2,
                4);
    }
}

#if MEGDNN_WITH_BENCHMARK
TEST_F(CUDA, BENCHMARK_GROUP_LOCAL_FORWARD)
{
    Benchmarker<GroupLocalForward> B(handle_cuda());
    B.execs({{2, 352, 4, 4}, {22, 4, 4, 16, 3, 3, 16}, {}});
    B.execs({{2, 192, 8, 8}, {12, 8, 8, 16, 3, 3, 16}, {}});
    B.execs({{2, 176, 4, 4}, {11, 4, 4, 16, 3, 3, 16}, {}});
}
#endif

TEST_F(CUDA, GROUP_LOCAL_BACKWARD_DATA)
{
    auto run = [&](size_t N, size_t IC, size_t IH, size_t IW,
            size_t FH, size_t FW,
            size_t OC, size_t OH, size_t OW,
            size_t PH, size_t PW,
            size_t SH, size_t SW,
            size_t group)
    {
        Checker<GroupLocalBackwardData> checker(handle_cuda());
        GroupLocal::Param param;
        param.pad_h = PH;
        param.pad_w = PW;
        param.stride_h = SH;
        param.stride_w = SW;
        auto ICg = IC / group;
        auto OCg = OC / group;
        checker.set_param(param).exec({{group, OH, OW, ICg, FH, FW, OCg},
                {N, OC, OH, OW},
                {N, IC, IH, IW},
                });
    };
    for (size_t N: {64}) {
        // normal case
        run(N, 64, 7, 7,
                3, 3,
                32, 5, 5,
                0, 0,
                1, 1,
                2);
        // padded case
        run(N, 32, 7, 7,
                3, 3,
                64, 7, 7,
                1, 1,
                1, 1,
                2);
        // strided case
        run(N, 64, 7, 7,
                3, 3,
                64, 3, 3,
                0, 0,
                2, 2,
                4);
    }
}

TEST_F(CUDA, GROUP_LOCAL_BACKWARD_FILTER)
{
    auto run = [&](size_t N, size_t IC, size_t IH, size_t IW,
            size_t FH, size_t FW,
            size_t OC, size_t OH, size_t OW,
            size_t PH, size_t PW,
            size_t SH, size_t SW,
            size_t group)
    {
        Checker<GroupLocalBackwardFilter> checker(handle_cuda());
        GroupLocal::Param param;
        param.pad_h = PH;
        param.pad_w = PW;
        param.stride_h = SH;
        param.stride_w = SW;
        auto ICg = IC / group;
        auto OCg = OC / group;
        checker.set_param(param).exec({{N, IC, IH, IW},
                {N, OC, OH, OW},
                {group, OH, OW, ICg, FH, FW, OCg},
                });
    };
    for (size_t N: {64}) {
        // normal case
        run(N, 64, 7, 7,
                3, 3,
                32, 5, 5,
                0, 0,
                1, 1,
                2);
        // padded case
        run(N, 32, 7, 7,
                3, 3,
                64, 7, 7,
                1, 1,
                1, 1,
                2);
        // strided case
        run(N, 64, 7, 7,
                3, 3,
                64, 3, 3,
                0, 0,
                2, 2,
                4);
    }
}

} // namespace test
} // namespace megdnn

// vim: syntax=cpp.doxygen
