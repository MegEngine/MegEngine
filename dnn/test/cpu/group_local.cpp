/**
 * \file dnn/test/cpu/group_local.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "megdnn/oprs/nn.h"

#include "test/cpu/fixture.h"
#include "test/common/checker.h"

namespace megdnn {
namespace test {

TEST_F(CPU, GROUP_LOCAL)
{
    auto run = [&](size_t N, size_t IC, size_t IH, size_t IW,
            size_t FH, size_t FW,
            size_t OC, size_t OH, size_t OW,
            size_t PH, size_t PW,
            size_t SH, size_t SW,
            size_t group)
    {
        Checker<GroupLocal> checker(handle());
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
    // simple groupped
    run(2, 6, 5, 5,
            2, 2,
            9, 4, 4,
            0, 0,
            1, 1,
            3);
    // ungroupped
    run(1, 1, 1, 1,
            1, 1,
            1, 1, 1,
            0, 0,
            1, 1,
            1);
    // normal case
    run(2, 64, 7, 7,
            3, 3,
            32, 5, 5,
            0, 0,
            1, 1,
            1);
    // padded and stridded case
    run(2, 32, 7, 7,
            3, 3,
            64, 9, 4,
            2, 1,
            1, 2,
            4);
    // strided case with larger batch
    run(7, 32, 7, 7,
            3, 3,
            64, 3, 3,
            0, 0,
            2, 2,
            8);
}

} // namespace test
} // namespace megdnn

// vim: syntax=cpp.doxygen
