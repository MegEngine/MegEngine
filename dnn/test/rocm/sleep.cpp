/**
 * \file dnn/test/rocm/sleep.cpp
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
#include "test/rocm/utils.h"
#include "megdnn/oprs.h"

#include <chrono>
#include <cstdio>

using namespace megdnn;
using namespace test;

#if !(MEGDNN_AARCH64)

TEST_F(ROCM, SLEEP) {
    auto opr = this->handle_rocm()->create_operator<Sleep>();

    auto run = [&](float time) -> double {
        opr->param() = {time};
        hip_check(hipDeviceSynchronize());
        auto t0 = std::chrono::high_resolution_clock::now();
        opr->exec();
        hip_check(hipDeviceSynchronize());
        auto t1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = t1 - t0;
        return diff.count();
    };

    // warmv7up
    run(0.01);

    for (auto i: {0.1, 0.3}) {
        auto get = run(i);
        ASSERT_GE(get, i);
        ASSERT_LE(get, i * 2);
    }
}

#endif


// vim: syntax=cpp.doxygen

