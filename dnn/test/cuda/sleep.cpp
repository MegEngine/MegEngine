/**
 * \file dnn/test/cuda/sleep.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "test/cuda/fixture.h"
#include "test/cuda/utils.h"
#include "megdnn/oprs.h"
#include "../src/common/utils.h"

#include <chrono>
#include <cstdio>

#include <cuda_runtime_api.h>

using namespace megdnn;
using namespace test;


TEST_F(CUDA, SLEEP) {
    auto opr = this->handle_cuda()->create_operator<megdnn::SleepForward>();

    auto run = [&](float time) -> double {
        opr->param() = {time};
        cuda_check(cudaDeviceSynchronize());
        auto t0 = std::chrono::high_resolution_clock::now();
        opr->exec();
        cuda_check(cudaDeviceSynchronize());
        auto t1 = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = t1 - t0;
        return diff.count();
    };

    // warmv7up
    run(0.01);

    for (auto i: {0.1, 0.3}) {
        auto get = run(i);
        // sleep kernel in cuda is easily affected by the frequency change of
        // GPU, so we just print warn log instead assert. more refer to
        // XPU-226
        if (get < i || get > i * 2) {
            megdnn_log_warn("expect time between [%f, %f], got %f", i, 2 * i,
                            get);
        }
    }
}



// vim: syntax=cpp.doxygen

