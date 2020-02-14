/**
 * \file dnn/test/cpu/task_executor.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/naive/handle.h"
#include "test/common/utils.h"
#include "test/cpu/fixture.h"

using namespace megdnn;
using namespace test;

TEST_F(CPU_MULTI_THREADS, THREAD_POOL) {
    auto single_thread_handle = create_cpu_handle(0);
    std::vector<int> data(100, 0);
    std::vector<int> result_singel_thread(100);
    std::vector<int> result_multi_thread(100);
    for (int i = 0; i < 100; i++) {
        data[i] = i;
    }
    auto single_run = [&data, &result_singel_thread]() {
        for (int i = 0; i < 100; i++) {
            result_singel_thread[i] = data[i];
        }
    };
    auto multi_thread_run = [&data, &result_multi_thread](size_t index,
                                                          size_t) {
        for (size_t i = index * 5; i < (index + 1) * 5; i++) {
            result_multi_thread[i] = data[i];
        }
    };
    MEGDNN_DISPATCH_CPU_KERN(
            static_cast<naive::HandleImpl*>(single_thread_handle.get()),
            single_run());
    MEGDNN_DISPATCH_MULTI_THREAD_CPU_KERN(
            static_cast<naive::HandleImpl*>(handle()), 20, multi_thread_run);
    for (int i = 0; i < 100; i++) {
        ASSERT_EQ(result_singel_thread[i], result_multi_thread[i]);
    }
}

// vim: syntax=cpp.doxygen
