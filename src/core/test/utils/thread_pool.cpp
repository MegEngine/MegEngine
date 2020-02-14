/**
 * \file src/core/test/utils/thread_pool.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "megbrain/utils/thread_pool.h"
#include "megbrain/comp_node.h"
#include "megbrain/system.h"
#include "megbrain/test/helper.h"
#include <atomic>
#include <random>

#if MGB_HAVE_THREAD
using namespace mgb;
TEST(TestThreadPool, BASIC) {
    auto thread_pool0 = std::make_shared<ThreadPool>(1u);
    auto thread_pool1 = std::make_shared<ThreadPool>(4u);
    ASSERT_EQ(thread_pool0->nr_threads(), static_cast<size_t>(1));
    ASSERT_EQ(thread_pool1->nr_threads(), static_cast<size_t>(4));

    std::vector<int> source(100), dst0(100), dst1(100), truth(100);
    std::atomic_size_t count0{0}, count1{0};
    for (int i = 0; i < 100; i++) {
        source[i] = i;
        dst0[i] = 0;
        dst1[i] = 0;
        truth[i] = i * i;
    }
    size_t total_task = 50;
    auto func0 = [&](size_t index, size_t) {
        count0++;
        size_t sub_task = 100 / total_task;
        for (size_t i = index * sub_task; i < (index + 1) * sub_task; i++) {
            dst0[i] = source[i] * source[i];
        }
    };
    auto func1 = [&](size_t index, size_t) {
        count1++;
        size_t sub_task = 100 / total_task;
        for (size_t i = index * sub_task; i < (index + 1) * sub_task; i++) {
            dst1[i] = source[i] * source[i];
        }
    };
    thread_pool0->active();
    thread_pool0->add_task({func0, total_task});
    thread_pool0->deactive();
    thread_pool1->active();
    thread_pool1->add_task({func1, total_task});
    thread_pool1->deactive();
    ASSERT_EQ(count0, total_task);
    ASSERT_EQ(count1, total_task);
    for (size_t i = 0; i < 100; i++) {
        ASSERT_EQ(dst0[i], truth[i]);
        ASSERT_EQ(dst1[i], truth[i]);
    }
}
#else
#pragma message "tests are disabled as thread is not enabled."
#endif  //  MGB_HAVE_THREAD

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
