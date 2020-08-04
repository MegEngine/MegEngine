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
#include "megbrain/opr/io.h"
#include "megbrain/opr/utility.h"
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

TEST(TestGraph, ParallelRunMultithreadMode) {
    // check race conditions when graphs are executed on multple threads
    std::atomic_size_t sync_counter{0};
    constexpr size_t NR_RUN = 50;
    size_t nr_worker = std::max(4, sys::get_cpu_count() / 4);
    if (auto setting = MGB_GETENV("TestGraphParallelRun_nr_worker")) {
        nr_worker = std::stoul(setting);
    }
    mgb_log("use %zu workers", nr_worker);

    auto sync_barrier = [&sync_counter, nr_worker](size_t& cnt) {
        ++sync_counter;
        ++cnt;
        while (sync_counter < cnt * nr_worker)
            ;
    };

    auto do_worker = [&sync_barrier](size_t sync_cnt) {
        auto cn = CompNode::load("multithread2:0");
        HostTensorGenerator<> gen;
        auto host_x = gen({23}, cn);
        HostTensorND host_y, y_expect;
        y_expect.copy_from(*host_x);
        {
            auto py = y_expect.ptr<float>();
            for (int i = 0; i < 23; ++i) {
                for (int j = 0; j < 5; ++j) {
                    py[i] = py[i] * 2 + 3;
                }
            }
        }

        sync_barrier(sync_cnt);
        auto graph = ComputingGraph::make();
        auto x = opr::Host2DeviceCopy::make(*graph, host_x), y = x;
        for (int i = 0; i < 5; ++i) {
            y = y * 2 + 3;
        }

        sync_barrier(sync_cnt);
        auto func = graph->compile({make_callback_copy(y, host_y)});

        sync_barrier(sync_cnt);
        func->execute();
        MGB_ASSERT_TENSOR_EQ(y_expect, host_y);
        memset(host_y.raw_ptr(), -1, 23 * sizeof(float));

        sync_barrier(sync_cnt);
        func->execute();
        MGB_ASSERT_TENSOR_EQ(y_expect, host_y);
        func->wait();
    };
    auto worker = [&]() {
        size_t scnt = 0;
        for (size_t run_id = 0; run_id < NR_RUN; ++run_id) {
            do_worker(scnt);
        }
    };

    std::vector<std::thread> workers;
    for (size_t i = 0; i < nr_worker; ++i)
        workers.emplace_back(worker);

    for (auto&& i : workers)
        i.join();
}
#else
#pragma message "tests are disabled as thread is not enabled."
#endif  //  MGB_HAVE_THREAD

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
