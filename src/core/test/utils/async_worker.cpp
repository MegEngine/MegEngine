/**
 * \file src/core/test/utils/async_worker.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/utils/async_worker.h"
#include "megbrain/test/helper.h"
#include "megbrain/utils/timer.h"

#include <atomic>
#include <chrono>
#include <thread>

#if MGB_HAVE_THREAD
using namespace mgb;

TEST(TestAsyncWorker, AsyncWorkerSet) {
    using namespace std::chrono;
    using namespace std::literals;
    milliseconds sleep_time(40);
    AsyncWorkerSet worker_set;
    int val0 = 0, val1 = 0;
    worker_set.add_worker("worker0",
        [&](){
            val0 ++;
            std::this_thread::sleep_for(sleep_time);
    });
    worker_set.add_worker("worker1",
        [&](){
            val1 ++;
            std::this_thread::sleep_for(sleep_time);
    });

    mgb_assert(val0 == 0 && val1 == 0);
    auto t0 = high_resolution_clock::now();
    worker_set.start();
    worker_set.start();
    worker_set.start();
    auto dt = high_resolution_clock::now() - t0;
    ASSERT_LT(dt, sleep_time);

    worker_set.wait_all();
    dt = high_resolution_clock::now() - t0;
    ASSERT_GT(dt, sleep_time * 3);
    ASSERT_LT(dt, sleep_time * 4);

    mgb_assert(val0 == 3 && val1 == 3);
}

TEST(TestAsyncWorker, FutureThreadPool) {
    auto worker = [](int n) {
        return n * n;
    };
    FutureThreadPool<int> pool;
    pool.start(3);
    std::vector<std::future<int>> futures;
    for (int i = 0; i < 1000; ++ i)
        futures.emplace_back(pool.launch(worker, i));

    for (auto &&i: futures)
        i.wait();

    for (int i = 0; i < 1000; ++ i)
        ASSERT_EQ(i * i, futures[i].get());

    auto sleep = []() {
        using namespace std::literals;
        std::this_thread::sleep_for(0.1s);
        return 0;
    };
    futures.clear();
    RealTimer timer;
    for (int i = 0; i < 6; ++ i)
        futures.push_back(pool.launch(sleep));
    for (auto &&i: futures)
        i.get();
    auto time = timer.get_secs();
    ASSERT_GT(time, 0.19);
    ASSERT_LT(time, 0.25);
}

#if MGB_ENABLE_EXCEPTION
TEST(TestAsyncWorker, AsyncWorkerSetException) {

    RealTimer timer;
    {
        AsyncWorkerSet worker_set;
        std::atomic_bool worker1_started{false};
        worker_set.add_worker("worker0",
                [&](){
                while(!worker1_started.load());
                throw std::runtime_error("exception test");
                });
        worker_set.add_worker("worker1",
                [&](){
                worker1_started.store(true);
                using namespace std::literals;
                std::this_thread::sleep_for(100ms);
                });

        timer.reset();
        worker_set.start();
        ASSERT_THROW(worker_set.wait_all(), std::runtime_error);
        ASSERT_LT(timer.get_msecs(), 100);
    }
    ASSERT_GT(timer.get_msecs(), 100);
}

TEST(TestAsyncWorker, FutureThreadPoolException) {
    auto worker = [](int n) {
        if (!n)
            throw std::runtime_error("x");
        return n * n;
    };
    FutureThreadPool<int> pool;
    pool.start(3);
    std::vector<std::future<int>> futures;
    for (int i = 0; i < 100; ++ i)
        futures.emplace_back(pool.launch(worker, i));

    for (auto &&i: futures)
        i.wait();

    ASSERT_THROW(futures[0].get(), std::runtime_error);
    for (int i = 1; i < 100; ++ i)
        ASSERT_EQ(i * i, futures[i].get());

    auto sleep = []() {
        using namespace std::literals;
        std::this_thread::sleep_for(0.1s);
        return 0;
    };
    futures.clear();
    RealTimer timer;
    for (int i = 0; i < 6; ++ i)
        futures.push_back(pool.launch(sleep));
    for (auto &&i: futures)
        i.get();
    auto time = timer.get_secs();
    ASSERT_GT(time, 0.19);
    ASSERT_LT(time, 0.21);
}
#endif

#else
#pragma message "tests are disabled as threads is not enabled."
#endif  // MGB_HAVE_THREAD

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
