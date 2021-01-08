/**
 * \file src/core/test/utils/thread.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */


#include "megbrain/utils/thread.h"
#include "megbrain/utils/timer.h"
#include "megbrain/test/helper.h"
#include <atomic>
#include <random>

#if MGB_HAVE_THREAD
using namespace mgb;

namespace {

#if MGB_ENABLE_EXCEPTION
    class ExcMaker final: public AsyncQueueSC<int, ExcMaker> {
        public:
            void process_one_task(int) {
                throw std::runtime_error("test");
            }
    };
#endif


    class FuncExecutor final: public AsyncQueueSC<
                              thin_function<void()>,
                              FuncExecutor> {
        public:
            void process_one_task(const thin_function<void()> &task) {
                task();
            }
    };

    template<int producer_sleep, int consumer_sleep>
    void test_scq_sync_multi_producer() {
        size_t nr_worker_call = 0;
        SCQueueSynchronizer sync(0);
        auto worker = [&]() {
            RNGxorshf rng{next_rand_seed()};
            while (auto nr = sync.consumer_fetch(1)) {
                nr_worker_call += nr;
                ASSERT_EQ(1u, nr);
                if (consumer_sleep) {
                    std::this_thread::sleep_for(std::chrono::microseconds(
                                rng() % consumer_sleep));
                }
                sync.consumer_commit(nr);
            }
        };
        sync.start_worker(std::thread{worker});

        constexpr size_t N = 500, M = 8;
        std::atomic_size_t nr_worker_started{0};
        auto producer_impl = [&]() {
            RNGxorshf rng{next_rand_seed()};
            ++ nr_worker_started;
            while (nr_worker_started.load() != M);
            for (size_t i = 0; i < N; ++ i) {
                if (producer_sleep) {
                    std::this_thread::sleep_for(std::chrono::microseconds(
                                rng() % producer_sleep));
                }
                sync.producer_add();
                if (i % 4 == 0)
                    sync.producer_wait();
            }
        };
        std::vector<std::thread> producer_threads;
        for (size_t i = 0; i < M; ++ i) {
            producer_threads.emplace_back(producer_impl);
        }
        for (auto &&i: producer_threads)
            i.join();
        sync.producer_wait();
        ASSERT_EQ(N * M, nr_worker_call);
    }
}

TEST(TestAsyncQueue, Synchronizer) {
    size_t nr_worker_call = 0;
    SCQueueSynchronizer sync(0);
    auto worker = [&]() {
        for (; ;) {
            auto nr = sync.consumer_fetch(1);
            if (!nr)
                return;
            nr_worker_call += nr;
            ASSERT_EQ(1u, nr);
            sync.consumer_commit(nr);
        }
    };
    sync.start_worker(std::thread{worker});

    constexpr size_t N = 3000000;
    RealTimer timer;
    for (size_t i = 0; i < N; ++ i) {
        sync.producer_add();
    }
    auto tadd = timer.get_secs_reset() * 1e9 / N;
    sync.producer_wait();
    auto twait = timer.get_secs_reset() * 1e9 / N;
    ASSERT_EQ(N, nr_worker_call);
    printf("tadd=%.3f twait=%.3f [ns]\n", tadd, twait);
}

TEST(TestAsyncQueue, SynchronizerWaitOverhead) {
    {
        size_t nr_worker_call = 0;
        SCQueueSynchronizer sync(0);
        auto worker = [&]() {
            for (;;) {
                auto nr = sync.consumer_fetch(1);
                if (!nr)
                    return;
                nr_worker_call += nr;
                ASSERT_EQ(1u, nr);
                sync.consumer_commit(nr);
            }
        };
        sync.start_worker(std::thread{worker});

        constexpr size_t N = 300000;
        RealTimer timer;
        for (size_t i = 0; i < N; ++i) {
            sync.producer_add();
            sync.producer_wait();
        }
        ASSERT_EQ(N, nr_worker_call);
        printf("avg_twait=%.3f [us]\n", timer.get_msecs() * 1e3 / N);
    }
    {
        double worker_time = 0, avg_await;
        {
            size_t nr_worker_call = 0;
            SCQueueSynchronizer sync(0);
            auto worker = [&]() {
                for (;;) {
                    auto nr = sync.consumer_fetch(1);
                    if (!nr)
                        return;
                    RealTimer timer;
                    nr_worker_call += nr;
                    ASSERT_EQ(1u, nr);
                    using namespace std::chrono_literals;
                    std::this_thread::sleep_for(100ms);
                    sync.consumer_commit(nr);
                    worker_time += timer.get_msecs();
                }
            };
            sync.start_worker(std::thread{worker});

            constexpr size_t N = 5;
            RealTimer timer;
            for (size_t i = 0; i < N; ++i) {
                sync.producer_add();
                sync.producer_wait();
            }
            ASSERT_EQ(N, nr_worker_call);
            avg_await = (timer.get_msecs() - worker_time) * 1e3 / N;
        }
        printf("with workload: avg_twait=%.3f [us]\n", avg_await);
    }
}

TEST(TestAsyncQueue, SynchronizerMultiProducer0) {
    test_scq_sync_multi_producer<0, 100>();
}

TEST(TestAsyncQueue, SynchronizerMultiProducer1) {
    test_scq_sync_multi_producer<100, 0>();
}

TEST(TestAsyncQueue, SynchronizerMultiProducer2) {
    test_scq_sync_multi_producer<0, 0>();
}

TEST(TestAsyncQueue, SynchronizerMultiProducer3) {
    test_scq_sync_multi_producer<100, 100>();
}

TEST(TestAsyncQueue, SynchronizerWaiterStarving) {
    SCQueueSynchronizer sync(0);
    std::atomic_size_t processed{0};
    auto worker = [&]() {
        while (sync.consumer_fetch(1)) {
            for (int volatile i = 0; i < 1000; ++ i);
            sync.consumer_commit(1);
            ++ processed;
        }
    };
    sync.start_worker(std::thread{worker});
    std::atomic_bool producer_run{true};
    std::atomic_size_t nr_added{0};
    auto producer = [&]() {
        while (producer_run) {
            size_t cur = ++ nr_added;
            while (cur - processed > 1000);
            sync.producer_add();
        }
    };
    std::thread th_producer{producer};

    while (nr_added.load() < 3);

    for (int i = 0; i < 10; ++ i) {
        sync.producer_wait();   // this should not block long
    }
    producer_run = false;
    th_producer.join();
    sync.producer_wait();
}

TEST(TestAsyncQueue, Correctness0) {
    class Adder final: public AsyncQueueSC<int, Adder> {
        int m_sum = 0;
        std::mt19937 m_rng;

        public:
            std::atomic_bool add_task_in_worker{true};
            std::atomic_size_t nr_task_added_in_worker{0};

            void process_one_task(int val) {
                if (add_task_in_worker && (m_rng() & 2)) {
                    ++ nr_task_added_in_worker;
                    add_task(val);
                } else {
                    m_sum += val;
                }
            }

            int sum() const {
                return m_sum;
            }
    };
    Adder adder;
    std::atomic_size_t nr_started{0};
    auto worker = [&](bool neg) {
        ++ nr_started;
        while (nr_started != 2);
        for (int i = 0; i < 10000; ++ i)
            adder.add_task(neg ? i : -i);
        adder.add_task(neg);
    };

    std::thread th0(worker, false), th1(worker, true);
    th0.join();
    th1.join();
    while (adder.nr_task_added_in_worker < 100);
    adder.add_task_in_worker = false;
    adder.wait_all_task_finish();
    ASSERT_EQ(1, adder.sum());
}

TEST(TestAsyncQueue, Correctness1) {
    class Adder final: public AsyncQueueSC<int, Adder> {
        int m_sum = 0;
        std::mt19937 m_rng;

        public:
            void process_one_task(int val) {
                if ((m_rng() & 2)) {
                    add_task(val);
                } else {
                    m_sum += val;
                }
            }

            int sum() const {
                return m_sum;
            }
    };
    Adder adder;
    std::atomic_size_t nr_started{0};
    auto worker = [&](bool neg) {
        ++ nr_started;
        while (nr_started != 2);
        for (int i = 0; i < 10000; ++ i)
            adder.add_task(neg ? i : -i);
        adder.add_task(neg);
    };

    std::thread th0(worker, false), th1(worker, true);
    th0.join();
    th1.join();
    adder.wait_task_queue_empty();
    ASSERT_EQ(1, adder.sum());
}

TEST(TestAsyncQueue, OutOfOrderCtor) {
    FuncExecutor fe;
    std::atomic_bool started{false};
    class Adder {
        int *m_sum = nullptr;
        bool m_slow_ctor = false;

        public:
            Adder(int *sum, bool slow_ctor):
                m_sum{sum},
                m_slow_ctor{slow_ctor}
            {
            }

            Adder(const Adder &src)
            {
                if (m_slow_ctor) {
                    using namespace std::literals;
                    std::this_thread::sleep_for(300us);
                }
                m_sum = src.m_sum;
            }

            void operator() (int i) {
                (*m_sum) += i;
            }
    };
    int sum = 0;
    std::atomic_size_t worker_ready{0};
    auto worker = [&sum, &worker_ready, &started, &fe](
            int n, std::mt19937::result_type seed) {
        Adder adder{&sum, !n};
        std::mt19937 rng{seed};
        ++ worker_ready;
        while (!started.load());
        for (int i = 0; i < 500; ++ i) {
            if (n) {
                using namespace std::literals;
                std::this_thread::sleep_for(300us);
            }
            fe.add_task(std::bind(adder, (n ^ (i&1)) ? i : -i));
        }
        fe.add_task(std::bind(adder, n));
    };

    std::thread
        th0(worker, 0, next_rand_seed()),
        th1(worker, 1, next_rand_seed());
    while (worker_ready.load() != 2);
    started.store(true);
    th0.join();
    th1.join();
    fe.wait_all_task_finish();
    ASSERT_EQ(1, sum);
}

#if MGB_ENABLE_EXCEPTION
TEST(TestAsyncQueue, Exception) {
    ExcMaker exc_maker;
    exc_maker.wait_all_task_finish();
    exc_maker.add_task(0);
    ASSERT_THROW(exc_maker.wait_all_task_finish(), std::runtime_error);
    exc_maker.wait_all_task_finish();
}
#endif

TEST(TestAsyncQueue, Benchmark) {
    struct Big {
        uint8_t data[16];
    };
    int nr_call = 0;
    auto func = [&](int i)  __attribute__((noinline)) {
        asm volatile ("" : : "r"(i / 12345));
        ++ nr_call;
    };
    Big big;
    for (int i = 0; i < 16; ++ i)
        big.data[i] = i;
    auto big_func = [b=big, &nr_call](int i) __attribute__((noinline)) {
        asm volatile ("" : : "r"(i / 12345), "r"(&b));
        ++ nr_call;
    };
    auto call = [](const thin_function<void()> &f) __attribute__((noinline)) {
        f();
    };
    FuncExecutor queue;
    constexpr int N = 100000;
    RealTimer timer;
    for (int i = 0; i < N; ++ i) {
        auto g = [func, i]() {
            func(i);
        };
        queue.add_task(g);
    }
    auto t0_add = timer.get_secs() * 1e9 / N;
    queue.wait_all_task_finish();
    auto t0_all = timer.get_secs_reset() * 1e9 / N;
    for (int i = 0; i < N; ++ i) {
        auto g = [func, i]() {
            func(i);
        };
        call(g);
    }
    auto t1 = timer.get_secs_reset() * 1e9 / N;
    for (int i = 0; i < N; ++ i)
        func(i);
    auto t2 = timer.get_secs_reset() * 1e9 / N;
    for (int i = 0; i < N; ++ i) {
        auto g = [big_func, i]() {
            big_func(i);
        };
        queue.add_task(g);
    }
    auto t3_add = timer.get_secs() * 1e9 / N;
    queue.wait_all_task_finish();
    auto t3_all = timer.get_secs_reset() * 1e9 / N;
    for (int i = 0; i < N; ++ i) {
        auto g = [big_func, i]() {
            big_func(i);
        };
        call(g);
    }
    auto t4 = timer.get_secs_reset() * 1e9 / N;
    // these profiling message should always be seen even if compiled without
    // logging support
    printf("time_per_iter: queue=(add=%.3f,all=%.3f) call=%.3f empty=%.3f "
            "big_queue=(add=%.3f,all=%.3f) big_call=%.3f [ns]\n",
            t0_add, t0_all, t1, t2, t3_add, t3_all, t4);
    ASSERT_EQ(N * 5, nr_call);
}

TEST(TestThread, Spinlock) {
    Spinlock lock;
    int cnt = 0;
    auto worker = [&](int tot) {
        for (int i = 0; i < tot; ++ i) {
            MGB_LOCK_GUARD(lock);
            ++ cnt;
        }
    };
    std::vector<std::thread> th;
    for (int i = 0; i < 10; ++ i) {
        th.emplace_back(worker, i + 1000);
    }
    for (auto &&i: th)
        i.join();
    ASSERT_EQ((1000 + 1009) * 5, cnt);
}

TEST(TestThread, RecursiveSpinlock) {
    RecursiveSpinlock lock;
    int cnt = 0;
    auto worker = [&](int tot) {
        for (int i = 0; i < tot; ++ i) {
            MGB_LOCK_GUARD(lock);
            {
                MGB_LOCK_GUARD(lock);
                {
                    MGB_LOCK_GUARD(lock);
                    ++ cnt;
                }
            }
        }
    };
    std::vector<std::thread> th;
    for (int i = 0; i < 10; ++ i) {
        th.emplace_back(worker, i + 1000);
    }
    for (auto &&i: th)
        i.join();
    ASSERT_EQ((1000 + 1009) * 5, cnt);
}

#else
#pragma message "tests are disabled as thread is not enabled."
#endif  //  MGB_HAVE_THREAD

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
