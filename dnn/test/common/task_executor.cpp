/**
 * \file dnn/test/common/task_executor.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "test/common/utils.h"

#if MEGDNN_ENABLE_MULTI_THREADS
#include <thread>
#include <mutex>
#include <atomic>
#include <vector>
#include <condition_variable>

#if defined(WIN32)
#include <windows.h>
#else
#ifdef __APPLE__
#include <mach/mach.h>
#include <mach/mach_host.h>
#else
#include <pthread.h>
#include <sched.h>
#include <stdint.h>
#include <sys/syscall.h>
#include <sys/sysinfo.h>
#include <unistd.h>
#endif
#endif
#endif

using namespace megdnn;
using namespace test;

namespace {

#if MEGDNN_ENABLE_MULTI_THREADS

#define SET_AFFINITY_CHECK(cond)                                 \
    do {                                                         \
        if (cond) {                                              \
            megdnn_log_warn("syscall for set affinity error\n"); \
        }                                                        \
    } while (0);

#if defined(WIN32)
DWORD do_set_cpu_affinity(const DWORD& mask) {
    auto succ = SetThreadAffinityMask(GetCurrentThread(), mask);
    return succ;
}

DWORD set_cpu_affinity(const std::vector<size_t>& cpuset) {
    auto nr = get_cpu_count();
    DWORD mask = 0;
    for (auto i : cpuset) {
        megdnn_assert(i < 64 && i < nr);
        mask |= 1 << i;
    }
    return do_set_cpu_affinity(mask);
}

#else  // not WIN32

#if defined(__APPLE__)
#pragma message("set_cpu_affinity is not enabled on apple platform")
int do_set_cpu_affinity(const int mask) {
    MEGDNN_MARK_USED_VAR(mask);
    return -1;
}
int set_cpu_affinity(const std::vector<size_t>& cpuset) {
    MEGDNN_MARK_USED_VAR(cpuset);
    return -1;
}

#else  // not __APPLE__

cpu_set_t do_set_cpu_affinity(const cpu_set_t& mask) {
    cpu_set_t prev_mask;
#if defined(ANDROID) || defined(__ANDROID__)
    SET_AFFINITY_CHECK(
            sched_getaffinity(gettid(), sizeof(prev_mask), &prev_mask));
    SET_AFFINITY_CHECK(sched_setaffinity(gettid(), sizeof(mask), &mask));
#else
    SET_AFFINITY_CHECK(sched_getaffinity(syscall(__NR_gettid),
                                         sizeof(prev_mask), &prev_mask));
    SET_AFFINITY_CHECK(
            sched_setaffinity(syscall(__NR_gettid), sizeof(mask), &mask));
#endif  // defined(ANDROID) || defined(__ANDROID__)
    return prev_mask;
}

cpu_set_t set_cpu_affinity(const std::vector<size_t>& cpuset) {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    auto nr = get_cpu_count();
    for (auto i : cpuset) {
        megdnn_assert(i < nr, "invalid CPU ID: nr_cpu=%zu id=%zu", nr, i);
        CPU_SET(i, &mask);
    }
    return do_set_cpu_affinity(mask);
}

#endif  // __APPLE__
#endif  // WIN32
#endif  // MEGDNN_ENABLE_MULTI_THREADS

}  // anonymous namespace

CpuDispatchChecker::TaskExecutor::TaskExecutor(TaskExecutorConfig* config) {
    if (config != nullptr) {
#if MEGDNN_ENABLE_MULTI_THREADS
        m_main_thread_affinity = false;
        m_stop = false;
        auto worker_threads_main_loop = [this](size_t i) {
            if (m_cpu_ids.size() > i)
                MEGDNN_MARK_USED_VAR(set_cpu_affinity({m_cpu_ids[i]}));
            while (!m_stop) {
                int index = -1;
                if (m_workers_flag[i]->load(std::memory_order_acquire)) {
                    while ((index = m_current_task_iter.fetch_sub(
                                    1, std::memory_order_acq_rel)) &&
                           index > 0) {
                        m_task(static_cast<size_t>(m_all_task_iter - index), i);
                    }
                    //! Flag worker is finished
                    m_workers_flag[i]->store(false, std::memory_order_release);
                }
                std::this_thread::yield();
            }
        };
        m_nr_threads = config->nr_thread;
        m_cpu_ids.insert(m_cpu_ids.end(), config->affinity_core_set.begin(),
                         config->affinity_core_set.end());
        if (m_cpu_ids.empty()) {
            megdnn_log_warn("Thread affinity was not set.");
        } else {
            megdnn_assert(m_cpu_ids.size() <= get_cpu_count(),
                          "The input affinity_core_set size exceed the "
                          "number of CPU cores, got: %zu cpu_count: %zu.",
                          m_cpu_ids.size(), get_cpu_count());
        }
        for (size_t i = 0; i < m_nr_threads - 1; i++) {
            m_workers_flag.emplace_back(new std::atomic_bool{false});
            m_workers.emplace_back(std::bind(worker_threads_main_loop, i));
        }
#else
        megdnn_throw(
                "Try to use multithreading with "
                "\'MEGDNN_ENABLE_MULTI_THREADS\' set to 0.");
#endif
    } else {
        m_nr_threads = 1;
    }
}

void CpuDispatchChecker::TaskExecutor::add_task(const MultiThreadingTask& task, size_t parallelism) {
#if MEGDNN_ENABLE_MULTI_THREADS
    if (!m_main_thread_affinity && m_cpu_ids.size() == m_nr_threads) {
        m_main_thread_prev_affinity_mask =
                set_cpu_affinity({m_cpu_ids[m_nr_threads - 1]});
        m_main_thread_affinity = true;
    }
#endif
    if (m_nr_threads == 1 || parallelism == 1) {
        for (size_t i = 0; i < parallelism; i++) {
            task(i, 0);
        }
    } else {
#if MEGDNN_ENABLE_MULTI_THREADS
        m_all_task_iter = parallelism;
        m_current_task_iter.exchange(parallelism, std::memory_order_acq_rel);
        m_task = task;

        //! Set flag to start thread working
        for (uint32_t i = 0; i < m_nr_threads - 1; i++) {
            *m_workers_flag[i] = true;
        }
        int index = -1;
        while ((index = m_current_task_iter.fetch_sub(
                        1, std::memory_order_acq_rel)) &&
               index > 0) {
            m_task(static_cast<size_t>(m_all_task_iter - index),
                   m_nr_threads - 1);
        }
        sync();
#else
        megdnn_throw(
                "Try to use multithreading with "
                "\'MEGDNN_ENABLE_MULTI_THREADS\' set to 0.");
#endif
    }
}

void CpuDispatchChecker::TaskExecutor::add_task(const Task& task) {
    task();
}

void CpuDispatchChecker::TaskExecutor::sync() {
#if MEGDNN_ENABLE_MULTI_THREADS
    bool no_finished = false;
    do {
        no_finished = false;
        for (uint32_t i = 0; i < m_nr_threads - 1; ++i) {
            if (*m_workers_flag[i]) {
                no_finished = true;
                break;
            }
        }
        if (no_finished) {
            std::this_thread::yield();
        }
    } while (no_finished);
#endif
}

CpuDispatchChecker::TaskExecutor::~TaskExecutor() {
#if MEGDNN_ENABLE_MULTI_THREADS
    m_stop = true;
    for (auto& worker : m_workers) {
        worker.join();
    }
    for (auto flag : m_workers_flag) {
        delete flag;
    }
    if (m_main_thread_affinity) {
        //! Restore the main thread affinity.
        MEGDNN_MARK_USED_VAR(
                do_set_cpu_affinity(m_main_thread_prev_affinity_mask));
    }
#endif
}

// vim: syntax=cpp.doxygen
