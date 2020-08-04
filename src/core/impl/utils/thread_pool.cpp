/**
 * \file src/core/impl/utils/thread_pool.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/utils/thread_pool.h"
#include <chrono>

using namespace mgb;

#if MGB_HAVE_THREAD
ThreadPool::ThreadPool(size_t threads_num)
        : m_nr_threads(threads_num),
          m_main_affinity_flag{false},
          m_stop{false},
          m_active{false} {
    if (m_nr_threads > 1) {
        if (m_nr_threads > static_cast<uint32_t>(sys::get_cpu_count())) {
            mgb_log_debug(
                    "The number of threads is bigger than number of "
                    "physical cpu cores, got: %zu core_number: %zu",
                    static_cast<size_t>(sys::get_cpu_count()), nr_threads());
        }
        for (uint32_t i = 0; i < m_nr_threads - 1; i++) {
            m_workers.push_back(new Worker([this, i]() {
                while (!m_stop) {
                    while (m_active) {
                        if (m_workers[i]->affinity_flag &&
                            m_core_binding_function != nullptr) {
                            m_core_binding_function(i);
                            m_workers[i]->affinity_flag = false;
                        }
                        //! if the thread should work
                        if (m_workers[i]->work_flag.load(
                                    std::memory_order_acquire)) {
                            int index = -1;
                            //! Get one task and execute
                            while ((index = m_task_iter.fetch_sub(
                                            1, std::memory_order_acq_rel)) &&
                                   index > 0) {
                                //! index is decrease, use
                                //! m_all_task_number - index to get the
                                //! increase id which will pass to task
                                m_task(static_cast<size_t>(m_nr_parallelism -
                                                           index),
                                       i);
                            }
                            //! Flag worker is finished
                            m_workers[i]->work_flag.store(
                                    false, std::memory_order_release);
                        }
                        //! Wait next task coming
                        std::this_thread::yield();
                    }
                    {
                        std::unique_lock<std::mutex> lock(m_mutex);
                        if (!m_stop && !m_active) {
                            m_cv.wait(lock,
                                      [this] { return m_stop || m_active; });
                        }
                    }
                }
            }));
        }
    }
}
void ThreadPool::add_task(const TaskElem& task_elem) {
    //! Make sure the main thread have bind
    if (m_main_affinity_flag &&
        m_core_binding_function != nullptr) {
        std::lock_guard<std::mutex> lock(m_mutex_task);
        m_core_binding_function(m_nr_threads - 1);
        m_main_affinity_flag = false;
    }
    size_t parallelism = task_elem.nr_parallelism;
    //! If only one thread or one task, execute directly
    if (task_elem.nr_parallelism == 1 || m_nr_threads == 1) {
        for (size_t i = 0; i < parallelism; i++) {
            task_elem.task(i, 0);
        }
        return;
    } else {
        std::lock_guard<std::mutex> lock(m_mutex_task);
        mgb_assert(m_task_iter.load(std::memory_order_acquire) <= 0,
                   "The init value of m_all_sub_task is not zero.");
        active();
        //! Set the task number, task iter and task
        m_nr_parallelism = parallelism;
        m_task_iter.exchange(parallelism, std::memory_order_relaxed);
        m_task = [&task_elem](size_t index, size_t thread_id) {
            task_elem.task(index, thread_id);
        };
        //! Set flag to start thread working
        for (uint32_t i = 0; i < m_nr_threads - 1; i++) {
            m_workers[i]->work_flag = true;
        }
        //! Main thread working
        int index = -1;
        while ((index = m_task_iter.fetch_sub(1, std::memory_order_acq_rel)) &&
               (index > 0)) {
            m_task(static_cast<size_t>(m_nr_parallelism - index),
                   m_nr_threads - 1);
        }
        //! make sure all threads done
        sync();
    }
}

void ThreadPool::set_affinity(AffinityCallBack affinity_cb) {
    mgb_assert(affinity_cb, "The affinity callback must not be nullptr");
    std::lock_guard<std::mutex> lock(m_mutex_task);
    m_core_binding_function = affinity_cb;
    for (size_t i = 0; i < m_nr_threads - 1; i++) {
        m_workers[i]->affinity_flag = true;
    }
    m_main_affinity_flag = true;
}

size_t ThreadPool::nr_threads() const {
    return m_nr_threads;
}

void ThreadPool::sync() {
    bool no_finished = false;
    do {
        no_finished = false;
        for (uint32_t i = 0; i < m_nr_threads - 1; ++i) {
            if (m_workers[i]->work_flag) {
                no_finished = true;
                break;
            }
        }
        if (no_finished) {
            std::this_thread::yield();
        }
    } while (no_finished);
}
void ThreadPool::active() {
    if (!m_active) {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_active = true;
        m_cv.notify_all();
    }
}
void ThreadPool::deactive() {
    std::lock_guard<std::mutex> lock_task(m_mutex_task);
    std::unique_lock<std::mutex> lock(m_mutex);
    m_active = false;
}
ThreadPool::~ThreadPool() {
    std::lock_guard<std::mutex> lock_task(m_mutex_task);
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        m_stop = true;
        m_active = false;
        m_cv.notify_all();
    }
    for (auto& worker : m_workers) {
        delete worker;
    }
}
#else
void ThreadPool::add_task(const TaskElem& task_elem) {
    for (size_t i = 0; i < task_elem.nr_parallelism; i++) {
        task_elem.task(i, 0);
    }
}
void ThreadPool::set_affinity(AffinityCallBack affinity_cb) {
    mgb_assert(affinity_cb != nullptr, "The affinity callback is nullptr");
    affinity_cb(0);
}
#endif
// vim: syntax=cpp.doxygen
