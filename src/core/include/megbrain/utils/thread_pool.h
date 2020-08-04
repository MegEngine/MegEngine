/**
 * \file src/core/include/megbrain/utils/thread_pool.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once
#include "megbrain/common.h"
#include "megbrain/system.h"
#include "megbrain/comp_node.h"

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace mgb {

using MultiThreadingTask = thin_function<void(size_t, size_t)>;
using AffinityCallBack = thin_function<void(size_t)>;
/**
 * \brief task element
 */
struct TaskElem {
    //! the task to be execute
    MultiThreadingTask task;
    //! number of the parallelism
    size_t nr_parallelism;
};

/**
 * \brief Worker and related flag
 */
struct Worker {
public:
    Worker(thin_function<void()>&& run) : thread{run} {}
    ~Worker() {
        thread.join();
    }
    //! Worker thread
    std::thread thread;
    //! Indicate whether the Worker thread need run
    std::atomic_bool work_flag{false};
    //! Indicate whether the Worker thread have binding core
    bool affinity_flag{false};
};

#if MGB_HAVE_THREAD
/**
 * \brief ThreadPool execute the task in multi-threads(nr_threads>1) mode , it
 * will fallback to single-thread mode if nr_thread is 1.
 */
class ThreadPool : public NonCopyableObj {
public:
    //! Create thread-pool nr_threads thread_pool
    ThreadPool(size_t nr_threads);
    //! The main thread set the task, parallelism and worker flag to
    //! notify other thread.
    void add_task(const TaskElem& task_elem);

    size_t nr_threads() const;

    //! Set the affinity of all the threads
    void set_affinity(AffinityCallBack affinity_cb);

    void sync();
    //! wake up all the threads from cv.wait(), when the thread pool is not
    //! active, all the threads will go to sleep.
    void active();
    //! all the threads go to sleep which will reduce CPU occupation
    void deactive();
    ~ThreadPool();

private:
    const size_t m_nr_threads = 0;
    //! Indicate whether the main thread have binding
    bool m_main_affinity_flag;
    //! The callback binding the threads to cores
    AffinityCallBack m_core_binding_function{nullptr};
    //! All the sub task number
    size_t m_nr_parallelism = 0;
    std::atomic_bool m_stop{false};
    std::atomic_bool m_active{false};
    //! The executable funcition pointer
    MultiThreadingTask m_task;

    std::vector<Worker*> m_workers;
    //! The task iter, when finished one, the m_all_task_iter sub 1
    std::atomic_int m_task_iter{0};
    //! The cv and mutex for threading activity
    std::condition_variable m_cv;
    std::mutex m_mutex;
    std::mutex m_mutex_task;
};
#else
/**
 * \brief ThreadPool execute the task in single thread mode
 */
class ThreadPool : public NonCopyableObj {
public:
    ThreadPool(size_t) {}
    void add_task(const TaskElem& task_elem);
    void set_affinity(AffinityCallBack affinity_cb);
    void active() {}
    void deactive() {}
    void sync() {}
    ~ThreadPool() {}
    size_t nr_threads() const { return 1_z; }
};

#endif
}  // namespace mgb
   // vim: syntax=cpp.doxygen
