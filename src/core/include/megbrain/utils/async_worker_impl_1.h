/**
 * \file src/core/include/megbrain/utils/async_worker_impl_1.h
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
#include "megbrain/utils/metahelper.h"

#include <string>
#include <thread>
#include <condition_variable>
#include "megbrain/utils/thin/function.h"
#include <atomic>
#include <exception>
#include <future>
#include <vector>
#include <deque>

namespace mgb {

/*!
 * \brief manage a set of asynchronous workers
 *
 * These workers can be started together and being waited by; when start() is
 * called once by the main thread, each worker in the background would call
 * given task once.
 *
 * Note that these workers are meant to cooperate which may, for example,
 * communicate and synchronize. So if one worker throws exception, all would be
 * requested to stop. And in such case, this AsyncWorkerSet can not be used
 * again.
 */
class AsyncWorkerSet final: public NonCopyableObj {
    public:
        using Task = thin_function<void()>;

        ~AsyncWorkerSet();

        //! whether there has been no worker added
        bool empty() const { return m_worker_threads.empty(); }

        /*!
         * \brief add a worker thread
         */
        void add_worker(const std::string &name, const Task &task);

        /*!
         * \brief start all the workers
         *
         * start() can be called multiple times without calling wait_all, and
         * for each call, task() would be invoked in the worker thread.
         */
        void start();

        /*!
         * \brief wait for all previously started workers to finish;
         *
         * Note that exceptions throw in previous start would be rethrown from
         * here.
         */
        void wait_all();

    private:
        bool volatile m_should_stop = false;
        MGB_IF_EXCEPTION(std::exception_ptr m_prev_exception = nullptr);
        size_t m_nr_start_call = 0;
        //! number of workers currently working (i.e. calling task())
        size_t volatile m_nr_worker_to_wait;
        std::atomic_bool m_worker_init_finished;
        std::mutex m_mtx;
        std::condition_variable m_cv_start, m_cv_finish;
        std::vector<std::thread> m_worker_threads;

        void check_exception();

        //! arrange all workers to exit
        void issue_stop_workers();

        /*!
         * \brief call given task repeatedly until m_should_stop
         *
         * Note: exception from task() would be propogated to
         * worker_impl_wrapper() to stop all workers
         */
        void worker_impl(const Task &task);

        //! set name, check exception, etc.
        void worker_impl_wrapper(const std::string *name, const Task *task);
};

/*!
 * \brief a thread pool with determined concurrency
 *
 * This class is intended to replace std::async by an implementation with an
 * underlying thread pool, so the number of concurrent tasks can be controlled.
 *
 * Control methods (i.e. start() and stop()) are NOT thread-safe.
 *
 * \tparam R return value of the tasks
 */
template<class R>
class FutureThreadPool final: public NonCopyableObj {
    using Task = std::packaged_task<R()>;
    std::deque<Task> m_tasks;
    std::mutex m_mtx;
    std::condition_variable m_cv_more_task;

    std::vector<std::thread> m_worker_threads;
    std::vector<std::thread::id> m_worker_tids;
    Maybe<std::string> m_name;
    bool m_should_stop = true;

    void worker_impl(size_t id) {
        {
            MGB_LOCK_GUARD(m_mtx);
            m_worker_tids.push_back(std::this_thread::get_id());
        }

        if (m_name.valid()) {
            sys::set_thread_name(
                    ssprintf("%s:%zu", m_name->c_str(), id));
        }

        for (; ; ) {
            Task task;
            for (; ; ) {
                std::unique_lock<std::mutex> lk(m_mtx);
                if (m_should_stop)
                    return;
                if (!m_tasks.empty()) {
                    task = std::move(m_tasks.front());
                    m_tasks.pop_front();
                    break;
                }

                m_cv_more_task.wait(lk);
            }
            task();
        }
    }

    public:
        using Future = std::future<R>;

        /*!
         * \param name thread name for the workers
         */
        FutureThreadPool(const Maybe<std::string> &name = None):
            m_name{name}
        {
        }

        ~FutureThreadPool() {
            stop();
        }

        /*!
         * \brief launch a task with given function and args
         */
        template<typename Func, typename ...Args>
        Future launch(Func&& func, Args&&... args) {
            auto bfunc = std::bind(
                    std::forward<Func>(func),
                    std::forward<Args>(args)...);

            MGB_LOCK_GUARD(m_mtx);
            m_tasks.emplace_back(std::move(bfunc));
            m_cv_more_task.notify_all();
            return m_tasks.back().get_future();
        }

        /*!
         * \brief start worker threads with given concurrency
         * \return thread IDs of the workers
         */
        const std::vector<std::thread::id>& start(size_t concurrency) {
            mgb_assert(concurrency > 0);
            mgb_assert(m_should_stop && m_worker_threads.empty() &&
                    m_worker_tids.empty());
            m_should_stop = false;
            m_worker_threads.reserve(concurrency);
            for (size_t i = 0; i < concurrency; ++ i) {
                m_worker_threads.emplace_back(std::bind(
                            &FutureThreadPool<R>::worker_impl, this, i));
            }

            for (; ; ) {
                {
                    MGB_LOCK_GUARD(m_mtx);
                    if (m_worker_tids.size() == concurrency)
                        return m_worker_tids;
                }
                std::this_thread::yield();
            }
        }

        /*!
         * \brief after all futures have been processed, call this method to
         *      stop the workers
         *
         * Note that this method would not wait for unfinished task.
         */
        void stop() {
            if (m_should_stop)
                return;

            {
                MGB_LOCK_GUARD(m_mtx);
                m_should_stop = true;
                m_cv_more_task.notify_all();
            }
            for (auto &&i: m_worker_threads)
                i.join();

            m_worker_threads.clear();
            m_worker_tids.clear();
        }
};

}


// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

