/**
 * \file src/core/impl/utils/async_worker.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/utils/async_worker.h"
#include "megbrain/common.h"

using namespace mgb;

#if MGB_HAVE_THREAD
#include "megbrain/utils/metahelper.h"
#include "megbrain/exception.h"
#include "megbrain/system.h"


void AsyncWorkerSet::issue_stop_workers() {
    MGB_LOCK_GUARD(m_mtx);
    m_should_stop = true;
    m_cv_start.notify_all();
}

AsyncWorkerSet::~AsyncWorkerSet() {
    issue_stop_workers();
    for (auto &&i: m_worker_threads)
        i.join();
}

void AsyncWorkerSet::add_worker(const std::string &name, const Task &task) {
    check_exception();

    m_worker_init_finished.store(false);
    m_worker_threads.emplace_back(&AsyncWorkerSet::worker_impl_wrapper,
            this, &name, &task);
    while(!m_worker_init_finished.load())
        std::this_thread::yield();
}

void AsyncWorkerSet::start() {
    check_exception();

    MGB_LOCK_GUARD(m_mtx);
    m_nr_start_call ++;
    m_nr_worker_to_wait = m_worker_threads.size();
    m_cv_start.notify_all();
}

void AsyncWorkerSet::wait_all() {
    for (; ; ) {
        std::unique_lock<std::mutex> lk(m_mtx);
        check_exception();
        if (!m_nr_worker_to_wait) {
            return;
        }
        m_cv_finish.wait(lk);
    }
}

void AsyncWorkerSet::worker_impl_wrapper(
        const std::string *name, const Task *taskptr) {

    std::string name_copy(*name);
    sys::set_thread_name(name_copy);
    Task task = *taskptr;

    MGB_IF_EXCEPTION(std::exception_ptr exc = nullptr);

    MGB_TRY {
        worker_impl(task);
    } MGB_CATCH_ALL_EXCEPTION(
            ssprintf("async worker `%s'", name_copy.c_str()).c_str(),
            exc);

#if MGB_ENABLE_EXCEPTION
    if (exc) {
        issue_stop_workers();
        MGB_LOCK_GUARD(m_mtx);
        m_nr_worker_to_wait -= 1;
        if (!m_prev_exception) {
            m_prev_exception = std::move(exc);
        }
        // notify with lock for predictable scheduling behavior; there should be
        // no runtime cost due to wait morphing
        m_cv_finish.notify_one();
    }
#endif
}

void AsyncWorkerSet::worker_impl(const Task &task) {
    size_t cur_finished_call = m_nr_start_call;
    m_worker_init_finished.store(true);

    for (; ; ) {
        std::unique_lock<std::mutex> lk(m_mtx);
        if (m_should_stop)
            return;
        size_t dst_nr_call = m_nr_start_call;

        if (cur_finished_call < dst_nr_call) {
            lk.unlock();

            while (cur_finished_call < dst_nr_call) {
                if (m_should_stop)
                    return;
                task();
                cur_finished_call ++;
            }

            lk.lock();

            // check stop flag whenever lock is acquired
            if (m_should_stop)
                return;

            if (cur_finished_call == m_nr_start_call) {
                mgb_assert(m_nr_worker_to_wait);
                m_nr_worker_to_wait --;
                m_cv_finish.notify_one();
                lk.unlock();
            }
        } else {
            m_cv_start.wait(lk);
        }
    }
}

void AsyncWorkerSet::check_exception() {
#if MGB_ENABLE_EXCEPTION
    if (m_prev_exception) {
        std::rethrow_exception(m_prev_exception);
    }
#endif
}

#else   // MGB_HAVE_THREAD

void AsyncWorkerSet::add_worker(const std::string &name, const Task &task) {
    mgb_assert(!m_task, "only one worker is allowed in single-thread mode");
    m_task = task;
}

void AsyncWorkerSet::start() {
    mgb_assert(m_task);
    m_task();
}

void AsyncWorkerSet::wait_all() {
}

#endif  // MGB_HAVE_THREAD

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

