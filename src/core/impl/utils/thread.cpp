/**
 * \file src/core/impl/utils/thread.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/utils/thread.h"
#include <thread>
#include <atomic>

using namespace mgb;

#if MGB_THREAD_SAFE
const std::thread::id RecursiveSpinlock::sm_none_owner = std::thread::id();

//! why not use initializer_list for global var, detail:
//! MGE-1738
RecursiveSpinlock::RecursiveSpinlock() {
    m_owner = sm_none_owner;
}

void RecursiveSpinlock::lock() {
    auto tid = std::this_thread::get_id();
    if (m_owner.load(std::memory_order_relaxed) != tid) {
        for (; ;) {
            auto id = sm_none_owner;
            if (m_owner.compare_exchange_weak(id, tid,
                        std::memory_order_acquire,
                        std::memory_order_relaxed)) {
                break;
            }
        }
    }
    ++ m_recur_count;
}

void RecursiveSpinlock::unlock() {
    mgb_assert(m_recur_count &&
            m_owner.load(std::memory_order_relaxed) ==
            std::this_thread::get_id());
    if (! (-- m_recur_count)) {
        m_owner.store(sm_none_owner, std::memory_order_release);
    }
}
#else
#if MGB_HAVE_THREAD
#error "can not disable thread safety while enabling thread support"
#endif
#endif

#if MGB_HAVE_THREAD
#include "megbrain/utils/timer.h"
#include <ctime>

namespace {
    class SpinlockReleaser {
        std::atomic_flag &m_lock;
        public:
            SpinlockReleaser(std::atomic_flag &lock):
                m_lock{lock}
            {}

            ~SpinlockReleaser() {
                m_lock.clear(std::memory_order_release);
            }
    };
}

/* =============== SCQueueSynchronizer ===============  */
size_t SCQueueSynchronizer::cached_max_spin = 0;
#ifdef WIN32
bool SCQueueSynchronizer::is_into_atexit = false;
#endif

size_t SCQueueSynchronizer::max_spin() {
    if (cached_max_spin)
        return cached_max_spin;

    if (MGB_GETENV("MGB_WORKER_NO_SLEEP")) {
        mgb_log_warn("worker would not sleep");
        return cached_max_spin = std::numeric_limits<size_t>::max();
    }

    if (auto spin_string = MGB_GETENV("MGB_WORKER_MAX_SPIN")) {
        auto spin = std::stoi(spin_string);
        mgb_log_warn("worker would execute with spin of %d", spin);
        return cached_max_spin = spin;
    }

    std::atomic_bool start{false}, stop{false};
    size_t cnt;
    double cnt_time;
    auto worker_fn = [&]() {
        start.store(true);
        volatile size_t cntv = 0;
        RealTimer timer;
        while (!stop.load() && (cntv < (1 << 24))) {
            ++ cntv;
        }
        cnt_time = timer.get_msecs();
        cnt = cntv;
    };
    std::thread worker{worker_fn};
    while (!start.load()) {
        std::this_thread::yield();
    }
    {
        using namespace std::chrono_literals;
        std::this_thread::sleep_for(5ms);
    }
    stop.store(true);
    worker.join();
    cached_max_spin = std::max<size_t>(cnt * (5 / cnt_time), 100000);
    return cached_max_spin;
}

SCQueueSynchronizer::SCQueueSynchronizer() = default;

SCQueueSynchronizer::~SCQueueSynchronizer() noexcept {
    if (!m_worker_started)
        return;
    if (!m_wait_finish_called) {
        mgb_log_error("async queue not finished in destructor");
        mgb_trap();
    }
    {
        MGB_LOCK_GUARD(m_mtx_more_task);
        m_should_exit = true;
        m_cv_more_task.notify_all();
    }
    m_worker_thread.join();
}

void SCQueueSynchronizer::start_worker(std::thread thread) {
    mgb_assert(!m_worker_started);
    m_worker_started = true;
    m_worker_thread = std::move(thread);
}

void SCQueueSynchronizer::producer_add() {
    m_wait_finish_called = false;
    m_tot_task.fetch_add(1, std::memory_order_release);

    if (m_consumer_waiting.test_and_set(std::memory_order_acquire)) {
        // m_consumer_waiting already acquired by consumer or another producer
        MGB_LOCK_GUARD(m_mtx_more_task);
        m_cv_more_task.notify_all();
    } else {
        m_consumer_waiting.clear(std::memory_order_release);
    }
}

void SCQueueSynchronizer::producer_wait() {
    auto wait_target = m_tot_task.load(std::memory_order_relaxed);
    if (m_worker_started &&
            m_finished_task.load(std::memory_order_acquire) < wait_target) {

        std::unique_lock<std::mutex> lock(m_mtx_finished);
        // update wait_target again in this critical section
        wait_target = m_tot_task.load(std::memory_order_relaxed);
        if (m_waiter_target_queue.empty()) {
            m_waiter_target.store(wait_target, std::memory_order_relaxed);
            m_waiter_target_queue.push_back(wait_target);
        } else {
            mgb_assert(wait_target >= m_waiter_target_queue.back());
            if (wait_target > m_waiter_target_queue.back()) {
                m_waiter_target_queue.push_back(wait_target);
            }
        }

        size_t done;
        for (; ;) {
            // ensure that m_waiter_target is visible in consumer
            std::atomic_thread_fence(std::memory_order_seq_cst);

            done = m_finished_task.load(std::memory_order_relaxed);
            if (done >= wait_target)
                break;
            m_cv_finished.wait(lock);
        }

        if (!m_waiter_target_queue.empty()) {
            size_t next_target = 0;
            while (done >= (next_target = m_waiter_target_queue.front())) {
                m_waiter_target_queue.pop_front();
                if (m_waiter_target_queue.empty()) {
                    next_target = std::numeric_limits<size_t>::max();
                    break;
                }
            }
            m_waiter_target.store(next_target, std::memory_order_release);
            // this is necessary in practice, although not needed logically
            m_cv_finished.notify_all();
        }
    }
    m_wait_finish_called = true;
}

size_t SCQueueSynchronizer::consumer_fetch(size_t max, size_t min) {
    mgb_assert(max >= min && min >= 1);
    size_t spin = 0, max_spin = SCQueueSynchronizer::max_spin(),
           cur_finished = m_finished_task.load(std::memory_order_relaxed);

    // relaxed mem order suffices because acquire would be called for ret
    while (m_tot_task.load(std::memory_order_relaxed) < cur_finished + min) {
        ++ spin;
        if (spin >= max_spin) {
            while (m_consumer_waiting.test_and_set(std::memory_order_relaxed));
            SpinlockReleaser releaser(m_consumer_waiting);

            std::unique_lock<std::mutex> lock(m_mtx_more_task);
            if (m_should_exit.load(std::memory_order_relaxed))
                return 0;
            if (m_tot_task.load(std::memory_order_relaxed) >=
                    cur_finished + min)
                break;
            m_cv_more_task.wait(lock);
        }
        if (m_should_exit.load(std::memory_order_relaxed))
            return 0;
    }
    auto ret = std::min(
            m_tot_task.load(std::memory_order_acquire) - cur_finished, max);
    mgb_assert(ret >= min);
    return ret;
}

void SCQueueSynchronizer::consumer_commit(size_t nr) {
    auto done = m_finished_task.fetch_add(nr, std::memory_order_relaxed) + nr;
    // pair with the thread fence in producer_wait()
    std::atomic_thread_fence(std::memory_order_seq_cst);
    if (done >= m_waiter_target.load(std::memory_order_relaxed)) {
        MGB_LOCK_GUARD(m_mtx_finished);
        m_cv_finished.notify_all();
    }
}

/* =============== SyncableCounter ===============  */

SyncableCounter::SyncableCounter() = default;

void SyncableCounter::incr(int delta) {
    MGB_LOCK_GUARD(m_mtx);
    m_val += delta;
    if (!m_val)
        m_cv.notify_all();
}


void SyncableCounter::wait_zero() {
    std::unique_lock<std::mutex> lk{m_mtx};
    for (; ; ) {
        if (!m_val)
            return;
        m_cv.wait(lk);
    }
}

#else   // MGB_HAVE_THREAD
#pragma message "threading support is disabled"
#if MGB_CUDA
#error "cuda must be disabled if threading is not available"
#endif
#endif  // MGB_HAVE_THREAD

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

