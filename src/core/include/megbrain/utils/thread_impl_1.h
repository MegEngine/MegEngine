/**
 * \file src/core/include/megbrain/utils/thread_impl_1.h
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
#include "megbrain/utils/metahelper.h"
#include "./thread_impl_spinlock.h"

#include <atomic>
#include <deque>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <limits>

namespace mgb {

    /*!
     * \brief a thread-safe counter that can be modified and waited to become
     *      zero
     */
    class SyncableCounter final: public NonCopyableObj {
        int m_val = 0;
        std::mutex m_mtx;
        std::condition_variable m_cv;

        public:
            SyncableCounter();

            void incr(int delta);

            //! wait for the counter to become zero
            void wait_zero();
    };


    /*!
     * \brief synchronization for single-consumer queue
     *
     * Note: there are internal size_t counters; on 32-bit platforms they may
     * wrap around within a practical time, which would crash the system.
     */
    class SCQueueSynchronizer {
        static size_t cached_max_spin;
        std::atomic_flag m_consumer_waiting = ATOMIC_FLAG_INIT;
        std::atomic_bool m_should_exit{false};
        bool m_worker_started = false, m_wait_finish_called = false;
        std::atomic_size_t m_finished_task{0}, m_tot_task{0};

        //! target m_finished_task values needed by callers of producer_wait()
        std::atomic_size_t m_waiter_target{std::numeric_limits<size_t>::max()};
        std::deque<size_t> m_waiter_target_queue;

        std::mutex m_mtx_more_task, m_mtx_finished;
        std::condition_variable m_cv_more_task, m_cv_finished;
        std::thread m_worker_thread;

        public:
            SCQueueSynchronizer();
            ~SCQueueSynchronizer() noexcept;

            bool worker_started() const {
                return m_worker_started;
            }

            static size_t max_spin();

            void start_worker(std::thread thread);

            //! add a new task in producer thread; require worker to have
            //! started
            void producer_add();

            //! wait for currently added tasks to finish
            void producer_wait();

            bool check_finished() const {
                return m_finished_task.load(std::memory_order_acquire) ==
                    m_tot_task.load(std::memory_order_acquire);
            }

            /*!
             * \brief blocking fetch tasks in consumer thread
             * \param max maximal number of tasks to be fetched
             * \param min minimal number of tasks to be fetched
             * \return number of tasks fetched; return 0 if worker should exit
             */
            size_t consumer_fetch(size_t max, size_t min = 1);

            /*!
             * \brief ack that tasks have been processed in consumer
             * \param nr numnber of tasks to be committed
             */
            void consumer_commit(size_t nr);
    };

    /*!
     * \brief multi producer, single consumer asynchronous queue, where SC means
     *      single consumer
     *
     * This queue allows task to be skipped and processed later; skipped tasks
     * would be appended to the end of the queue.
     *
     * The worker would be started when first task is added.
     *
     * Note: there are internal size_t counters; on 32-bit platforms they may
     * wrap around within a practical time, which would crash the system.
     *
     * \tparam Param single param for a task
     * \tparam TaskImpl a subclass that provides the following public method:
     *
     *      void process_one_task(Param &);
     *
     *      Note that add_task() can be called within this callback
     */
    template<typename Param, class TaskImpl>
    class AsyncQueueSC: public NonCopyableObj {
        class SyncedParam {
            typename
                std::aligned_storage<sizeof(Param), alignof(Param)>::type
                m_storage;

            public:
                std::atomic_bool init_done{false};

                Param* get() {
                    return aliased_ptr<Param>(&m_storage);
                }

                void fini() {
                    init_done.store(false, std::memory_order_relaxed);
                    get()->~Param();
                }
        };

        public:
            void add_task(const Param &param) {
                SyncedParam* p = allocate_task();
                new (p->get()) Param(param);
                p->init_done.store(true, std::memory_order_release);
                m_synchronizer.producer_add();
            }

            void add_task(Param &&param) {
                SyncedParam* p = allocate_task();
                new (p->get()) Param(std::move(param));
                p->init_done.store(true, std::memory_order_release);
                m_synchronizer.producer_add();
            }

            /*!
             * \brief wait for the worker to process all already issued tasks
             *
             * Note: new tasks issued during this call would not be waited
             */
            void wait_all_task_finish() {
                auto tgt = m_queue_tail_tid.load(std::memory_order_acquire);
                do {
                    // we need a loop because other threads might be adding new
                    // tasks, and m_queue_tail_tid is increased before
                    // producer_add()
                    m_synchronizer.producer_wait();
                } while (m_finished_task.load(std::memory_order_acquire) < tgt);
                check_exception();
                on_sync_all_task_finish();
            }

            /*!
             * \brief wait until the task queue becomes empty
             *
             * Note: new tasks can be also added from the worker. This method is
             * mostly useful in the case where only the worker but no other
             * threads might add new tasks
             */
            void wait_task_queue_empty() {
                size_t tgt, done;
                do {
                    m_synchronizer.producer_wait();
                    // producer_wait() only waits for tasks that are added upon
                    // entrance of the function, and new tasks might be added
                    // during waiting, so we have to loop
                    done = m_finished_task.load(std::memory_order_relaxed);
                    std::atomic_thread_fence(std::memory_order_seq_cst);
                    tgt = m_queue_tail_tid.load(std::memory_order_relaxed);
                } while (tgt != done);
                // wait again to ensure m_wait_finish_called is set
                m_synchronizer.producer_wait();
            }

            /*!
             * \brief check for exception in worker thread and rethrow it to the
             *      caller thread
             */
            void check_exception() {
#if MGB_ENABLE_EXCEPTION
                if (m_worker_exc) {
                    std::exception_ptr exc;
                    std::swap(m_worker_exc, exc);
                    std::rethrow_exception(exc);
                }
#else
#endif
            }

            /*!
             * \brief check whether all tasks are finished
             */
            MGB_WARN_UNUSED_RESULT bool all_task_finished() const {
                return m_synchronizer.check_finished();
            }

        protected:
            ~AsyncQueueSC() noexcept = default;

            /*!
             * \brief callback when worker thread starts; this function is
             *      invoked from the worker thread
             */
            virtual void on_async_queue_worker_thread_start() {
            }

            /*!
             * \brief callback when worker thread end; this function is
             *      invoked from the worker thread
             */
            virtual void on_sync_all_task_finish() {}

        private:
            static constexpr size_t BLOCK_SIZE = 256;
            struct TaskBlock {
                size_t first_tid;   //! task id of first task
                TaskBlock *prev = nullptr;
                std::unique_ptr<TaskBlock> next;

                SyncedParam params[BLOCK_SIZE];
            };
            // write at queue tail and read at head
            size_t m_new_block_first_tid = 0;
            std::unique_ptr<TaskBlock> m_queue_head;
            TaskBlock *m_queue_tail = nullptr;
            std::atomic_size_t m_queue_tail_tid{0},    //!< id of next task
                m_finished_task{0};
            std::vector<std::unique_ptr<TaskBlock>> m_free_task_block;
            Spinlock m_mutex;
            SyncedParam *m_cur_task = nullptr;
            SCQueueSynchronizer m_synchronizer;
#if MGB_ENABLE_EXCEPTION
            std::exception_ptr m_worker_exc;    //!< exception caught in worker
#endif

            MGB_NOINLINE
            SyncedParam* allocate_task() {
                TaskBlock *tail = m_queue_tail;
                const size_t tid = m_queue_tail_tid.fetch_add(
                        1, std::memory_order_relaxed);
                int offset;
                if (!tail ||
                        (offset = static_cast<ptrdiff_t>(tid) -
                         static_cast<ptrdiff_t>(tail->first_tid)) < 0 ||
                        offset >= static_cast<int>(BLOCK_SIZE)) {

                    MGB_LOCK_GUARD(m_mutex);
                    // reload newest tail
                    tail = m_queue_tail;
                    if (!m_synchronizer.worker_started()) {
                        m_synchronizer.start_worker(std::thread{
                                &AsyncQueueSC::worker_thread_impl, this});
                    }
                    if (!tail) {
                        m_queue_head = allocate_task_block_unsafe(nullptr);
                        tail = m_queue_tail = m_queue_head.get();
                    } else if (tid >= tail->first_tid + BLOCK_SIZE) {
                        for (; ; ) {
                            tail->next = allocate_task_block_unsafe(tail);
                            tail = tail->next.get();
                            m_queue_tail = tail;
                            if (tid < tail->first_tid + BLOCK_SIZE) {
                                break;
                            }
                        }
                    } else {
                        while (tid < tail->first_tid) {
                            tail = tail->prev;
                        }
                    }
                    offset = tid - tail->first_tid;
                }
                return &tail->params[offset];
            }

            //! allocate TaskBlock with m_mutex held
            MGB_NOINLINE
            std::unique_ptr<TaskBlock> allocate_task_block_unsafe(
                    TaskBlock *prev) {
                std::unique_ptr<TaskBlock> ret;
                if (!m_free_task_block.empty()) {
                    ret = std::move(m_free_task_block.back());
                    m_free_task_block.pop_back();
                } else {
                    ret = std::make_unique<TaskBlock>();
                }
                ret->first_tid = m_new_block_first_tid;
                m_new_block_first_tid += BLOCK_SIZE;
                ret->prev = prev;
                return ret;
            }

            void worker_thread_impl() {
                on_async_queue_worker_thread_start();
                size_t qh = 0;

                for (; ; ) {
                    MGB_TRY {
                        worker_thread_impl_no_exc(&qh);
                        return;
                    } MGB_CATCH_ALL_EXCEPTION("AsyncQueueSC", m_worker_exc);

                    if (m_cur_task) {
                        m_cur_task->fini();
                        m_cur_task = nullptr;
                    }
                    m_synchronizer.consumer_commit(1);
                    m_finished_task.fetch_add(1, std::memory_order_release);
                }
            }

            void worker_thread_impl_no_exc(size_t * __restrict__ qh_ptr) {
                size_t &qh = *qh_ptr;
                for (; ; ) {
                    if (!m_synchronizer.consumer_fetch(1))
                        return;

                    if (qh == BLOCK_SIZE) {
                        qh = 0;
                        MGB_LOCK_GUARD(m_mutex);
                        m_free_task_block.emplace_back(std::move(m_queue_head));
                        m_queue_head = std::move(
                                m_free_task_block.back()->next);
                        if (m_queue_head) {
                            m_queue_head->prev = nullptr;
                        } else {
                            m_queue_tail = nullptr;
                        }
                    }

                    SyncedParam &cur = m_queue_head->params[qh ++];
                    while (!cur.init_done.load(std::memory_order_acquire));
                    cur.init_done.store(false, std::memory_order_relaxed);
                    m_cur_task = &cur;
                    static_cast<TaskImpl*>(this)->process_one_task(*cur.get());
                    m_cur_task = nullptr;
                    cur.fini();
                    m_synchronizer.consumer_commit(1);
                    m_finished_task.fetch_add(1, std::memory_order_release);
                }
            }
    };

    //! a thread would block until all threads reach this barrier
    class Barrier {
        bool m_need_clear = false;
        std::mutex m_mtx;
        std::condition_variable m_cv;
        size_t m_nr_reached = 0;

        public:
            void wait(size_t nr_participants) {
                std::unique_lock<std::mutex> lk{m_mtx};
                if (m_need_clear) {
                    m_need_clear = false;
                    m_nr_reached = 0;
                }
                auto nr = ++ m_nr_reached;
                mgb_assert(nr <= nr_participants);
                if (nr == nr_participants) {
                    m_need_clear = true;
                    m_cv.notify_all();
                    return;
                }
                m_cv.wait(lk, [&]() {return m_nr_reached == nr_participants;});
            }
    };
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
