/**
 * \file src/core/include/megbrain/utils/thread_impl_0.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include <thread>
#include <atomic>
#include "megbrain/common.h"
#include "megbrain/utils/metahelper.h"

#if MGB_THREAD_SAFE
#include "./thread_impl_spinlock.h"
#else
namespace mgb{
class Spinlock final: public NonCopyableObj {
    public:
        void lock() {}
        void unlock() {}
};

class RecursiveSpinlock final: public NonCopyableObj {
    public:
        void lock() {}
        void unlock() {}
};
}
#endif

namespace mgb {
    class SyncableCounter final: public NonCopyableObj {
        public:
            void incr(int) {
            }

            void wait_zero() {
            }
    };

    class SCQueueSynchronizer {
        public:
            static size_t max_spin() {
                return 0;
            }
    };

    // tasks would be dispatched inplace
    template<typename Param, class TaskImpl>
    class AsyncQueueSC: public NonCopyableObj {
        public:
            virtual ~AsyncQueueSC() = default;

            void add_task(const Param &param) {
                static_cast<TaskImpl*>(this)->process_one_task(param);
            }

            void add_task(Param &&param) {
                static_cast<TaskImpl*>(this)->process_one_task(param);
            }

            void wait_all_task_finish() {
            }

            void wait_task_queue_empty() {
            }

            void check_exception() {
            }

            /*!
             * \brief check whether all tasks are finished
             */
            MGB_WARN_UNUSED_RESULT bool all_task_finished() const {
                return true;
            }

        protected:
            virtual void on_sync_all_task_finish() {}
            virtual void on_async_queue_worker_thread_start() {}
    };
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

