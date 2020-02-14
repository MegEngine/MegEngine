/**
 * \file src/core/include/megbrain/utils/thread_impl_spinlock.h
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
#include <thread>
#include <atomic>

namespace mgb {

//! lightweight spinlock
class Spinlock final: public NonCopyableObj {
    std::atomic_flag m_state = ATOMIC_FLAG_INIT;

    public:

        void lock() {
            while (m_state.test_and_set(std::memory_order_acquire));
        }

        void unlock() {
            m_state.clear(std::memory_order_release);
        }
};

//! recursive spinlock
class RecursiveSpinlock final: public NonCopyableObj {
    static const std::thread::id sm_none_owner;
    std::atomic<std::thread::id> m_owner{sm_none_owner};
    size_t m_recur_count = 0;

    public:

        void lock();
        void unlock();
};

}  // namespace mgb
// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

