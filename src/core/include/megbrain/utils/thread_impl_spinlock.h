#pragma once

#include <atomic>
#include <thread>
#include "megbrain/common.h"
#include "megbrain/utils/metahelper.h"

namespace mgb {

//! lightweight spinlock
class Spinlock final : public NonCopyableObj {
    std::atomic_flag m_state = ATOMIC_FLAG_INIT;

public:
    void lock() {
        while (m_state.test_and_set(std::memory_order_acquire)) {
        };
    }

    void unlock() { m_state.clear(std::memory_order_release); }
};

//! recursive spinlock
class RecursiveSpinlock final : public NonCopyableObj {
    static const std::thread::id sm_none_owner;
    std::atomic<std::thread::id> m_owner;
    size_t m_recur_count = 0;

public:
    RecursiveSpinlock();
    void lock();
    void unlock();
};

}  // namespace mgb
// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
