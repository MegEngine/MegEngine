/**
 * \file src/core/impl/utils/comp_node_sync_manager.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/utils/comp_node_sync_manager.h"
#include "megbrain/utils/thread.h"

using namespace mgb;

CompNodeSyncManager& CompNodeSyncManager::comp_node(CompNode cn) {
    mgb_assert(cn.valid());
    if (m_comp_node != cn) {
        clear_waiter_record();
        m_comp_node = cn;
    }
    return *this;
}

CompNode::Event& CompNodeSyncManager::busy_wait_set_ready_and_get_event() {
    busy_wait_set_ready();
    mgb_assert(m_ready_event);
    return *m_ready_event;
}

CompNodeSyncManager& CompNodeSyncManager::add_waiter_record(
        bool need_ready_event, size_t nr_waiter) {
    mgb_assert(!m_have_been_waited && m_comp_node.valid() && nr_waiter);
    if (need_ready_event && !m_ready_event) {
        m_ready_event = m_comp_node.create_event();
    }
    m_nr_waiter += nr_waiter;
    return *this;
}

#if MGB_HAVE_THREAD

CompNodeSyncManager& CompNodeSyncManager::clear_waiter_record() {
    mgb_assert(!m_nr_ready.load(), "there are unused ready events");
    m_have_been_waited = false;
    m_ready_event.reset();
    m_nr_waiter = 0;
    return *this;
}

void CompNodeSyncManager::do_set_ready() {
    // nr_waiter must be nonzero (checked outside)
    mgb_assert(m_comp_node.valid());
    m_have_been_waited = true;
    auto nr_ready = m_nr_ready.load();
    mgb_assert(!nr_ready,
               "new ready event while"
               " previous ones have not been fetched (%zu prev)",
               nr_ready);
    if (m_ready_event)
        m_ready_event->record();

    {
        MGB_LOCK_GUARD(m_mtx);
        m_nr_ready.store(m_nr_waiter);
        m_cv.notify_all();
    }
}

CompNodeSyncManager& CompNodeSyncManager::busy_wait_set_ready() {
    mgb_assert(m_nr_waiter,
               "before actually waiting on a tensor,"
               " you must call set_has_waiter first");

    size_t spin = 0, max_spin = SCQueueSynchronizer::get_default_max_spin();
    while (!m_nr_ready.load()) {
        ++spin;
        if (spin >= max_spin) {
            std::unique_lock<std::mutex> lock(m_mtx);
            if (m_nr_ready.load())
                break;
            m_cv.wait(lock);
        }
    }

    auto v = m_nr_ready.fetch_sub(1);
    mgb_assert(v, "more waiters than add_waiter_record calls");
    return *this;
}

#else  // MGB_HAVE_THREAD

CompNodeSyncManager& CompNodeSyncManager::clear_waiter_record() {
    m_have_been_waited = false;
    m_ready_event.reset();
    m_nr_waiter = 0;
    return *this;
}

void CompNodeSyncManager::do_set_ready() {
    m_have_been_waited = true;
    if (m_ready_event)
        m_ready_event->record();
}

CompNodeSyncManager& CompNodeSyncManager::busy_wait_set_ready() {
    // We can't wait, we can only ensure that set_ready has already been called.
    mgb_assert(m_have_been_waited);
    return *this;
}

#endif  // MGB_HAVE_THREAD

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

