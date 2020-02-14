/**
 * \file src/core/include/megbrain/utils/comp_node_sync_manager.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/comp_node.h"
#include "megbrain/utils/metahelper.h"

#if MGB_HAVE_THREAD
#include <atomic>
#include <condition_variable>
#endif

namespace mgb {
namespace cg {
class EagerEvalManager;
}

/*!
 * \brief synchronization between multile comp nodes / CPU threads
 */
class CompNodeSyncManager : public NonCopyableObj {
    friend class cg::EagerEvalManager;
    CompNode m_comp_node;
    std::unique_ptr<CompNode::Event> m_ready_event;
    bool m_have_been_waited = false;
    size_t m_nr_waiter = 0;

#if MGB_HAVE_THREAD
    //! number of "ready" events; consumed by each call to
    //! busy_wait_set_ready
    std::atomic_size_t m_nr_ready{0};

    std::mutex m_mtx;
    std::condition_variable m_cv;
#endif  // MGB_HAVE_THREAD

    void do_set_ready();

public:
    CompNodeSyncManager() = default;
    CompNodeSyncManager(CompNode cn) { comp_node(cn); }

    /*!
     * \brief reset comp node
     *
     * If new comp node is different from the old one, clear_waiter_record()
     * would be called.
     */
    CompNodeSyncManager& comp_node(CompNode cn);

    /*!
     * \brief add a waiter record, so busy_wait_set_ready could be
     *      called
     * \param need_ready_event if true, get_ready_event() could be
     *      called
     * \param nr_waiter number of waiter records to be added; it is the
     *      number of corresponding busy_wait_set_ready() calls
     */
    CompNodeSyncManager& add_waiter_record(bool need_ready_event,
                                           size_t nr_waiter = 1);

    /*!
     * \brief clear waiter status
     */
    CompNodeSyncManager& clear_waiter_record();

    /*!
     * \brief called when host computing finished and device command
     *      issued
     */
    CompNodeSyncManager& set_ready() {
        if (m_nr_waiter)
            do_set_ready();
        return *this;
    }

    /*!
     * \brief block the host thread until another thread calls set_ready
     *
     * Note that ready count would be decreased; calls to this method
     * must match calls to add_waiter_record()
     */
    CompNodeSyncManager& busy_wait_set_ready();

    /*!
     * \brief call busy_wait_set_ready() and then get ready event for
     *      deviec sync
     *
     * There must be one call to add_waiter_record() with
     * need_ready_event == true
     */
    CompNode::Event& busy_wait_set_ready_and_get_event();
};

}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

