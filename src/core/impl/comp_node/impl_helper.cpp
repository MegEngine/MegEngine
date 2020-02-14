/**
 * \file src/core/impl/comp_node/impl_helper.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./impl_helper.h"

using namespace mgb;

void CompNodeImplHelper::EventImplHelper::record() {
    MGB_LOCK_GUARD(m_mtx);

    do_record();
    m_recorded = true;
    m_finished = false;
}

bool CompNodeImplHelper::EventImplHelper::finished() {
    if (m_finished)
        return true;

    MGB_LOCK_GUARD(m_mtx);

    if (m_finished)
        return true;
    mgb_assert(m_recorded);
    if (do_finished()) {
        m_finished = true;
        m_recorded = false;
        return true;
    }
    return false;
}

void CompNodeImplHelper::EventImplHelper::host_wait() {
    if (sm_cpu_sync_level >= 2) {
        while (!finished())
            ;
        return;
    }
    if (sm_cpu_sync_level >= 1) {
        while (!finished()) {
            std::this_thread::yield();
        }
        return;
    }
    mgb_assert(!sm_cpu_sync_level, "invalid cpu sync level: %d",
               sm_cpu_sync_level);

    host_wait_cv();
}

void CompNodeImplHelper::EventImplHelper::host_wait_cv() {
    while (!finished()) {
        std::this_thread::yield();
    }
}

double CompNodeImplHelper::EventImplHelper::elapsed_time_until(Event& end_) {
    mgb_assert(m_create_flags & NEED_TIMER);
    auto&& end = static_cast<EventImplHelper&>(end_);
    mgb_assert(m_comp_node_impl == end.m_comp_node_impl);
    mgb_assert(finished() && end_.finished());
    return do_elapsed_time_until(end);
}

void CompNodeImplHelper::EventImplHelper::device_wait_by(CompNode cn) {
    mgb_assert(m_recorded);
    do_device_wait_by(static_cast<Impl*>(cn.m_impl));
}

CompNode CompNodeImplHelper::EventImplHelper::comp_node() const {
    return CompNodeImplHelper::make_comp_node_from_impl(m_comp_node_impl);
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
