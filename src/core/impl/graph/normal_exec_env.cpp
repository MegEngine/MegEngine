/**
 * \file src/core/impl/graph/normal_exec_env.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./normal_exec_env.h"

#include "megbrain/graph/exc_extra_info.h"

#include <thread>

using namespace mgb;
using namespace mgb::cg;

void NormalExecEnv::pause_exec() {
#if MGB_HAVE_THREAD
    m_exec_paused.store(true, std::memory_order_relaxed);
#else
    mgb_throw(InternalError, "pause_exec() without thread support");
#endif
}

void NormalExecEnv::resume_exec() {
#if MGB_HAVE_THREAD
    MGB_LOCK_GUARD(m_exec_paused_resume_mtx);
    m_exec_paused.store(false, std::memory_order_relaxed);
    m_exec_paused_resume_cv.notify_all();
#else
    mgb_throw(InternalError, "resume_exec() without thread support");
#endif
}

void NormalExecEnv::wait_resume_if_paused() {
#if MGB_HAVE_THREAD
    if (m_exec_paused.load(std::memory_order_relaxed)) {
        for (;;) {
            std::unique_lock<std::mutex> lock{m_exec_paused_resume_mtx};
            if (!m_exec_paused.load()) {
                break;
            }
            m_exec_paused_resume_cv.wait(lock);
        }
    }
#else
    mgb_throw(InternalError, "wait_resume_if_paused() without thread support");
#endif
}

void NormalExecEnv::normalize_comp_node(CompNode& cn) {
    // we need to use different issuing queues for comp nodes which  has an
    // implicit (and seemingly unconfigurable) limit of the driver's queue.
    // For example, CUDA comp nodes. For other comp nodes, we can use a single
    // queue
    if (!cn.contain_flag(CompNode::Flag::QUEUE_LIMITED)) {
        static auto default_cpu = CompNode::default_cpu();
        if (!(m_async_level & 0b10)) {
            cn = default_cpu;
        }
    }
}

void NormalExecEnv::add_comp_node(CompNode cn) {
    normalize_comp_node(cn);
    m_worker_task_queue[cn];  // insert task seq
}

template <bool check_exec_pause, bool check_exec_mask>
void NormalExecEnv::run_task_seq_impl(const TaskSeq& seq) {
    OperatorNodeBase* cur_opr = nullptr;
    MGB_MARK_USED_VAR(cur_opr);
    MGB_TRY {
        for (auto&& i : seq) {
            cur_opr = i.opr;
#if MGB_ENABLE_COND_EXEC
            if (check_exec_mask) {
                if (i.mask && !i.mask->enabled()) {
                    continue;
                }
            }
#endif
            i.task();

            if (check_exec_pause) {
                wait_resume_if_paused();
            }
        }
    }
    MGB_CATCH(MegBrainError & exc, {
        if (cur_opr && !exc.extra_info())
            OperatorNodeExcExtraInfo::record(cur_opr, exc);
        throw;
    })
}

template <bool check_exec_pause>
void NormalExecEnv::run_task_seq(const TaskSeq& seq) {
#if MGB_ENABLE_COND_EXEC
    if (m_has_exec_mask) {
        return run_task_seq_impl<check_exec_pause, true>(seq);
    }
#endif
    return run_task_seq_impl<check_exec_pause, false>(seq);
}

void NormalExecEnv::dispatch_on_comp_node(CompNode cn, Task&& task) {
    ExecutionMask* mask = nullptr;
    MGB_IF_COND_EXEC(mask = m_cur_active_opr_mask);
    dispatch_on_comp_node_with_mask(cn, std::move(task), mask);
}

void NormalExecEnv::dispatch_on_comp_node_with_mask(CompNode cn, Task&& task,
                                                    ExecutionMask* mask) {
    if (m_async_level) {
        normalize_comp_node(cn);
        m_worker_task_queue.at(cn).emplace_back(
                std::move(task), m_cur_active_opr MGB_IF_COND_EXEC(, mask));
    } else {
        m_sync_task_queue.emplace_back(
                std::move(task), m_cur_active_opr MGB_IF_COND_EXEC(, mask));
    }
}

void NormalExecEnv::start_exec() {
#if MGB_HAVE_THREAD
    resume_exec();
#endif

    if (m_async_level) {
        mgb_assert(!m_worker_task_queue.empty());
        if (m_worker_task_queue.size() > 1 || (m_async_level & 0b100)) {
            if (m_worker_set.empty()) {
                // init async dispatch workers
                for (auto&& i : m_worker_task_queue) {
                    auto runner = [ this, cn = i.first ]() {
                        run_task_seq<true>(m_worker_task_queue.at(cn));
                    };
                    m_worker_set.add_worker(
                            "comp_node_dispatch:" + i.first.to_string(),
                            runner);
                }
            }
            m_worker_set.start();
        } else {
            run_task_seq<false>(m_worker_task_queue.begin()->second);
        }
    } else {
        run_task_seq<false>(m_sync_task_queue);
    }
}

void NormalExecEnv::wait_all() {
    if (!m_worker_task_queue.empty()) {
        if (m_worker_task_queue.size() > 1 || (m_async_level & 0b100)) {
            m_worker_set.wait_all();
        }
    }
}

void NormalExecEnv::clear() {
    for (auto&& i : m_worker_task_queue)
        i.second.clear();
    m_sync_task_queue.clear();
    m_cur_active_opr = nullptr;
    MGB_IF_COND_EXEC(m_has_exec_mask = false);
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
