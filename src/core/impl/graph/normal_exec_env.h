/**
 * \file src/core/impl/graph/normal_exec_env.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/graph/execution_mask.h"
#include "megbrain/graph/operator_node.h"
#include "megbrain/utils/async_worker.h"

namespace mgb {
namespace cg {

//! normal ExecEnv impl (its counterpart is EagerExecEnv)
class NormalExecEnv final : public GraphExecutable::ExecEnv {
    struct TaskSeqElem {
        Task task;
        OperatorNodeBase* opr;
        MGB_IF_COND_EXEC(ExecutionMask* mask);

        TaskSeqElem(Task task_, OperatorNodeBase* opr_ MGB_IF_COND_EXEC(
                                        , ExecutionMask* mask_))
                : task{std::move(task_)},
                  opr{opr_} MGB_IF_COND_EXEC(, mask{mask_}) {}

        TaskSeqElem(const TaskSeqElem&) = default;

        // add noexcept so it can be moved in vector
        TaskSeqElem(TaskSeqElem&& rhs) noexcept
                : task{std::move(rhs.task)},
                  opr{rhs.opr} MGB_IF_COND_EXEC(, mask{rhs.mask}) {}

        TaskSeqElem& operator=(const TaskSeqElem&) = default;

        TaskSeqElem& operator=(TaskSeqElem&& rhs) noexcept {
            task = std::move(rhs.task);
            opr = rhs.opr;
            MGB_IF_COND_EXEC(mask = rhs.mask);
            return *this;
        }
    };

    using TaskSeq = std::vector<TaskSeqElem>;

    int m_async_level = 1;

#if MGB_HAVE_THREAD
    std::atomic_bool m_exec_paused{false};
    std::mutex m_exec_paused_resume_mtx;
    std::condition_variable m_exec_paused_resume_cv;
#endif

    AsyncWorkerSet m_worker_set;
    CompNode::UnorderedMap<TaskSeq> m_worker_task_queue;
    TaskSeq m_sync_task_queue;
    OperatorNodeBase* m_cur_active_opr = nullptr;
    MGB_IF_COND_EXEC(ExecutionMask* m_cur_active_opr_mask = nullptr);
    MGB_IF_COND_EXEC(bool m_has_exec_mask = false);

    inline void wait_resume_if_paused();

    void normalize_comp_node(CompNode& cn);

    template <bool check_exec_pause>
    void run_task_seq(const TaskSeq& seq);

    template <bool check_exec_pause, bool check_exec_mask>
    void run_task_seq_impl(const TaskSeq& seq);

public:
    //! see ComputingGraph::Options::async_exec_level
    void set_async_level(int level) {
        mgb_assert(m_worker_task_queue.empty() && m_sync_task_queue.empty());
        m_async_level = level;
    }

    /*!
     * \brief register a compuing node
     */
    void add_comp_node(CompNode cn);

    /*!
     * \brief set active operator, so all following calls to
     *      dispatch_on_comp_node() would be assumbed to be issued by this
     *      opr
     */
    void set_active_opr(OperatorNodeBase* opr) {
        m_cur_active_opr = opr;
#if MGB_ENABLE_COND_EXEC
        m_cur_active_opr_mask =
                opr ? ExecutionMask::get_from_opr(opr) : nullptr;
        if (m_cur_active_opr_mask) {
            m_has_exec_mask = true;
        }
#endif
    }

    void dispatch_on_comp_node(CompNode cn, Task&& task) override;

    void dispatch_on_comp_node_with_mask(CompNode cn, Task&& task,
                                         ExecutionMask* mask) override;

    /*!
     * \brief start running of all added tasks; if there is only one task
     *      queue, it would be executed syncrhonouly
     */
    void start_exec();

    /*!
     * \brief wait for previous start_exec() to finish
     */
    void wait_all();

    /*!
     * \brief clear all tasks
     *
     * Note that the task queues are not cleared, and add_comp_node() does
     * not need to be called again.
     */
    void clear();

    /*!
     * \brief pause execution on all threads if there are async dispatch
     *      threads
     */
    void pause_exec() override;

    /*!
     * \brief resume execution (cancel previous pause_exec())
     */
    void resume_exec() override;
};

}  // namespace cg
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
