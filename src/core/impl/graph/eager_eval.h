/**
 * \file src/core/impl/graph/eager_eval.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "./impl_common.h"
#include "./var_node_mem_mgr.h"
#include "megbrain/graph/cg.h"

namespace mgb {
namespace cg {

#if MGB_BUILD_SLIM_SERVING
class EagerEvalManager {
public:
    explicit EagerEvalManager(ComputingGraphImpl*) {}

    void on_opr_insert(OperatorNodeBase*) {}

    const ComputingGraph::VarReceiverInfo* var_receiver_info(
            const VarNode*) const {
        return nullptr;
    }

    GraphExecutable::ExecEnv* exec_env() { return nullptr; }

    const CompSeqExtraInfo* comp_seq_extra_info() { return nullptr; }

    bool enabled() const { return false; }

    size_t get_var_nr_readers(VarNode*) const { return REFCNT_INF; }
};

#else
class EagerEvalManager {
    class EagerExecEnv;
    struct VersionTrait {
        enum Flag : uint8_t {
            // never re-eval, all outputs of this operator could be treated
            // as constant; conflicts with MUTABLE
            CONST = 1 << 0,
            // always re-eval; conflicts with CONST
            MUTABLE = 1 << 1,
            // always re-eval and would mark all readers of this op as MUTABLE;
            // used together with MUTABLE
            MUTABLE_SOURCE = 1 << 2
        };
        Flag flag = static_cast<Flag>(0);
        bool need_reeval;
        void update_version() {
            mgb_assert(need_reeval);
            if (!(flag & Flag::MUTABLE)) {
                need_reeval = false;
            }
            for (auto &&i : readers) {
                i->need_reeval = true;
            }
        }
        SmallVector<VersionTrait*> readers;
    };
    //! -1: uninitialized (before first opr insertion); 0/1: disabled/enabled
    int m_first_opr_enable_status = -1;
    ComputingGraph* const m_owner_graph;
    std::unique_ptr<EagerExecEnv> m_exec_env;
    CompSeqExtraInfo m_comp_seq_extra_info;
    MemPool<CompNodeSyncManager> m_var_sync_mgr_pool;
    MemPool<VersionTrait> m_version_trait_pool;
    ThinHashMap<OperatorNodeBase*, VersionTrait*> m_opr2version;

    bool m_record_mode = false;
    ThinHashSet<OperatorNodeBase*> m_record_oprs;
    ThinHashMap<VarNode*, size_t> m_var2nr_readers;

    //! run ID used for static memory allocator and would not get increased
    size_t m_run_id = 1;

    void do_on_opr_insert(OperatorNodeBase* opr);
    void update_static_infer_result(OperatorNodeBase *opr);
    void prepare_for_exec(OperatorNodeBase* opr);
    void alloc_output_mem(OperatorNodeBase* opr);
    void init_waiting_spec(OperatorNodeBase* opr);

    //! copy var tensor as contiguous if layout constraint is not satisified
    void ensure_input_layout(VarNode* var);

    //! check version of the given operator and return opr's current status
    //! -1: uninitilized / 0: version unchanged / 1: version changed
    int check_version(OperatorNodeBase* opr);

public:
    explicit EagerEvalManager(ComputingGraph* graph);
    ~EagerEvalManager() noexcept;

    bool enabled() const { return m_owner_graph->options().eager_evaluation; }

    //! called after an operator is inserted; output vars would be evaluated if
    //! eager_eval is enabled
    //! re-evaluation would be triggered if a previously inserted operator
    //! was reinserted and its version was changed
    void on_opr_insert(OperatorNodeBase* opr);

    /*!
     * \brief return faked VarReceiverInfo; or nullptr if not enabled
     *
     * VarReceiverInfo should be faked so that all vars would be considered as
     * being used
     */
    const ComputingGraph::VarReceiverInfo* var_receiver_info(
            const VarNode* var) const;

    /*!
     * \brief get curresponding ExecEnv if enabled; return nullptr if not
     *      enabled
     */
    GraphExecutable::ExecEnv* exec_env();

    /*!
     * \brief get a suitable CompSeqExtraInfo if enabled; return nullptr if not
     *      enabled
     */
    const CompSeqExtraInfo* comp_seq_extra_info() {
        if (enabled()) {
            return &m_comp_seq_extra_info;
        }
        return nullptr;
    }

    /*!
     * \brief record oprs rather than really execute them when insert oprs
     * into graph, which only use in symbolic gradients computing.
     */
    bool enter_record_mode() {
        bool old = m_record_mode;
        mgb_assert(old || m_record_oprs.empty());
        m_record_mode = true;
        return old;
    }

    /*!
     * \brief flush all oprs recorded and execute the oprs which were depended on
     * dest_vars. Note it would also turn off record mode after calling this method.
     */
    void flush_record_oprs(const VarNodeArray &dest_vars);

    /*!
     * \brief get the reader numbers of a var. return REFCNT_INF if var is not an
     * intermediate result when calculating grad.
     */
    size_t get_var_nr_readers(VarNode* var) const;
};
#endif  // MGB_BUILD_SLIM_SERVING

}  // namespace cg
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
