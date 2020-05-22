/**
 * \file src/core/impl/graph/seq_sublinear_memory.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "./memory_optimizer.h"
#include "megbrain/graph/cg.h"
#include "megbrain/utils/async_worker.h"

#if MGB_ENABLE_SUBLINEAR
namespace mgb {
namespace cg {

/*!
 * \brief modifying computing sequence, with basically the same idea of Training
 *      Deep Nets with Sublinear Memory Cost
 */
class SeqModifierForSublinearMemory {
    /*!
     * describes modifications that should be applied to an operator sequnce:
     * maps from an opr to the oprs that should be duplicated and inserted
     * before it.
     */
    using SeqModifyAction = std::unordered_map<OperatorNodeBase*, OprNodeArray>;
    using SplitPointSet = std::shared_ptr<std::vector<size_t>>;

    //! Config options
    using Config = mgb::cg::ComputingGraph::Options::SublinearMemConfig;
    Config* m_config;

    //! get modifications to be taken under some specific constraints
    class ModifyActionPlanner;

    //! search best modify action for opr seq on a single comp node
    class ActionSearcherSingleCN;

    struct Opr;
    struct Var;

    struct InternalDeleter {
        void operator()(ActionSearcherSingleCN*) const;
        void operator()(ModifyActionPlanner*) const;
    };

    struct OprReplaceInfo {
        OperatorNodeBase
                *recomp = nullptr,  //!< recomp operator from replaced input
                *dup = nullptr;     //!< duplicated operator due to discarding
    };

    //! map from original operator to its replace info; used for sanity check
    ThinHashMap<OperatorNodeBase*, OprReplaceInfo> m_opr2replace_info;

    //! map from thread ID to corresponding ModifyActionPlanner as a worker
    std::unordered_map<std::thread::id,
                       std::unique_ptr<ModifyActionPlanner, InternalDeleter>>
            m_thread2planner;

    //! thread pool to run ModifyActionPlanner
    FutureThreadPool<void> m_planner_thread_pool;

    //! map from original var to replaced var
    ThinHashMap<VarNode*, VarNode*> m_var_map;

    VarNodeArray m_new_inputs;  //!< setup by replace_vars

    MemoryOptimizerHelper m_mem_opt;

    ComputingGraphImpl* const m_owner_graph = nullptr;

    CompNode::UnorderedMap<size_t> m_prev_min_bottleneck;

    /*!
     * \brief replace input vars according to m_var_map, and store results in
     *      m_new_inputs;
     * \return whether any var is changed
     */
    bool replace_vars(const VarNodeArray& inputs);

    /*!
     * \brief copy opr and set inputs to m_new_inputs, and add outputs in
     *     m_var_map
     * \return new operator
     */
    OperatorNodeBase* copy_opr_from_new_inputs(OperatorNodeBase* opr,
                                               bool recomp);

    //! restore computing sequence and modify operator priority
    void reset_opr_seq(const OprNodeArray& oprseq);

    //! search for best action based on *cn2oprseq*
    SeqModifyAction search_action(const CompNode::UnorderedMap<OprNodeArray>* cn2oprseq);

    //! apply action and store result to m_var_map
    void apply_action(SeqModifyAction& action, const OprNodeArray& oprseq);

    template <typename... Args>
    static SplitPointSet make_split_point_set(Args&&... args) {
        return std::make_shared<SplitPointSet::element_type>(
                std::forward<Args>(args)...);
    }

public:
    SeqModifierForSublinearMemory(ComputingGraphImpl* owner, Config* config_g);

    //! see memory_optimizer set_priority_before_opt
    void set_priority_before_opt(const VarNodeArray& endpoints) {
        m_mem_opt.set_priority_before_opt(endpoints);
    }

    //! see memory_optimizer restore_graph_option
    void restore_graph_option() {
        m_mem_opt.restore_graph_option();
    }

    //! replace endpoint vars by the ones that require more computing
    void modify_endpoint_vars(VarNodeArray& endpoints);

    //! check whether actual opr_seq is what we expect; throw InternalError
    void sanity_check(const OprNodeArray& opr_seq);

    const CompNode::UnorderedMap<size_t>& prev_min_bottleneck();
};

}  // namespace cg
}  // namespace mgb

#endif  //  MGB_ENABLE_SUBLINEAR

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
