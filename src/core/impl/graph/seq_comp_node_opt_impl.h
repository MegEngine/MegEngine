/**
 * \file src/core/impl/graph/seq_comp_node_opt_impl.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megbrain/graph/seq_comp_node_opt.h"
#include "./impl_common.h"

namespace mgb {
namespace cg {

class SeqCompNodeOptimizerImpl final: public SeqCompNodeOptimizer {
    ThinHashMap<VarNode*, StreamPropType> m_var2prop_type;
    ThinHashMap<VarNode*, CompNodeSyncManager> m_var2sync_mgr;
    ComputingGraphImpl *m_owner_graph;
    std::vector<std::pair<VarNode*, CompNode>> m_comp_node_to_restore;
    ThinHashSet<OperatorNodeBase*> m_comp_node_changed_oprs;
    ThinHashMap<VarNode*, PropFunction> m_var2prop_func;

    /*!
     * cn0 -> (cn1 -> [(a, b)]): an opr at step \p a on \p cn0 is known to start
     * after step \b p on \p cn1; step numbers are stored in ascending order
     *
     * this is initialized by init_ready_event() and used by
     * get_opr_other_cn_nr_finish()
     */
    CompNode::UnorderedMap<CompNode::UnorderedMap<std::vector<
        std::pair<size_t, size_t>>>> m_cnpair2opr_step;

    //! change certain vars to the stream as instructed by
    //! register_specific_stream_var
    void change_to_specific_stream(const VarNodeArray &endpoints);

    //! move a single var to specific stream and record in
    //! m_comp_node_to_restore
    void var_to_specific_stream(VarNode *var, const int stream);

    public:
        SeqCompNodeOptimizerImpl(ComputingGraphImpl *graph):
            m_owner_graph(graph)
        {}

        void init_ready_event(const CompSeqExtraInfo &extra_info,
                              const OprNodeArray &seq);

        void optimize_comp_nodes(const VarNodeArray &endpoints);

        void restore_comp_nodes();

        void register_stream_var(VarNode* var, StreamPropType prop_type) override;

        void register_propagate_function(VarNode* var, PropFunction prop_func) override;

        StreamPropType stream_prop_type(VarNode *var) override {
            auto iter = m_var2prop_type.find(var);
            return iter == m_var2prop_type.end()
                           ? StreamPropType{0, StreamPropType::PropType::NONE}
                           : iter->second;
        }

        /*!
         * \brief get max \p x so that an opr at \p step on \p cn is known to
         *      start after \p x oprs have finished on \p other_cn
         *
         * Note: all step numbers are defined in the serialized computing
         * sequence (as returned by
         * ComputingGraph::opr_step_num_in_cur_comp_seq)
         */
        size_t get_opr_other_cn_nr_finish(
                CompNode cn, size_t step, CompNode other_cn) const;
};

}
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
