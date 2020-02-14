/**
 * \file src/core/impl/graph/memory_optimizer.h
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

namespace mgb {
namespace cg {

/*! \brief helper class for getting opr sequence, set/restore priority
 *   and split oprseq into cn2oprseq.
 */
class MemoryOptimizerHelper {
    ComputingGraphImpl* const m_owner_graph = nullptr;

    ThinHashMap<VarNode*, size_t> m_var_memsize;

    ThinHashMap<OperatorNodeBase*, int> m_saved_priority;

    bool m_graph_option_changed = false;

    CompNode::UnorderedMap<OprNodeArray> m_cn2oprseq;

public:
    //! get operator sequence in computing(topological) order.
    struct CompSeq {
        const OprNodeArray* m_seq;
        ComputingGraphImpl* const m_owner_graph;
        CompSeq(ComputingGraphImpl* owner, const VarNodeArray& endpoints);
        ~CompSeq();
    };

    //! marking the 'bad' opr/vars which should be ignored.
    struct SubGraphConfig {
        VarNode::Flag bad_var_flag;
        OperatorNodeBase::NodeProp::Flag bad_opr_flag;
        SubGraphConfig& add_bad_opr_flag(
                OperatorNodeBase::NodeProp::Flag flag) {
            bad_opr_flag |= flag;
            return *this;
        };
        SubGraphConfig& add_bad_var_flag(VarNode::Flag flag) {
            bad_var_flag |= flag;
            return *this;
        };
    };

    MemoryOptimizerHelper(ComputingGraphImpl* owner);
    //! valid after `split_into_cn2oprseq` called
    const ThinHashMap<VarNode*, size_t>* var2memsize() const {
        return &m_var_memsize;
    }

    //! modify priority of given operator and record original value
    void set_priority(OperatorNodeBase* opr, int pri);

    //! split *oprseq* into *m_cn2oprseq*
    const CompNode::UnorderedMap<OprNodeArray>* split_into_cn2oprseq(
            const OprNodeArray& oprseq, const SubGraphConfig& config);

    //! called before graph optimization to set opr priority
    void set_priority_before_opt(const VarNodeArray& endpoints);

    //! restore graph options to the version before modified by
    //! modify_endpoint_vars()
    void restore_graph_option();
};

}  // namespace cg
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
