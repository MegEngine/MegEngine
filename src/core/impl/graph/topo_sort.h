/**
 * \file src/core/impl/graph/topo_sort.h
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
#include "megbrain/graph/cg.h"
#include "megbrain/utils/shared_set.h"

namespace mgb {
namespace cg {

class TopoSorter {
public:
    struct PriorityItem;
    using PriorityRemapper =
            thin_function<void(const VarNodeArray& dest_vars,
                               const PriorityItem* seq, size_t seq_len)>;

    TopoSorter(ComputingGraphImpl* graph);
    ~TopoSorter() noexcept;

    /*!
     * \brief get a computing sequence satisifying topology requirement
     * \param extra_info output param, extra info for the comp seq
     */
    const OprNodeArray* get_comp_seq(CompSeqExtraInfo& extra_info,
                                     const VarNodeArray& dest);

    //! undo modifications on opr node props
    void restore_opr_prop();

    /*!
     * \brief set a callback function to modify opr priorities
     *
     * Note that remapper would be cleard during next call of get_comp_seq()
     */
    void set_priority_remapper(PriorityRemapper remapper) {
        m_priority_remapper = std::move(remapper);
    }

private:
    //! node information in bfs queue
    struct NodeTrait;

    //! perform DFS to discover opr deps
    class DFSDepDiscover;

    //! element in BFS priority queue
    class BFSQueueElem;

    //! current sorting state
    struct State;

    using OprNodeProp = OperatorNodeBase::NodeProp;

    OprNodeArray m_seq;
    ComputingGraphImpl* m_owner_graph;
    CompSeqExtraInfo* m_cur_extra_info = nullptr;
    State* m_state = nullptr;

    //! record original dep map value that has been modified
    std::vector<std::tuple<OperatorNodeBase*, VarNode*, OprNodeProp::DepType>>
            m_modified_dep_map_log;

    PriorityRemapper m_priority_remapper;

    /*!
     * \brief make final sequence satisfying topological order and
     *      used-defined priority; result is written to m_seq
     */
    void bfs_make_seq();

    /*!
     * \brief add computing order requriment on opr that var must finish
     *      before it
     */
    void add_extra_comp_order_dep(OperatorNodeBase* opr, VarNode* var);
};

struct TopoSorter::PriorityItem {
    const OperatorNodeBase* opr;
    //! pointer to priority that can be directly modified
    int* priority;
    //! a timestamp for when processing of this opr is finished during DFS
    size_t dfs_step_num;
};

}  // namespace cg
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

