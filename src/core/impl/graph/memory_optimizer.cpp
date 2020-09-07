/**
 * \file src/core/impl/graph/memory_optimizer.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./memory_optimizer.h"
#include "./cg_impl.h"

namespace mgb {
namespace cg {

MemoryOptimizerHelper::CompSeq::CompSeq(ComputingGraphImpl* owner,
                                        const VarNodeArray& endpoints)
        : m_owner_graph(owner) {
    CompSeqExtraInfo extra_info;
    m_seq = owner->topo_sorter().get_comp_seq(extra_info, endpoints);
}
MemoryOptimizerHelper::CompSeq::~CompSeq() {
    m_owner_graph->topo_sorter().restore_opr_prop();
}

void MemoryOptimizerHelper::set_priority(OperatorNodeBase* opr, int pri) {
    int& val = opr->node_prop().attribute().priority;
    m_saved_priority.insert({opr, val});
    val = pri;
}

void MemoryOptimizerHelper::set_priority_before_opt(
        const VarNodeArray& endpoints) {
    mgb_assert(!m_graph_option_changed, "restore_graph_option() not called");
    mgb_assert(m_saved_priority.empty());
    m_graph_option_changed = true;

    CompSeqExtraInfo extra_info;
    const OprNodeArray* seq;
    MGB_TRY {
        seq = m_owner_graph->topo_sorter().get_comp_seq(extra_info, endpoints);
    }
    MGB_FINALLY(m_owner_graph->topo_sorter().restore_opr_prop());
    int pri = std::numeric_limits<int>::min();

    // fix priorities of original operator, so grad operator can be grouped.
    // Note that the priorities are negative, because grad() would use negative
    // priority and we want all grad oprs to execute after fwd oprs
    for (auto i : *seq) {
        set_priority(i, ++pri);
    }
}

const CompNode::UnorderedMap<OprNodeArray>*
MemoryOptimizerHelper::split_into_cn2oprseq(const OprNodeArray& oprseq,
                                            const SubGraphConfig& config) {
    auto BAD_OPR_FLAG = config.bad_opr_flag;
    auto BAD_VAR_FLAG = config.bad_var_flag;

    m_cn2oprseq.clear();
    m_var_memsize.clear();
    for (auto i : oprseq) {
        if (i->node_prop().contain(BAD_OPR_FLAG)) {
            continue;
        }

        auto cn = i->output(0)->comp_node();
        auto cn_loc = cn.locator();

        bool have_static_shape_out = false, multi_out_cn = false,
             different_device_inp = false;

        // check whether there are inputs from different device (if so, this opr
        // should never be duplciated)
        for (auto j : i->input()) {
            auto loc = j->comp_node().locator();
            if (loc.type != cn_loc.type || loc.device != cn_loc.device) {
                different_device_inp = true;
                break;
            }
        }

        if (different_device_inp) {
            continue;
        }

        // check same comp node and known shape for outputs
        for (auto j : i->output()) {
            if (j->comp_node() != cn) {
                multi_out_cn = true;
            }
        }

        if (multi_out_cn) {
            continue;
        }

        auto&& infer_mgr = m_owner_graph->static_infer_manager();
        for (auto j : i->output()) {
            if (!j->contain_flag(BAD_VAR_FLAG)) {
                // omit infer type check
                // inferred shape will be used as-is
                if (auto shape = infer_mgr.infer_shape_fallible(j)) {
                    have_static_shape_out = true;
                    m_var_memsize[j] = j->dtype().size(shape->total_nr_elems());
                }
            }
        }

        if (have_static_shape_out) {
            m_cn2oprseq[cn].push_back(i);
        }
    }
    return &m_cn2oprseq;
}

void MemoryOptimizerHelper::restore_graph_option() {
    if (!m_graph_option_changed)
        return;

    for (auto&& i : m_saved_priority) {
        i.first->node_prop().attribute().priority = i.second;
    }
    m_saved_priority.clear();
    m_graph_option_changed = false;
}

MemoryOptimizerHelper::MemoryOptimizerHelper(ComputingGraphImpl* owner)
        : m_owner_graph(owner) {}

}  // namespace cg
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
