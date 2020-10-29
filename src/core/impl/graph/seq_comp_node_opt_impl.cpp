/**
 * \file src/core/impl/graph/seq_comp_node_opt_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./seq_comp_node_opt_impl.h"
#include "./var_node_mem_mgr.h"
#include "./cg_impl.h"

#include <queue>

using namespace mgb;
using namespace cg;

void SeqCompNodeOptimizerImpl::optimize_comp_nodes(
        const VarNodeArray &endpoints) {
    mgb_assert(m_comp_node_to_restore.empty() &&
            m_comp_node_changed_oprs.empty(), "restore_comp_nodes not called");
    change_to_specific_stream(endpoints);

    for (auto &&i: m_comp_node_to_restore) {
        auto opr = i.first->owner_opr();
        if (m_comp_node_changed_oprs.insert(opr).second) {
            opr->on_output_comp_node_stream_changed();
        }
    }
}

void SeqCompNodeOptimizerImpl::restore_comp_nodes() {
    for (auto &&i: m_comp_node_to_restore)
        i.first->comp_node(i.second);
    for (auto i: m_comp_node_changed_oprs)
        i->on_output_comp_node_stream_changed();

    m_comp_node_to_restore.clear();
    m_comp_node_changed_oprs.clear();
}

void SeqCompNodeOptimizerImpl::var_to_specific_stream(VarNode* var,
                                                      const int stream) {
    auto old_cn = var->comp_node();
    if (old_cn.locator().stream == stream)
        return;
    if (!old_cn.contain_flag(CompNode::Flag::HAS_COPY_STREAM))
        return;
    auto new_cn = old_cn.change_stream(stream);
    mgb_assert(old_cn != new_cn);
    m_comp_node_to_restore.emplace_back(var, old_cn);
    var->comp_node(new_cn);
}

void SeqCompNodeOptimizerImpl::change_to_specific_stream(
        const VarNodeArray &endpoints) {
    if (!m_owner_graph->options().seq_opt.enable_seq_comp_node_opt) {
        mgb_log_debug("sequence computing node optimization disabled");
        return;
    }

    ThinHashMap<VarNode*, StreamPropType> changed_vars;
    std::pair<OperatorNodeBase*, StreamPropType> prop_type_storage;
    std::pair<OperatorNodeBase*, SmallVector<StreamPropType>> input_props_storage;

    // both `propagate_single_opr` and `get_input_props` might be called any number
    // of times(>=0) with the same \p opr in a cb(opr) function call, so we cache
    // the result of the first call.
    auto propagate_single_opr = [&](OperatorNodeBase* opr) {
        mgb_assert(opr);
        if (prop_type_storage.first == opr) {
            return prop_type_storage.second;
        }
        prop_type_storage.first = opr;

        bool any_strong_changed = false, all_weak_changed = true,
                all_weak_changed_valid = false;
        auto &&dep_map = opr->node_prop().dep_map();

        ThinHashSet<int> inp_streams;
        for (auto i: opr->input()) {
            if (!need_device_computing_on_var(i, dep_map.at(i))) {
                // opr does not have dev comp dep on i, so do not consider it
                // for comp node change
                continue;
            }

            auto iter = changed_vars.find(i);
            if (iter == changed_vars.end()) {
                all_weak_changed = false;
            } else {
                mgb_assert(iter->second.prop_type != StreamPropType::NONE);
                if (iter->second.prop_type == StreamPropType::STRONG) {
                    any_strong_changed = true;
                } else {
                    all_weak_changed_valid = true;
                }
                inp_streams.insert(iter->second.stream);
            }
        }

        auto type = StreamPropType::NONE;
        int stream = 0;
        if (any_strong_changed ||
                (all_weak_changed && all_weak_changed_valid)) {
            type = any_strong_changed ?
                StreamPropType::STRONG : StreamPropType::WEAK;
            int copy_stream = CompNode::Stream::COPY;
            if (inp_streams.count(copy_stream))
                stream = copy_stream;
            mgb_assert(type != StreamPropType::NONE && stream != 0);
        }
        return prop_type_storage.second = StreamPropType{stream, type};
    };

    auto get_input_props = [&](OperatorNodeBase *opr) {
        mgb_assert(opr);
        if (input_props_storage.first == opr) {
            return input_props_storage.second;
        }
        input_props_storage.first = opr;

        auto &&props = input_props_storage.second;
        props.clear();
        for (auto i : opr->input()) {
            auto &&iter = changed_vars.find(i);
            if (iter != changed_vars.end()) {
                props.push_back(iter->second);
            } else {
                props.push_back(StreamPropType{0, StreamPropType::NONE});
            }
        }
        return input_props_storage.second;
    };

    auto cb = [&](OperatorNodeBase *opr) {
        if (opr->node_prop().contain(
                    OperatorNodeBase::NodeProp::Flag::
                    DISALLOW_COMP_NODE_OPTIMIZE)) {
            return;
        }

        // first check whether any output var is registered for change
        bool output_changed = false;
        for (auto i: opr->output()) {
            auto iter = m_var2prop_type.find(i);
            if (iter != m_var2prop_type.end()) {
                output_changed = true;
                var_to_specific_stream(i, iter->second.stream);
                changed_vars[i] = iter->second;
            }
        }
        if (output_changed)
            return;

        for (auto i: opr->output()) {
            StreamPropType prop;
            auto &&iter = m_var2prop_func.find(i);
            if (iter != m_var2prop_func.end()) {
                iter->second(prop, get_input_props(opr));
            }
            else {
                prop = propagate_single_opr(opr);
            }
            if (prop.prop_type != StreamPropType::NONE) {
                var_to_specific_stream(i, prop.stream);
                changed_vars[i] = prop;
            }
        }
    };

    DepOprIter dep_iter{cb};
    for (auto i: endpoints) {
        dep_iter.add(i->owner_opr());
    }
}

void SeqCompNodeOptimizerImpl::register_stream_var(
        VarNode *var, StreamPropType stream_prop_type) {
    int stream = stream_prop_type.stream;
    auto prop_type = stream_prop_type.prop_type;
    mgb_assert(var->owner_graph() == m_owner_graph &&
            (prop_type == StreamPropType::WEAK ||
             prop_type == StreamPropType::STRONG));
    mgb_assert(stream == CompNode::Stream::COPY);

    auto ins = m_var2prop_type.insert({var, {stream, prop_type}});
    if (!ins.second) {
        mgb_assert(ins.first->second.stream == stream);
        ins.first->second.prop_type =
                std::max(ins.first->second.prop_type, prop_type);
    }
}

void SeqCompNodeOptimizerImpl::register_propagate_function(
        VarNode *var, PropFunction prop_func) {
    mgb_assert(var->owner_graph() == m_owner_graph);
    mgb_assert(m_var2prop_func.emplace(var, prop_func).second);
}

void SeqCompNodeOptimizerImpl::init_ready_event(
        const CompSeqExtraInfo &extra_info, const OprNodeArray &seq) {
    // clear existing synchronizers
    for (OperatorNodeBase* opr : seq) {
        for (auto i : opr->output()) {
            VarNodeMemManager::set_var_node_cn_sync_manager(i, nullptr);
            m_var2sync_mgr.erase(i);
        }
    }
    m_cnpair2opr_step.clear();

    // opr step, idx of output
    using VarStep = std::pair<size_t, size_t>;

    // cn0 -> (cn1 -> step): step on cn1 is known to have finished for current
    // opr on cn0
    CompNode::UnorderedMap<CompNode::UnorderedMap<VarStep>> cnpair2step;

    // vars to be waited on for current opr; only the latest var needs to be
    // waited for each comp node
    CompNode::UnorderedMap<VarNode*> vars_to_wait;

    CompNode::UnorderedSet cur_used_cn;
    ThinHashMap<VarNode*, VarStep> var2step;
    size_t cur_step = 0;

    using OprNodeProp = OperatorNodeBase::NodeProp;

    // init opr waiting spec and add waiter record
    for (OperatorNodeBase *opr: seq) {
        if (opr->node_prop().contain(
                    OperatorNodeBase::NodeProp::Flag::NO_INPUT_WAITING)) {
            opr->input_waiting_spec({});
            ++ cur_step;
            continue;
        }

        cur_used_cn.clear();
        OperatorNodeBase::InputWaitingSpec waiting_spec;
        for (auto ovar: opr->output()) {
            auto cn = ovar->comp_node();
            if (!cur_used_cn.insert(cn).second)
                continue;

            auto &&dep2step = cnpair2step[cn];
            vars_to_wait.clear();

            for (auto &&i: opr->node_prop().dep_map()) {
                // It can ignore PERSISTENT_DEVICES_VALUE(PDV) vars for most cases.
                // But if some opr depends on PDV var's host value, it should ensure
                // the PDV var has already synchronized the host value from device
                // while executing the opr.
                bool pdv_need_sync_host = false;
                if (i.first->contain_flag(
                            VarNode::Flag::PERSISTENT_DEVICE_VALUE)) {
                    // do not wait on PERSISTENT_DEVICE_VALUE vars
                    if (extra_info.missing_for_value.count(i.first)) {
                        pdv_need_sync_host = true;
                    } else {
                        continue;
                    }
                }
                if ((OprNodeProp::is_device_comp_order_dep(i.second) &&
                        i.first->comp_node() != cn) || pdv_need_sync_host) {
                    auto step = var2step.at(i.first);
                    auto ins = dep2step.insert({i.first->comp_node(), step});
                    // only wait for var if it is beyond currently known
                    // synchronized step
                    if (ins.second || step > ins.first->second) {
                        ins.first->second = step;
                        vars_to_wait[i.first->comp_node()] = i.first;
                    }
                }
            }

            if (!vars_to_wait.empty()) {
                waiting_spec.emplace_back();
                waiting_spec.back().comp_node = cn;
                for (auto&& i : vars_to_wait) {
                    VarNode* var = i.second;
                    auto&& mgr = m_var2sync_mgr[var];
                    VarNodeMemManager::set_var_node_cn_sync_manager(var, &mgr);
                    mgr.comp_node(var->comp_node()).add_waiter_record(true);
                    waiting_spec.back().dev_ready.push_back(var);
                }

                auto&& record = m_cnpair2opr_step[cn];
                for (auto&& i : vars_to_wait) {
                    auto step_done = var2step.at(i.second).first;
                    auto&& seq = record[i.first];
                    // for multi-output operator, there might be multiple other
                    // operators which depand on different output varnodes, and
                    // those output vars share the same opr step number
                    mgb_assert(seq.empty() || step_done >= seq.back().second);
                    if (seq.empty() || step_done > seq.back().second) {
                        seq.emplace_back(cur_step, step_done);
                    }
                }
            }
        }

        opr->input_waiting_spec(std::move(waiting_spec));
        for (size_t i = 0; i < opr->output().size(); ++ i) {
            var2step[opr->output(i)] = {cur_step, i};
        }
        cur_step ++;
    }
    mgb_assert(cur_step == seq.size());
}

size_t SeqCompNodeOptimizerImpl::get_opr_other_cn_nr_finish(
        CompNode cn, size_t step, CompNode other_cn) const {
    auto iter0 = m_cnpair2opr_step.find(cn);
    if (iter0 == m_cnpair2opr_step.end())
        return 0;
    auto iter1 = iter0->second.find(other_cn);
    if(iter1 == iter0->second.end())
        return 0;

    auto data = iter1->second.data();

    // find maximal x satisfying data[x].first <= step
    size_t begin = 0, end = iter1->second.size();
    while (begin + 1 != end) {
        auto mid = (begin + end) / 2;
        if (data[mid].first <= step) {
            begin = mid;
        } else {
            end = mid;
        }
    }

    if (data[begin].first > step) {
        mgb_assert(!begin);
        return 0;
    }
    mgb_assert(begin + 1 == iter1->second.size() ||
            data[begin + 1].first > step);
    return data[begin].second + 1;
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
