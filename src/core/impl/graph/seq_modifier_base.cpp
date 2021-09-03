/**
 * \file src/core/impl/graph/seq_modifier_base.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./seq_modifier_base.h"

#if MGB_ENABLE_SUBLINEAR || MGB_ENABLE_DTR

using namespace mgb;
using namespace cg;

void SeqModifierBase::ModifyActionPlannerBase::init_seq(const OprNodeArray& opr_seq, bool remove_unused_output) {
    m_orig_opr_seq = &opr_seq;

    m_var_storage.clear();
    m_seq.clear();
    m_var_mempool.reorder_free();
    m_opr_mempool.reorder_free();
    m_nr_endpoint_oprs = 0;

    ThinHashMap<VarNode*, Var*> varmap;
    ThinHashMap<VarNode*, Opr*> var_used;
    for (auto orig_opr : *m_orig_opr_seq) {
        auto time = m_seq.size();
        m_seq.emplace_back(m_opr_mempool.alloc_unique(orig_opr, time));
        auto opr = m_seq.back().get();
        m_nr_endpoint_oprs += opr->is_endpoint;
        for (auto&& dep : orig_opr->node_prop().dep_map()) {
            if (!OperatorNodeBase::NodeProp::is_device_value_dep(dep.second))
                continue;

            auto iter = varmap.find(dep.first);
            if (iter == varmap.end()) {
                // input var needs not to be considered
                size_t size = dep.first->dtype().size(dep.first->shape().total_nr_elems());
                if (!var_used[dep.first]) {
                    opr->inputs_size.push_back(size);
                }
                var_used[dep.first] = opr;
                continue;
            }

            auto ivar = iter->second;
            bool exist = false;
            for (auto i : opr->input) {
                if (i == ivar) {
                    exist = true;
                    break;
                }
            }
            if (exist) {
                // same var for different inputs
                continue;
            }

            opr->input.push_back(ivar);
            auto&& prev_rec = ivar->access_rec.back();
            prev_rec.stride = time - prev_rec.opr->time;
            ivar->access_rec.emplace_back(opr);
        }

        for (auto i : orig_opr->output()) {
            auto var2memsize = m_par_modifier->m_mem_opt.var2memsize();
            auto iter = var2memsize->find(i);
            if (iter == var2memsize->end()) {
                // some vars are ignored; see split_into_cn2oprseq()
                continue;
            }
            m_var_storage.emplace_back(
                    m_var_mempool.alloc_unique(i, iter->second, opr));
            auto ovar = m_var_storage.back().get();
            varmap[i] = ovar;
            opr->output.push_back(ovar);
        }
        mgb_assert(!opr->output.empty());
    }
    for (auto x : var_used) {
        size_t size = x.first->dtype().size(x.first->shape().total_nr_elems());
        var_used[x.first]->inputs_size.push_back(-static_cast<ptrdiff_t>(size));
    }
    if (remove_unused_output) {
        for (auto&& i : m_seq) {
            auto&& oarr = i->output;
            for (size_t j = 0; j < oarr.size();) {
                if (oarr[j]->access_rec.size() == 1) {
                    std::swap(oarr[j], oarr.back());
                    oarr.pop_back();
                } else
                    ++j;
            }
        }
    }
}

bool SeqModifierBase::replace_vars(const VarNodeArray& inputs) {
    m_new_inputs.assign(inputs.begin(), inputs.end());
    bool changed = false;
    for (auto&& i : m_new_inputs) {
        auto iter = m_var_map.find(i);
        if (iter != m_var_map.end()) {
            i = iter->second;
            changed = true;
        }
    }
    return changed;
}

OperatorNodeBase* SeqModifierBase::copy_opr_from_new_inputs(
        OperatorNodeBase* opr, bool recomp, size_t recomp_cnt) {
    auto config = opr->config();
    // update operator instance id to bybass the shallow copy's cache because
    // some pair of recomp-opr and dup-opr have the same inputs, params and
    // config, we use instance id to differentiate them. To be safe, we update
    // instance id whatever reason is `recomp` or `dup`
    config.name(opr->name() + (recomp ? ":recomp" : ":dup") + std::to_string(recomp_cnt));
    config.update_instance_id(reinterpret_cast<void*>(
                                reinterpret_cast<size_t>(this) + 
                                (recomp_cnt << 1 | (recomp & 1))));

    // Note: if all outputs of op were placed on the same comp_node, since its
    // stream maybe changed during seq_comp_node_opt, output's comp_node has
    // higher priority than opr->config()
    auto out_cn = opr->output(0)->comp_node();
    for (auto i : opr->output()) {
        auto cn = i->comp_node();
        if (out_cn != cn) {
            out_cn = {};
            break;
        }
    }
    if (out_cn.valid())
        config.comp_node(out_cn);

    auto opr_new = serialization::copy_opr_shallow(*opr, m_new_inputs, config);
    mgb_assert(opr_new != opr);

    auto&& out0 = opr->output();
    auto&& out1 = opr_new->output();
    mgb_assert(out0.size() == out1.size());
    bool stream_changed = false;
    for (size_t i = 0; i < out0.size(); ++i) {
        auto &&cn0 = out0[i]->comp_node(),
             &&cn1 = out1[i]->comp_node();
        if (cn0 != cn1) {
            mgb_assert(recomp);
            mgb_assert(cn0.locator().type == cn1.locator().type &&
                       cn0.locator().device == cn1.locator().device);
            out1[i]->comp_node(cn0);
            stream_changed = true;
        }
        m_var_map[out0[i]] = out1[i];
    }
    if (stream_changed) {
        opr_new->on_output_comp_node_stream_changed();
    }
    return opr_new;
}

#endif  //  MGB_ENABLE_SUBLINEAR || MGB_ENABLE_DTR

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
