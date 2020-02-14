/**
 * \file src/core/impl/graph/graph_opt.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./graph_opt.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/opr/basic_arith.h"
#include "megbrain/serialization/serializer.h"

using namespace mgb;
using namespace cg;

constexpr size_t MAX_CONST_FOLDING_SIZE = 1024;

OperatorNodeBase* GraphOptimizer::insert_pre(OperatorNodeBase *opr) {
    auto hash = opr->hash();
    auto iter = m_opr_hash_list.find(hash);
    if (iter != m_opr_hash_list.end()) {
        for (auto i: iter->second) {
            if (i->is_same(*opr)) {
                if (opr->owner_graph()->options().log_level >= 2) {
                    mgb_log_debug("opr %s{%s} already exists as %s, "
                            "do not insert again",
                            opr->cname(), opr->dyn_typeinfo()->name,
                            i->cname());
                }
                mgb_assert(i->output().size() == opr->output().size());
                if (opr->usable_output().size() == 1) {
                    auto c = m_const_map.find(i->output(0));
                    if (c != m_const_map.end())
                        return c->second;
                }
                return i;
            }
        }
    }
    return nullptr;
}

OperatorNodeBase* GraphOptimizer::insert_post(OperatorNodeBase *opr) {
    bool already_inserted = false;
    auto hash = opr->hash();
    auto iter = m_opr_hash_list.find(hash);
    if (iter != m_opr_hash_list.end()) {
        for (auto i: iter->second) {
            if (i->is_same(*opr)) {
                already_inserted = true;
                // If the hash of the operator to be saved is already saved in
                // m_opr_hash_list, we validate that the to-be-saved operator
                // is original one which we saved.
                // If this fails, it usually means insert_post is not paired
                // with a corresponding insert_pre, or the caller didn't use
                // the saved operator returned by insert_pre.
                mgb_assert(i == opr);
            }
        }
    }
    if (!already_inserted) {
        m_opr_hash_list[hash].push_back(opr);
    }

#if !MGB_BUILD_SLIM_SERVING
    // For eager mode, return the original opr without the opt pass
    if (opr->owner_graph()->options().eager_evaluation) return opr;
#endif

    OperatorNodeBase* ret = nullptr;
    static const std::array<OperatorNodeBase* (GraphOptimizer::*) (VarNode*), 3> passes = {
            &GraphOptimizer::merge_bcast,
            &GraphOptimizer::swap_typecvt_and_bcast,
            &GraphOptimizer::replace_const_var,
    };

    for (auto pass : passes) {
        if (opr->usable_output().size() > 1)
            break;

        ret = (this->*pass)(opr->output(0));
        opr = ret ? ret : opr;
    }
    return opr;
}

namespace {

Maybe<std::pair<OperatorNodeBase*, OperatorNodeBase*>> match_oprs_in_chain(
        VarNode* var, Typeinfo* type, Typeinfo* prev_type) {
    auto opr = var->owner_opr();
    if (opr->input().size() == 0)
        return {};

    if (opr->dyn_typeinfo() != type)
        return {};

    auto prev_opr = opr->input(0)->owner_opr();
    if (prev_opr->dyn_typeinfo() != prev_type)
        return {};

    return std::pair<OperatorNodeBase*, OperatorNodeBase*>{opr, prev_opr};
}
}  // namespace

OperatorNodeBase* GraphOptimizer::merge_bcast(VarNode* var) {
    if (!is_const_var_value(var))
        return nullptr;

    auto bcast_type = opr::Broadcast::typeinfo();
    auto oprs = match_oprs_in_chain(var, bcast_type, bcast_type);
    if (!oprs.valid())
        return nullptr;

    auto opr = oprs->first;
    auto prev_opr = oprs->second;
    auto new_bcast = opr::Broadcast::make(
            prev_opr->input(0), opr->output(0)->shape(), opr->config());
    return new_bcast.node()->owner_opr();
}

OperatorNodeBase* GraphOptimizer::swap_typecvt_and_bcast(VarNode* var) {
    if (!is_const_var_value(var))
        return nullptr;

    auto oprs = match_oprs_in_chain(var, opr::TypeCvt::typeinfo(),
                                    opr::Broadcast::typeinfo());
    if (!oprs.valid())
        return nullptr;

    auto opr = oprs->first;
    auto prev_opr = oprs->second;
    auto new_cvt =
            opr::TypeCvt::make(prev_opr->input(0), var->dtype(), opr->config());
    auto new_bcast = opr::Broadcast::make(new_cvt, prev_opr->output(0)->shape(),
                                          prev_opr->config());
    return new_bcast.node()->owner_opr();
}

OperatorNodeBase* GraphOptimizer::replace_const_var(VarNode* var) {
    if (!is_const_var_value(var))
        return nullptr;

    {
        auto type = var->owner_opr()->dyn_typeinfo();
        if (type == opr::ImmutableTensor::typeinfo())
            return nullptr;
    }

    auto&& mgr = var->owner_graph()->static_infer_manager();
    auto&& shp = mgr.infer_shape(var);
    if (shp.total_nr_elems() >= MAX_CONST_FOLDING_SIZE)
        return nullptr;

    auto&& infer_val = mgr.infer_value(var);
    if (!infer_val.layout().is_contiguous()) {
        return nullptr;
    }

    HostTensorND val;
    val.copy_from(infer_val);
    auto imm = opr::ImmutableTensor::make(
                       *var->owner_graph(), val,
                       OperatorNodeConfig{}.comp_node(var->comp_node()))
                       .node()
                       ->owner_opr();
    m_const_map[var] = imm;
    mgb_assert(imm->output(0)->dtype() == var->dtype());
    return imm;
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
