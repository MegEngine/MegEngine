/**
 * \file src/jit/impl/internal_graph.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/gopt/framework.h"
#include "megbrain/jit/executor_opr.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/serialization/opr_shallow_copy.h"

#if MGB_JIT
using namespace mgb;
using namespace jit;
using namespace cg;
using namespace gopt;

namespace {
void recursive_replace(ThinHashMap<VarNode*, VarNode*>& old2new,
                       OperatorNodeBase* opr) {
    VarNodeArray rewritten_inputs;
    for (auto inp : opr->input()) {
        if (!old2new.count(inp))
            recursive_replace(old2new, inp->owner_opr());
        rewritten_inputs.push_back(old2new[inp]);
    }
    auto new_opr = serialization::copy_opr_shallow(*opr, rewritten_inputs,
                                                   opr->config());
    for (size_t i = 0; i < opr->output().size(); ++i) {
        if (!opr->output(i)->contain_flag(VarNode::Flag::VOLATILE_CONTENT)) {
            old2new[opr->output(i)] = new_opr->output(i);
        }
    }
}

InternalGraphPtr expand_executor_opr(const InternalGraphPtr& prev_igraph) {
    bool has_jit_executor = false;
    SymbolVarArray endpoints{SymbolVar{prev_igraph->output()}};
    SubGraph sub_graph{endpoints};
    SubGraph::Rewriter rewriter{&sub_graph};

    auto on_opr = [&](OperatorNodeBase* opr) {
        if (auto jit_exctor = try_cast_as_op<JITExecutor>(opr)) {
            has_jit_executor = true;
            auto& igraph = jit_exctor->internal_graph();
            mgb_assert(igraph.output());

            ThinHashMap<VarNode*, VarNode*> old2new;
            for (size_t i = 0; i < opr->input().size(); ++i) {
                auto inp = rewriter.get_var(opr->input(i));
                auto ph = igraph.placeholders().at(i)->output(0);
                auto iter = old2new.emplace(ph, inp);
                if (!iter.second) {
                    mgb_assert(iter.first->second == inp);
                }
            }
            recursive_replace(old2new, igraph.output()->owner_opr());

            rewriter.replace_var(opr->output(0), old2new[igraph.output()],
                                 mgb_cstr_log("update internal graph"));
        } else {
            rewriter.auto_replace_outputs(opr);
        }
    };
    sub_graph.iter(on_opr);
    if (!has_jit_executor)
        return prev_igraph;
    rewriter.apply_inplace();

    return std::make_shared<InternalGraph>(
            rewriter.get_var(prev_igraph->output()),
            rewriter.get_var(prev_igraph->shape_infer()),
            rewriter.get_var(prev_igraph->value_infer()),
            prev_igraph->placeholders());
}

}  // namespace

InternalGraphGenerator::InternalGraphGenerator(cg::OperatorNodeBase* opr)
        : m_output{opr->output(0)} {
    add_opr(opr);
}

VarNode* InternalGraphGenerator::replace_graph_by_placeholder() {
    ThinHashMap<VarNode*, VarNode*> old2new;
    auto cpu_default = CompNode::default_cpu();
    auto igraph_copy_opr_shallow = [cpu_default](OperatorNodeBase* opr,
                                                 const VarNodeArray& inputs) {
        OperatorNodeConfig config = opr->config();
        // reset instance_id.
        config.reset_instance_id();
        if (auto imm = gopt::try_cast_as_op<opr::ImmutableTensor>(opr)) {
            HostTensorND hval{cpu_default};
            hval.copy_from(imm->value()).sync();
            return opr::ImmutableTensor::make(*opr->owner_graph(), hval).node();
        }
        auto new_opr = serialization::copy_opr_shallow(*opr, inputs, config);
        return new_opr->output(0);
    };

    m_orig_inps.clear();
    m_placeholders.clear();
    VarNodeArray new_inp;
    ThinHashSet<cg::OperatorNodeBase*> graph_input_opr_set;
    for (auto i: m_graph_input_set)
        graph_input_opr_set.insert(i->owner_opr());

    auto on_opr = [&](cg::OperatorNodeBase* opr) {
        bool any_output_in_internal_graph = false;
        for (auto nd : opr->output()) {
            // skip the varnode that is not part of the internal graph, it could
            // happen when opr is the graph_input's owner_opr
            if (nd->contain_flag(VarNode::Flag::VOLATILE_CONTENT) ||
                !m_var_dep_type.count(nd) ||
                (graph_input_opr_set.count(opr) && !m_graph_input_set.count(nd)))
                continue;
            any_output_in_internal_graph = true;
            auto dep_type = m_var_dep_type.at(nd);
            dep_type &= ~DepType::VALUE_ALLOW_EMPTY;
            bool is_shape_input = dep_type == DepType::HOST_VALUE;
            mgb_assert(is_shape_input || dep_type == DepType::DEV_VALUE,
                       "unhandled dep type: %d", static_cast<int>(dep_type));
            VarNode* new_nd;
            if (m_graph_input_set.count(nd)) {
                using IT = JITPlaceholder::InpType;
                new_nd = JITPlaceholder::make(nd, m_input_idx++,
                                              is_shape_input
                                                      ? IT::HOST_VALUE_FOR_SHAPE
                                                      : IT::DEV_VALUE)
                                 .node();
                m_orig_inps.push_back(nd);
                m_placeholders.push_back(new_nd);
            } else {
                mgb_assert(!is_shape_input);
                mgb_assert(m_opr_set.count(opr));
                new_inp.clear();
                for (auto i : opr->input()) {
                    new_inp.push_back(old2new.at(i));
                }
                new_nd = igraph_copy_opr_shallow(nd->owner_opr(), new_inp);
            }
            mgb_assert(new_nd->comp_node() == cpu_default);
            old2new[nd] = new_nd;
        }
        mgb_assert(any_output_in_internal_graph,
                   "at least one output should be in the internal graph.");
    };
    cg::DepOprIter iter{on_opr};
    for (auto i : m_graph_input_set) {
        for (auto j : i->owner_opr()->input()) {
            if (!graph_input_opr_set.count(j->owner_opr()) &&
                !m_opr_set.count(j->owner_opr())) {
                iter.set_visited(j->owner_opr());
            }
        }
    }
    iter.add(m_output);
    return old2new.at(m_output);
}

InternalGraphPtr InternalGraphGenerator::generate() {
    m_input_idx = 0;

    auto new_nd = replace_graph_by_placeholder();
    auto igraph = std::make_shared<InternalGraph>(
            new_nd, m_output, m_output, to_placeholder_opr_arr(m_placeholders));
    return expand_executor_opr(igraph);
}

size_t InternalGraphGenerator::get_cnt_input_if_add(
        cg::OperatorNodeBase* opr) const {
    // minus 1 first because this opr should be removed from subgraph's input
    size_t new_cnt_input = m_graph_input_set.size() - 1;
    for (auto inp : opr->input()) {
        if (m_graph_input_set.count(inp) == 0)
            new_cnt_input += 1;
    }
    return new_cnt_input;
}

void InternalGraphGenerator::add_opr(cg::OperatorNodeBase* opr) {
    if (m_opr_set.count(opr)) {
        // ignore duplicated oprs (which occur in tests)
        return;
    }

    if (opr->input().empty()) {
        mgb_assert(opr->same_type<opr::ImmutableTensor>(),
                   "should not add net source opr %s{%s}", opr->cname(),
                   opr->dyn_typeinfo()->name);
    }

    // currently only single-output opr is supported; ensure it here
    for (size_t i = 1; i < opr->output().size(); ++i) {
        mgb_assert(opr->output()[i]->contain_flag(
                VarNode::Flag::VOLATILE_CONTENT));
    }

    if (!m_opr_set.empty()) {
        auto nr_remove = m_graph_input_set.erase(opr->output(0));
        mgb_assert(nr_remove == 1, "opr output not added");
    } else {
        // opr_set is empty, so this is the endpoint opr
        m_var_dep_type[opr->output(0)] = DepType::DEV_VALUE;
    }

    m_opr_set.insert(opr);
    for (auto inp : opr->input()) {
        m_graph_input_set.insert(inp);
    }

    for (auto&& i : opr->node_prop().dep_map()) {
        DepType dt = i.second & ~DepType::VALUE_ALLOW_EMPTY;
        mgb_assert(dt == DepType::DEV_VALUE || dt == DepType::HOST_VALUE,
                   "unsupported dep type: opr %s{%s} on input %s dt=%d",
                   opr->cname(), opr->dyn_typeinfo()->name, i.first->cname(),
                   static_cast<int>(dt));
        m_var_dep_type[i.first] |= i.second;
    }

    if (opr->same_type<opr::Reduce>()) {
        if (!has_reduce()) {
            m_before_reduce_shape = opr->input(0)->shape();
            m_feature_bits |= JITFeatureBits::REDUCE;
        }
        mgb_assert(opr->input(0)->shape().eq_shape(m_before_reduce_shape));
        find_reduce_opr_deps(opr);
    }
    if (opr->same_type<opr::Dimshuffle>()) {
        m_feature_bits |= JITFeatureBits::DIMSHUFFLE;
        find_oprs_depended_by_dimshuffle(opr);
    }
    if (opr->same_type<mgb::jit::JITExecutor>()) {
        auto jit = &opr->cast_final<mgb::jit::JITExecutor>();
        if (jit->has_reduce()) {
            if (!has_reduce()) {
                m_before_reduce_shape = jit->broadcasted_input_shape();
                m_feature_bits |= JITFeatureBits::REDUCE;
            }
            mgb_assert(jit->broadcasted_input_shape().eq_shape(
                    m_before_reduce_shape));
            find_reduce_opr_deps(opr);
        }
        if (jit->has_dimshuffle()) {
            m_feature_bits |= JITFeatureBits::REDUCE;
            find_oprs_depended_by_dimshuffle(opr);
        }
    }
}

void InternalGraphGenerator::find_reduce_opr_deps(cg::OperatorNodeBase* opr) {
    mgb_assert(opr->same_type<opr::Reduce>() ||
               (opr->same_type<jit::JITExecutor>() &&
                try_cast_as_op<jit::JITExecutor>(opr)->has_reduce()));
    VarNode* nd = opr->output(0);
    auto cb = [this, &nd](cg::OperatorNodeBase* opr) {
        m_reduce_out_var_deps[nd].insert(opr);
    };
    cg::DepOprIter{cb}.add(opr);
}

void InternalGraphGenerator::find_oprs_depended_by_dimshuffle(
        cg::OperatorNodeBase* dimshuffle) {
    mgb_assert(
            dimshuffle->same_type<opr::Dimshuffle>() ||
            (dimshuffle->same_type<jit::JITExecutor>() &&
             try_cast_as_op<jit::JITExecutor>(dimshuffle)->has_dimshuffle()));
    auto cb = [this, dimshuffle](cg::OperatorNodeBase* opr) {
        if (!m_oprs_depended_by_dimshuffle.count(opr)) {
            // No dimshuffle depend on the opr.
            mgb_assert(!m_oprs_depended_by_dimshuffle.count(dimshuffle));
            m_oprs_depended_by_dimshuffle[opr] = dimshuffle;
        } else {
            // Already be depended by dimshuffle.
            if (m_oprs_depended_by_dimshuffle.count(dimshuffle) &&
                m_oprs_depended_by_dimshuffle.at(opr) ==
                        m_oprs_depended_by_dimshuffle.at(dimshuffle)) {
                m_oprs_depended_by_dimshuffle[opr] = dimshuffle;
            }
        }
    };
    cg::DepOprIter{cb}.add(dimshuffle);
}

PlaceholderArray InternalGraphGenerator::to_placeholder_opr_arr(
        const VarNodeArray& vars) {
    PlaceholderArray ret(vars.size());
    for (size_t i = 0; i < vars.size(); ++i) {
        ret[i] = &vars[i]->owner_opr()->cast_final_safe<JITPlaceholder>();
    }
    return ret;
}

#endif  // MGB_JIT

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
