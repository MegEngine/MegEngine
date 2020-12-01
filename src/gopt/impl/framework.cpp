/**
 * \file src/gopt/impl/framework.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/gopt/framework.h"
#include "megbrain/gopt/basic_arith.h"
#include "megbrain/gopt/gtrans.h"
#include "megbrain/gopt/inference.h"
#include "megbrain/gopt/misc.h"
#include "megbrain/gopt/weights_preprocess.h"
#include "megbrain/graph/cg.h"
#include "megbrain/graph/event.h"
#include "megbrain/graph/exc_extra_info.h"
#include "megbrain/serialization/opr_shallow_copy.h"
#include "megbrain/serialization/serializer.h"
#include "megbrain/utils/timer.h"

#if MGB_JIT
#include "megbrain/jit/fusion_pass.h"
#endif

#if MGB_ENABLE_TENSOR_RT
#include "megbrain/tensorrt/opr_replace.h"
#endif

using namespace mgb;
using namespace gopt;

/* ================ SubGraph ================ */

OperatorNodeBase* SubGraph::Rewriter::auto_replace_outputs(
        OperatorNodeBase *opr) {
    auto &&new_inp = m_opr_new_inp_cache;
    new_inp.clear();
    new_inp.reserve(opr->input().size());
    bool has_replaced_inp = false;

    for (auto i: opr->input()) {
        auto new_var = get_var(i);
        if (new_var != i) {
            has_replaced_inp = true;
            new_inp.push_back(new_var);
        } else {
            new_inp.push_back(i);
        }
    }

    if (has_replaced_inp) {
        auto new_opr = serialization::copy_opr_shallow(
                *opr, new_inp, opr->config());
        auto &&out0 = opr->output(), &&out1 = new_opr->output();
        size_t i = 0;
        auto err_msg = [opr, new_opr] {
            return ssprintf("bad opr copy: src=%s{%s} dst=%s{%s}",
                    opr->cname(), opr->dyn_typeinfo()->name,
                    new_opr->cname(), new_opr->dyn_typeinfo()->name);
        };
        MGB_MARK_USED_VAR(err_msg);
        // opr output size mismatch may be caused by:
        //     0) inplace arith optimization (e.g. PowC need an extra workspace)
        //     1) other post-insert optimization (e.g. const folding)
        // we can't handle only usable_output here, since some output var with
        // volatile flag could be the graph's endpoint (e.g. RemoteSend)
        for (; i < std::min(out0.size(), out1.size()); ++ i) {
            bool v0 = out0[i]->contain_flag(VarNode::Flag::VOLATILE_CONTENT),
                 v1 = out1[i]->contain_flag(VarNode::Flag::VOLATILE_CONTENT);
            mgb_assert(v0 == v1, "%s", err_msg().c_str());

            auto &&ins = m_varmap.insert({out0[i], {true, nullptr}});
            mgb_assert(ins.second || ins.first->second.first,
                       "opr output already replaced");
            // handle repeated call on the same opr
            ins.first->second.second = out1[i];
            on_var_replaced(out0[i], out1[i], nullptr);
        }
        for (; i < out0.size(); ++ i) {
            mgb_assert(out0[i]->contain_flag(VarNode::Flag::VOLATILE_CONTENT),
                    "%s", err_msg().c_str());
        }
        for (; i < out1.size(); ++ i) {
            mgb_assert(out1[i]->contain_flag(VarNode::Flag::VOLATILE_CONTENT),
                    "%s", err_msg().c_str());
        }
        return new_opr;
    }
    return opr;
}

void SubGraph::Rewriter::replace_var(
        VarNode *src, VarNode *dst, const char *msg) {
    if (src == dst)
        return;

    // Optimizers should not create a loop in varaible replace map.
    mgb_throw_if(
            get_var_internal(dst).second == src, InternalError,
            "dst %s maps back to src %s in SubGraph::Rewriter::replace_var",
            dst->cname(), src->cname());

    auto &&ins = m_varmap.insert({src, {false, dst}});
    if (!ins.second) {
        auto &&old_rep = ins.first->second;
        mgb_assert(old_rep.first || old_rep.second == dst,
                "can not replace a var twice");
        old_rep.first = false;
        old_rep.second = dst;
    }
    on_var_replaced(src, dst, msg);
}

void SubGraph::Rewriter::on_var_replaced(
        VarNode* src, VarNode* dst, const char* msg) {
    if (auto state = m_owner_graph->owner_opt_state()) {
        state->on_var_replaced(src, dst, msg);
    }
}

void SubGraph::Rewriter::apply_inplace() const {
    m_owner_graph->m_endpoint_oprs.clear();
    m_owner_graph->m_endpoint_vars_set.clear();
    for (auto &&var: m_owner_graph->m_endpoint_vars) {
        var = get_var(var.node());
        m_owner_graph->m_endpoint_oprs.insert(var.node()->owner_opr());
        m_owner_graph->m_endpoint_vars_set.insert(var.node());
    }
}

std::pair<bool, VarNode*> SubGraph::Rewriter::get_var_internal(VarNode* var) {
    // The implementation is (manually) unrolled once, background:
    // git-core/brain-sdk/MegBrain/merge_requests/486#note_76971
    auto it = m_varmap.find(var);
    if (it == m_varmap.end()) {
        return {true, var};
    }
    mgb_assert(it->second.second != var, "loop detected in m_varmap");
    auto it_next = m_varmap.find(it->second.second);
    if (it_next == m_varmap.end()) {
        return it->second;
    }
    mgb_assert(it_next->second.second != it->second.second,
               "loop detected in m_varmap");
    auto next = get_var_internal(it_next->second.second);
    it_next->second = {next.first & it_next->second.first, next.second};
    return it->second = {it_next->second.first & it->second.first, next.second};
}

SubGraph::SubGraph(const SymbolVarArray &endpoint_vars):
    m_endpoint_vars(endpoint_vars)
{
    mgb_assert(!endpoint_vars.empty(), "endpoints can not be empty");
    m_comp_graph = endpoint_vars[0].node()->owner_graph();
    for (auto i: endpoint_vars) {
        m_endpoint_oprs.insert(i.node()->owner_opr());
        m_endpoint_vars_set.insert(i.node());
        mgb_assert(m_comp_graph == i.node()->owner_graph(),
                "endpoints belong to different computing graphs");
    }
}

void SubGraph::iter(
        const Callback& cb,
        std::shared_ptr<ExtraDep> extra_dep) const {
    Callback on_opr;

    if (m_owner_opt_state) {
        on_opr = [state=m_owner_opt_state, &cb](OperatorNodeBase *opr) {
            state->m_opr_property_flag = OprPropertyFlag::ALL;
            state->m_cur_iter_src_opr = cg::get_opr_root_source_opr(opr);
            state->m_cur_iter_opr_priority =
                opr->node_prop().attribute().priority;
            state->m_cur_iter_opr_stream_prop_type =
                state->m_comp_node_opt.stream_prop_type(
                        opr->output(0));
            mgb_assert(state->m_oprs_inserted.empty());
            cb(opr);
            state->m_opr_property_flag = OprPropertyFlag::NONE;
            state->m_cur_iter_src_opr = nullptr;
            state->m_oprs_inserted.clear();
        };
    } else {
        on_opr = cb;
    }

    cg::DepOprIter dep_iter{on_opr, std::move(extra_dep)};
    for (auto i: m_endpoint_oprs)
        dep_iter.add(i);
}

ThinHashMap<VarNode*, size_t> SubGraph::get_var2nr_val_dep_oprs() const {
    ThinHashMap<VarNode*, size_t> ret;
    auto cb = [&](OperatorNodeBase *opr) {
        for (auto &&i: opr->node_prop().dep_map()) {
            if (OperatorNodeBase::NodeProp::is_device_value_dep(i.second)) {
                ++ ret.at(i.first);
            }
        }
        for (auto i: opr->output()) {
            if (!i->contain_flag(VarNode::Flag::VOLATILE_CONTENT)) {
                auto ins = ret.insert({i, 0});
                mgb_assert(ins.second);
            }
        }
    };
    iter(cb);
    for (auto i: m_endpoint_vars_set) {
        auto iter = ret.find(i);
        if (iter == ret.end()) {
            mgb_assert(i->contain_flag(VarNode::Flag::VOLATILE_CONTENT));
            ret[i] = 1;
        } else {
            ++ ret.at(i);
        }
    }
    return ret;
}

/* ================ UniqReaderCheck ================ */

UniqReaderCheck::UniqReaderCheck(const SubGraph &graph):
    m_var2nr_val_dep{graph.get_var2nr_val_dep_oprs()}
{
}

void UniqReaderCheck::update_on_opr_auto_replace(OperatorNodeBase* opr,
                                                 OperatorNodeBase* repl_opr) {
    auto non_volatile_size = [](const VarNodeArray& vars) -> size_t {
        size_t size = 0;
        for (size_t i = 0; i < vars.size(); ++i) {
            if (!vars[i]->contain_flag(VarNode::Flag::VOLATILE_CONTENT)) {
                size++;
            }
        }
        return size;
    };
    if (opr != repl_opr) {
        auto &&o0 = opr->output(), &&o1 = repl_opr->output();
        mgb_assert(non_volatile_size(o0) == non_volatile_size(o1));
        for (size_t i = 0; i < o0.size(); ++i) {
            auto iter = m_var2nr_val_dep.find(o0[i]);
            if (iter != m_var2nr_val_dep.end()) {
                auto n = iter->second;
                m_var2nr_val_dep[o1[i]] = n;
            }
        }
    }
}

/* ================ OptState ================ */

OptState::OptState(
        const GraphOptimizer *owner_optimizer, const SubGraph& graph):
    m_owner_optimizer{owner_optimizer},
    m_var_replace_map{
        const_cast<ThinHashMap<VarNode*, VarNode*>*>(
                &GraphOptimizer::var_replace_map(*graph.comp_graph()))},
    m_comp_node_opt{graph.comp_graph()->seq_comp_node_optimizer()},
    m_graph{graph}
{
    mgb_assert(!m_graph.m_owner_opt_state);
    m_var_replace_map->clear();
    m_graph.m_owner_opt_state = this;
    m_oprs_inserted.clear();

    auto on_opr_insert = [this](const cg::event::OprInserted &ev) {
        auto need_src_opr = m_opr_property_flag & OprPropertyFlag::SOURCE_OPR,
             need_priority = m_opr_property_flag & OprPropertyFlag::PRIORITY;
        if (need_src_opr)
            mgb_assert(m_cur_iter_src_opr, "opr %s{%s} created outside from "
                    "SubGraph::iter",
                    ev.opr->cname(), ev.opr->dyn_typeinfo()->name);
        if (ev.exc || ev.is_dedup)
            return;

        auto &&new_attr = ev.opr->node_prop().attribute();
        auto &&ins = m_oprs_inserted.insert({ev.opr, OprPropertyFlag::NONE});
        mgb_assert(ins.second);

        if (need_src_opr && !new_attr.src_opr) {
            auto src_opr = m_cur_iter_src_opr;
            if (ev.opr != src_opr)
                new_attr.src_opr = src_opr;
            ins.first->second |= OprPropertyFlag::SOURCE_OPR;
        }
        if (need_priority) {
            new_attr.priority = m_cur_iter_opr_priority;
            if (!ev.opr->update_priority()) {
                ins.first->second |= OprPropertyFlag::PRIORITY;
            }
        }

        auto csp = m_cur_iter_opr_stream_prop_type;
        if (csp.prop_type != cg::SeqCompNodeOptimizer::StreamPropType::NONE) {
            for (auto i: ev.opr->output())
                m_comp_node_opt.register_stream_var(i, csp);
        }
    };
    m_on_opr_insert_handler = graph.comp_graph()->event().register_receiver<
        cg::event::OprInserted>(on_opr_insert);
}

void OptState::on_var_replaced(VarNode *src, VarNode *dst, const char *msg) {
    if (src->contain_flag(VarNode::Flag::VOLATILE_CONTENT)) {
        // this can only happen in auto_replace_outputs()
        mgb_assert(dst->contain_flag(VarNode::Flag::VOLATILE_CONTENT) &&
                src->owner_opr()->dyn_typeinfo() ==
                dst->owner_opr()->dyn_typeinfo());
        mgb_assert(!msg);
        return;
    }

    //! check_property
    {
        auto iter = m_oprs_inserted.find(dst->owner_opr());
        if (iter != m_oprs_inserted.end()) {
            auto &&src_attr = src->owner_opr()->node_prop().attribute(),
                 &&dst_attr = dst->owner_opr()->node_prop().attribute();
            auto opr_info = [&](OperatorNodeBase* opr) {
                return opr ? opr->name() + "(" + std::to_string(opr->id()) + ")"
                           : "NULL";
            };
            auto err_msg = [&] {
                std::string ret = "Please contact Engine group:\n";
                ret += "src opr: ";
                ret += opr_info(src->owner_opr());
                ret += ", dst opr: ";
                ret += opr_info(dst->owner_opr());
                return ret;
            };
            MGB_MARK_USED_VAR(err_msg);
            if (iter->second & OprPropertyFlag::SOURCE_OPR) {
                auto &&src_rt = get_opr_root_source_opr(src->owner_opr()),
                     &&dst_rt = get_opr_root_source_opr(dst->owner_opr());
                mgb_assert(dst_rt == src_rt,
                           "%s\nsrc source_opr: %s, dst source_opr: %s\n",
                           err_msg().c_str(), opr_info(src_rt).c_str(),
                           opr_info(dst_rt).c_str());
            }
            if (iter->second & OprPropertyFlag::PRIORITY) {
                mgb_assert(src_attr.priority == dst_attr.priority,
                           "%s\nsrc priority: %d, dst priority %d\n",
                           err_msg().c_str(), src_attr.priority,
                           dst_attr.priority);
            }
        }
    }

    {
        bool suc = true;
        SmallVector<std::string> fail_chks;
        if (m_var_replace_check_flag & VarReplaceCheckFlag::CHECK_INFER_TYPE) {
            auto&& mgr = src->owner_graph()->static_infer_manager();
            auto it0 = mgr.get_infer_type(src), it1 = mgr.get_infer_type(dst);
            using cg::static_infer::InferType;
            // only check wheter inferable
            auto norm = [](InferType::Flag f) -> bool {
                return f & (InferType::RT_STATIC | InferType::CONST);
            };
            if (!(norm(it0.shape) == norm(it1.shape) &&
                norm(it0.value) <= norm(it1.value))) {
                suc = false;
                fail_chks.push_back("infer-type");
            }
        }
        if (m_var_replace_check_flag & VarReplaceCheckFlag::CHECK_DTYPE) {
            if (src->dtype() != dst->dtype()) {
                suc = false;
                fail_chks.push_back("dtype");
            }
        }
        if (m_var_replace_check_flag & VarReplaceCheckFlag::CHECK_SHAPE) {
            if (!(src->shape().eq_shape(dst->shape()))) {
                suc = false;
                fail_chks.push_back("shape");
            }
        }
        if (!suc) {
            std::string fail_msg = "{";
            for (size_t i = 0; i < fail_chks.size(); i++) {
                fail_msg += fail_chks[i];
                if (i < fail_chks.size() - 1) {
                    fail_msg += ",";
                }
            }
            fail_msg += "}";
            mgb_throw_raw(
                    cg::OperatorNodeExcExtraInfo::ExcMaker{src->owner_opr()}
                            .make<InternalError>(ssprintf(
                                    "%s mismatch for replace_var: %s",
                                    fail_msg.c_str(),
                                    cg::dump_var_info({src, dst}).c_str())));
        }
    }

    if (src->has_name_set() && !dst->has_name_set()) {
        dst->name(src->name());
    }
    (*m_var_replace_map)[src] = dst;
    // dst should be considered as newly inserted, and previous replace
    // record should be ignored
    m_var_replace_map->erase(dst);

#if MGB_ENABLE_LOGGING
    if (msg && m_owner_optimizer->verbosity()) {
        m_log_msg.
            append("\n ").
            append(std::to_string(m_log_nr_item)).
            append(": ").
            append(src->owner_opr()->cname()).
            append(" => ").
            append(dst->owner_opr()->cname()).
            append(" (").
            append(msg).
            append(")");
    }
    ++ m_log_nr_item;
#endif
}

size_t OptState::flush_log(const char *title) {
    if (m_owner_optimizer->verbosity() >= 2) {
        if (m_log_msg.empty()) {
            m_log_msg = mgb_cstr_log(" no var replacement logged");
        }
        mgb_log("%s%s", title, m_log_msg.c_str());
        m_log_msg.clear();
    }
    auto ret = m_log_nr_item;
    m_log_nr_item = 0;
    return ret;
}

void OptState::call_with_opr(OperatorNodeBase *opr, thin_function<void(void)> func,
                             OprPropertyFlag opr_property_flag) {
    auto src_opr = cg::get_opr_root_source_opr(opr);
    auto opr_priority = opr->node_prop().attribute().priority;
    auto stream_prop_type = m_comp_node_opt.stream_prop_type(opr->output(0));
    ThinHashMap<OperatorNodeBase*, OprPropertyFlag> oprs_inserted;

    auto swap_properties = [&,
        need_src_opr = opr_property_flag & OprPropertyFlag::SOURCE_OPR,
        need_priority = opr_property_flag & OprPropertyFlag::PRIORITY] {
        if (need_src_opr) {
            std::swap(m_cur_iter_src_opr, src_opr);
        }
        if (need_priority) {
            std::swap(m_cur_iter_opr_priority, opr_priority);
        }
        std::swap(m_cur_iter_opr_stream_prop_type, stream_prop_type);
        std::swap(m_opr_property_flag, opr_property_flag);
        std::swap(m_oprs_inserted, oprs_inserted);
    };
    MGB_TRY {
        swap_properties();
        func();
    } MGB_FINALLY({
        swap_properties();
    });
}

/* ================ RecursiveSubGraphRewriteHelper ================ */
RecursiveSubGraphRewriteHelper::
~RecursiveSubGraphRewriteHelper() noexcept = default;

RecursiveSubGraphRewriteHelper::RecursiveSubGraphRewriteHelper(OptState &state):
    m_opt_state{state}, m_rewriter{state.graph().make_rewriter()}
{
}

void RecursiveSubGraphRewriteHelper::apply() {
    using namespace std::placeholders;
    m_opt_state.graph().iter(
            std::bind(&RecursiveSubGraphRewriteHelper::on_opr, this, _1));
    m_rewriter.apply_inplace();
}

void RecursiveSubGraphRewriteHelper::on_opr(OperatorNodeBase *opr) {
    auto on_new_opr = [this](OperatorNodeBase *opr) {
        auto repl_opr = m_rewriter.auto_replace_outputs(opr);
        return on_new_opr_check_should_process(opr, repl_opr);
    };

    if (!on_new_opr(opr))
        return;

    auto orig_out = get_opr_single_output_var(opr);
    if (!orig_out)
        return;

    mgb_assert(m_opr_stack.empty());
    m_opr_stack.push_back({
            orig_out, m_rewriter.get_var(orig_out)->owner_opr()});

    bool first = true;
    while (!m_opr_stack.empty()) {
        auto cur_frame = m_opr_stack.back();
        m_opr_stack.pop_back();
        auto cur_opr = cur_frame.opr;
        bool should_process;
        if (first) {
            should_process = true;
            first = false;
        } else {
            should_process = on_new_opr(cur_opr);
        }
        auto cur_out = get_opr_single_output_var(cur_opr);
        mgb_assert(cur_out);
        cur_out = m_rewriter.get_var(cur_out);

        if (should_process) {
            auto trans = process_opr(cur_out);
            if (trans.valid()) {
                m_opr_stack.push_back({
                        cur_frame.orig_var, trans->result->owner_opr()});
                for (auto i: reverse_adaptor(trans->internal)) {
                    if (i)
                        m_opr_stack.push_back({i, i->owner_opr()});
                }
                if (trans->msg) {
                    if (!m_log_msg.empty())
                        m_log_msg.push_back(';');
                    m_log_msg.append(trans->msg);
                }
                continue;
            }
        }

        auto src = cur_frame.orig_var;
        if (m_rewriter.get_var(src) != cur_out) {
            const char *msg = nullptr;
            if (m_opr_stack.empty()) {
                msg = m_log_msg.c_str();
            }
            m_rewriter.replace_var(src, cur_out, msg);
            after_replace_var(src, cur_out);
            if (m_opr_stack.empty()) {
                m_log_msg.clear();
                break;
            }
        }
    }
}

/* ================ GraphOptimizer ================ */

GraphOptimizer::~GraphOptimizer() noexcept = default;

class GraphOptimizer::VarReplaceMapStorage :public UserDataContainer::UserData {
    MGB_TYPEINFO_OBJ_DECL;

    public:
        ThinHashMap<VarNode*, VarNode*> map;
};
MGB_TYPEINFO_OBJ_IMPL(GraphOptimizer::VarReplaceMapStorage);

GraphOptimizer& GraphOptimizer::add_pass(std::unique_ptr<Pass> pass) {
    mgb_assert(!pass->m_owner_optimizer);
    pass->m_owner_optimizer = this;
    m_passes.emplace_back(std::move(pass));
    return *this;
}

SubGraph GraphOptimizer::apply(const SubGraph &graph) const {
    RealTimer timer;
    OptState state{this, graph};

    size_t tot_nr_replace = 0;

    // first update output var shapes of all oprs
    state.graph().iter(cg::update_output_var_shapes);

    auto &&opt = graph.comp_graph()->options();
    auto orig_setting = opt.graph_opt_level;
    Pass *cur_pass = nullptr;
    MGB_MARK_USED_VAR(cur_pass);
    MGB_TRY {
        for (auto &&i: m_passes) {
            state.set_var_replace_check_flag(VarReplaceCheckFlag::CHECK_ALL);
            cur_pass = i.get();
            opt.graph_opt_level = 1;
            i->apply(state);
            tot_nr_replace += state.flush_log(
                    mgb_ssprintf_log(
                        "apply optimization pass %s:", i->name()).c_str());
        }
    } MGB_CATCH(std::exception &exc, {
        mgb_log_error("error while applying optimization pass %s: %s",
                cur_pass->name(), exc.what());
        opt.graph_opt_level = orig_setting;
        throw;
    })
    MGB_FINALLY(
        opt.graph_opt_level = orig_setting
    );
    if (verbosity() >= 1) {
        mgb_log_debug("graph optimization: applied %zu passes, "
                "total %zu var(s) replaced; time=%.2fms",
                m_passes.size(), tot_nr_replace, timer.get_msecs());
    }
    return state.graph();
}

const GraphOptimizer& GraphOptimizer::apply_inplace(VarNodeArray &vars) const {
    if (m_passes.empty()) {
        // this check is necessary, since OptState would clear
        // var_replace_map()
        return *this;
    }

    auto g = apply({{vars.begin(), vars.end()}});
    for (size_t i = 0; i < vars.size(); ++ i) {
        vars[i] = g.endpoint_vars()[i].node();
    }
    return *this;
}

GraphOptimizer& GraphOptimizer::add_preset_passes(
        bool after_grad, const OptimizeForInferenceOptions* inference_opt,
        const ComputingGraph::Options* comp_graph_opt) {
    auto cv_type = inference_opt ? ConstVarType::IMMUTABLE_AND_PARAM
                                 : ConstVarType::IMMUTABLE;
    if (inference_opt) {
        add_pass<ConvertBatchNormToElemwisePass>();
    }
    if (!after_grad || inference_opt) {
        add_pass<CondExecConstPredicateFolding>();
    }
    if (after_grad || inference_opt) {
        add_pass<RemoveNonComputingOprPass>();
    }
    add_pass<DelayBroadcastPass>();
    add_pass<ExpandFusedArithPass>();
    add_pass<NormalizeArithChainPass>();
    if (inference_opt) {
        add_pass<ParamRedistributePass>();
        add_pass<ParamFusePass>();
    }
    add_pass<ArithMulDistributePass>();
    add_pass<ReorderArithChainPass>(cv_type);

    add_pass<ArithFusePass>();
    // reorder again because shapes of fused oprs might change
    add_pass<ReorderArithChainPass>(cv_type);
    add_pass<FinalArithTransformPass>();
    add_pass<RemoveRedundantTypeCvtPass>();
    add_pass<RemoveRedundantCopyPass>();

#if MGB_JIT
    bool need_jit = false;
    if (comp_graph_opt && (std::abs(comp_graph_opt->graph_opt_level) >= 3 ||
            comp_graph_opt->graph_opt.jit)) {
        need_jit = true;
    }
    if (need_jit && after_grad) {
        add_pass<gopt::RecompTypeCvtPass>();
    }
#endif

    // combine astype and reduce.
    // Note: apply this pass before JITFusion, so the TypeCvt which
    // read by both Reduce and Elemwise could be fused correctly.
    add_pass<CombineAstypeAndReducePass>();

#if MGB_JIT
    if (need_jit) {
        add_pass<gopt::JITFusionPass>(
                after_grad,
                std::max<uint8_t>(comp_graph_opt->graph_opt.jit, 1));
    }
#endif

    if (inference_opt) {
        add_pass<ParamFusePass>();
        add_passes_for_optimize_options(*inference_opt);
    }


    if (inference_opt) {
        // merge params to reduce loading time and graph overhead
        add_pass<ParamMergePass>();
        add_pass<FuseDeconvCvtPass>();
    }
    return *this;
}

const ThinHashMap<VarNode*, VarNode*>& GraphOptimizer::var_replace_map(
        ComputingGraph &graph) {
    auto storage = graph.options().user_data.get_user_data_or_create<
        VarReplaceMapStorage>();
    return storage->map;
}

VarNode* GraphOptimizer::var_replace_lookup(VarNode *var) {
    auto &&map = var_replace_map(*(var->owner_graph()));
    for (; ; ) {
        auto iter = map.find(var);
        if (iter == map.end())
            return var;
        var = iter->second;
    }
}


const GraphOptimizer& GraphOptimizer::add_passes_for_optimize_options(
        const cg::GraphCommonOptimizeOptions& options) {
    return add_passes_for_optimize_options(
            const_cast<cg::GraphCommonOptimizeOptions&>(options));
}

const GraphOptimizer& GraphOptimizer::add_passes_for_optimize_options(
        cg::GraphCommonOptimizeOptions& options, bool reset) {
    bool need_param_fuse = false;

#define cb(_option, _passes)             \
    if (options.has_set_##_option()) {   \
        _passes need_param_fuse = true;  \
        if (reset) {                     \
            options.disable_##_option(); \
        }                                \
    }
    
    cb(fuse_preprocess, {add_pass(FuseNCHW4Int8Preprocess::make());});
    cb(f16_io_comp, { add_pass(ConvertF32ToF16Pass::make(false)); });
    cb(f16_io_f32_comp, { add_pass(ConvertF32ToF16Pass::make(true)); });


    cb(nchw4, {
        add_pass<FuseConvBiasNonlinPass>();
        add_pass<FuseConvBiasZPass>();
        add_pass(EnableNCHW4Pass::make_nchw4_converter());
        add_pass<ShuffleShuffleRemovePass>();
        add_pass<RemoveRedundantTypeCvtPass>();
    });
    cb(nhwcd4, {
        add_pass<FuseConvBiasNonlinPass>();
        add_pass(ConvertFormatPass::make_nhwcd4_converter());
    });
    cb(nchw88, {
        add_pass<FuseConvBiasNonlinPass>();
        add_pass(EnableNchwxxPass::make_nchwxx_converter(8));
        add_pass<ShuffleShuffleRemovePass>();
    });
    cb(nchw44, {
        add_pass<FuseConvBiasNonlinPass>();
        add_pass(EnableNchwxxPass::make_nchwxx_converter(4));
        add_pass<ShuffleShuffleRemovePass>();
    });
    cb(nchw44_dot, {
        add_pass<FuseConvBiasNonlinPass>();
        add_pass(EnableNchw44DotPass::make_nchw44_dot_converter());
        add_pass<ShuffleShuffleRemovePass>();
    });
    cb(nchw32, {
        add_pass<FuseConvBiasNonlinPass>();
        add_pass<FuseConvBiasZPass>();
        add_pass(EnableNCHW4Pass::make_nchw4_converter());
        add_pass(EnableTensorCorePass::make_tensorcore_converter());
        add_pass<ShuffleShuffleRemovePass>();
        add_pass<RemoveRedundantTypeCvtPass>();
        add_pass(FuseNCHW4Int8Preprocess::make());
    });
    cb(chwn4, {
        add_pass<FuseConvBiasNonlinPass>();
        add_pass<FuseConvBiasZPass>();
        add_pass(EnableNCHW4Pass::make_nchw4_converter());
        add_pass(EnableCHWN4Pass::make_chwn4_converter());
        add_pass<ShuffleShuffleRemovePass>();
        add_pass<RemoveRedundantTypeCvtPass>();
    });

    cb(fuse_conv_bias_nonlinearity, { add_pass<FuseConvBiasNonlinPass>(); });
    cb(fuse_conv_bias_with_z, {
        add_pass<FuseConvBiasNonlinPass>();
        add_pass<FuseConvBiasZPass>();
    });

    cb(weight_winograd_transform,
       { add_pass<WinogradTransformReplacePass>(); });
#undef cb

    if (need_param_fuse) {
        add_pass<ParamFusePass>();
    }
    return *this;
}

/* ================ ConstVarPropogateBase ================ */

ConstVarPropogate::AddOprResult ConstVarPropogate::add_opr(
        OperatorNodeBase *opr) {
    using ProfFlag = OperatorNodeBase::NodeProp::Flag;
    auto &&info = m_oprinfo[opr];
    if (info.processed)
        return info.result;
    info.processed = true;

#if MGB_ENABLE_JSON
    (*opr->to_json_extra_json)["gopt::cvprop"] = json::Bool::make(false);
#endif

    AddOprResult ret{false, false, false};
    auto make_ret = [&ret, &info]() {
        info.result = ret;
        return ret;
    };

    if (is_const_var(m_const_var_type, opr)) {
        auto sz = var_mem_size(opr->output(0));
        mgb_assert(sz || opr->output(0)->contain_flag(
                                 VarNode::Flag::ALLOW_EMPTY_SHAPE));
        info.is_const = true;
        info.max_size = sz;
        return make_ret();
    }

    if (opr->input().empty())
        return make_ret();

    if (opr->node_prop().contain(
                ProfFlag::FORCE_UPDATE_INPUT_VAR |
                ProfFlag::IMPURE_FUNC)) {
        return make_ret();
    }

    size_t max_input_size = 0;
    ret.all_const_inp = true;
    for (auto i: opr->input()) {
        auto io = i->owner_opr();
        auto iter = m_oprinfo.find(io);
        if (iter == m_oprinfo.end()) {
            add_opr(io);
            iter = m_oprinfo.find(io);
            mgb_assert(iter != m_oprinfo.end());
        }
        auto &&src = iter->second;
        if (src.is_const) {
            update_max(max_input_size, src.max_size);
            ret.has_const_inp = true;
            if (!is_const_var(m_const_var_type, i->owner_opr())) {
                ret.has_midconst_inp = true;
            }
        } else {
            ret.all_const_inp = false;
        }
    }
    if (ret.all_const_inp) {
#if MGB_ENABLE_JSON
        (*opr->to_json_extra_json)["gopt::cvprop"] = json::Bool::make(true);
#endif
        info.max_size = max_input_size;
        info.is_const = true;
    }
    return make_ret();
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
