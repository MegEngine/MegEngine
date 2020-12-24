/**
 * \file src/core/impl/graph/helper.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/graph/helper.h"
#include "megbrain/gopt/framework.h"
#include "megbrain/opr/utility.h"
#include "megbrain/serialization/opr_shallow_copy.h"
#include "./cg_impl.h"

using namespace mgb;
using namespace cg;

/* =================== global functions =================== */

CompNode::UnorderedSet cg::get_opr_comp_node_set(OperatorNodeBase *opr) {
    CompNode::UnorderedSet rst;
    for (auto i: opr->output())
        rst.insert(i->comp_node());
    if (opr->node_prop().contain(
                OperatorNodeBase::NodeProp::Flag::SINGLE_COMP_NODE))
        mgb_assert(rst.size() == 1);
    return rst;
}

bool cg::is_all_input_static_storage(OperatorNodeBase* opr) {
    for (auto&& i : opr->node_prop().dep_map())
        if (i.second != OperatorNodeBase::NodeProp::DepType::DEV_COMP_ORDER &&
            !is_static_var_storage(i.first))
            return false;
    return true;
}

VarNodeArray cg::to_var_node_array(const SymbolVarArray& symbol_var_array) {
    VarNodeArray var_node_array(symbol_var_array.size());
    for (size_t i = 0; i < symbol_var_array.size(); ++i) {
        var_node_array[i] = symbol_var_array[i].node();
    }
    return var_node_array;
}

SymbolVarArray cg::to_symbol_var_array(const VarNodeArray& var_node_array) {
    SymbolVarArray symbol_var_array(var_node_array.size());
    for (size_t i = 0; i < var_node_array.size(); ++i) {
        symbol_var_array[i] = var_node_array[i];
    }
    return symbol_var_array;
}

std::string cg::dump_var_info(const VarNodeArrayView &vars) {
    std::string rst;
    int idx = 0;
    for (auto i: vars) {
        if (!rst.empty())
            rst.append(" ");
        auto opr = i->owner_opr();
        if (vars.size() > 1)
            rst.append(ssprintf("%d=", idx ++));
        bool valid = i->dev_tensor_valid();
        auto slot = find(opr->output(), i) - opr->output().begin();
        auto &&it = i->owner_graph()->static_infer_manager().get_infer_type(i);
        rst.append(ssprintf(
                    "{id:%zu, %s:%s, %s, "
                    "owner:%s{%s}, name:%s, slot:%td, %s, %c, %d, %d}",
                    i->id(),
                    valid ? "layout": "shape",
                    valid ? i->layout().to_string().c_str() :
                        i->shape().to_string().c_str(),
                    i->dtype().name(),
                    opr->cname(), opr->dyn_typeinfo()->name,
                    i->cname(),
                    slot,
                    i->comp_node().to_string().c_str(),
                    cg::is_static_var_storage(i) ? 's' : 'd',
                    static_cast<int>(it.shape), static_cast<int>(it.value)
                    ));
    }
    return rst;
}

SymbolVar cg::grad(SymbolVar target, SymbolVar wrt, bool warn_mid_wrt,
        bool return_zero_for_nodep) {
    return grad(target, SymbolVarArray{wrt},
            warn_mid_wrt, return_zero_for_nodep)[0];
}

SymbolVarArray cg::grad(SymbolVar target_, SymbolVarArray wrts_, bool warn_mid_wrt,
        bool return_zero_for_nodep) {
#if MGB_ENABLE_GRAD
    auto target = target_.node();
    SymbolVarArray grads;
    grads.reserve(wrts_.size());
    VarNodeArray dest_vars;
    auto&& graph = target->owner_graph();
    auto&& eager_mgr = ComputingGraphImpl::downcast(graph)->eager_eval_manager();
    auto&& grad_mgr = ComputingGraphImpl::downcast(graph)->grad_manager();
    bool already_recorded = eager_mgr.enter_record_mode();
    for (auto&& wrt_ : wrts_) {
        auto wrt = wrt_.node();
        if (warn_mid_wrt && wrt->owner_opr()->input().size()) {
            mgb_log_warn("taking gradient with respect to an intermediate node may "
                    "produce incorrect results (for example, when it is produced "
                    "by subtensor); node: %s",
                    cg::dump_var_info({wrt}).c_str());
        }
        mgb_throw_if(graph != wrt->owner_graph(), GraphError,
                "target and wrt must belong to the same graph");
        auto rst = grad_mgr.grad(target, wrt);
        if (!rst && return_zero_for_nodep) {
            mgb_log_warn("target node (%s) does not depend on wrt node (%s), "
                    "return zeros as grad", cg::dump_var_info({target}).c_str(),
                    cg::dump_var_info({wrt}).c_str());
            rst = (wrt_ * 0).node();
        }
        if (rst)
            dest_vars.push_back(rst);
        grads.emplace_back(rst);
    }
    if (!already_recorded && eager_mgr.enabled()) {
        eager_mgr.flush_record_oprs(dest_vars);
        grad_mgr.clean_cache();
    }
    return grads;
#else
    MGB_MARK_USED_VAR(target_);
    MGB_MARK_USED_VAR(wrts_);
    MGB_MARK_USED_VAR(warn_mid_wrt);
    MGB_MARK_USED_VAR(return_zero_for_nodep);
    mgb_throw(MegBrainError, "grad disabled at compile time");
#endif
}

SymbolVar cg::current_grad_target(ComputingGraph &graph) {
#if MGB_ENABLE_GRAD
    auto var = ComputingGraphImpl::downcast(&graph)->grad_manager(
            ).current_grad_target();
    mgb_throw_if(!var, GraphError, "current_grad_target() called outside "
            "grad computing environment");
    return var;
#else
    MGB_MARK_USED_VAR(graph);
    mgb_throw(MegBrainError, "grad disabled at compile time");
#endif
}

SymbolVarArray cg::get_dest_vars_with_extra_deps(
        const SymbolVarArray& dest_vars, SpecialOprStat* sopr_stat) {
    return ExtraDependencyMerger{sopr_stat}.add(dest_vars);
}

namespace {

SymbolVarArray replace_vars_internal(
        const SymbolVarArray& dest,
        thin_function<void(OperatorNodeBase*,
                gopt::SubGraph::Rewriter&)> on_opr) {
    if (dest.empty()) {
        return dest;
    }

    // check that they belong to the same graph
    mgb_assert(dest[0].node());
    auto og = dest[0].node()->owner_graph();
    for (auto i : dest) {
        mgb_assert(i.node() && i.node()->owner_graph() == og);
    }

    auto dest_with_extra_deps = get_dest_vars_with_extra_deps(dest);

    // do the replace
    gopt::SubGraph graph{dest_with_extra_deps};
    auto rewriter = graph.make_rewriter();
    graph.iter([&](OperatorNodeBase* opr){ on_opr(opr, rewriter); });

    auto new_og = rewriter.get_var(dest[0].node())->owner_graph();
    auto &&old_extra_vardeps = og->options().extra_vardeps,
         &&new_extra_vardeps = new_og->options().extra_vardeps;
    auto on_opr_replace_dep = [&](OperatorNodeBase* opr) {
        for (auto i : opr->output()) {
            auto new_node = rewriter.get_var(i);
            auto iter = old_extra_vardeps.find(i);
            if (iter == old_extra_vardeps.end())
                continue;

            if (new_node == i) {
                for (const auto& dep : iter->second) {
                    auto new_dep = rewriter.get_var(dep);
                    mgb_assert(dep == new_dep,
                               "var %s is not replaced, but its extra "
                               "dependency %s is replaced by %s ",
                               cg::dump_var_info({i}).c_str(),
                               cg::dump_var_info({dep}).c_str(),
                               cg::dump_var_info({new_dep}).c_str());
                }
            } else {
                auto& new_deps = new_extra_vardeps[new_node];
                for (const auto& dep : iter->second) {
                    new_deps.push_back(rewriter.get_var(dep));
                }
            }
        }
    };

    if (dest_with_extra_deps.size() != dest.size())
        graph.iter(on_opr_replace_dep);

    rewriter.apply_inplace();
    auto ret = graph.endpoint_vars();
    ret.resize(dest.size());
    return ret;
}
} //namespace

SymbolVarArray cg::replace_oprs(
        const SymbolVarArray& dest,
        const ThinHashMap<OperatorNodeBase*, OperatorNodeBase*>& oprmap) {
    if (oprmap.empty() || dest.empty()) {
        return dest;
    }

    mgb_assert(dest[0].node());
    auto graph = dest[0].node()->owner_graph();
    for (auto i : dest) {
        mgb_assert(i.node() && i.node()->owner_graph() == graph,
                   "Dest should all be in same graph");
    }
    for (auto&& i : oprmap) {
        mgb_assert(i.first->owner_graph() == graph &&
                           i.second->owner_graph() == graph,
                   "Original and dest operators in oprmap should all be in "
                   "same graph");
    }

    ThinHashMap<SymbolVar, SymbolVar> varmap;
    for (auto&& p : oprmap) {
        const auto& outputs0 = p.first->usable_output();
        const auto& outputs1 = p.second->usable_output();
        mgb_assert(outputs0.size() == outputs1.size(),
                   "Number of outputs differ: old operator %s has %zu outputs, "
                   "while new operator %s has %zu outputs.",
                   p.first->name().c_str(), outputs0.size(),
                   p.second->name().c_str(), outputs1.size());
        for (size_t i = 0; i < outputs0.size(); i++) {
            varmap[outputs0[i]] = outputs1[i];
        }
    }
    return replace_vars(dest, varmap);
}

SymbolVarArray cg::replace_vars(
        const SymbolVarArray& dest,
        const ThinHashMap<SymbolVar, SymbolVar>& varmap) {
    if (varmap.empty())
        return dest;
    auto og = dest[0].node()->owner_graph();
    for (auto&& i : varmap) {
        mgb_assert(i.first.node() && i.second.node() &&
                   i.first.node()->owner_graph() == og &&
                   i.second.node()->owner_graph() == og);
    }
    auto on_opr = [&](OperatorNodeBase* opr,
            gopt::SubGraph::Rewriter& rewriter) {
        for (auto i : opr->output()) {
            auto viter = varmap.find(i);
            if (viter != varmap.end()) {
                rewriter.replace_var(i, viter->second.node(), nullptr);
            }
        }
        rewriter.auto_replace_outputs(opr);
    };
    return replace_vars_internal(dest, on_opr);
}

SymbolVarArray cg::replace_vars_comp_graph(
    const SymbolVarArray &dest, ComputingGraph* new_graph) {
    ComputingGraph *orig_graph = dest[0].node()->owner_graph();
    mgb_assert(new_graph != orig_graph);
    auto on_opr = [&](OperatorNodeBase* opr,
            gopt::SubGraph::Rewriter& rewriter) {
        OperatorNodeBase* new_opr;
        if (opr->input().size()) {
            rewriter.auto_replace_outputs(opr);
        } else {
            mgb_assert(opr->owner_graph() != new_graph);
            new_opr = serialization::copy_opr_shallow(
                    *opr, {}, opr->config(), {new_graph});
            auto &&out0 = opr->output(), &&out1 = new_opr->output();
            mgb_assert(out0.size() == out1.size());
            for (size_t i = 0; i < out0.size(); ++ i) {
                rewriter.replace_var(out0[i], out1[i], "replace comp graph.");
            }
        }
    };
    return replace_vars_internal(dest, on_opr);
}

SymbolVarArray cg::find_h2d(const SymbolVarArray& dest) {
    mgb_assert(!dest.empty());
    SymbolVarArray h2d;
    auto on_opr = [&](OperatorNodeBase* opr) {
        if (opr->same_type<opr::Host2DeviceCopy>()) {
            h2d.emplace_back(opr->output(0));
        }
    };

    // check that they belong to the same graph
    mgb_assert(dest[0].node());
    auto og = dest[0].node()->owner_graph();
    for (auto i : dest) {
        mgb_assert(i.node() && i.node()->owner_graph() == og);
    }

    auto dest_with_extra_deps = get_dest_vars_with_extra_deps(dest);

    gopt::SubGraph graph{dest_with_extra_deps};
    graph.iter([&](OperatorNodeBase* opr){ on_opr(opr); });

    return h2d;
}

OperatorNodeBase* cg::get_opr_root_source_opr(OperatorNodeBase *opr) {
    auto &&attr = opr->node_prop().attribute();
    if (!attr.src_opr)
        return opr;
    auto orig = attr.src_opr;
    mgb_assert(orig != opr);
    return attr.src_opr = get_opr_root_source_opr(orig);
}

cg::MemPlanIntersectionType cg::get_mem_plan_intersection_type(
        VarNode* a, VarNode *b) {
    auto &&m0 = a->mem_plan(), &&m1 = b->mem_plan();
    if (&m0.chunk() != &m1.chunk())
        return MemPlanIntersectionType::DISJOINT;

    auto get_real_span = [](const MemAllocPlan &p) {
        auto span = p.layout().span();
        return std::make_pair(span.low_byte + p.offset_in_chunk_byte(),
                span.high_byte + p.offset_in_chunk_byte());
    };
    auto s0 = get_real_span(m0), s1 = get_real_span(m1);
    if (s0.first == s1.first && s0.second == s1.second)
        return MemPlanIntersectionType::IDENTICAL;
    if (s0.second <= s1.first || s1.second <= s0.first)
        return MemPlanIntersectionType::DISJOINT;
    return MemPlanIntersectionType::OVERLAP;
}

void cg::request_fwd_in2out_writable_if_no_mem_ovelap(
        OperatorNodeBase *opr, size_t inp, size_t out) {
    auto ivar = opr->input(inp), ovar = opr->output(out);
    if (is_static_var_storage(ivar) != is_static_var_storage(ovar)) {
        // If ovar is dynamic but there are other outputs of opr with static
        // storage, this function would be called during the static allocation
        // phase, and get_mem_plan_intersection_type() would fail.
        // So we just return here
        return;
    }

    auto &&dep_map = opr->node_prop().dep_map();
    using NP = OperatorNodeBase::NodeProp;
    mgb_assert(NP::is_device_value_dep(dep_map.at(ivar)));

    if (!ivar->layout().is_contiguous())
        return;

    using IT = MemPlanIntersectionType;
    for (size_t i = 0; i < opr->input().size(); ++ i) {
        auto iv = opr->input()[i];
        if (i != inp && NP::is_device_value_dep(dep_map.at(iv)) &&
                get_mem_plan_intersection_type(iv, ivar) != IT::DISJOINT) {
            return;
        }
    }
    ovar->set_fwd_in2out_writable(ivar);
}

void cg::add_workspace_output(OperatorNodeBase *opr) {
    opr->add_output("workspace")
        ->add_flag(VarNode::Flag::VOLATILE_CONTENT)
        .add_flag(VarNode::Flag::ALLOW_EMPTY_SHAPE)
        .dtype(dtype::Byte());
}

void cg::copy_shape_to_tensor_value(
        DeviceTensorND &dest, const TensorShape &shp) {

    dest.comp_node(CompNode::default_cpu()).
        dtype(dtype::Int32()).
        resize({std::max<size_t>(1, shp.ndim)});
    auto ptr = dest.ptr<dt_int32>();
    if (!shp.ndim)
        ptr[0] = 0;
    else {
        for (size_t i = 0; i < shp.ndim; i ++)
            ptr[i] = shp.shape[i];
    }
}

void cg::copy_tensor_value_to_shape(
        TensorShape &dest, const DeviceTensorND &val) {
    constexpr size_t MAX_DT_SIZE = 4;
    mgb_assert(val.dtype().size() <= MAX_DT_SIZE);

    mgb_assert(val.shape().ndim == 1, "shape tensor must be 1-dim, got %s",
               val.shape().to_string().c_str());
    mgb_assert(val.comp_node().device_type() == CompNode::DeviceType::CPU);
    dest.ndim = val.shape(0);
    mgb_assert(dest.ndim <= TensorShape::MAX_NDIM);
    auto vptr = val.raw_ptr();
    dt_byte contig[MAX_DT_SIZE * TensorShape::MAX_NDIM];
    if (val.layout().stride[0] != 1) {
        auto dst = contig;
        auto dst_strd = val.dtype().size();
        auto src = val.raw_ptr();
        auto src_strd = val.layout().stride[0] * dst_strd;
        for (size_t i = 0; i < dest.ndim; ++ i) {
            memcpy(dst, src, dst_strd);
            dst += dst_strd;
            src += src_strd;
        }
        vptr = contig;
    }
    static_cast_dtype_safe(dest.shape, val.dtype(), vptr, dest.ndim);
}

SymbolVar cg::var_from_tensor_shape(
        ComputingGraph &graph, const OperatorNodeConfig &config,
        const char *opr_name, const TensorShape &shape) {
    auto cn = config.get_single_comp_node();
    mgb_throw_if(!cn.valid(), GraphError,
            "must specify comp node in %s config", opr_name);
    DeviceTensorND dv;
    copy_shape_to_tensor_value(dv, shape);
    HostTensorND hv{cn};
    hv.copy_from(dv);
    return opr::ImmutableTensor::make(graph, hv);
}

/* =================== DepOprIter =================== */
void cg::DepOprIter::push_stack(OperatorNodeBase* opr) {
    if (m_visited.insert(opr).second) {
        if (m_extra_dep) {
            auto it = m_extra_dep->find(opr);
            if (it != m_extra_dep->end()) {
                m_stack.push_back({opr, opr->input().data(), it->second.data(),
                                   0, opr->input().size(), it->second.size()});
                return;
            }
        }
        m_stack.push_back(
                {opr, opr->input().data(), nullptr, 0, opr->input().size(), 0});
    }
}

void cg::DepOprIter::add(OperatorNodeBase *dest) {
    if (!m_owner_graph) {
        m_owner_graph = dest->owner_graph();
    } else {
        mgb_assert(m_owner_graph == dest->owner_graph(),
                "dest oprs belong to different graphs");
    }
    push_stack(dest);
    while (!m_stack.empty()) {
        auto &&frame = m_stack.back();
        if (frame.inp_idx == frame.nr_input + frame.nr_extra_dep) {
            m_cb(frame.opr);
            m_stack.pop_back();
        } else {
            VarNode* inp = nullptr;
            if (frame.inp_idx < frame.nr_input) {
                inp = frame.inputs[frame.inp_idx ++];
            } else {
                inp = frame.extra_deps[frame.inp_idx - frame.nr_input];
                frame.inp_idx++;
            }
            push_stack(inp->owner_opr());
        }
    }
}


/* =================== InterGraphVarTransformer =================== */

MGB_TYPEINFO_OBJ_IMPL(InterGraphVarTransformer);

void InterGraphVarTransformer::register_to(ComputingGraph *dest,
        const ComputingGraph *src, const TransFunc &trans) {
    mgb_assert(dest && src && trans);
    mgb_assert(dest->id() > src->id(),
            "inter-graph trans only allowed from old graph to new graph");
    auto mk = []() {
        return std::shared_ptr<InterGraphVarTransformer>(
                new InterGraphVarTransformer);
    };
    auto ptr = dest->options().user_data.
        get_user_data_or_create<InterGraphVarTransformer>(mk);
    mgb_assert(!ptr->m_trans_func, "InterGraphVarTransformer on graph #%zu{%p} "
            "already registered", dest->id(), dest);
    ptr->m_graph_dest = dest;
    ptr->m_graph_src = src;
    ptr->m_trans_func = trans;
}

const InterGraphVarTransformer*
InterGraphVarTransformer::get(const ComputingGraph &graph) {
    auto ret = graph.options().user_data.get_user_data<
        InterGraphVarTransformer>();
    if (!ret.second)
        return nullptr;
    mgb_assert(ret.second == 1);
    return ret.first[0];
}

VarNode* InterGraphVarTransformer::trans(VarNode *src) const {
    if (src->owner_graph() != m_graph_src) {
        auto strans = get(*m_graph_src);
        mgb_throw_if(!strans, GraphError,
                "no InterGraphVarTransformer registered for var %s, "
                "which belongs to graph #%zu{%p}",
                dump_var_info({src}).c_str(),
                src->owner_graph()->id(), src->owner_graph());
        src = strans->trans(src);
    }
    auto ret = m_trans_func(src);
    mgb_assert(ret && ret->owner_graph() == m_graph_dest);
    return ret;
}

/* =================== ExtraDependencyMerger =================== */
ExtraDependencyMerger::ExtraDependencyMerger(SpecialOprStat* sopr_stat)
        : m_sopr_stat{sopr_stat}, m_opr_iter{[this](OperatorNodeBase* opr) {
              on_opr(opr);
          }} {}

ExtraDependencyMerger::~ExtraDependencyMerger() = default;

void ExtraDependencyMerger::on_opr(OperatorNodeBase* opr) {
    if (!m_owner_graph) {
        m_owner_graph = opr->owner_graph();
    }
    mgb_assert(m_owner_graph == opr->owner_graph(),
               "owner graph changes in ExtraDependencyMerger; opr: %s{%s}",
               opr->cname(), opr->dyn_typeinfo()->name);
    auto&& extra_deps = m_owner_graph->options().extra_vardeps;
    auto sopr_stat = m_sopr_stat;
    MGB_MARK_USED_VAR(sopr_stat);
    auto&& new_deps = m_new_deps;
    for (auto i : opr->output()) {
        auto&& iter = extra_deps.find(i);
        if (iter != extra_deps.end()) {
            new_deps.insert(new_deps.end(), iter->second.begin(),
                            iter->second.end());
        }
#if !MGB_BUILD_SLIM_SERVING && MGB_ENABLE_GRAD
        if (sopr_stat && opr->same_type<opr::VirtualGrad>()) {
            sopr_stat->has_virtual_grad = true;
        }
#endif
        if (sopr_stat && opr->same_type<opr::ShapeHint>()) {
            sopr_stat->has_shape_hint = true;
        }
    }
}

SymbolVarArray& ExtraDependencyMerger::add(const SymbolVarArray& vars) {
    m_result.reserve(m_result.size() + vars.size());
    for (auto&& i : vars) {
        m_result.push_back(i);
        m_opr_iter.add(i);
    }
    while (!m_new_deps.empty()) {
        auto opr = m_new_deps.back()->owner_opr();
        m_new_deps.pop_back();
        if (!m_opr_iter.visited(opr)) {
            m_opr_iter.add(opr);
            m_result.push_back(opr->output(0));
        }
    }
    return m_result;
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
