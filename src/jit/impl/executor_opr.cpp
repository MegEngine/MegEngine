/**
 * \file src/jit/impl/executor_opr.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/jit/executor_opr.h"
#include "megbrain/common.h"
#include "megbrain/comp_node_env.h"
#include "megbrain/gopt/framework.h"
#include "megbrain/graph/grad_impl.h"
#include "megbrain/graph/helper.h"
#include "megbrain/jit/compiler.h"
#include "megbrain/jit/param_elem_visitor.h"
#include "megbrain/jit/placeholder_opr.h"
#include "megbrain/opr/basic_arith_wrapper.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/opr/utility.h"
#include "megbrain/utils/hash.h"
#include "megbrain/serialization/opr_shallow_copy.h"

#if MGB_JIT

using namespace mgb;
using namespace jit;

using CPFlag = Compiler::Property::Flag;
/* =================== Fusion ==================== */

MGB_DYN_TYPE_OBJ_FINAL_IMPL(JITExecutor);
JITExecutor::JITExecutor(const InternalGraphPtr& internal_graph,
                         const VarNodeArray& inputs,
                         const OperatorNodeConfig& config)
        : Super(internal_graph->output()->owner_graph(), config,
                ssprintf("JIT-Fusion{%zu}",
                         internal_graph->placeholders().size()),
                inputs),
          m_internal_graph{internal_graph},
          m_compiler{Compiler::get(*inputs[0]->owner_graph(),
                                   inputs[0]->comp_node())} {
    for (auto inp : inputs) {
        add_input({inp});
    }
    m_input_broadcastable.resize(inputs.size());
    auto&& placeholders = m_internal_graph->placeholders();
    mgb_assert(placeholders.size() == inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        mgb_assert(placeholders[i]->output(0) != internal_graph->output());
        if (placeholders[i]->is_host_value_shape_input() ||
            input()[i]
                    ->owner_opr()
                    ->same_type<opr::MarkNoBroadcastElemwise>()) {
            m_input_broadcastable[i] = false;
        } else {
            m_input_broadcastable[i] = true;
        }
    }
    if (inputs.size() == 1) {
        m_input_broadcastable[0] = false;
    } else {
        Maybe<size_t> non_scalar;
        for (size_t i = 0; i < input().size(); ++i) {
            if (placeholders[i]->is_host_value_shape_input())
                continue;
            if (!(cg::is_const_var_shape(input(i)) &&
                  input(i)->shape().is_scalar())) {
                if (non_scalar.valid()) {
                    non_scalar.invalidate();
                    break;
                }
                non_scalar = i;
            }
        }
        if (non_scalar.valid()) {
            // exactly one input is non-scalar
            m_input_broadcastable[non_scalar.val()] = false;
        }
    }
    add_output(None)->dtype(m_internal_graph->output()->dtype());
    add_equivalence_component<ScalarHash<void*>>(internal_graph->output());
    for (size_t i = 0, it = m_compiler->get_nr_workspace_outputs(this); i < it;
         ++i) {
        cg::add_workspace_output(this);
    }

    // check if output of internal_graph is depend on all placeholders
    size_t nr_placeholders = internal_graph_ptr()->placeholders().size();
    std::vector<bool> used(nr_placeholders, false);
    // check if there is reduce or dimshuffle opr
    cg::DepOprIter{[this, nr_placeholders, &used](cg::OperatorNodeBase* opr) {
        if (opr->same_type<opr::Reduce>()) {
            m_feature_bits |= JITFeatureBits::REDUCE;
        }
        if (opr->same_type<opr::Dimshuffle>()) {
            m_feature_bits |= JITFeatureBits::DIMSHUFFLE;
        }
        if (auto ph = opr->try_cast_final<JITPlaceholder>()) {
            mgb_assert(ph->input_id() < nr_placeholders,
                "bad placeholders %s in JITExecutor %s",
                ph->cname(), cname());
            used[ph->input_id()] = true;
        }
    }}.add(internal_graph->output());

    for (size_t i = 0; i < nr_placeholders; ++ i) {
        mgb_assert(used[i],
            "placeholder %s is not depended on the output of %s",
            internal_graph_ptr()->placeholders()[i]->cname(), cname());
    }

    if (has_dimshuffle()) {
        prepare_dimshuffle();
    }
}

void JITExecutor::add_input_layout_constraint() {
    if (m_compiler->property().contain_flag(CPFlag::NEED_INPUT_CONTIG)) {
        for (auto i : input()) {
            i->add_layout_constraint_contiguous();
        }
    } else {
        for (auto i : input()) {
            i->add_layout_constraint_monotone();
        }
    }
}

void JITExecutor::init_output_mem_plan(bool dynamic) {
    Super::init_output_mem_plan(dynamic);
    m_args.need_update = true;
}

void JITExecutor::mem_plan_fwd_in2out_writable() {
    //! currently mem fwd only support elemwise fusion
    if (m_feature_bits != JITFeatureBits::NONE) return;
    mixin_mem_plan_fwd_in2out_writable(*this);
}


SymbolVar JITExecutor::make(const InternalGraphPtr& internal_graph,
                            const VarNodeArray& inputs,
                            const OperatorNodeConfig& config) {
    return internal_graph->output()
            ->owner_graph()
            ->insert_opr(std::make_unique<JITExecutor>(internal_graph, inputs,
                                                       config))
            ->output(0);
}

void JITExecutor::init_output_static_infer_desc() {
    using namespace cg::static_infer;
    auto&& mgr = owner_graph()->static_infer_manager();
    mgr.register_shape_infer(
            output(0),
            ShapeInferDesc::make_identity(m_internal_graph->shape_infer()));
    m_compiler->init_workspace_size_infer(this);
    if (m_internal_graph->value_infer()) {
        mgr.register_value_infer(
                output(0),
                ValueInferDesc::make_identity(m_internal_graph->value_infer()));
    }
}

void JITExecutor::scn_do_execute() {
    if (m_executable == nullptr || m_args.need_update) {
        m_executable = m_compiler->compile(this);
    }
    m_executable->execute(this);
}

//! change the inputs which depend on dimshuffle opr, make sure dimshuffles
//! can be ignored
void JITExecutor::do_dimshuffle() {

    static auto get_dimshuffled_layout = [](const TensorLayout& ily,
            std::vector<int> pattern) {

        TensorLayout oly{ily.dtype};
        oly.ndim = pattern.size();

        bool input_used[TensorLayout::MAX_NDIM] = {0};
        for (uint32_t idx = 0; idx < pattern.size(); ++idx) {
            auto i = pattern[idx];
            if (i < 0) {
                oly.shape[idx] = 1;
                oly.stride[idx] = 1;
            } else {
                input_used[i] = true;
                oly.shape[idx] = ily.shape[i];
                oly.stride[idx] = ily.stride[i];
            }
        }

        for (size_t i = 0; i < ily.ndim; ++i) {
            mgb_assert(input_used[i] || ily.shape[i] == 1,
                       "non-1 dim discarded in Dimshuffle: ishp=%s dim=%zd",
                       static_cast<const TensorShape&>(ily).to_string().c_str(),
                       i);
        }
        return oly;
    };

    for (auto&& i : m_internal_graph->placeholders()) {
        auto&& input = m_args.inputs[i->input_id()];
        auto&& iter = m_jitph2dimshuffle.find(i);
        if (iter == m_jitph2dimshuffle.end()) continue;
        auto&& param = iter->second;
        mgb_assert(input.layout.ndim == param.second,
                    "input ndim mismatch for Dimshuffle: "
                    "expect=%u "
                    "actual=%zu",
                    param.second, input.layout.ndim);
        auto dimshuffled_layout = get_dimshuffled_layout(
                input.layout, param.first);
        input.layout = dimshuffled_layout;
    }
}

void JITExecutor::update_args() {
    m_args.outputs.clear();
    for (auto out : output()) {
        m_args.outputs.push_back({out, out->layout(), -1});
    }
    m_args.inputs.resize(input().size());

    auto is_host_value_shape_input = [this](size_t idx) {
        return m_internal_graph->placeholders()
                .at(idx)
                ->is_host_value_shape_input();
    };

    for (size_t i = 0; i < input().size(); i++) {
        auto&& dst_data = m_args.inputs[i];
        dst_data.from = input(i);
        dst_data.idx = i;
        if (is_host_value_shape_input(i)) {
            auto&& mgr = owner_graph()->static_infer_manager();
            auto&& shpval_inp_val = &mgr.infer_value(input(i));
            cg::copy_tensor_value_to_shape(dst_data.layout, *shpval_inp_val);
            dst_data.layout.dtype = {};
            for (size_t i = 0; i < dst_data.layout.ndim; ++i) {
                dst_data.layout.stride[i] = 0;
            }
        } else {
            dst_data.layout = input(i)->layout();
        }
    }

    //! dimshuffle opr need to change the input.
    if (has_dimshuffle()) {
        do_dimshuffle();
    }

    if (m_compiler->property().contain_flag(CPFlag::NEED_INPUT_COLLAPSE)) {
        // collective collapse datum layout, try to reduce the output ndim
        opr::Elemwise::TensorLayoutPtrArray inp_layouts;
        inp_layouts.reserve(m_args.inputs.size());
        for (size_t i = 0; i < m_args.inputs.size(); i++) {
            if (!is_host_value_shape_input(i)) {
                inp_layouts.push_back(&m_args.inputs[i].layout);
            }
        }
        opr::Elemwise::broadcast_collective_collapse(inp_layouts,
                                                     &m_args.outputs[0].layout);
    }

    // compute and update hash
    XXHash hstate;

    //  update layout info
    auto prop = m_compiler->property();
    if (prop.contain_flag(CPFlag::BIND_NDIM | CPFlag::BIND_SHAPE)) {
        mgb_assert(prop.contain_flag(CPFlag::BIND_NDIM),
                   "BIND_NDIM must be set if bind_shape is set");
        std::vector<size_t> buf;
        buf.reserve(1024);
        buf.push_back(m_args.inputs.size());
        for (auto&& i : m_args.inputs) {
            buf.push_back(i.layout.ndim);
            if (prop.contain_flag(CPFlag::BIND_SHAPE)) {
                for (size_t j = 0; j < i.layout.ndim; ++j) {
                    buf.push_back(i.layout[j]);
                }
            }
        }
        hstate.update(buf.data(), sizeof(buf[0]) * buf.size());
    }
    m_args.hash = hstate.digest();

    // update version number
    static std::atomic_uint_fast64_t global_version;
    m_args.version = global_version.fetch_add(1);

    m_args.need_update = false;
}

void JITExecutor::prepare_dimshuffle() {
    std::unordered_set<OperatorNodeBase*> visited;
    std::vector<OperatorNodeBase*> stack(0);
    std::vector<uint8_t> idx(0);  // input index
    using Param = DimshuffleParam;
    std::vector<Param> dimshuffle_stack;

    auto merge_dimshuffle = [&](const opr::Dimshuffle::Param& p) {
        if (dimshuffle_stack.empty()) {
            dimshuffle_stack.emplace_back();
            auto&& param = dimshuffle_stack.back();
            param.first.insert(param.first.end(), p.pattern, p.pattern + p.pattern_len);
            param.second = p.ndim;
        } else {
            // merge(p, src) -> param and it has performing dimshuffle(dimshuffle(x, p), src)
            // is equivalent to dimshuffle(x, param)
            dimshuffle_stack.emplace_back();
            auto&& param = dimshuffle_stack.back();
            auto&& src = dimshuffle_stack[dimshuffle_stack.size() - 2];
            mgb_assert(p.pattern_len == src.second);
            param.first.resize(src.first.size());
            for (size_t i = 0; i < src.first.size(); ++ i) {
                if (src.first[i] == -1) {
                    param.first[i] = -1;
                } else {
                    param.first[i] = p.pattern[src.first[i]];
                }
            }
            param.second = p.ndim;
        }
    };
    auto push_back = [&](cg::OperatorNodeBase* op) {
        mgb_assert(!op->same_type<jit::JITPlaceholder>());
        if (auto o = op->try_cast_final<opr::Dimshuffle>()) {
            merge_dimshuffle(o->param());
        }
        stack.push_back(op);
        idx.push_back(0);
    };
    auto pop_back = [&]() {
        auto&& op = stack.back();
        if (op->same_type<opr::Dimshuffle>()) {
            dimshuffle_stack.pop_back();
        }
        stack.pop_back();
        idx.pop_back();
    };

    push_back(m_internal_graph->output()->owner_opr());

    while (!stack.empty()) {
        if (idx.back() < stack.back()->input().size()) {
            auto cur_opr = stack.back()->input(idx.back())->owner_opr();
            if (visited.insert(cur_opr).second) {
                if (auto jitph = cur_opr->try_cast_final<jit::JITPlaceholder>()) {
                    if (!dimshuffle_stack.empty()) {
                        mgb_assert(
                            m_jitph2dimshuffle.emplace(jitph, dimshuffle_stack.back()).second,
                            "already visited JITPlaceholder %s",
                            jitph->cname());
                    }
                    ++ idx.back();
                } else {
                    push_back(cur_opr);
                }
            } else {
                ++ idx.back();
            }
        } else {
            pop_back();
            if (!stack.empty())
                ++ idx.back();
        }
    }
}

const JITExecutor::Args& JITExecutor::args() const {
    if (m_args.need_update) {
        const_cast<JITExecutor*>(this)->update_args();
    }
    return m_args;
}

bool JITExecutor::Args::operator==(const Args& rhs) const {
    auto&& lhs = *this;
    mgb_assert(!lhs.need_update && !rhs.need_update);
    if (lhs.hash != rhs.hash) {
        return false;
    }
    if (lhs.version == rhs.version) {
        return true;
    }
    if (lhs.outputs.size() != rhs.outputs.size())
        return false;
    if (lhs.inputs.size() != rhs.inputs.size())
        return false;

    auto prop = owner->m_compiler->property();

    if (prop.contain_flag(CPFlag::BIND_NDIM | CPFlag::BIND_SHAPE)) {
        bool (*chk_layout)(const TensorLayout&, const TensorLayout&);
        if (prop.contain_flag(CPFlag::BIND_SHAPE)) {
            chk_layout = [](const TensorLayout& lhs, const TensorLayout& rhs) {
                return lhs.eq_shape(rhs);
            };
        } else {
            chk_layout = [](const TensorLayout& lhs, const TensorLayout& rhs) {
                return lhs.ndim == rhs.ndim;
            };
        }
        for (size_t i = 0; i < lhs.inputs.size(); i++) {
            if (!chk_layout(lhs.inputs[i].layout, rhs.inputs[i].layout))
                return false;
        }
        for (size_t i = 0; i < lhs.outputs.size(); i++) {
            if (!chk_layout(lhs.outputs[i].layout, rhs.outputs[i].layout))
                return false;
        }
    }

    // elect a common version so next check can be fast
    lhs.version = rhs.version = std::min(lhs.version, rhs.version);

    return true;
}

JITExecutor::NodeProp* JITExecutor::do_make_node_prop() const {
    auto ret = Super::do_make_node_prop();
    using DepType = NodeProp::DepType;
    SmallVector<DepType> dt(input().size());
    auto&& placeholders = internal_graph().placeholders();
    for (size_t i = 0; i < dt.size(); ++i) {
        dt[i] = placeholders[i]->is_host_value_shape_input()
                        ? DepType::HOST_VALUE
                        : DepType::DEV_VALUE;
    }
    ret->reset_dep_type(input(), dt);
    return ret;
}

megdnn::TensorShape JITExecutor::broadcasted_input_shape() const {
    megdnn::TensorShapeArray inp_shps;
    megdnn::TensorShape brdcast_shp;
    auto placeholders = m_internal_graph->placeholders();
    for (auto ph : placeholders) {
        if (!ph->is_host_value_shape_input()) {
            inp_shps.push_back(input(ph->input_id())->shape());
        }
    }
    megdnn::Elemwise::deduce_shape(inp_shps, brdcast_shp);
    return brdcast_shp;
}


#if MGB_ENABLE_GRAD
namespace {
class InternalGraphRewriter {
    ThinHashMap<VarNode*, VarNode*> m_var_map;
    VarNode* m_dest_var;
    VarNodeArray m_new_inp;
    VarNode* get_var(VarNode* var) {
        auto&& iter = m_var_map.find(var);
        if (iter != m_var_map.end()) {
            return iter->second;
        }
        return var;
    }
public:
    InternalGraphRewriter(VarNode* dest_var)
        :m_dest_var{dest_var}{}
    void iter(thin_function<void(cg::OperatorNodeBase*)>&& cb) {
        m_var_map.clear();
        cg::DepOprIter{std::move(cb)}.add(m_dest_var->owner_opr());
        m_dest_var = get_var(m_dest_var);
    }
    VarNode* dest_var() {
        return m_dest_var;
    }
    void replace_var(VarNode* src, VarNode* dst) {
        // Note: do not perform var replacing recursively
        // when we extract used placeholders from internal graph, we don't
        // consider placeholder replacement pair (a to b), (b to c) as a
        // var replacing chain (a to b to c) but as a injective function
        // from (a, b) to (b, c)
        // in other cases, each var node would be passed as \p src or
        // \p dst at most once
        m_var_map[src] = dst;
    }
    void auto_replace_outputs(cg::OperatorNodeBase* opr) {
        // in JIT internal graph, output size of opr is always 1
        mgb_assert(opr->usable_output().size() == 1);
        m_new_inp.clear();
        bool need_replace = false;
        for (auto&& i : opr->input()) {
            auto inp = get_var(i);
            m_new_inp.push_back(inp);
            need_replace |= (inp != i);
        }
        if (need_replace) {
            auto new_op = serialization::copy_opr_shallow(*opr, m_new_inp);
            replace_var(opr->output(0), new_op->output(0));
        }
    }
};
} // anonymous namespace
MGB_IMPL_OPR_GRAD(JITExecutor) {
    VarNodeArray grad_inputs;
    for (auto input : opr.input())
        grad_inputs.push_back(input);
    mgb_assert(out_grad[0]);
    grad_inputs.push_back(opr.output(0));
    grad_inputs.push_back(out_grad[0]);
    auto fwd_igraph_ptr = opr.internal_graph_ptr();
    auto output_ph = JITPlaceholder::make(
            fwd_igraph_ptr->output(), fwd_igraph_ptr->placeholders().size());
    auto og_ph = JITPlaceholder::make(
            out_grad[0], fwd_igraph_ptr->placeholders().size() + 1);
    auto loss = opr::VirtualLoss::make({fwd_igraph_ptr->output()}, {og_ph});
    auto gx = cg::grad(loss, fwd_igraph_ptr->placeholders()[wrt_idx]->output(0),
                       false, false);
    if (!gx.node()) {
        return nullptr;
    }
    if (gx.node()->owner_opr()->same_type<opr::InvalidGrad>()) {
        return opr::InvalidGrad::make(opr, wrt_idx);
    }
    // early return if grad expression is single node
    for (size_t i = 0; i < fwd_igraph_ptr->placeholders().size(); ++i) {
        if (gx.node() == fwd_igraph_ptr->placeholders()[i]->output(0)) {
            return grad_inputs[i];
        }
    }
    if (gx.node() == og_ph.node()) {
        return out_grad[0];
    }
    if (gx.node() == fwd_igraph_ptr->output()) {
        return opr.output(0);
    }
    if (auto imm = gopt::try_cast_as_op<opr::ImmutableTensor>(gx.node()->owner_opr())) {
        HostTensorND hval{grad_inputs[0]->comp_node()};
        hval.copy_from(imm->value()).sync();
        return opr::ImmutableTensor::make(*imm->owner_graph(), hval).node();
    }

    // replace output var in internal graph with output placeholder, so
    // we could forward opr.output(computeed by forward JITExecutor) into
    // placeholder to avoid redundant computation
    InternalGraphRewriter rewriter{gx.node()};
    rewriter.iter([&rewriter, &fwd_igraph_ptr,
            &output_ph](cg::OperatorNodeBase* opr) {
        if (opr == fwd_igraph_ptr->output()->owner_opr()) {
            rewriter.replace_var(opr->output(0), output_ph.node());
            return;
        }
        rewriter.auto_replace_outputs(opr);
    });

    auto expand_into_origin_graph = [&rewriter](
        cg::OperatorNodeBase* opr, const VarNodeArray& grad_inputs) {
        if (auto ph = gopt::try_cast_as_op<JITPlaceholder>(opr)) {
            rewriter.replace_var(
                opr->output(0), grad_inputs.at(ph->input_id()));
            return;
        }
        if (auto imm = gopt::try_cast_as_op<opr::ImmutableTensor>(opr)) {
            HostTensorND hval{grad_inputs[0]->comp_node()};
            hval.copy_from(imm->value()).sync();
            rewriter.replace_var(opr->output(0),
                opr::ImmutableTensor::make(*opr->owner_graph(), hval).node());
            return;
        }
        rewriter.auto_replace_outputs(opr);
    };

    if (opr.compiler()->property().feature_bits & JITFeatureBits::REDUCE) {
        // expand the gradient graph into the original graph to handle bcast
        // oprs
        using namespace std::placeholders;
        rewriter.iter(std::bind(expand_into_origin_graph, _1,
                std::cref(grad_inputs)));
        return rewriter.dest_var();
    } else {
        VarNodeArray new_grad_inputs;
        PlaceholderArray placeholders;
        bool all_inp_const = true;
        // gx was not depend on all JITPlaceholders so we need to extract used
        // placeholders and build a new internal graph
        rewriter.iter([&rewriter, &grad_inputs, &new_grad_inputs,
                &placeholders, &all_inp_const](cg::OperatorNodeBase* opr) {
            if (auto ph = gopt::try_cast_as_op<JITPlaceholder>(opr)) {
                new_grad_inputs.push_back(grad_inputs[ph->input_id()]);
                auto new_ph = JITPlaceholder::make(
                        new_grad_inputs.back(), placeholders.size())
                        .node()->owner_opr();
                placeholders.push_back(new_ph->try_cast_final<JITPlaceholder>());
                mgb_assert(placeholders.back());
                rewriter.replace_var(opr->output(0), new_ph->output(0));
                if (!cg::is_const_var_value(new_grad_inputs.back())) {
                    all_inp_const = false;
                }
                return;
            }
            rewriter.auto_replace_outputs(opr);
        });
        if (all_inp_const) {
            // if all_inp_const, expand grad graph into origin graph by replace
            // placeholders with const inputs, so it could benefit from static
            // infer and const folding mechanism
            using namespace std::placeholders;
            rewriter.iter(std::bind(expand_into_origin_graph, _1,
                    std::cref(new_grad_inputs)));
            return rewriter.dest_var();
        }
        gx = rewriter.dest_var();

        auto shape_infer = fwd_igraph_ptr->shape_infer();
        if (opr.has_dimshuffle()) {
            auto&& iter = opr.dimshuffle_params().find(
                    fwd_igraph_ptr->placeholders()[wrt_idx]);
            if (iter != opr.dimshuffle_params().end()) {
                auto&& pattern = iter->second.first;
                auto&& ndim = iter->second.second;
                std::vector<int> back(ndim, -1);
                for (size_t i = 0; i < pattern.size(); i ++) {
                    // outdim[i] is indim[j]
                    auto j = pattern[i];
                    if (j >= 0) {
                        mgb_assert(back[j] == -1,
                                "taking grad for Dimshuffle with duplicated "
                                "input axis unsupported");
                        back[j] = i;
                    }
                }
                shape_infer = opr::Dimshuffle::make(shape_infer, back, pattern.size()).node();
            }
        }
        auto grad_ig = std::make_shared<InternalGraph>(
                gx.node(), shape_infer, nullptr,
                std::move(placeholders));
        auto grad_jit = JITExecutor::make(grad_ig, new_grad_inputs);

        if (opr.input_broadcastable()[wrt_idx]) {
            grad_jit = opr::reduce_sum(
                    grad_jit, opr::GetVarShape::make(opr.input(wrt_idx)));
        }
        return grad_jit.node();
    }
}
#endif  // MGB_ENABLE_GRAD

#endif  // MGB_JIT

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
