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

    // check if there is reduce or dimshuffle opr
    cg::DepOprIter{[this](cg::OperatorNodeBase* opr) {
        if (opr->same_type<opr::Reduce>()) {
            m_feature_bits |= JITFeatureBits::REDUCE;
        }
        if (opr->same_type<opr::Dimshuffle>()) {
            m_feature_bits |= JITFeatureBits::DIMSHUFFLE;
        }
    }}.add(internal_graph->output());
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

    auto get_dimshuffled_layout = [](const TensorLayout& ily, int32_t* pattern,
                                 size_t pattern_len) {

        TensorLayout oly{ily.dtype};
        oly.ndim = pattern_len;

        bool input_used[TensorLayout::MAX_NDIM] = {0};
        for (uint32_t idx = 0; idx < pattern_len; ++idx) {
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

    // DFS to make sure traverse the dimshuffles in one branch
    std::unordered_set<VarNode*> visited;
    std::vector<OperatorNodeBase*> stack(0);
    std::vector<uint8_t> idx(0);  // input index
    stack.push_back(m_internal_graph->output()->owner_opr());
    idx.push_back(0);

    while (!stack.empty()) {
        if (idx.back() < stack.back()->input().size() &&
            !visited.count(stack.back()->input(idx.back()))) {
            visited.insert(stack.back()->input(idx.back()));
            stack.push_back(stack.back()->input(idx.back())->owner_opr());
            if (stack.back()->same_type<jit::JITPlaceholder>()) {
                auto jitph = gopt::try_cast_as_op<JITPlaceholder>(stack.back());
                size_t input_id = jitph->input_id();
                auto&& input = m_args.inputs[input_id];

                for (int i = stack.size() - 1; i >= 0; --i) {
                    if (stack[i]->same_type<opr::Dimshuffle>()) {
                        auto param =
                                stack[i]->cast_final_safe<opr::Dimshuffle>()
                                        .param();

                        mgb_assert(input.layout.ndim == param.ndim,
                                   "input ndim mismatch for Dimshuffle: "
                                   "expect=%u "
                                   "actual=%zu",
                                   param.ndim, input.layout.ndim);
                        auto dimshuffled_layout = get_dimshuffled_layout(
                                input.layout, param.pattern, param.pattern_len);
                        input.layout = dimshuffled_layout;
                    }
                }

                stack.pop_back();
                ++idx.back();
            } else {
                idx.push_back(0);
            }
        } else {
            stack.pop_back();
            idx.pop_back();
            if (!stack.empty())
                ++idx.back();
        }
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
    do_dimshuffle();

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
    if (opr.compiler()->property().feature_bits & JITFeatureBits::REDUCE) {
        // expand the gradient graph into the original graph to handle bcast
        // oprs
        ThinHashMap<VarNode*, VarNode*> old2new;
        VarNodeArray new_inp;
        auto on_opr = [&old2new, &grad_inputs,
                       &new_inp](cg::OperatorNodeBase* opr) {
            if (auto ph = gopt::try_cast_as_op<JITPlaceholder>(opr)) {
                old2new[opr->output(0)] = grad_inputs.at(ph->input_id());
                return;
            }
            if (auto imm = gopt::try_cast_as_op<opr::ImmutableTensor>(opr)) {
                HostTensorND hval{grad_inputs[0]->comp_node()};
                hval.copy_from(imm->value()).sync();
                old2new[opr->output(0)] =
                        opr::ImmutableTensor::make(*opr->owner_graph(), hval)
                                .node();
                return;
            }
            new_inp.clear();
            for (auto inp : opr->input()) {
                new_inp.push_back(old2new.at(inp));
            }
            auto new_opr = serialization::copy_opr_shallow(*opr, new_inp);
            old2new[opr->output(0)] = new_opr->output(0);
        };
        cg::DepOprIter{on_opr}.add(gx.node());
        return old2new.at(gx.node());
    } else {
        PlaceholderArray placeholders = fwd_igraph_ptr->placeholders();
        for (SymbolVar i : {output_ph, og_ph}) {
            placeholders.push_back(
                    &i.node()->owner_opr()->cast_final_safe<JITPlaceholder>());
        }
        for (size_t i = 0; i < placeholders.size(); ++i) {
            if (gx.node() == placeholders[i]->output(0)) {
                return grad_inputs[i];
            }
        }
        auto grad_ig = std::make_shared<InternalGraph>(
                gx.node(), fwd_igraph_ptr->shape_infer(), nullptr,
                std::move(placeholders));
        auto grad_jit = JITExecutor::make(grad_ig, grad_inputs);

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
