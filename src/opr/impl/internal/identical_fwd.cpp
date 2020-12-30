/**
 * \file src/opr/impl/internal/identical_fwd.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/opr/internal/identical_fwd.h"
#include "megbrain/opr/io.h"

using namespace mgb;
using namespace opr;
using namespace mixin;

void mixin::init_rt_force_dynamic_mem_alloc_imply_chain_for_dyn_pass_i2o(
        OperatorNodeBase &opr) {
    VarNode *valid_out = nullptr;
    for (auto i: opr.output()) {
        if (!i->contain_flag(VarNode::Flag::VOLATILE_CONTENT)) {
            mgb_assert(!valid_out);
            valid_out = i;
        }
    }
    mgb_assert(valid_out);

    for (auto i: opr.input()) {
        i->add_rt_force_dynamic_mem_alloc_imply_chain(valid_out);
    }
    valid_out->add_rt_force_dynamic_mem_alloc_imply_chain(opr.input(0));
}

/* ===================== FwdIn2OutWritableHelper  ===================== */
void FwdIn2OutWritableHelper::mixin_mem_plan_fwd_in2out_writable(
        OperatorNodeBase& opr) {
    auto&& inp = opr.input();
    auto isize = inp.size();
    std::vector<bool> have_conflict(isize, false);
    for (size_t i = 0; i < isize; ++i) {
        for (size_t j = i + 1; j < isize; ++j) {
            auto type = cg::get_mem_plan_intersection_type(inp[i], inp[j]);
            using Type = cg::MemPlanIntersectionType;
            bool overlap = type == Type::OVERLAP;
            bool self_fwd = type == Type::IDENTICAL &&
                            (!inp[i]->layout().is_contiguous() ||
                             !inp[j]->layout().is_contiguous());
            if (overlap || self_fwd) {
                have_conflict[i] = true;
                have_conflict[j] = true;
            }
        }
    }
    auto o = opr.output(0);
    for (size_t idx = 0; idx < isize; ++ idx) {
        auto i = inp[idx];
        // equal shape means no broadcast
        if (!have_conflict[idx] && o->shape().eq_shape(i->shape()) &&
            o->dtype().enumv() == i->dtype().enumv() &&
            i->layout().is_contiguous())
            o->set_fwd_in2out_writable(i);
    }
}

/* ===================== ReadonlyFwdHelper ===================== */

void ReadonlyFwdHelper::mixin_rofwd_init_mem_plan(OperatorNodeBase &opr) {
    mgb_assert(m_rofwd_subspec.layout().eq_shape(opr.output(0)->shape()),
            "shape mismatch in ReadonlyFwdHelper: "
            "inp=%s sub_spec=%s output=%s (this=%s)",
            opr.input(0)->shape().to_string().c_str(),
            m_rofwd_subspec.layout().to_string().c_str(),
            opr.output(0)->shape().to_string().c_str(),
            opr.dyn_typeinfo()->name);
    m_mem_fwd_success = opr.output(0)->set_fwd_in2out_readonly(
            opr.input(0), m_rofwd_subspec);
}

void ReadonlyFwdHelper::mixin_rofwd_execute(OperatorNodeBase &opr) {
    mgb_assert(m_rofwd_subspec.layout().ndim, "rofwd uninitialized");

    auto &&out = opr.output(0)->dev_tensor(),
         &&inp = opr.input(0)->dev_tensor();
    if (m_mem_fwd_success) {
        mgb_assert(inp.raw_ptr() + m_rofwd_subspec.offset_byte() ==
                out.raw_ptr() &&
                out.layout().eq_layout(m_rofwd_subspec.layout()));
    } else {
        out.copy_from_fixlayout(inp.sub(m_rofwd_subspec));
    }
}

/* ===================== ForwardInputToOutput ===================== */

class ForwardInputToOutput::MutableSrc : public UserDataContainer::UserData {
    MGB_TYPEINFO_OBJ_DECL;
    VarNode* m_var;

public:
    MutableSrc(CompNode cn, ComputingGraph* cg)
            : m_var{opr::Host2DeviceCopy::make(
                            *cg, std::make_shared<HostTensorND>(
                                         cn, TensorShape{1}, dtype::Float32{}))
                            .node()}

    {}
    VarNode* var() const { return m_var; }
};

MGB_TYPEINFO_OBJ_IMPL(ForwardInputToOutput::MutableSrc);

void ForwardInputToOutput::mixin_init_rt_force_dynamic_mem_alloc_imply_chain(
        OperatorNodeBase &opr) {
    VarNode *valid_out = nullptr;
    for (auto i: opr.output()) {
        if (!i->contain_flag(VarNode::Flag::VOLATILE_CONTENT)) {
            mgb_assert(!valid_out);
            valid_out = i;
        }
    }
    mgb_assert(valid_out);

    // There may be many inputs such as in opr::VirtualDep, but we only forward first one
    opr.input(0)->add_rt_force_dynamic_mem_alloc_imply_chain(valid_out);
    valid_out->add_rt_force_dynamic_mem_alloc_imply_chain(opr.input(0));
}

void ForwardInputToOutput::mixin_mem_plan_fwd_in2out_readonly(
        OperatorNodeBase& opr) {
    m_mem_fwd_success = opr.output(0)->set_fwd_in2out_readonly(
            opr.input(0),
            SubTensorSpec::make_from_layout(opr.input(0)->layout()));
}

void ForwardInputToOutput::set_ignore_side_effect() {
    mgb_assert(!m_static_infer_called,
               "can not call set_ignore_side_effect() after static infer has "
               "been used");
    m_ignore_side_effect = true;
}

cg::static_infer::ValueInferDesc
ForwardInputToOutput::mixin_get_static_infer_desc(OperatorNodeBase &opr) {
    using namespace cg::static_infer;
    auto infer_val = [](DeviceTensorND& dst, const InpVal& iv) {
        dst = iv.val[0].value();
        return true;
    };
    return {SourceType::DEP,{{opr.input(0), DepType::VALUE}},infer_val};
}

void ForwardInputToOutput::mixin_init_output_static_infer_desc(
        OperatorNodeBase& opr) {
    using namespace cg::static_infer;
    m_static_infer_called = true;

    auto&& mgr = opr.owner_graph()->static_infer_manager();
    auto ivar = opr.input(0), ovar = opr.output(0);
    mgr.register_shape_infer(ovar, ShapeInferDesc::make_identity(ivar));
    m_append_one_more_shape = false;
    ValueInferDesc desc = this->mixin_get_static_infer_desc(opr);
    if (!m_ignore_side_effect) {
        m_append_one_more_shape = ensure_not_replaced_by_const_folding(desc);
    }
    mgr.register_value_infer(ovar, desc);
}

bool ForwardInputToOutput::ensure_not_replaced_by_const_folding(
        cg::static_infer::ValueInferDesc& desc) {
    using namespace cg::static_infer;
    mgb_assert(!desc.deps.empty());
    VarNode* ivar = desc.deps[0].dest;
    auto graph = ivar->owner_graph();
    auto&& mgr = graph->static_infer_manager();

    for (auto&& i : desc.deps) {
        auto infer_type = mgr.get_infer_type(i.dest);
        if (i.type == DepType::VALUE) {
            if (infer_type.value != InferType::CONST) {
                return false;
            }
        } else {
            mgb_assert(i.type == DepType::SHAPE);
            if (infer_type.shape != InferType::CONST) {
                return false;
            }
        }
    }

    // all inputs are constant, so we add a mutable shape dep

    auto make_mutable_src = [graph, ivar]() {
        return std::make_shared<MutableSrc>(ivar->comp_node(), graph);
    };
    auto src = graph->options()
                       .user_data
                       .get_user_data_or_create<MutableSrc>(make_mutable_src)
                       ->var();
    desc.deps.push_back({src, DepType::SHAPE});
    return true;
}

void ForwardInputToOutput::mixin_scn_do_execute(OperatorNodeBase &opr) {
    auto &&odev = opr.output(0)->dev_tensor(),
         &&idev = opr.input(0)->dev_tensor();
    if (m_mem_fwd_success) {
        mgb_assert(odev.raw_ptr() == idev.raw_ptr());
    } else {
        odev.copy_from_fixlayout(idev);
    }
    scn_do_execute_finish(odev);
}

void ForwardInputToOutput::scn_do_execute_finish(const DeviceTensorND&) {}

void ForwardInputToOutput::register_stream_propagate_in2out(OperatorNodeBase &opr) {
    auto &&ovar = opr.output(0);
    auto&& mgr = ovar->owner_graph()->seq_comp_node_optimizer();
    using PropType = cg::SeqCompNodeOptimizer::StreamPropType;
    auto func = [](PropType& dst, const SmallVector<PropType>& inp) {
        dst = inp[0];
    };
    mgr.register_propagate_function(ovar, func);
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
