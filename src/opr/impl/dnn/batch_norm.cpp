/**
 * \file src/opr/impl/dnn/batch_norm.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/opr/dnn/batch_norm.h"
#include "megbrain/graph/grad_impl.h"
#include "megbrain/opr/basic_arith.h"
#include "megbrain/opr/io.h"
#include "megbrain/opr/tensor_manip.h"

#include "../internal/megdnn_opr_wrapper.inl"

using namespace mgb;
using namespace opr;

namespace mgb {
namespace opr {
namespace intl {
template <>
struct AutoAddWorkspaceNeedLimitGetter<megdnn::BNForward> {
    static constexpr bool val = true;
};

template <>
struct AutoAddWorkspaceNeedLimitGetter<megdnn::BNBackward> {
    static constexpr bool val = true;
};
}  // namespace intl
}  // namespace opr
}  // namespace mgb

MGB_DYN_TYPE_OBJ_FINAL_IMPL(BatchNormForward);

BatchNormForward::BatchNormForward(
        VarNode* x, VarNode* scale, VarNode* bias, VarNode* mean, VarNode* variance,
        const Param& param, const OperatorNodeConfig& config)
        : Super{x->owner_graph(),
                config,
                "batch_norm",
                {x, scale, bias, mean, variance}} {
    if (owner_graph()->options().no_force_inplace) {
        m_force_inplace = false;
    }

    if (m_force_inplace && param.fwd_mode == Param::FwdMode::TRAINING) {
        auto check_dest = [&](VarNode* dest) {
            auto dest_opr = dest->owner_opr();
            mgb_throw_if(
                    !(dest_opr->same_type<SharedDeviceTensor>() ||
                      dest_opr->same_type<VolatileSharedDeviceTensor>()),
                    GraphError,
                    "mean and variance in training mode BatchNorm must be"
                    "SharedDeviceTensor or VolatileSharedDeviceTensor;"
                    "got %s{%s} actually",
                    dest_opr->cname(), dest_opr->dyn_typeinfo()->name);
        };
        check_dest(mean);
        check_dest(variance);
    }

    init_megdnn_opr(*this, param);

    add_input({x, scale, bias, mean, variance});

    output(4)->add_flag(VarNode::Flag::ALLOW_EMPTY_SHAPE);  // reserve
    output(5)->add_flag(VarNode::Flag::ALLOW_EMPTY_SHAPE);
    // running mean/var
    if (param.fwd_mode == Param::FwdMode::INFERENCE) {
        auto mark_empty_var = [&](VarNode* var) {
            var->add_flag(VarNode::Flag::ALLOW_EMPTY_SHAPE)
                    .add_flag(VarNode::Flag::VOLATILE_CONTENT);
        };
        mark_empty_var(output(0));
        mark_empty_var(output(1));
    } else if (m_force_inplace) {
        output(0)->set_fwd_in2out_writable_force(input(3)).add_flag(
                VarNode::Flag::NO_MEM_RECLAIM);

        output(1)->set_fwd_in2out_writable_force(input(4)).add_flag(
                VarNode::Flag::NO_MEM_RECLAIM);
    }
}

BatchNormForward::BatchNormForward(
        VarNode* x, VarNode* scale, VarNode* bias, const Param& param,
        const OperatorNodeConfig& config)
        : Super{x->owner_graph(), config, "batch_norm", {x, scale, bias}} {
    init_megdnn_opr(*this, param);

    add_input({x, scale, bias});
    output(4)->add_flag(VarNode::Flag::ALLOW_EMPTY_SHAPE);  // reserve
    output(5)->add_flag(VarNode::Flag::ALLOW_EMPTY_SHAPE);
    auto mark_empty_var = [&](VarNode* var) {
        var->add_flag(VarNode::Flag::ALLOW_EMPTY_SHAPE)
                .add_flag(VarNode::Flag::VOLATILE_CONTENT);
    };
    mark_empty_var(output(0));
    mark_empty_var(output(1));
}

SymbolVarArray BatchNormForward::make(
        SymbolVar x, SymbolVar scale, SymbolVar bias, SymbolVar mean,
        SymbolVar variance, const Param& param, const OperatorNodeConfig& config) {
    auto&& out = x.node()->owner_graph()
                         ->insert_opr(std::make_unique<BatchNormForward>(
                                 x.node(), scale.node(), bias.node(), mean.node(),
                                 variance.node(), param, config))
                         ->output();
    SymbolVarArray ret(out.size());
    for (size_t i = 0; i < ret.size(); i++) {
        ret[i] = out[i];
    }
    return ret;
}

SymbolVarArray BatchNormForward::make(
        SymbolVar x, SymbolVar scale, SymbolVar bias, const Param& param,
        const OperatorNodeConfig& config) {
    auto&& out = x.node()->owner_graph()
                         ->insert_opr(std::make_unique<BatchNormForward>(
                                 x.node(), scale.node(), bias.node(), param, config))
                         ->output();
    SymbolVarArray ret(out.size());
    for (size_t i = 0; i < ret.size(); i++) {
        ret[i] = out[i];
    }
    return ret;
}

cg::OperatorNodeBase::NodeProp* BatchNormForward::do_make_node_prop() const {
    auto ret = Super::do_make_node_prop();
    ret->add_dep_type_existing_var(input(0), NodeProp::DepType::VALUE_ALLOW_EMPTY);
    if (need_stats() && m_force_inplace) {
        ret->add_flag(NodeProp::Flag::FORCE_UPDATE_INPUT_VAR);
    }
    return ret;
}

void BatchNormForward::scn_do_execute() {
    auto&& x = input(0)->dev_tensor();
    auto&& y = output(5)->dev_tensor();
    if (need_stats()) {
        auto &&o0 = output(0)->dev_tensor(), &&o1 = output(1)->dev_tensor(),
             &&i0 = input(3)->dev_tensor(), &&i1 = input(4)->dev_tensor();
        mgb_assert(o0.raw_ptr() && o1.raw_ptr());  // non-empty tensor
        mgb_assert(
                o0.comp_node() == i0.comp_node() && o1.comp_node() == i1.comp_node() &&
                o0.layout().eq_layout(i0.layout()) &&
                o1.layout().eq_layout(i1.layout()));
        if (!m_force_inplace) {
            if (o0.raw_ptr() != i0.raw_ptr()) {
                o0.copy_from_fixlayout(i0);
            }
            if (o1.raw_ptr() != i1.raw_ptr()) {
                o1.copy_from_fixlayout(i1);
            }
        } else {
            mgb_assert(o0.raw_ptr() == i0.raw_ptr() && o1.raw_ptr() == i1.raw_ptr());
        }
    }
    mgb_assert(x.layout().eq_layout(y.layout()));
    if (x.layout().is_empty()) {
        return;
    }
    mgb_assert(x.layout().is_contiguous() && y.layout().is_contiguous());
    auto scale = input(1)->dev_tensor().as_megdnn();
    auto bias = input(2)->dev_tensor().as_megdnn();
    megdnn::TensorND mean, variance;
    if (param().fwd_mode == Param::FwdMode::INFERENCE) {
        mean = input(3)->dev_tensor().as_megdnn();
        variance = input(4)->dev_tensor().as_megdnn();
    } else {
        mean = output(0)->dev_tensor().as_megdnn();
        variance = output(1)->dev_tensor().as_megdnn();
    }
    auto save_mean = output(2)->dev_tensor().as_megdnn();
    auto save_variance = output(3)->dev_tensor().as_megdnn();
    auto reserve = output(4)->dev_tensor().as_megdnn();
    auto workspace = intl::get_megdnn_workspace_from_var(output().back());
    megdnn_opr()->exec(
            x.as_megdnn(), scale, bias, mean, variance, save_mean, save_variance,
            reserve, y.as_megdnn(), workspace);
}

void BatchNormForward::add_input_layout_constraint() {
    mixin::megdnn_utils::add_input_layout_constraint_contig(*this);
}

void BatchNormForward::get_output_var_shape(
        const TensorShapeArray& inp_shape, TensorShapeArray& out_shape) const {
    mgb_assert(
            inp_shape[0].ndim == 4 && inp_shape[0].ndim == 4 && inp_shape[1].ndim == 4,
            "expect input, scale and bias to be 4 dim tensor, but "
            "got input dim: %zu, scale dim: %zu, bias dim: %zu",
            inp_shape[0].ndim, inp_shape[1].ndim, inp_shape[2].ndim);

    size_t channel_idx;
    if (param().param_dim == Param::ParamDim::DIM_111C) {
        channel_idx = 3;
    } else {
        channel_idx = 1;
    }
    size_t inp_c = inp_shape[0][channel_idx], scale_c = inp_shape[1][channel_idx],
           bias_c = inp_shape[2][channel_idx];
    mgb_assert(
            inp_c == scale_c && inp_c == bias_c,
            "inconsistent channel size, input chennel: %zu, scale channel: %zu, bias "
            "channel: %zu",
            inp_c, scale_c, bias_c);

    out_shape[5] = inp_shape[0];
    for (size_t i = 0; i < 4; ++i) {
        out_shape[i] = inp_shape[1];
    }
    if (!need_stats()) {
        out_shape[0] = out_shape[1] = {0};
    }
    if (inp_shape[0].is_empty()) {
        out_shape[4] = {0};
    } else {
        out_shape[4] = {
                megdnn_opr()->get_reserve_in_bytes({inp_shape[0], input(0)->dtype()})};
    }
}

size_t BatchNormForward::get_workspace_size_bytes(
        const TensorShapeArray& input_shapes,
        const TensorShapeArray& output_shapes) const {
    if (input_shapes[0].is_empty())
        return 0;
#define in(x) \
    { input_shapes[x], input(x)->dtype() }
#define out(x) \
    { output_shapes[x], output(x)->dtype() }
    return megdnn_opr()->get_workspace_in_bytes(
            in(0), in(1), in(2), out(0), out(1), out(2), out(3), out(4), out(5));
#undef in
#undef out
}

void BatchNormForward::init_output_static_infer_desc() {
    Super::set_nr_managed_outputs(this->output().size() - 1);
    Super::init_output_static_infer_desc();
    this->init_output_static_infer_desc_workspace(
            intl::AutoAddWorkspaceNeedLimitGetter<megdnn::BNForward>::val);
}

void BatchNormForward::init_output_dtype() {
    size_t nr_inp = input().size();
    mgb_assert(input(0)->dtype().category() == input(1)->dtype().category());
    for (size_t i = 2; i < nr_inp; ++i) {
        mgb_assert(input(1)->dtype() == input(i)->dtype());
    }
    output(4)->dtype(dtype::Byte());      // reserve
    output(5)->dtype(input(0)->dtype());  // output
    for (size_t i = 0; i < 4; ++i) {
        output(i)->dtype(input(1)->dtype());
    }
}

void BatchNormForward::mem_plan_fwd_in2out_writable() {
    if (need_stats() && !m_force_inplace) {
        // TODO: testing
        output(0)->set_fwd_in2out_writable(input(3));
        output(1)->set_fwd_in2out_writable(input(4));
    }
}

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(BatchNormForward) {
    mgb_assert(wrt_idx < 5, "wrt_idx %zu is out of range", wrt_idx);
    VarNodeArray ret(opr.input().size(), nullptr);
    SymbolVarArray grad;
    switch (opr.param().fwd_mode) {
        case BatchNorm::Param::FwdMode::TRAINING:
            grad = BatchNormBackward::make(
                    opr.input(0), out_grad[5], opr.output(2), opr.output(3),
                    opr.input(1), opr.output(4),  // reserve
                    opr.param());
            for (size_t i = 0; i < 3; ++i) {
                ret[i] = grad[(i + 2) % 3].node();
            }
            return ret;
        case BatchNorm::Param::FwdMode::INFERENCE:
            auto sqrt_var = PowC::make(
                    (SymbolVar{opr.input(4)} +
                     static_cast<dt_float32>(opr.param().epsilon)),
                    0.5, opr.config());
            auto d_bn_scale_unreduced =
                    SymbolVar{out_grad[5]} *
                    (SymbolVar{opr.input(0)} - SymbolVar{opr.input(3)}) / sqrt_var;
            auto d_bn_scale = Reduce::make(
                    d_bn_scale_unreduced, Reduce::Param::Mode::SUM,
                    GetVarShape::make(opr.input(1)));
            auto d_bn_bias = Reduce::make(
                    out_grad[5], Reduce::Param::Mode::SUM,
                    GetVarShape::make(opr.input(2)));
            auto dx = SymbolVar{out_grad[5]} * SymbolVar{opr.input(1)} / sqrt_var;

            ret[0] = dx.node();
            ret[1] = d_bn_scale.node();
            ret[2] = d_bn_bias.node();
            return ret;
    }
    return ret;
}
#endif

MGB_DYN_TYPE_OBJ_FINAL_IMPL(BatchNormBackward);

BatchNormBackward::BatchNormBackward(
        VarNode* x, VarNode* y_grad, VarNode* save_mean, VarNode* save_variance,
        VarNode* scale, VarNode* reserve, const Param& param,
        const OperatorNodeConfig& config)
        : Super({x->owner_graph(),
                 config,
                 "batch_norm_bwd",
                 {x, y_grad, save_mean, save_variance, scale, reserve}},
                0, true) {
    init_megdnn_opr(*this, param);
    add_input({x, y_grad, save_mean, save_variance, scale, reserve});
}

SymbolVarArray BatchNormBackward::make(
        SymbolVar x, SymbolVar y_grad, SymbolVar save_mean, SymbolVar save_variance,
        SymbolVar scale, SymbolVar reserve, const Param& param,
        const OperatorNodeConfig& config) {
    auto&& out = x.node()->owner_graph()
                         ->insert_opr(std::make_unique<BatchNormBackward>(
                                 x.node(), y_grad.node(), save_mean.node(),
                                 save_variance.node(), scale.node(), reserve.node(),
                                 param, config))
                         ->output();
    SymbolVarArray ret(out.size());
    for (size_t i = 0; i < ret.size(); i++) {
        ret[i] = out[i];
    }
    return ret;
}

void BatchNormBackward::init_output_static_infer_desc() {
    using namespace cg::static_infer;
    auto&& mgr = owner_graph()->static_infer_manager();

    mgr.register_shape_infer(output(0), ShapeInferDesc::make_identity(input(4)));
    mgr.register_shape_infer(output(1), ShapeInferDesc::make_identity(input(4)));
    mgr.register_shape_infer(output(2), ShapeInferDesc::make_identity(input(0)));
    this->init_output_static_infer_desc_workspace(
            intl::AutoAddWorkspaceNeedLimitGetter<megdnn::BNBackward>::val);
}

void BatchNormBackward::init_output_dtype() {
    mgb_assert(input(0)->dtype().category() == input(2)->dtype().category());
    mgb_assert(input(0)->dtype() == input(1)->dtype());
    mgb_assert(input(2)->dtype() == input(3)->dtype());
    mgb_assert(input(2)->dtype() == input(4)->dtype());
    output(0)->dtype(input(2)->dtype());
    output(1)->dtype(input(2)->dtype());
    output(2)->dtype(input(0)->dtype());
}

cg::OperatorNodeBase::NodeProp* BatchNormBackward::do_make_node_prop() const {
    auto ret = Super::do_make_node_prop();
    ret->add_dep_type_existing_var(input(5), NodeProp::DepType::VALUE_ALLOW_EMPTY);
    return ret;
}
// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
