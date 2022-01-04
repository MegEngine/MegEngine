/**
 * \file src/opr/impl/dnn/rnn.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "megbrain/opr/dnn/rnn.h"
#include "../internal/megdnn_opr_wrapper.inl"
#include "megbrain/graph/grad_impl.h"
#include "megbrain/opr/basic_arith_wrapper.h"
#include "megbrain/opr/blas.h"
#include "megbrain/opr/internal/out_shape_by_sym_var.h"
#include "megbrain/opr/tensor_manip.h"
#include "megbrain/opr/utility.h"

using namespace mgb;
using namespace opr;

/* ================= RNNCell =================  */
MGB_DYN_TYPE_OBJ_FINAL_IMPL(RNNCellForward);
RNNCellForward::RNNCellForward(
        VarNode* input, VarNode* weight_ih, VarNode* bias_ih, VarNode* hx,
        VarNode* weight_hh, VarNode* bias_hh, const Param& param,
        const OperatorNodeConfig& config)
        : Super{input->owner_graph(),
                config,
                "rnn_cell",
                {input, weight_ih, bias_ih, hx, weight_hh, bias_hh}} {
    init_megdnn_opr(*this, param);
    add_input({input, weight_ih, bias_ih, hx, weight_hh, bias_hh});
}

SymbolVar RNNCellForward::make(
        SymbolVar input, SymbolVar weight_ih, SymbolVar bias_ih, SymbolVar hx,
        SymbolVar weight_hh, SymbolVar bias_hh, const Param& param,
        const OperatorNodeConfig& config) {
    return input.insert_single_output_opr<RNNCellForward>(
            input.node(), weight_ih.node(), bias_ih.node(), hx.node(), weight_hh.node(),
            bias_hh.node(), param, config);
}

#if MGB_ENABLE_GRAD

VarNode* rnnCellBackward(
        const SymbolVar& input, const SymbolVar& weight_ih, const SymbolVar& hx,
        const SymbolVar& weight_hh, const SymbolVar& out,
        RNNCell::NonlineMode nonlineMode, size_t wrt_idx, const SymbolVar& og) {
    SymbolVar tmp;
    // activation
    using NonlineMode = RNNCell::NonlineMode;
    using Mode = Elemwise::Mode;
    switch (nonlineMode) {
        case NonlineMode::IDENTITY:
            tmp = og;
            break;
        case NonlineMode::TANH:
            tmp = Elemwise::make({out, og}, Mode::TANH_GRAD);
            break;
        case NonlineMode::RELU:
            tmp = Elemwise::make({out, og}, Mode::SWITCH_GT0);
            break;
        default:
            mgb_throw(GraphError, "Activation method not supported");
    }
    // now grad is in tmp
    if (wrt_idx == 2 || wrt_idx == 5)
        return tmp.node();  // bias

    SymbolVar result;
    // A * Bt = C, A' = C' * B, B' = C't * A
    if (wrt_idx == 0) {  // input
        result = MatrixMul::make(
                tmp, weight_ih,
                {false, false});  // transpose a false, transpose b false
    } else if (wrt_idx == 1) {    // weight_ih
        result = MatrixMul::make(tmp, input, {true, false});
    } else if (wrt_idx == 3) {  // hx
        result = MatrixMul::make(tmp, weight_hh, {false, false});
    } else if (wrt_idx == 4) {  // weight_hh
        result = MatrixMul::make(tmp, hx, {true, false});
    }
    return result.node();
}

MGB_IMPL_OPR_GRAD(RNNCell) {
    SymbolVar input(opr.input(0)), weight_ih(opr.input(1)), hx(opr.input(3)),
            weight_hh(opr.input(4));
    SymbolVar out(opr.output(0)), og{out_grad.at(0)};
    return rnnCellBackward(
            input, weight_ih, hx, weight_hh, out, opr.param().nonlineMode, wrt_idx, og);
}
#endif

/* ================= LSTMCell =================  */
MGB_DYN_TYPE_OBJ_FINAL_IMPL(LSTMCell);
LSTMCellForward::LSTMCellForward(
        VarNode* input, VarNode* weight_ih, VarNode* bias_ih, VarNode* hx,
        VarNode* weight_hh, VarNode* bias_hh, VarNode* cx, const Param& param,
        const OperatorNodeConfig& config)
        : Super{input->owner_graph(),
                config,
                "lstm_cell",
                {input, weight_ih, bias_ih, hx, weight_hh, bias_hh, cx}} {
    init_megdnn_opr(*this, param);
    add_input({input, weight_ih, bias_ih, hx, weight_hh, bias_hh, cx});
}

SymbolVar LSTMCellForward::make(
        SymbolVar input, SymbolVar weight_ih, SymbolVar bias_ih, SymbolVar hx,
        SymbolVar weight_hh, SymbolVar bias_hh, SymbolVar cx, const Param& param,
        const OperatorNodeConfig& config) {
    return input.insert_single_output_opr<LSTMCellForward>(
            input.node(), weight_ih.node(), bias_ih.node(), hx.node(), weight_hh.node(),
            bias_hh.node(), cx.node(), param, config);
}

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(LSTMCell) {
    SymbolVar input(opr.input(0)), weight_ih(opr.input(1)), hx(opr.input(3)),
            weight_hh(opr.input(4)), cx(opr.input(6));
    SymbolVar h_out(opr.output(0)), c_out(opr.output(1)), gates(opr.output(2)),
            h_og{out_grad.at(0)}, c_og{out_grad.at(1)};
    size_t ghs = gates.shape()[1] / 4;  // gate_hidden_size
    SymbolVarArray gates_array = Split::make(
            gates, Split::Options::make_partition(gates, 1, {ghs, ghs, ghs, ghs}));
    mgb_assert(gates_array.size() == 4);
    using Mode = Elemwise::Mode;
    const SymbolVar &i(Elemwise::make({gates_array.at(0)}, Mode::SIGMOID)),
            f(Elemwise::make({gates_array.at(1)}, Mode::SIGMOID)),
            o(Elemwise::make({gates_array.at(2)}, Mode::SIGMOID)),
            g(Elemwise::make({gates_array.at(3)}, Mode::TANH));
    SymbolVar i_grad, f_grad, o_grad, g_grad;

    SymbolVar tanh_c_out = Elemwise::make({c_out}, Mode::TANH);
    o_grad = Elemwise::make({o, h_og * tanh_c_out}, Mode::SIGMOID_GRAD);
    c_og = c_og + Elemwise::make({tanh_c_out, h_og * o}, Mode::TANH_GRAD);
    f_grad = Elemwise::make({f, c_og * cx}, Mode::SIGMOID_GRAD);
    i_grad = Elemwise::make({i, c_og * g}, Mode::SIGMOID_GRAD);
    g_grad = Elemwise::make({g, c_og * i}, Mode::TANH_GRAD);
    SymbolVar rnn_cell_grad = Concat::make({i_grad, f_grad, o_grad, g_grad}, -1);

    SymbolVar result;
    if (wrt_idx < 6) {
        using NonlineMode = RNNCell::NonlineMode;
        result = rnnCellBackward(
                input, weight_ih, hx, weight_hh, gates, NonlineMode::IDENTITY, wrt_idx,
                rnn_cell_grad);
    } else {  // cx
        result = c_og * f;
    }
    return result.node();
}
#endif

/* ================= RNN =================  */
MGB_DYN_TYPE_OBJ_FINAL_IMPL(RNN);
MEGDNN_OPR_INIT3(RNNForward, "rnn_fwd");

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(RNN) {
    mgb_assert(
            opr.param().fwd_mode == RNN::Param::FwdMode::TRAINING,
            "RNN could only take grad in training mode");
    SymbolVarArray grads = RNNBackward::make(
            opr.input(0), opr.output(0), opr.input(1), out_grad.at(0), out_grad.at(1),
            opr.input(2), opr.output(2), opr.param());
    // return grads.at(wrt_idx).node(); // input, hx, weights
    VarNodeArray ret(opr.input().size(), nullptr);
    for (size_t i = 0; i < ret.size(); ++i) {
        ret[i] = grads[i].node();
    }
    return ret;
}
#endif

/* ================= RNNBackward =================  */
MGB_DYN_TYPE_OBJ_FINAL_IMPL(RNNBackward);

RNNBackward::RNNBackward(
        VarNode* x, VarNode* y, VarNode* hx, VarNode* dy, VarNode* dhy,
        VarNode* flatten_weights, VarNode* reserve_space, const Param& param,
        const OperatorNodeConfig& config)
        : Super({x->owner_graph(),
                 config,
                 "rnn_bwd",
                 {x, y, hx, dy, dhy, flatten_weights, reserve_space}},
                0, true) {
    init_megdnn_opr(*this, param);
    add_input({x, y, hx, dy, dhy, flatten_weights, reserve_space});
}

SymbolVarArray RNNBackward::make(
        SymbolVar x, SymbolVar y, SymbolVar hx, SymbolVar dy, SymbolVar dhy,
        SymbolVar flatten_weights, SymbolVar reserve_space, const Param& param,
        const OperatorNodeConfig& config) {
    auto&& out = x.node()->owner_graph()
                         ->insert_opr(std::make_unique<RNNBackward>(
                                 x.node(), y.node(), hx.node(), dy.node(), dhy.node(),
                                 flatten_weights.node(), reserve_space.node(), param,
                                 config))
                         ->output();
    SymbolVarArray ret(out.size());
    for (size_t i = 0; i < ret.size(); ++i) {
        ret[i] = out[i];
    }
    return ret;
}

RNNBackward::Super::NodeProp* RNNBackward::do_make_node_prop() const {
    auto ret = Super::do_make_node_prop();
    ret->add_dep_type_existing_var(input(6), NodeProp::DepType::VALUE_ALLOW_EMPTY);
    return ret;
}

void RNNBackward::init_output_static_infer_desc() {
    using namespace cg::static_infer;
    auto&& mgr = owner_graph()->static_infer_manager();

    mgr.register_shape_infer(output(0), ShapeInferDesc::make_identity(input(0)));
    mgr.register_shape_infer(output(1), ShapeInferDesc::make_identity(input(2)));
    mgr.register_shape_infer(output(2), ShapeInferDesc::make_identity(input(5)));
    this->init_output_static_infer_desc_workspace(
            intl::AutoAddWorkspaceNeedLimitGetter<megdnn::RNNBackward>::val);
}

void RNNBackward::init_output_dtype() {
    output(0)->dtype(input(0)->dtype());
    output(1)->dtype(input(2)->dtype());
    output(2)->dtype(input(5)->dtype());
}

/* ================= LSTM =================  */
MGB_DYN_TYPE_OBJ_FINAL_IMPL(LSTM);
LSTMForward::LSTMForward(
        VarNode* input, VarNode* hx, VarNode* cx, VarNode* flatten_weights,
        const Param& param, const OperatorNodeConfig& config)
        : Super{input->owner_graph(),
                config,
                "lstm",
                {input, hx, cx, flatten_weights}} {
    init_megdnn_opr(*this, param);
    add_input({input, hx, cx, flatten_weights});
}

SymbolVar LSTMForward::make(
        SymbolVar input, SymbolVar hx, SymbolVar cx, SymbolVar flatten_weights,
        const Param& param, const OperatorNodeConfig& config) {
    return input.insert_single_output_opr<LSTMForward>(
            input.node(), hx.node(), cx.node(), flatten_weights.node(), param, config);
}

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(LSTM) {
    SymbolVarArray grads = LSTMBackward::make(
            opr.input(0), opr.output(0), opr.input(1), opr.input(2), out_grad.at(0),
            out_grad.at(1), out_grad.at(2), opr.input(3), opr.output(3), opr.param());
    return grads.at(wrt_idx).node();  // input, hx, cx, weights
}
#endif

/* ================= LSTMBackward =================  */
MGB_DYN_TYPE_OBJ_FINAL_IMPL(LSTMBackward);
LSTMBackward::LSTMBackward(
        VarNode* x, VarNode* y, VarNode* hx, VarNode* cx, VarNode* dy, VarNode* dhy,
        VarNode* dcy, VarNode* flatten_weights, VarNode* reserve_space,
        const Param& param, const OperatorNodeConfig& config)
        : Super({x->owner_graph(),
                 config,
                 "lstm_bwd",
                 {x, y, hx, cx, dy, dhy, dcy, flatten_weights, reserve_space}},
                1, true) {
    init_megdnn_opr(*this, param);
    add_input({x, y, hx, cx, dy, dhy, dcy, flatten_weights, reserve_space});
}

SymbolVarArray LSTMBackward::make(
        SymbolVar x, SymbolVar y, SymbolVar hx, SymbolVar cx, SymbolVar dy,
        SymbolVar dhy, SymbolVar dcy, SymbolVar flatten_weights,
        SymbolVar reserve_space, const Param& param, const OperatorNodeConfig& config) {
    auto&& out = x.node()->owner_graph()
                         ->insert_opr(std::make_unique<LSTMBackward>(
                                 x.node(), y.node(), hx.node(), cx.node(), dy.node(),
                                 dhy.node(), dcy.node(), flatten_weights.node(),
                                 reserve_space.node(), param, config))
                         ->output();
    SymbolVarArray ret(out.size());
    for (size_t i = 0; i < ret.size(); ++i) {
        ret[i] = out[i];
    }
    return ret;
}

LSTMBackward::Super::NodeProp* LSTMBackward::do_make_node_prop() const {
    auto ret = Super::do_make_node_prop();
    ret->add_dep_type_existing_var(
            input(8),  // reserve space
            NodeProp::DepType::VALUE_ALLOW_EMPTY);
    return ret;
}

void LSTMBackward::init_output_static_infer_desc() {
    using namespace cg::static_infer;
    auto&& mgr = owner_graph()->static_infer_manager();

    mgr.register_shape_infer(output(0), ShapeInferDesc::make_identity(input(0)));
    mgr.register_shape_infer(output(1), ShapeInferDesc::make_identity(input(2)));
    mgr.register_shape_infer(output(2), ShapeInferDesc::make_identity(input(3)));
    mgr.register_shape_infer(output(3), ShapeInferDesc::make_identity(input(7)));
    this->init_output_static_infer_desc_workspace(
            intl::AutoAddWorkspaceNeedLimitGetter<megdnn::LSTMBackward>::val);
}

void LSTMBackward::init_output_dtype() {
    output(0)->dtype(input(0)->dtype());
    output(1)->dtype(input(2)->dtype());
    output(2)->dtype(input(3)->dtype());
    output(3)->dtype(input(7)->dtype());
}