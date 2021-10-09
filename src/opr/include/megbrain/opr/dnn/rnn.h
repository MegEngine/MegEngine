/**
 * \file src/opr/include/megbrain/opr/dnn/rnn.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include "megbrain/opr/internal/megdnn_opr_wrapper.h"
#include "megbrain/opr/internal/out_shape_by_sym_var.h"
#if MGB_CUDA
#include "../../../../impl/nvof/denseflownvidia.h"
#include "megbrain/opr/param_defs.h"
#endif
#include "megdnn/oprs.h"

namespace mgb {
namespace opr {
MGB_DEFINE_OPR_CLASS(
        RNNCellForward, intl::MegDNNOprWrapperFwd<megdnn::RNNCellForward>) // {
public:
    using NonlineMode = Param::NonlineMode;

    RNNCellForward(
            VarNode* input, VarNode* weight_ih, VarNode* bias_ih, VarNode* hx,
            VarNode* weight_hh, VarNode* bias_hh, const Param& param,
            const OperatorNodeConfig& config);
    static SymbolVar make(
            SymbolVar input, SymbolVar weight_ih, SymbolVar bias_ih, SymbolVar hx,
            SymbolVar weight_hh, SymbolVar bias_hh, const Param& param = {},
            const OperatorNodeConfig& config = {});
};
using RNNCell = RNNCellForward;

MGB_DEFINE_OPR_CLASS(
        LSTMCellForward, intl::MegDNNOprWrapperFwd<megdnn::LSTMCellForward>) // {
public:
    LSTMCellForward(
            VarNode* input, VarNode* weight_ih, VarNode* bias_ih, VarNode* hx,
            VarNode* weight_hh, VarNode* bias_hh, VarNode* cx, const Param& param,
            const OperatorNodeConfig& config);
    static SymbolVar make(
            SymbolVar input, SymbolVar weight_ih, SymbolVar bias_ih, SymbolVar hx,
            SymbolVar weight_hh, SymbolVar bias_hh, SymbolVar cx,
            const Param& param = {}, const OperatorNodeConfig& config = {});
};
using LSTMCell = LSTMCellForward;

MGB_DEFINE_OPR_CLASS(RNNForward, intl::MegDNNOprWrapperFwd<megdnn::RNNForward>) // {
    /*private:
            SymbolVarArray weight_ih_arr;  // 1d, idx: direction * num_layers + layer
            SymbolVarArray weight_hh_arr;
            SymbolVarArray bias_arr;
    */

public:
    RNNForward(
            VarNode* input, VarNode* hx, VarNode* flatten_weights, const Param& param,
            const OperatorNodeConfig& config);
    static SymbolVar make(
            SymbolVar input, SymbolVar hx, SymbolVar flatten_weights,
            const Param& param = {}, const OperatorNodeConfig& config = {});
};
using RNN = RNNForward;

MGB_DEFINE_OPR_CLASS(
        RNNBackward, intl::MegDNNOprWrapperBwd<megdnn::RNNBackward>) // {
public:
    RNNBackward(
            VarNode* x, VarNode* y, VarNode* hx, VarNode* dy, VarNode* dhy,
            VarNode* flatten_weights, VarNode* reserve_space, const Param& param,
            const OperatorNodeConfig& config);
    static SymbolVarArray make(
            SymbolVar x, SymbolVar y, SymbolVar hx, SymbolVar dy, SymbolVar dhy,
            SymbolVar flatten_weights, SymbolVar reserve_space, const Param& param = {},
            const OperatorNodeConfig& config = {});
    Super::NodeProp* do_make_node_prop() const override;

private:
    void init_output_static_infer_desc() override;
    void init_output_dtype() override;
};

MGB_DEFINE_OPR_CLASS(
        LSTMForward, intl::MegDNNOprWrapperFwd<megdnn::LSTMForward>) // {
public:
    LSTMForward(
            VarNode* input, VarNode* hx, VarNode* cx, VarNode* flatten_weights,
            const Param& param, const OperatorNodeConfig& config);
    static SymbolVar make(
            SymbolVar input, SymbolVar hx, SymbolVar cx, SymbolVar flatten_weights,
            const Param& param = {}, const OperatorNodeConfig& config = {});
};
using LSTM = LSTMForward;

MGB_DEFINE_OPR_CLASS(
        LSTMBackward, intl::MegDNNOprWrapperBwd<megdnn::LSTMBackward>) // {
public:
    LSTMBackward(
            VarNode* x, VarNode* y, VarNode* hx, VarNode* cx, VarNode* dy, VarNode* dhy,
            VarNode* dcy, VarNode* flatten_weights, VarNode* reserve_space,
            const Param& param, const OperatorNodeConfig& config);
    static SymbolVarArray make(
            SymbolVar x, SymbolVar y, SymbolVar hx, SymbolVar cx, SymbolVar dy,
            SymbolVar dhy, SymbolVar dcy, SymbolVar flatten_weights,
            SymbolVar reserve_space, const Param& param = {},
            const OperatorNodeConfig& config = {});
    Super::NodeProp* do_make_node_prop() const override;

private:
    void init_output_static_infer_desc() override;
    void init_output_dtype() override;
};

}  // namespace opr
}  // namespace mgb