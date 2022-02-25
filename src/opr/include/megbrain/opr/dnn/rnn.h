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

    MGE_WIN_DECLSPEC_FUC RNNCellForward(
            VarNode* input, VarNode* weight_ih, VarNode* bias_ih, VarNode* hx,
            VarNode* weight_hh, VarNode* bias_hh, const Param& param,
            const OperatorNodeConfig& config);
    MGE_WIN_DECLSPEC_FUC static SymbolVar make(
            SymbolVar input, SymbolVar weight_ih, SymbolVar bias_ih, SymbolVar hx,
            SymbolVar weight_hh, SymbolVar bias_hh, const Param& param = {},
            const OperatorNodeConfig& config = {});
};
using RNNCell = RNNCellForward;

MGB_DEFINE_OPR_CLASS(
        LSTMCellForward, intl::MegDNNOprWrapperFwd<megdnn::LSTMCellForward>) // {
public:
    MGE_WIN_DECLSPEC_FUC LSTMCellForward(
            VarNode* input, VarNode* weight_ih, VarNode* bias_ih, VarNode* hx,
            VarNode* weight_hh, VarNode* bias_hh, VarNode* cx, const Param& param,
            const OperatorNodeConfig& config);
    MGE_WIN_DECLSPEC_FUC static SymbolVar make(
            SymbolVar input, SymbolVar weight_ih, SymbolVar bias_ih, SymbolVar hx,
            SymbolVar weight_hh, SymbolVar bias_hh, SymbolVar cx,
            const Param& param = {}, const OperatorNodeConfig& config = {});
};
using LSTMCell = LSTMCellForward;

MGB_DEFINE_OPR_CLASS(RNNForward, intl::MegDNNOprWrapperFwd<megdnn::RNNForward>) // {
public:
    MGE_WIN_DECLSPEC_FUC RNNForward(
            VarNode* input, VarNode* hx, VarNode* flatten_weights, const Param& param,
            const OperatorNodeConfig& config);
    MGE_WIN_DECLSPEC_FUC static SymbolVar make(
            SymbolVar input, SymbolVar hx, SymbolVar flatten_weights,
            const Param& param = {}, const OperatorNodeConfig& config = {});
};
using RNN = RNNForward;

MGB_DEFINE_OPR_CLASS(
        RNNBackward, intl::MegDNNOprWrapperBwd<megdnn::RNNBackward>) // {
public:
    MGE_WIN_DECLSPEC_FUC RNNBackward(
            VarNode* x, VarNode* y, VarNode* hx, VarNode* dy, VarNode* dhy,
            VarNode* flatten_weights, VarNode* reserve_space, const Param& param,
            const OperatorNodeConfig& config);
    MGE_WIN_DECLSPEC_FUC static SymbolVarArray make(
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
    MGE_WIN_DECLSPEC_FUC LSTMForward(
            VarNode* input, VarNode* hx, VarNode* cx, VarNode* flatten_weights,
            const Param& param, const OperatorNodeConfig& config);
    MGE_WIN_DECLSPEC_FUC static SymbolVar make(
            SymbolVar input, SymbolVar hx, SymbolVar cx, SymbolVar flatten_weights,
            const Param& param = {}, const OperatorNodeConfig& config = {});
};
using LSTM = LSTMForward;

MGB_DEFINE_OPR_CLASS(
        LSTMBackward, intl::MegDNNOprWrapperBwd<megdnn::LSTMBackward>) // {
public:
    MGE_WIN_DECLSPEC_FUC LSTMBackward(
            VarNode* x, VarNode* y, VarNode* hx, VarNode* cx, VarNode* dy, VarNode* dhy,
            VarNode* dcy, VarNode* flatten_weights, VarNode* reserve_space,
            const Param& param, const OperatorNodeConfig& config);
    MGE_WIN_DECLSPEC_FUC static SymbolVarArray make(
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