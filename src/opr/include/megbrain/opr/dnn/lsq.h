#pragma once
#include "megbrain/opr/internal/megdnn_opr_wrapper.h"
#include "megdnn/oprs.h"
namespace mgb {
namespace opr {

MGB_DEFINE_OPR_CLASS(LSQForward, intl::MegDNNOprWrapperFwd<megdnn::LSQForward>) // {
public:
    MGE_WIN_DECLSPEC_FUC LSQForward(
            VarNode* src, VarNode* scale, VarNode* zero_point, VarNode* grad_scale,
            const Param& param, const OperatorNodeConfig& config);

    MGE_WIN_DECLSPEC_FUC static SymbolVar make(
            SymbolVar src, SymbolVar scale, SymbolVar zero_point, SymbolVar grad_scale,
            const Param& param = {}, const OperatorNodeConfig& config = {});
};
using LSQ = LSQForward;

MGB_DEFINE_OPR_CLASS(
        LSQBackward, intl::MegDNNOprWrapperBwd<megdnn::LSQBackward>) // {
public:
    MGE_WIN_DECLSPEC_FUC LSQBackward(
            VarNode* y_grad, VarNode* x, VarNode* scale, VarNode* zero_point,
            VarNode* grad_scale, const Param& param, const OperatorNodeConfig& config);

    MGE_WIN_DECLSPEC_FUC static SymbolVarArray make(
            SymbolVar y_grad, SymbolVar x, SymbolVar scale, SymbolVar zero_point,
            SymbolVar grad_scale, const Param& param = {},
            const OperatorNodeConfig& config = {});

private:
    void init_output_static_infer_desc() override;
    void init_output_dtype() override;
};

}  // namespace opr
}  // namespace mgb
