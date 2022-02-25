#pragma once
#include "megbrain/opr/internal/megdnn_opr_wrapper.h"
#include "megdnn/oprs.h"
namespace mgb {
namespace opr {
MGB_DEFINE_OPR_CLASS(
        FakeQuantForward, intl::MegDNNOprWrapperFwd<megdnn::FakeQuantForward>) // {
public:
    MGE_WIN_DECLSPEC_FUC FakeQuantForward(
            VarNode* src, VarNode* scale, VarNode* zero_point, const Param& param,
            const OperatorNodeConfig& config);

    MGE_WIN_DECLSPEC_FUC static SymbolVar make(
            SymbolVar src, SymbolVar scale, SymbolVar zero_point,
            const Param& param = {}, const OperatorNodeConfig& config = {});
};  // namespace opr
using FakeQuant = FakeQuantForward;

MGB_DEFINE_OPR_CLASS(
        FakeQuantBackward, intl::MegDNNOprWrapperBwd<megdnn::FakeQuantBackward>) // {
public:
    MGE_WIN_DECLSPEC_FUC FakeQuantBackward(
            VarNode* diff, VarNode* input, VarNode* scale, VarNode* zero_point,
            const Param& param, const OperatorNodeConfig& config);

    MGE_WIN_DECLSPEC_FUC static SymbolVar make(
            SymbolVar diff, SymbolVar input, SymbolVar scale, SymbolVar zero_point,
            const Param& param = {}, const OperatorNodeConfig& config = {});
};

}  // namespace opr
}  // namespace mgb
