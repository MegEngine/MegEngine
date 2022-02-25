#pragma once

#include "megbrain/opr/internal/megdnn_opr_wrapper.h"
#include "megdnn/oprs.h"

namespace mgb {
namespace opr {

MGB_DEFINE_OPR_CLASS(LRNForward, intl::MegDNNOprWrapperFwd<megdnn::LRNForward>) // {
public:
    MGE_WIN_DECLSPEC_FUC LRNForward(
            VarNode* src, const Param& param, const OperatorNodeConfig& config);
    MGE_WIN_DECLSPEC_FUC static SymbolVar make(
            SymbolVar src, const Param& param, const OperatorNodeConfig& config = {});
};
using LRN = LRNForward;

MGB_DEFINE_OPR_CLASS(
        LRNBackward, intl::MegDNNOprWrapperBwd<megdnn::LRNBackward>) // {
public:
    MGE_WIN_DECLSPEC_FUC LRNBackward(
            VarNode* src, VarNode* dst, VarNode* diff, const Param& param,
            const OperatorNodeConfig& config);
    MGE_WIN_DECLSPEC_FUC static SymbolVar make(
            SymbolVar src, SymbolVar dst, SymbolVar diff, const Param& param,
            const OperatorNodeConfig& config = {});
};

}  // namespace opr
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
