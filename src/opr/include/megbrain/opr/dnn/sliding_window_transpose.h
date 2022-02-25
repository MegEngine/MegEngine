#pragma once

#include "megbrain/opr/internal/megdnn_opr_wrapper.h"
#include "megdnn/oprs.h"

namespace mgb {
namespace opr {

MGB_DEFINE_OPR_CLASS(
        SlidingWindowTransposeForward,
        intl::MegDNNOprWrapperFwd<megdnn::SlidingWindowTransposeForward>) // {
public:
    MGE_WIN_DECLSPEC_FUC SlidingWindowTransposeForward(
            VarNode* src, const Param& param, const OperatorNodeConfig& config);

    MGE_WIN_DECLSPEC_FUC static SymbolVar make(
            SymbolVar src, const Param& param = {},
            const OperatorNodeConfig& config = {});
};
using SlidingWindowTranspose = SlidingWindowTransposeForward;

MGB_DEFINE_OPR_CLASS(
        SlidingWindowTransposeBackward,
        intl::MegDNNOprWrapperBwd<megdnn::SlidingWindowTransposeBackward>) // {
public:
    MGE_WIN_DECLSPEC_FUC SlidingWindowTransposeBackward(
            VarNode* diff, VarNode* src_for_shape, const Param& param,
            const OperatorNodeConfig& config);

    MGE_WIN_DECLSPEC_FUC static SymbolVar make(
            SymbolVar diff, SymbolVar src_for_shape, const Param& param = {},
            const OperatorNodeConfig& config = {});
};

}  // namespace opr
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
