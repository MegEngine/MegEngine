#pragma once

#include "megbrain/opr/internal/megdnn_opr_wrapper.h"
#include "megdnn/oprs.h"

namespace mgb {
namespace opr {

MGB_DEFINE_OPR_CLASS(
        Images2NeibsForward, intl::MegDNNOprWrapperFwd<megdnn::Images2NeibsForward>) // {
public:
    MGE_WIN_DECLSPEC_FUC Images2NeibsForward(
            VarNode* src, const Param& param, const OperatorNodeConfig& config);

    MGE_WIN_DECLSPEC_FUC static SymbolVar make(
            SymbolVar src, const Param& param = {},
            const OperatorNodeConfig& config = {});
};
using Images2Neibs = Images2NeibsForward;

MGB_DEFINE_OPR_CLASS(
        Images2NeibsBackward, intl::MegDNNOprWrapperBwd<megdnn::Images2NeibsBackward>) // {
public:
    MGE_WIN_DECLSPEC_FUC Images2NeibsBackward(
            VarNode* diff, VarNode* src_for_shape, const Param& param,
            const OperatorNodeConfig& config);

    MGE_WIN_DECLSPEC_FUC static SymbolVar make(
            SymbolVar diff, SymbolVar src_for_shape, const Param& param = {},
            const OperatorNodeConfig& config = {});
};

}  // namespace opr
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
