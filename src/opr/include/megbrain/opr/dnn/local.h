#pragma once

#include "megbrain/opr/internal/megdnn_opr_wrapper.h"
#include "megdnn/oprs.h"

namespace mgb {
namespace opr {

//! param: src, filter
MGB_DEFINE_MEGDNN_OPR_WRAPPER_FWD2(LocalForward);
MGB_DEFINE_MEGDNN_OPR_WRAPPER_FWD2(GroupLocalForward);

//! param: filter, diff, src
MGB_DEFINE_MEGDNN_OPR_WRAPPER_BWD3(LocalBackwardData);
MGB_DEFINE_MEGDNN_OPR_WRAPPER_BWD3(GroupLocalBackwardData);

//! param: src, diff, filter
MGB_DEFINE_MEGDNN_OPR_WRAPPER_BWD3(LocalBackwardFilter);
MGB_DEFINE_MEGDNN_OPR_WRAPPER_BWD3(GroupLocalBackwardFilter);

using Local = LocalForward;
using GroupLocal = GroupLocalForward;

}  // namespace opr
}  // namespace mgb

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
