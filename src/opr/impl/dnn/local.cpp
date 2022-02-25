#include "megbrain/opr/dnn/local.h"
#include "./helper.h"
#include "megbrain/graph/grad_impl.h"

#include "../internal/megdnn_opr_wrapper.inl"

using namespace mgb;
using namespace opr;

MGB_DYN_TYPE_OBJ_FINAL_IMPL(LocalForward);
MEGDNN_OPR_INIT2(LocalForward, "local")

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(LocalForward) {
    return intl::conv_grad<LocalBackwardData, LocalBackwardFilter>(
            opr, wrt_idx, out_grad);
}
#endif

MGB_DYN_TYPE_OBJ_FINAL_IMPL(LocalBackwardData);
MEGDNN_OPR_INIT3(LocalBackwardData, "local_bwd_data", 2, false);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(LocalBackwardFilter);
MEGDNN_OPR_INIT3(LocalBackwardFilter, "local_bwd_filter", 2, false);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(GroupLocalForward);
MEGDNN_OPR_INIT2(GroupLocalForward, "glocal")

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(GroupLocalForward) {
    return intl::conv_grad<GroupLocalBackwardData, GroupLocalBackwardFilter>(
            opr, wrt_idx, out_grad);
}
#endif

MGB_DYN_TYPE_OBJ_FINAL_IMPL(GroupLocalBackwardData);
MEGDNN_OPR_INIT3(GroupLocalBackwardData, "glocal_bwd_data", 2, false);

MGB_DYN_TYPE_OBJ_FINAL_IMPL(GroupLocalBackwardFilter);
MEGDNN_OPR_INIT3(GroupLocalBackwardFilter, "glocal_bwd_filter", 2, false);

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
