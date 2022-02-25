#include "megbrain/opr/dnn/images2neibs.h"
#include "megbrain/graph/grad_impl.h"

#include "../internal/megdnn_opr_wrapper.inl"

using namespace mgb;
using namespace opr;

MGB_DYN_TYPE_OBJ_FINAL_IMPL(Images2NeibsForward);
MEGDNN_OPR_INIT1(Images2NeibsForward, "images2neibs")

#if MGB_ENABLE_GRAD
MGB_IMPL_OPR_GRAD(Images2NeibsForward) {
    mgb_assert(wrt_idx == 0 && out_grad.size() == 2 && !out_grad[1]);
    return Images2NeibsBackward::make(out_grad[0], opr.input(0), opr.param()).node();
}
#endif

MGB_DYN_TYPE_OBJ_FINAL_IMPL(Images2NeibsBackward);
MEGDNN_OPR_INIT2(Images2NeibsBackward, "images2neibs_grad", 1, false);

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
