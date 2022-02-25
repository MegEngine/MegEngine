#include "megdnn/oprs.h"

#include "src/common/utils.h"

namespace megdnn {

void DotForward::check_exec(
        const TensorLayout& A, const TensorLayout& B, const TensorLayout& C,
        size_t workspace_in_bytes) {
    auto errmsg = [&]() {
        return megdnn_layout_msg(A) + ", " + megdnn_layout_msg(B) + ", " +
               megdnn_layout_msg(C);
    };
    MEGDNN_MARK_USED_VAR(errmsg);
    megdnn_assert(A.ndim == 1_z && A.stride[0] >= 0, "%s", errmsg().c_str());
    megdnn_assert(B.ndim == 1_z && B.stride[0] >= 0, "%s", errmsg().c_str());
    megdnn_assert(A.shape[0] == B.shape[0], "%s", errmsg().c_str());
    megdnn_assert(C.is_scalar(), "%s", errmsg().c_str());

    megdnn_assert(A.dtype == B.dtype && A.dtype == C.dtype);

    auto required_workspace_in_bytes = get_workspace_in_bytes(A, B, C);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
}

MGE_WIN_DECLSPEC_FUC void DotForward::deduce_layout(
        const TensorLayout& A, const TensorLayout&, TensorLayout& C) {
    C = TensorLayout(TensorShape{1}, A.dtype);
}

}  // namespace megdnn

// vim: syntax=cpp.doxygen
