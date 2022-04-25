
#include "megdnn/oprs.h"

#include "src/common/utils.h"

namespace megdnn {

void CumprodForward::deduce_layout(const TensorLayout& src, TensorLayout& dst) {
    megdnn_assert_contiguous(src);
    dst = src;
}

void CumprodForward::check_exec(
        const TensorLayout& src, const TensorLayout& dst, size_t workspace_in_bytes) {
    megdnn_assert_contiguous(src);
    megdnn_assert_eq_layout(src, dst);
    megdnn_assert(param().axis >= 0);
    megdnn_assert(static_cast<size_t>(param().axis) < src.ndim);
    auto required_workspace_in_bytes = get_workspace_in_bytes(src, dst);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
}

}  // namespace megdnn

// vim: syntax=cpp.doxygen
