#include "megdnn/oprs.h"

#include "src/common/utils.h"

namespace megdnn {

void LRNBase::check_param() {
    megdnn_assert(param().n & 1);
}

void LRNForward::deduce_layout(const TensorLayout& src, TensorLayout& dst) {
    dst = src;
}

void LRNForward::check_exec(
        const TensorLayout& src, const TensorLayout& dst, size_t workspace_in_bytes) {
    check_param();
    megdnn_assert_contiguous(src);
    megdnn_assert_eq_layout(src, dst);

    megdnn_assert(src.dtype.category() == DTypeCategory::FLOAT);
    auto required_workspace_in_bytes = get_workspace_in_bytes(src, dst);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
}

void LRNBackward::check_exec(
        const TensorLayout& src, const TensorLayout& dst, const TensorLayout& diff,
        const TensorLayout& grad, size_t workspace_in_bytes) {
    check_param();
    megdnn_assert_contiguous(src);
    megdnn_assert_eq_layout(src, dst);
    megdnn_assert_eq_layout(src, diff);
    megdnn_assert_eq_layout(src, grad);
    megdnn_assert(src.dtype.category() == DTypeCategory::FLOAT);
    auto required_workspace_in_bytes = get_workspace_in_bytes(src, dst, diff, grad);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
}

}  // namespace megdnn
// vim: syntax=cpp.doxygen
