#include "megdnn/oprs.h"

#include "src/common/utils.h"

namespace megdnn {

void SoftmaxBase::deduce_layout_fwd(const TensorLayout& src, TensorLayout& dst) {
    megdnn_assert(
            param().axis >= -static_cast<int32_t>(src.ndim) &&
                    param().axis < static_cast<int32_t>(src.ndim),
            "axis: %d ndim: %zu", param().axis, src.ndim);
    megdnn_assert_contiguous(src);
    dst = src;

    dst.dtype = src.dtype;
    dst.format = src.format;
    dst.init_contiguous_stride();
}

void SoftmaxBase::check_layout_fwd(const TensorLayout& src, const TensorLayout& dst) {
    TensorLayout dst_expected;
    megdnn_assert_eq_dtype(src, dst);
    deduce_layout_fwd(src, dst_expected);
    megdnn_assert_eq_layout(dst_expected, dst);
    megdnn_assert(src.dtype == dst.dtype);
}

void SoftmaxForward::deduce_layout(const TensorLayout& src, TensorLayout& dst) {
    deduce_layout_fwd(src, dst);
}

void SoftmaxForward::check_exec(
        const TensorLayout& src, const TensorLayout& dst, size_t workspace_in_bytes) {
    check_layout_fwd(src, dst);
    auto required_workspace_in_bytes = get_workspace_in_bytes(src, dst);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
}

void SoftmaxBackward::check_exec(
        const TensorLayout& src, const TensorLayout& diff, const TensorLayout& grad,
        size_t workspace_in_bytes) {
    megdnn_assert_eq_layout(src, diff);
    megdnn_assert_eq_layout(src, grad);
    auto required_workspace_in_bytes = get_workspace_in_bytes(src, diff, grad);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
}

}  // namespace megdnn

// vim: syntax=cpp.doxygen