#include "megdnn/oprs.h"

#include "src/common/utils.h"

namespace megdnn {

void WhereBase::deduce_layout_fwd(
        const TensorLayout& mask, const TensorLayout& data1, const TensorLayout& data2,
        TensorLayout& dst) {
    if (!mask.is_empty())
        megdnn_assert(mask.is_physical_contiguous());
    if (!data1.is_empty())
        megdnn_assert(data1.is_physical_contiguous());
    if (!data2.is_empty())
        megdnn_assert(data2.is_physical_contiguous());
    if (!dst.is_empty())
        megdnn_assert(dst.is_physical_contiguous());

    auto errmsg = [&]() {
        return megdnn_layout_msg(mask) + ", " + megdnn_layout_msg(data1) + ", " +
               megdnn_layout_msg(data2) + ", " + megdnn_layout_msg(dst);
    };
    auto mask_dtype = mask.dtype, data1_dtype = data1.dtype, data2_dtype = data2.dtype;
    megdnn_assert(mask_dtype.category() == DTypeCategory::BOOL);
    megdnn_assert(
            data1_dtype == data2_dtype &&
            (data1_dtype.category() == DTypeCategory::INT ||
             data1_dtype.category() == DTypeCategory::FLOAT ||
             data1_dtype.category() == DTypeCategory::BOOL));
    megdnn_assert(data1.ndim == data2.ndim, "%s", errmsg().c_str());
    megdnn_assert(data1.ndim == mask.ndim, "%s", errmsg().c_str());

    dst = TensorLayout{data1};
}

void WhereBase::check_layout_fwd(
        const TensorLayout& mask, const TensorLayout& data1, const TensorLayout& data2,
        const TensorLayout& dst) {
    TensorLayout dst_expected;
    megdnn_assert_eq_shape(mask, data1);
    megdnn_assert_eq_dtype(data1, dst);
    megdnn_assert_eq_shape(data1, data2);
    deduce_layout_fwd(mask, data1, data2, dst_expected);
    megdnn_assert_eq_shape(dst_expected, dst);
}

void WhereForward::deduce_layout(
        const TensorLayout& mask, const TensorLayout& data1, const TensorLayout& data2,
        TensorLayout& dst) {
    deduce_layout_fwd(mask, data1, data2, dst);
}

void WhereForward::check_exec(
        const TensorLayout& mask, const TensorLayout& data1, const TensorLayout& data2,
        const TensorLayout& dst, size_t workspace_in_bytes) {
    check_layout_fwd(mask, data1, data2, dst);
    auto required_workspace_in_bytes = get_workspace_in_bytes(mask, data1, data2, dst);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
}

void WhereBase::deduce_layout_bwd(
        const TensorLayout& diff, const TensorLayout& mask, TensorLayout& grad_data1,
        TensorLayout& grad_data2) {
    if (!diff.is_empty())
        megdnn_assert(diff.is_physical_contiguous());
    if (!mask.is_empty())
        megdnn_assert(mask.is_physical_contiguous());
    if (!grad_data1.is_empty())
        megdnn_assert(grad_data1.is_physical_contiguous());
    if (!grad_data2.is_empty())
        megdnn_assert(grad_data2.is_physical_contiguous());

    auto errmsg = [&]() {
        return megdnn_layout_msg(diff) + ", " + megdnn_layout_msg(mask) + ", " +
               megdnn_layout_msg(grad_data1) + megdnn_layout_msg(grad_data2);
    };
    auto diff_dtype = diff.dtype, mask_dtype = mask.dtype;
    megdnn_assert(mask_dtype.category() == DTypeCategory::BOOL);
    megdnn_assert(
            diff_dtype.category() == DTypeCategory::INT ||
            diff_dtype.category() == DTypeCategory::FLOAT);
    megdnn_assert(diff.ndim == mask.ndim, "%s", errmsg().c_str());

    grad_data1 = TensorLayout{diff};
    grad_data2 = TensorLayout{diff};
}

void WhereBase::check_layout_bwd(
        const TensorLayout& diff, const TensorLayout& mask,
        const TensorLayout& grad_data1, const TensorLayout& grad_data2) {
    TensorLayout grad_expected1;
    TensorLayout grad_expected2;

    megdnn_assert_eq_shape(diff, mask);
    megdnn_assert_eq_shape(diff, grad_data1);
    megdnn_assert_eq_dtype(diff, grad_data1);
    megdnn_assert_eq_shape(diff, grad_data2);
    megdnn_assert_eq_dtype(diff, grad_data2);

    deduce_layout_bwd(diff, mask, grad_expected1, grad_expected2);

    megdnn_assert_eq_shape(grad_expected1, grad_data1);
    megdnn_assert_eq_dtype(grad_expected1, grad_data1);
    megdnn_assert_eq_shape(grad_expected2, grad_data2);
    megdnn_assert_eq_dtype(grad_expected2, grad_data2);
}

void WhereBackward::deduce_layout(
        const TensorLayout& diff, const TensorLayout& mask, TensorLayout& grad_data1,
        TensorLayout& grad_data2) {
    deduce_layout_bwd(diff, mask, grad_data1, grad_data2);
}

void WhereBackward::check_exec(
        const TensorLayout& diff, const TensorLayout& mask,
        const TensorLayout& grad_data1, const TensorLayout& grad_data2,
        size_t workspace_in_bytes) {
    check_layout_bwd(diff, mask, grad_data1, grad_data2);
    auto required_workspace_in_bytes =
            get_workspace_in_bytes(diff, mask, grad_data1, grad_data2);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
}

}  // namespace megdnn
