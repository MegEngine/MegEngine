#include "megdnn/oprs.h"
#include "unroll_macro.h"

#include "src/common/utils.h"

namespace megdnn {

using Param = GeneralNormBase::Param;

void GeneralNormBase::deduce_layout_fwd_impl(
        const TensorLayout& data, const Param& p, TensorLayout& dst, TensorLayout& mean,
        TensorLayout& rstd) {
    TensorLayout unnormalized_layout = data;
    unnormalized_layout.remove_axis_inplace(p.normalized_axis);
    unnormalized_layout.init_contiguous_stride();
    dst = data;
    mean = unnormalized_layout;
    rstd = unnormalized_layout;
}

void GeneralNormBase::deduce_layout_fwd(
        const TensorLayout& data, const TensorLayout& weight, const TensorLayout& bias,
        TensorLayout& dst, TensorLayout& mean, TensorLayout& rstd) {
    MEGDNN_MARK_USED_VAR(weight);
    MEGDNN_MARK_USED_VAR(bias);
    deduce_layout_fwd_impl(data, param(), dst, mean, rstd);
}

void GeneralNormBase::check_layout_fwd(
        const TensorLayout& data, const TensorLayout& weight, const TensorLayout& bias,
        const TensorLayout& dst, const TensorLayout& mean, const TensorLayout& rstd) {
    megdnn_assert_contiguous(data);
    megdnn_assert_contiguous(weight);
    megdnn_assert_contiguous(bias);
    megdnn_assert_contiguous(dst);
    megdnn_assert_contiguous(mean);
    megdnn_assert_contiguous(rstd);
    auto errmsg = [&]() {
        return megdnn_layout_msg(data) + ", " + megdnn_layout_msg(weight) + ", " +
               megdnn_layout_msg(bias) + ", " + megdnn_layout_msg(dst) + ", " +
               megdnn_layout_msg(mean) + ", " + megdnn_layout_msg(rstd);
    };
    MEGDNN_MARK_USED_VAR(errmsg);

    auto equal_layout = [](const TensorLayout& lhs, const TensorLayout& rhs) -> bool {
        if (!(lhs.ndim == rhs.ndim && lhs.dtype == rhs.dtype &&
              lhs.format == rhs.format))
            return false;
        for (size_t i = 0; i < lhs.ndim; ++i) {
            if (lhs.shape[i] != rhs.shape[i] || lhs.stride[i] != rhs.stride[i]) {
                return false;
            }
        }
        return true;
    };

    megdnn_assert(equal_layout(data, dst), "%s", errmsg().c_str());
    megdnn_assert(equal_layout(weight, bias), "%s", errmsg().c_str());
    megdnn_assert(equal_layout(mean, rstd), "%s", errmsg().c_str());

    auto p = param();
    megdnn_assert(
            p.normalized_axis < data.ndim,
            "the axis of normalized should be smaller than the dimension of input");

    TensorLayout unnormalized_layout = data;
    unnormalized_layout.remove_axis_inplace(p.normalized_axis);
    for (size_t i = 0; i < data.ndim - 1; ++i) {
        megdnn_assert(
                unnormalized_layout.shape[i] == mean.shape[i], "%s", errmsg().c_str());
    }
    if (p.affine) {
        megdnn_assert(
                data.shape[p.normalized_axis] == weight.shape[0], "%s",
                errmsg().c_str());
    }
}

void GeneralNormForward::deduce_layout(
        const TensorLayout& data, const TensorLayout& weight, const TensorLayout& bias,
        TensorLayout& dst, TensorLayout& mean, TensorLayout& rstd) {
    deduce_layout_fwd(data, weight, bias, dst, mean, rstd);
}

void GeneralNormForward::check_exec(
        const TensorLayout& data, const TensorLayout& weight, const TensorLayout& bias,
        const TensorLayout& dst, const TensorLayout& mean, const TensorLayout& rstd,
        size_t workspace_in_bytes) {
    check_layout_fwd(data, weight, bias, dst, mean, rstd);
    auto required_workspace_in_bytes =
            get_workspace_in_bytes(data, weight, bias, dst, mean, rstd);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
}

void GeneralNormBackward::deduce_layout(
        const TensorLayout& diff, const TensorLayout& data, const TensorLayout& weight,
        const TensorLayout& mean, const TensorLayout& rstd, TensorLayout& ddata,
        TensorLayout& dweight, TensorLayout& dbias) {
    MEGDNN_MARK_USED_VAR(diff);
    MEGDNN_MARK_USED_VAR(mean);
    MEGDNN_MARK_USED_VAR(rstd);
    ddata = data;
    dweight = weight;
    dbias = weight;
}

void GeneralNormBackward::check_exec(
        const TensorLayout& diff, const TensorLayout& data, const TensorLayout& weight,
        const TensorLayout& mean, const TensorLayout& rstd, const TensorLayout& ddata,
        const TensorLayout& dweight, const TensorLayout& dbias,
        size_t workspace_in_bytes) {
    auto p = param();
    auto required_workspace_in_bytes = get_workspace_in_bytes(
            diff, data, weight, mean, rstd, ddata, dweight, dbias);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);

    megdnn_assert_contiguous(diff);
    megdnn_assert_contiguous(data);
    megdnn_assert_contiguous(mean);
    megdnn_assert_contiguous(rstd);
    megdnn_assert_contiguous(ddata);
    if (p.affine) {
        megdnn_assert_contiguous(weight);
        megdnn_assert_contiguous(dweight);
        megdnn_assert_contiguous(dbias);
    }

    auto errmsg = [&]() {
        return megdnn_layout_msg(diff) + ", " + megdnn_layout_msg(data) + ", " +
               megdnn_layout_msg(weight) + ", " + megdnn_layout_msg(mean) + ", " +
               megdnn_layout_msg(rstd) + ", " + megdnn_layout_msg(ddata) + ", " +
               megdnn_layout_msg(dweight) + ", " + megdnn_layout_msg(dbias);
    };
    MEGDNN_MARK_USED_VAR(errmsg);

    auto equal_layout = [](const TensorLayout& lhs, const TensorLayout& rhs) -> bool {
        if (!(lhs.ndim == rhs.ndim && lhs.dtype == rhs.dtype &&
              lhs.format == rhs.format))
            return false;
        for (size_t i = 0; i < lhs.ndim; ++i) {
            if (lhs.shape[i] != rhs.shape[i] || lhs.stride[i] != rhs.stride[i]) {
                return false;
            }
        }
        return true;
    };

    megdnn_assert(equal_layout(data, ddata), "%s", errmsg().c_str());
    megdnn_assert(equal_layout(mean, rstd), "%s", errmsg().c_str());
    if (p.affine) {
        megdnn_assert(equal_layout(weight, dweight), "%s", errmsg().c_str());
        megdnn_assert(equal_layout(weight, dbias), "%s", errmsg().c_str());
    }

    megdnn_assert(
            p.normalized_axis < data.ndim,
            "the axis of normalized should be smaller than the dimension of input");

    TensorLayout unnormalized_layout = data;
    unnormalized_layout.remove_axis_inplace(p.normalized_axis);
    for (size_t i = 0; i < data.ndim - 1; ++i) {
        megdnn_assert(
                unnormalized_layout.shape[i] == mean.shape[i], "%s", errmsg().c_str());
    }
    if (p.affine) {
        megdnn_assert(
                data.shape[p.normalized_axis] == weight.shape[0], "%s",
                errmsg().c_str());
    }
}

}  // namespace megdnn

// vim: syntax=cpp.doxygen
