#include "megdnn/basic_types.h"
#include "megdnn/oprs.h"
#include "unroll_macro.h"

#include "src/common/utils.h"

namespace megdnn {

using Param = GeneralNormBase::Param;

void GeneralNormBase::deduce_layout_fwd(
        const TensorLayout& data, const TensorLayout& weight, const TensorLayout& bias,
        TensorLayout& dst, TensorLayout& mean, TensorLayout& rstd) {
    MEGDNN_MARK_USED_VAR(weight);
    MEGDNN_MARK_USED_VAR(bias);

    TensorShape unnormalized_shape;
    size_t normalized_axis_start = param().axis_start;
    size_t normalized_axis_end = param().axis_end;
    size_t idx = 0;
    for (size_t i = 0; i < normalized_axis_start; ++i, ++idx)
        unnormalized_shape[idx] = data.shape[i];
    for (size_t i = normalized_axis_end; i < data.ndim; ++i, ++idx)
        unnormalized_shape[idx] = data.shape[i];
    TensorLayout unnormalized_layout =
            TensorLayout(unnormalized_shape, dtype::Float32());
    unnormalized_layout.ndim = idx;
    unnormalized_layout.init_contiguous_stride();

    dst = data;
    mean = unnormalized_layout;
    rstd = unnormalized_layout;
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

    TensorLayout unnormalized_layout = data;
    size_t normalized_axis_start = param().axis_start;
    size_t normalized_axis_end = param().axis_end;
    size_t idx = 0;
    for (size_t i = 0; i < normalized_axis_start; ++i, ++idx)
        unnormalized_layout.shape[idx] = unnormalized_layout.shape[i];
    for (size_t i = normalized_axis_end; i < unnormalized_layout.ndim; ++i, ++idx)
        unnormalized_layout.shape[idx] = unnormalized_layout.shape[i];
    unnormalized_layout.ndim = idx == 0 ? 1 : idx;

    megdnn_assert(unnormalized_layout.ndim == mean.ndim, "%s", errmsg().c_str());
    for (size_t i = 0; i < unnormalized_layout.ndim; ++i) {
        megdnn_assert(
                unnormalized_layout.shape[i] == mean.shape[i], "%s", errmsg().c_str());
    }
    if (param().affine) {
        for (size_t i = normalized_axis_start, j = 0; i < normalized_axis_end; ++i, ++j)
            megdnn_assert(data.shape[i] == weight.shape[j], "%s", errmsg().c_str());
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
    size_t required_workspace_in_bytes =
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

    TensorLayout unnormalized_layout = data;
    size_t normalized_axis_start = param().axis_start;
    size_t normalized_axis_end = param().axis_end;
    size_t idx = 0;
    for (size_t i = 0; i < normalized_axis_start; ++i, ++idx)
        unnormalized_layout.shape[idx] = unnormalized_layout.shape[i];
    for (size_t i = normalized_axis_end; i < unnormalized_layout.ndim; ++i, ++idx)
        unnormalized_layout.shape[idx] = unnormalized_layout.shape[i];
    unnormalized_layout.ndim = idx == 0 ? 1 : idx;

    megdnn_assert(unnormalized_layout.ndim == mean.ndim, "%s", errmsg().c_str());
    for (size_t i = 0; i < unnormalized_layout.ndim; ++i) {
        megdnn_assert(
                unnormalized_layout.shape[i] == mean.shape[i], "%s", errmsg().c_str());
    }
    if (param().affine) {
        for (size_t i = normalized_axis_start, j = 0; i < normalized_axis_end; ++i, ++j)
            megdnn_assert(data.shape[i] == weight.shape[j], "%s", errmsg().c_str());
    }
}

}  // namespace megdnn

// vim: syntax=cpp.doxygen
