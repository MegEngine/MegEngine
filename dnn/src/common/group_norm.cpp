#include "megdnn/oprs.h"

#include "src/common/utils.h"

namespace megdnn {

using Param = GroupNormBase::Param;

void GroupNormBase::deduce_layout_fwd(
        const TensorLayout& data, const TensorLayout& weight, const TensorLayout& bias,
        TensorLayout& dst, TensorLayout& mean, TensorLayout& rstd) {
    MEGDNN_MARK_USED_VAR(weight);
    MEGDNN_MARK_USED_VAR(bias);
    size_t N = data.shape[0];
    size_t group = param().group;
    TensorLayout unnormalized_layout({N, group}, dtype::Float32());
    dst = data;
    mean = unnormalized_layout;
    rstd = unnormalized_layout;
}

void GroupNormBase::check_layout_fwd(
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

    megdnn_assert(data.eq_layout(dst), "%s", errmsg().c_str());
    megdnn_assert(weight.eq_layout(bias), "%s", errmsg().c_str());
    megdnn_assert(mean.eq_layout(rstd), "%s", errmsg().c_str());

    auto p = param();
    size_t C = data.shape[1];
    size_t group = p.group;
    megdnn_assert(
            group > 0, "Expected num groups to be greater than 0, got %zu", group);
    megdnn_assert(
            C % group == 0,
            "Expected number of channels in input to be divisible by num_groups, but "
            "got Channel of shape %zu and num_groups= %zu",
            C, group);
}

void GroupNormForward::deduce_layout(
        const TensorLayout& data, const TensorLayout& weight, const TensorLayout& bias,
        TensorLayout& dst, TensorLayout& mean, TensorLayout& rstd) {
    deduce_layout_fwd(data, weight, bias, dst, mean, rstd);
}

void GroupNormForward::check_exec(
        const TensorLayout& data, const TensorLayout& weight, const TensorLayout& bias,
        const TensorLayout& dst, const TensorLayout& mean, const TensorLayout& rstd,
        size_t workspace_in_bytes) {
    check_layout_fwd(data, weight, bias, dst, mean, rstd);
    auto required_workspace_in_bytes =
            get_workspace_in_bytes(data, weight, bias, dst, mean, rstd);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
}

void GroupNormBackward::deduce_layout(
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

void GroupNormBackward::check_exec(
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

    megdnn_assert(data.eq_layout(ddata), "%s", errmsg().c_str());
    megdnn_assert(mean.eq_layout(rstd), "%s", errmsg().c_str());
    if (p.affine) {
        megdnn_assert(weight.eq_layout(dweight), "%s", errmsg().c_str());
        megdnn_assert(weight.eq_layout(dbias), "%s", errmsg().c_str());
    }
}

}  // namespace megdnn

// vim: syntax=cpp.doxygen
