#include "megdnn/oprs.h"
#include "src/common/cv/common.h"
#include "src/common/cv/helper.h"
#include "src/common/utils.h"

namespace megdnn {

void RemapBase::deduce_layout_fwd(
        const TensorLayout& src, const TensorLayout& map_xy, TensorLayout& dst) {
    size_t n = src.shape[0];
    size_t c, oh, ow;
    oh = map_xy.shape[1];
    ow = map_xy.shape[2];
    if (param().format == param::Remap::Format::NHWC) {
        c = src.shape[3];
        dst = TensorLayout(TensorShape({n, oh, ow, c}), src.dtype);
    } else if (param().format == param::Remap::Format::NCHW) {
        c = src.shape[1];
        dst = TensorLayout(TensorShape{n, c, oh, ow}, src.dtype, src.format);
    } else if (param().format == param::Remap::Format::NHWCD4) {
        c = src.shape[2];
        dst = TensorLayout{{n, oh, c, ow, 4}, src.dtype, src.format};
    } else {
        megdnn_throw("unsupport format");
    }
}

void RemapBase::check_layout_fwd(
        const TensorLayout& src, const TensorLayout& map_xy, const TensorLayout& dst) {
    auto errmsg = [&]() {
        return megdnn_layout_msg(src) + ", " + megdnn_layout_msg(map_xy) + ", " +
               megdnn_layout_msg(dst);
    };
    MEGDNN_MARK_USED_VAR(errmsg);
    megdnn_assert(src.ndim == dst.ndim);
    megdnn_assert(dst.dtype == src.dtype);
    megdnn_assert(dst.shape[0] == src.shape[0], "%s", errmsg().c_str());
    megdnn_assert(map_xy.shape[3] == 2);
    megdnn_assert(map_xy.shape[0] == src.shape[0]);
    megdnn_assert_contiguous(src);

    // map_xy only support floa32 type
    // map_xy always in NHWC format
    megdnn_assert(map_xy.dtype.enumv() == DTypeEnum::Float32);

    // In remap opr, H, W is same as H W in map_xy.
    if (param().format == param::Remap::Format::NHWC) {
        megdnn_assert(src.shape[3] == dst.shape[3], "%s", errmsg().c_str());
        megdnn_assert(
                dst.shape[2] == map_xy.shape[2] && dst.shape[1] == map_xy.shape[1],
                "%s", errmsg().c_str());
    } else if (param().format == param::Remap::Format::NCHW) {
        megdnn_assert(src.shape[1] == dst.shape[1], "%s", errmsg().c_str());
        megdnn_assert(
                dst.shape[2] == map_xy.shape[1] && dst.shape[3] == map_xy.shape[2],
                "%s", errmsg().c_str());
    } else if (param().format == param::Remap::Format::NHWCD4) {
        megdnn_assert(src.shape[2] == dst.shape[2], "%s", errmsg().c_str());
        megdnn_assert(src.ndim == 5_z, "%s", errmsg().c_str());
        megdnn_assert(dst.ndim == 5_z, "%s", errmsg().c_str());
        megdnn_assert(param().format == Param::Format::NHWCD4);
    } else {
        megdnn_throw("unsupport format");
    }
}

void Remap::deduce_layout(
        const TensorLayout& src, const TensorLayout& map_xy, TensorLayout& dst) {
    deduce_layout_fwd(src, map_xy, dst);
}

void Remap::check_exec(
        const TensorLayout& src, const TensorLayout& map_xy, const TensorLayout& dst,
        size_t workspace_in_bytes) {
    check_layout_fwd(src, map_xy, dst);
    auto required_workspace_in_bytes = get_workspace_in_bytes(src, map_xy, dst);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
}

void RemapBackwardData::check_exec(
        const TensorLayout& map_xy, const TensorLayout& diff, const TensorLayout& grad,
        size_t workspace_in_bytes) {
    check_layout_fwd(grad, map_xy, diff);
    megdnn_assert(
            grad.dtype ==
                    dtype::Float32() DNN_INC_FLOAT16(|| grad.dtype == dtype::BFloat16())
                            DNN_INC_FLOAT16(|| grad.dtype == dtype::Float16()),
            "Backward Remap only supports Float32/BFloat16.");
    auto required_workspace_in_bytes = get_workspace_in_bytes(map_xy, diff, grad);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
}

void RemapBackwardMat::check_exec(
        const TensorLayout& src, const TensorLayout& map_xy, const TensorLayout& diff,
        const TensorLayout& grad, size_t workspace_in_bytes) {
    check_layout_fwd(src, map_xy, diff);
    megdnn_assert_eq_layout(map_xy, grad);
    megdnn_assert(
            grad.dtype ==
                    dtype::Float32() DNN_INC_FLOAT16(|| grad.dtype == dtype::BFloat16())
                            DNN_INC_FLOAT16(|| grad.dtype == dtype::Float16()),
            "Backward Remap only supports Float32/BFloat16.");
    auto required_workspace_in_bytes = get_workspace_in_bytes(src, map_xy, diff, grad);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
}

}  // namespace megdnn

// vim: syntax=cpp.doxygen
