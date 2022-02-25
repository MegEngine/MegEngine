#include "megdnn/oprs/nn.h"
#include "src/common/utils.h"

using namespace megdnn;

void MaskConvForward::deduce_dtype(DType src, DType filter, DType, DType& dst) {
    check_or_deduce_dtype_fwd(src, filter, dst);
}

void MaskConvForward::deduce_layout(
        const TensorLayout& src, const TensorLayout& filter, const TensorLayout& mask,
        TensorLayout& dst) {
    deduce_layout_fwd(src, filter, dst);
    megdnn_assert(dst[2] == mask[0]);
    megdnn_assert(dst[3] == mask[1]);
}

MaskConvForward::CanonizedFilterMeta MaskConvForward::check_exec(
        const TensorLayout& src, const TensorLayout& filter, const TensorLayout& mask,
        const TensorLayout& dst, size_t workspace_in_bytes) {
    auto ret = check_layout_fwd(src, filter, dst);
    megdnn_assert(dst[2] == mask[0]);
    megdnn_assert(dst[3] == mask[1]);
    auto required_workspace_in_bytes = get_workspace_in_bytes(src, filter, mask, dst);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
    return ret;
}

void MaskPropagate::deduce_layout(const TensorLayout& src, TensorLayout& dst) {
    size_t oh, ow;
    auto p = param();
    infer_conv_shape2d(
            src[0], src[1], (p.kernel_h - 1) * p.dilate_h + 1,
            (p.kernel_w - 1) * p.dilate_w + 1, p.stride_h, p.stride_w, p.pad_h, p.pad_w,
            oh, ow);
    dst = TensorLayout{{oh, ow}, src.dtype};
}

// vim: syntax=cpp.doxygen
