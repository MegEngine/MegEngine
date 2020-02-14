/**
 * \file dnn/src/common/deformable_conv.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megdnn/oprs/nn.h"
#include "src/common/utils.h"

using namespace megdnn;

using CanonizedFilterMeta = DeformableConvBase::CanonizedFilterMeta;

namespace {

template <typename Param>
std::string get_errmsg(const TensorLayout& src, const TensorLayout& filter,
                       const TensorLayout& offset, const TensorLayout& mask,
                       const TensorLayout& dst, const Param& param) {
    MEGDNN_MARK_USED_VAR(src);
    MEGDNN_MARK_USED_VAR(filter);
    MEGDNN_MARK_USED_VAR(dst);
    return megdnn_layout_msg(src) + ", " + megdnn_layout_msg(filter) + ", " +
           megdnn_layout_msg(offset) + ", " + megdnn_layout_msg(mask) + ", " +
           megdnn_layout_msg(dst) + ", " + megdnn_mangle("only support nchw") +
           ", " + megdnn_mangle("group=") + std::to_string(param.group) + ", " +
           megdnn_mangle("deformable_group=") +
           std::to_string(param.deformable_group) + ", " +
           megdnn_mangle("pad_h=") + std::to_string(param.pad_h) + ", " +
           megdnn_mangle("pad_w=") + std::to_string(param.pad_w) + ", " +
           megdnn_mangle("stride_h=") + std::to_string(param.stride_h) + ", " +
           megdnn_mangle("stride_w=") + std::to_string(param.stride_w) + ", " +
           megdnn_mangle("dilate_h=") + std::to_string(param.dilate_h) + ", " +
           megdnn_mangle("dilate_w=") + std::to_string(param.dilate_w);
}

template <typename Param>
void make_canonized_filter_meta_nchw(size_t src_ndim,
                                     const TensorLayout& filter,
                                     const Param& param,
                                     CanonizedFilterMeta& ret) {
    megdnn_assert(param.mode == Param::Mode::CROSS_CORRELATION,
                  "only support CROSS_CORRELATION mode");

    megdnn_assert(param.format == Param::Format::NCHW,
                  "only support nchw input layout");

    size_t flt_start, flt_spatial_start, ocpg_pos, icpg_pos;

    flt_start = 0, flt_spatial_start = 2;
    ocpg_pos = 0, icpg_pos = 1;

    if (param.sparse == Param::Sparse::GROUP)
        flt_start = 1;

    ret.spatial_ndim = src_ndim - 2;

    megdnn_assert(
            ret.spatial_ndim == 2,
            "only 2D convolution is supported, and imput should be 4-dim; "
            "got input dim = %zu",
            src_ndim);

    ret.ocpg = filter[flt_start + ocpg_pos];
    ret.icpg = filter[flt_start + icpg_pos];

    auto dilation = ret.dilation;

    for (size_t i = 0; i < ret.spatial_ndim; ++i) {
        megdnn_assert(dilation[i] > 0,
                      "invalid dilation on spatial dim %zu, %u", i,
                      dilation[i]);
        ret.spatial[i] = filter[i + flt_start + flt_spatial_start];
        ret.dilated_spatial[i] = (ret.spatial[i] - 1) * dilation[i] + 1;
    }
}

}  // namespace

namespace megdnn {

CanonizedFilterMeta DeformableConvBase::make_canonized_filter_meta(
        size_t src_ndim, const TensorLayout& filter,
        const TensorLayout& offset) const {
    megdnn_assert_contiguous(filter);

    CanonizedFilterMeta ret;
    ret.group = 1;
    ret.dtype = filter.dtype;
    ret.stride[0] = param().stride_h;
    ret.stride[1] = param().stride_w;
    ret.padding[0] = param().pad_h;
    ret.padding[1] = param().pad_w;
    ret.dilation[0] = param().dilate_h;
    ret.dilation[1] = param().dilate_w;

    if (param().sparse == Param::Sparse::GROUP) {
        megdnn_assert(filter.ndim == 5,
                      "filter dim should be 5 for group conv");
        ret.group = filter[0];
    }

    make_canonized_filter_meta_nchw(src_ndim, filter, param(), ret);

    auto fh = ret.spatial[0];
    auto fw = ret.spatial[1];

    ret.deformable_group = offset[1] / (2 * fh * fw);

    return ret;
}

void DeformableConvBase::deduce_layout_fwd(const TensorLayout& im,
                                           const TensorLayout& filter,
                                           const TensorLayout& offset,
                                           const TensorLayout& mask,
                                           TensorLayout& dst) {
    // im shape: (n, IC, IH, IW)
    megdnn_assert(im.ndim == 4, "invalid src layout: %s",
                  megdnn_layout_msg(im).c_str());
    // filter shape: (OC, IC, FH, FW) or (g, OC/g, IC/g, FH, FW)
    megdnn_assert(filter.ndim == 4 || filter.ndim == 5,
                  "invalid filter layout: %s",
                  megdnn_layout_msg(filter).c_str());
    // offset shape: (N, 2*dg*FH*FW, OH, OW)
    megdnn_assert(offset.ndim == 4, "invalid offset layout: %s",
                  megdnn_layout_msg(offset).c_str());
    // mask shape: (N, dg*FH*FW, OH, OW)
    megdnn_assert(mask.ndim == 4, "invalid mask layout: %s",
                  megdnn_layout_msg(mask).c_str());

    size_t n = im.shape[0], ic = im.shape[1];
    size_t ih = im.shape[2], iw = im.shape[3];
    size_t dh = param().dilate_h, dw = param().dilate_w;
    size_t ph = param().pad_h, pw = param().pad_w;
    size_t sh = param().stride_h, sw = param().stride_w;

    auto&& fm = make_canonized_filter_meta(im.ndim, filter, offset);
    size_t fh = fm.spatial[0], fw = fm.spatial[1];

    size_t kh = 1 + (fh - 1) * dh;
    size_t kw = 1 + (fw - 1) * dw;

    size_t group = fm.group;
    size_t deformable_group = fm.deformable_group;

    size_t icpg = fm.icpg, ocpg = fm.ocpg;
    size_t oc = group * ocpg;
    size_t oh = (ih + ph * 2 - kh) / sh + 1;
    size_t ow = (iw + pw * 2 - kw) / sw + 1;

    megdnn_assert(group > 0 && deformable_group > 0,
                  "group and deformable group should > 0");
    megdnn_assert(ic == icpg * group, "im ic != group * icpg of filter");
    megdnn_assert(ic % deformable_group == 0, "ic %% deformable_group != 0");
    megdnn_assert(oc % deformable_group == 0, "oc %% deformable_group != 0");

    megdnn_assert(
            (offset[1] % (2 * fh * fw) == 0) && (mask[1] % (fh * fw) == 0),
            "invalid deformable group deduced from offset(%s) or mask(%s)",
            megdnn_layout_msg(offset).c_str(), megdnn_layout_msg(mask).c_str());

    megdnn_assert((offset[1] / (2 * fh * fw)) == (mask[1] / (fh * fw)),
                  "offset(%s) and mask(%s) should have same deformable group",
                  megdnn_layout_msg(offset).c_str(),
                  megdnn_layout_msg(mask).c_str());

    megdnn_assert((offset[2] == mask[2]) && (offset[3] == mask[3]),
                  "offset(%s) and mask(%s) should have same spatial dim",
                  megdnn_layout_msg(offset).c_str(),
                  megdnn_layout_msg(mask).c_str());
    megdnn_assert(oh == offset[2], "deduced oh(%zu) != offset oh(%zu)", oh,
                  offset[2]);
    megdnn_assert(ow == offset[3], "deduced ow(%zu) != offset ow(%zu)", ow,
                  offset[3]);
    dst.ndim = 4;

    dst = {{n, oc, oh, ow}, im.dtype};
}
void DeformableConvBase::check_layout_fwd(const TensorLayout& im,
                                          const TensorLayout& filter,
                                          const TensorLayout& offset,
                                          const TensorLayout& mask,
                                          const TensorLayout& dst) {
    auto& im_dtype = im.dtype;
    TensorLayout dst_expected;
    megdnn_assert(im_dtype.enumv() == DTypeEnum::Float32,
                  "DeformableConv only support float32 input");
    megdnn_assert_eq_dtype(im, dst);
    megdnn_assert_eq_dtype(im, filter);
    megdnn_assert_eq_dtype(im, dst);
    megdnn_assert_eq_dtype(im, offset);
    megdnn_assert_eq_dtype(im, mask);
    deduce_layout_fwd(im, filter, offset, mask, dst_expected);
    megdnn_assert_eq_layout(dst_expected, dst);
}

void DeformableConvForward::deduce_layout(const TensorLayout& im,
                                          const TensorLayout& filter,
                                          const TensorLayout& offset,
                                          const TensorLayout& mask,
                                          TensorLayout& dst) {
    deduce_layout_fwd(im, filter, offset, mask, dst);
    return;
}

CanonizedFilterMeta DeformableConvForward::check_exec(
        const TensorLayout& im, const TensorLayout& filter,
        const TensorLayout& offset, const TensorLayout& mask,
        const TensorLayout& dst, size_t workspace_in_bytes) {
    auto ret = make_canonized_filter_meta(im.ndim, filter, offset);
    auto required_workspace_in_bytes =
            get_workspace_in_bytes(im, filter, offset, mask, dst);
    check_layout_fwd(im, filter, offset, mask, dst);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
    return ret;
}

CanonizedFilterMeta DeformableConvBackwardFilter::check_exec(
        const TensorLayout& im, const TensorLayout& offset,
        const TensorLayout& mask, const TensorLayout& out_grad,
        const TensorLayout& filter_grad, size_t workspace_in_bytes) {
    check_layout_fwd(im, filter_grad, offset, mask, out_grad);
    // check dtype
    megdnn_assert_eq_dtype(im, filter_grad);

    auto ret = make_canonized_filter_meta(im.ndim, filter_grad, offset);
    auto required_workspace_in_bytes =
            get_workspace_in_bytes(im, offset, mask, out_grad, filter_grad);

    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
    return ret;
}

CanonizedFilterMeta DeformableConvBackwardData::check_exec(
        const TensorLayout& im, const TensorLayout& filter,
        const TensorLayout& offset, const TensorLayout& mask,
        const TensorLayout& out_grad, const TensorLayout& im_grad,
        const TensorLayout& offset_grad, const TensorLayout& mask_grad,
        size_t workspace_in_bytes) {
    check_layout_fwd(im, filter, offset, mask, out_grad);

    // check dtype
    megdnn_assert_eq_dtype(im, im_grad);
    megdnn_assert_eq_dtype(im, offset_grad);
    megdnn_assert_eq_dtype(im, mask_grad);

    // check layout
    megdnn_assert(im.shape == im_grad.shape, "invalid im_grad shape: %s",
                  megdnn_layout_msg(im_grad).c_str());
    megdnn_assert(offset.shape == offset_grad.shape,
                  "invalid offset_grad shape: %s",
                  megdnn_layout_msg(offset_grad).c_str());
    megdnn_assert(mask.shape == mask_grad.shape, "invalid mask_grad shape: %s",
                  megdnn_layout_msg(mask_grad).c_str());

    auto ret = make_canonized_filter_meta(im.ndim, filter, offset);
    auto required_workspace_in_bytes =
            get_workspace_in_bytes(im, filter, offset, mask, out_grad, im_grad,
                                   offset_grad, mask_grad);

    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
    return ret;
}
}  // namespace megdnn

// vim: syntax=cpp.doxygen
