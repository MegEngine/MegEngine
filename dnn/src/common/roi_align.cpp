/**
 * \file dnn/src/common/roi_align.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "megdnn/oprs.h"

#include "src/common/utils.h"

namespace megdnn {

void ROIAlignBase::deduce_layout_fwd(const TensorLayout& src,
                                     const TensorLayout& rois,
                                     TensorLayout& dst, TensorLayout& index) {
    megdnn_assert_contiguous(src);
    megdnn_assert_contiguous(rois);
    megdnn_assert_contiguous(dst);
    megdnn_assert_contiguous(index);
    auto errmsg = [&]() {
        return megdnn_layout_msg(src) + ", " + megdnn_layout_msg(rois) + ", " +
               megdnn_layout_msg(dst) + ", " + megdnn_layout_msg(index);
    };
    MEGDNN_MARK_USED_VAR(errmsg);
    using Format = ROIAlignBase::Param::Format;
    megdnn_assert(param().format == Format::NCHW);
    auto src_dtype = src.dtype, rois_dtype = rois.dtype;
    megdnn_assert(src_dtype == rois_dtype &&
                  src_dtype.category() == DTypeCategory::FLOAT);
    megdnn_assert(src.ndim == 4_z, "%s", errmsg().c_str());
    size_t channels = src.shape[1];
    megdnn_assert(rois.ndim == 2_z, "%s", errmsg().c_str());
    // rois shape: bid, x0, y0, x1, y1
    megdnn_assert(rois[1] == 5_z, "%s", errmsg().c_str());
    size_t M = rois[0];
    size_t pooled_height = param().pooled_height;
    size_t pooled_width = param().pooled_width;
    dst = TensorLayout{{M, channels, pooled_height, pooled_width}, src.dtype};
    index = dst;
    index.dtype = dtype::Int32();
}

void ROIAlignBase::check_layout_fwd(const TensorLayout& src,
                                    const TensorLayout& rois,
                                    const TensorLayout& dst,
                                    const TensorLayout& index) {
    TensorLayout dst_expected, index_expected;
    megdnn_assert_eq_dtype(src, dst);
    deduce_layout_fwd(src, rois, dst_expected, index_expected);
    megdnn_assert_eq_shape(dst_expected, dst);
    megdnn_assert_eq_shape(index_expected, index);
    megdnn_assert(index.dtype == dtype::Int32());
}

void ROIAlignForward::deduce_layout(const TensorLayout& src,
                                    const TensorLayout& rois, TensorLayout& dst,
                                    TensorLayout& index) {
    deduce_layout_fwd(src, rois, dst, index);
}

void ROIAlignForward::check_exec(const TensorLayout& src,
                                 const TensorLayout& rois,
                                 const TensorLayout& dst,
                                 const TensorLayout& index,
                                 size_t workspace_in_bytes) {
    check_layout_fwd(src, rois, dst, index);
    auto required_workspace_in_bytes =
            get_workspace_in_bytes(src, rois, dst, index);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
}

void ROIAlignBackward::check_exec(const TensorLayout& diff,
                                  const TensorLayout& rois,
                                  const TensorLayout& index,
                                  const TensorLayout& grad,
                                  size_t workspace_in_bytes) {
    check_layout_fwd(grad, rois, diff, index);
    auto required_workspace_in_bytes =
            get_workspace_in_bytes(diff, rois, index, grad);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
}

}  // namespace megdnn

// vim: syntax=cpp.doxygen
