/**
 * \file dnn/src/common/roi_pooling.cpp
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

void ROIPoolingBase::check_layout_fwd(const TensorLayout &src,
        const TensorLayout &rois,
        const TensorLayout &dst,
        const TensorLayout &index)
{
    // all should be contiguous
    megdnn_assert_contiguous(src);
    megdnn_assert_contiguous(rois);
    megdnn_assert_contiguous(dst);
    megdnn_assert_contiguous(index);
    auto errmsg = [&]() {
        return megdnn_layout_msg(src) + ", "
        + megdnn_layout_msg(rois) + ", "
        + megdnn_layout_msg(dst) + ", "
        + megdnn_layout_msg(index);
    };
    MEGDNN_MARK_USED_VAR(errmsg);
    // src
    megdnn_assert(src.ndim == 4_z, "%s", errmsg().c_str());
    auto C = src.shape[1];
    // rois
    megdnn_assert(rois.ndim == 2_z, "%s", errmsg().c_str());
    auto M = rois.shape[0];
    megdnn_assert(rois[1] == 5_z, "%s", errmsg().c_str());
    // dst
    megdnn_assert(dst[0] == M, "%s", errmsg().c_str());
    megdnn_assert(dst[1] == C, "%s", errmsg().c_str());
    // index
    megdnn_assert_eq_shape(index, dst);

    megdnn_assert(src.dtype.category() == DTypeCategory::FLOAT);
    megdnn_assert(rois.dtype.category() == DTypeCategory::FLOAT);
    megdnn_assert(dst.dtype.category() == DTypeCategory::FLOAT);
    megdnn_assert(index.dtype == dtype::Int32());
}

void ROIPoolingForward::check_exec(const TensorLayout &src,
        const TensorLayout &rois,
        const TensorLayout &dst,
        const TensorLayout &index,
        size_t workspace_in_bytes)
{
    check_layout_fwd(src, rois, dst, index);
    auto required_workspace_in_bytes = get_workspace_in_bytes(src,
            rois, dst, index);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
}

void ROIPoolingBackward::check_exec(const TensorLayout &diff,
        const TensorLayout &src,
        const TensorLayout &rois,
        const TensorLayout &index,
        const TensorLayout &grad,
        size_t workspace_in_bytes)
{
    check_layout_fwd(src, rois, diff, index);
    megdnn_assert_eq_layout(src, grad);
    auto required_workspace_in_bytes = get_workspace_in_bytes(diff,
            src, rois, index, grad);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
}

} // namespace megdnn

// vim: syntax=cpp.doxygen
