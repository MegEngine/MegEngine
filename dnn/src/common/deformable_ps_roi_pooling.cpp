/**
 * \file dnn/src/common/deformable_ps_roi_pooling.cpp
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

namespace megdnn {

void DeformablePSROIPoolingBase::deduce_layout_fwd(const TensorLayout& data,
                                                   const TensorLayout& rois,
                                                   const TensorLayout& trans,
                                                   TensorLayout& out_data,
                                                   TensorLayout& out_count) {
    megdnn_assert_contiguous(data);
    megdnn_assert_contiguous(rois);
    megdnn_assert_contiguous(trans);

    auto errmsg = [&]() {
        return std::string("data: ") + megdnn_layout_msg(data) +
               ", rois: " + megdnn_layout_msg(rois) +
               ", trans: " + megdnn_layout_msg(trans) +
               ", out_data: " + megdnn_layout_msg(out_data) +
               ", out_count: " + megdnn_layout_msg(out_count);
    };

    MEGDNN_MARK_USED_VAR(data);
    MEGDNN_MARK_USED_VAR(rois);
    MEGDNN_MARK_USED_VAR(trans);
    MEGDNN_MARK_USED_VAR(out_data);
    MEGDNN_MARK_USED_VAR(out_count);
    MEGDNN_MARK_USED_VAR(out_count);
    MEGDNN_MARK_USED_VAR(errmsg);

    megdnn_assert(data.dtype.enumv() == DTypeEnum::Float32,
                  "DeformablePSROIPooling only support float32 input");
    megdnn_assert(data.ndim == 4_z, "invalid data shape, %s", errmsg().c_str());
    megdnn_assert(rois.ndim == 2_z && rois[1] == 5, "invalid rois shape, %s",
                  errmsg().c_str());
    megdnn_assert(trans.ndim == 4_z, "invalid trans shape, %s",
                  errmsg().c_str());

    if (!param().no_trans) {
        megdnn_assert(trans[1] == 2_z && trans[2] == param().pooled_h &&
                              trans[3] == param().pooled_w,
                      "invalid trans shape: %s", errmsg().c_str());
    }

    out_data = {{rois[0], data[1], param().pooled_h, param().pooled_w},
                data.dtype};
    out_count = out_data;
}

void DeformablePSROIPoolingBase::check_layout_fwd(const TensorLayout& data,
                                                  const TensorLayout& rois,
                                                  const TensorLayout& trans,
                                                  const TensorLayout& out_data,
                                                  const TensorLayout& out_count,
                                                  size_t workspace_in_bytes) {
    MEGDNN_MARK_USED_VAR(workspace_in_bytes);

    TensorLayout exp_out_data, exp_out_count;
    deduce_layout_fwd(data, rois, trans, exp_out_data, exp_out_count);

    megdnn_assert_eq_layout(out_data, exp_out_data);
    megdnn_assert_eq_layout(out_count, exp_out_count);
}

void DeformablePSROIPoolingForward::deduce_layout(const TensorLayout& data,
                                                  const TensorLayout& rois,
                                                  const TensorLayout& trans,
                                                  TensorLayout& out_data,
                                                  TensorLayout& out_count) {
    deduce_layout_fwd(data, rois, trans, out_data, out_count);
}

void DeformablePSROIPoolingForward::check_exec(const TensorLayout& data,
                                               const TensorLayout& rois,
                                               const TensorLayout& trans,
                                               const TensorLayout& out_data,
                                               const TensorLayout& out_count,
                                               size_t workspace_in_bytes) {
    check_layout_fwd(data, rois, trans, out_data, out_count,
                     workspace_in_bytes);
    auto required_workspace_in_bytes =
            get_workspace_in_bytes(data, rois, trans, out_data, out_count);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
}

void DeformablePSROIPoolingBackward::check_exec(
        const TensorLayout& data, const TensorLayout& rois,
        const TensorLayout& trans, const TensorLayout& out_diff,
        const TensorLayout& out_count, const TensorLayout& data_diff,
        const TensorLayout& trans_diff, size_t workspace_in_bytes) {
    check_layout_fwd(data_diff, rois, trans_diff, out_diff, out_count,
                     workspace_in_bytes);
    megdnn_assert_eq_layout(data, data_diff);
    megdnn_assert_eq_layout(trans, trans_diff);
    auto required_workspace_in_bytes = get_workspace_in_bytes(
            data, rois, trans, out_diff, out_count, data_diff, trans_diff);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
}

}  // namespace megdnn

// vim: syntax=cpp.doxygen
