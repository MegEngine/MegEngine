/**
 * \file dnn/src/common/correlation.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#include "megdnn/oprs.h"

#include "src/common/utils.h"

namespace megdnn {

void CorrelationBase::deduce_layout_fwd(
        const TensorLayout& data1, const TensorLayout& data2, TensorLayout& dst) {
    megdnn_assert_contiguous(data1);
    megdnn_assert_contiguous(data2);
    megdnn_assert_contiguous(dst);
    auto errmsg = [&]() {
        return megdnn_layout_msg(data1) + ", " + megdnn_layout_msg(data2) + ", " +
               megdnn_layout_msg(dst);
    };
    MEGDNN_MARK_USED_VAR(errmsg);
    using Format = CorrelationBase::Param::Format;
    megdnn_assert(param().format == Format::NCHW);
    auto data1_dtype = data1.dtype, data2_dtype = data2.dtype;
    megdnn_assert(
            data1_dtype == data2_dtype &&
            data1_dtype.category() == DTypeCategory::FLOAT);
    megdnn_assert(data1.ndim == 4_z, "%s", errmsg().c_str());
    megdnn_assert(data2.ndim == 4_z, "%s", errmsg().c_str());

    uint32_t pad_size = param().pad_size;
    uint32_t kernel_size = param().kernel_size;
    uint32_t stride1 = param().stride1;
    uint32_t stride2 = param().stride2;
    uint32_t max_displacement = param().max_displacement;

    int paddedbottomheight = data1[2] + 2 * pad_size;
    int paddedbottomwidth = data1[3] + 2 * pad_size;
    uint32_t kernel_radius = (kernel_size - 1) / 2;
    uint32_t border_size = max_displacement + kernel_radius;
    uint32_t top_width =
            ceil(static_cast<float>(paddedbottomwidth - border_size * 2) /
                 static_cast<float>(stride1));
    uint32_t top_height =
            ceil(static_cast<float>(paddedbottomheight - border_size * 2) /
                 static_cast<float>(stride1));
    uint32_t neighborhood_grid_radius = max_displacement / stride2;
    uint32_t neighborhood_grid_width = neighborhood_grid_radius * 2 + 1;
    uint32_t top_channels = neighborhood_grid_width * neighborhood_grid_width;
    megdnn_assert(top_width >= 1 && top_height >= 1);

    dst = TensorLayout{{data1[0], top_channels, top_height, top_width}, data1.dtype};
}

void CorrelationBase::check_layout_fwd(
        const TensorLayout& data1, const TensorLayout& data2, const TensorLayout& dst) {
    TensorLayout dst_expected;
    megdnn_assert_eq_dtype(data1, dst);
    megdnn_assert_eq_shape(data1, data2);
    deduce_layout_fwd(data1, data2, dst_expected);
    megdnn_assert_eq_shape(dst_expected, dst);
}

void CorrelationForward::deduce_layout(
        const TensorLayout& data1, const TensorLayout& data2, TensorLayout& dst) {
    deduce_layout_fwd(data1, data2, dst);
}

void CorrelationForward::check_exec(
        const TensorLayout& data1, const TensorLayout& data2, const TensorLayout& dst,
        size_t workspace_in_bytes) {
    check_layout_fwd(data1, data2, dst);
    auto required_workspace_in_bytes = get_workspace_in_bytes(data1, data2, dst);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
}

void CorrelationBackwardData1::check_exec(
        const TensorLayout& diff, const TensorLayout& data1, const TensorLayout& data2,
        const TensorLayout& grad1, size_t workspace_in_bytes) {
    check_layout_fwd(grad1, data2, diff);
    megdnn_assert_eq_shape(data1, data2);
    auto required_workspace_in_bytes =
            get_workspace_in_bytes(diff, data1, data2, grad1);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
}

void CorrelationBackwardData2::check_exec(
        const TensorLayout& diff, const TensorLayout& data1, const TensorLayout& data2,
        const TensorLayout& grad2, size_t workspace_in_bytes) {
    check_layout_fwd(data1, grad2, diff);
    megdnn_assert_eq_shape(data1, data2);
    auto required_workspace_in_bytes =
            get_workspace_in_bytes(diff, data1, data2, grad2);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
}

void CorrelationBackwardData2::deduce_layout(
        const TensorLayout& diff, const TensorLayout& data1, const TensorLayout& data2,
        TensorLayout& grad) {
    megdnn_assert_eq_shape(data1, data2);
    check_layout_fwd(data1, data2, diff);
    grad = data2;
}

void CorrelationBackwardData1::deduce_layout(
        const TensorLayout& diff, const TensorLayout& data1, const TensorLayout& data2,
        TensorLayout& grad) {
    megdnn_assert_eq_shape(data1, data2);
    check_layout_fwd(data1, data2, diff);
    grad = data1;
}

}  // namespace megdnn

// vim: syntax=cpp.doxygen
