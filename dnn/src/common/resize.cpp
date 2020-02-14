/**
 * \file dnn/src/common/resize.cpp
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

void ResizeBase::check_layout_fwd(const TensorLayout& src,
                                  const TensorLayout& dst) {
    auto errmsg = [&]() {
        return megdnn_layout_msg(src) + ", " + ", " + megdnn_layout_msg(dst);
    };
    MEGDNN_MARK_USED_VAR(errmsg);

    megdnn_assert(dst.dtype == src.dtype && dst.shape[0] == src.shape[0], "%s",
                  errmsg().c_str());
    if (param().format == Param::Format::NCHW) {
        megdnn_assert(dst.shape[1] == src.shape[1], "%s", errmsg().c_str());
        megdnn_assert(param().imode ==
                      param::Resize::InterpolationMode::INTER_LINEAR);
    } else if (param().format == Param::Format::NHWC) {
        megdnn_assert(dst.shape[3] == src.shape[3], "%s", errmsg().c_str());
    } else if (param().format == Param::Format::NCHW4) {
        megdnn_assert(src.ndim == 5);
        megdnn_assert(src.dtype.enumv() == DTypeEnum::QuantizedS8);
        megdnn_assert(src.shape[4] == 4);
        megdnn_assert(dst.shape[4] == 4);
    } else {
        megdnn_assert(param().format == Param::Format::NHWCD4,
                      "invalid resize tensor format");
        megdnn_assert(param().imode ==
                      param::Resize::InterpolationMode::INTER_LINEAR);
        megdnn_assert(dst.shape[2] == src.shape[2], "%s", errmsg().c_str());
    }
}

void Resize::check_exec(const TensorLayout& src, const TensorLayout& dst,
                        size_t workspace_in_bytes) {
    check_layout_fwd(src, dst);
    auto required_workspace_in_bytes = get_workspace_in_bytes(src, dst);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
}

void ResizeBackward::check_exec(const TensorLayout& diff,
                                const TensorLayout& grad,
                                size_t workspace_in_bytes) {
    check_layout_fwd(grad, diff);
    auto required_workspace_in_bytes = get_workspace_in_bytes(diff, grad);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
    megdnn_assert(param().format == Param::Format::NCHW &&
                          grad.dtype == dtype::Float32(),
                  "Backward resize only supports Float32 and NCHW.");
}

std::pair<float, int> ResizeBase::get_origin_coord(float scale, int size,
                                                   int idx) {
    //! copy from resize_cv.cpp
    float alpha = (idx + 0.5f) / scale - 0.5f;
    int origin_idx = static_cast<int>(floor(alpha));
    alpha -= origin_idx;
    if (origin_idx < 0) {
        origin_idx = 0;
        alpha = 0;
    } else if (origin_idx + 1 >= size) {
        origin_idx = size - 2;
        alpha = 1;
    }

    return {alpha, origin_idx};
}

}  // namespace megdnn

// vim: syntax=cpp.doxygen
