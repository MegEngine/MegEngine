/**
 * \file dnn/src/common/separableFilter.cpp
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

void SeparableFilterBase::deduce_layout_fwd(const TensorLayout& src,
                                            const TensorLayout& filter_x,
                                            const TensorLayout& filter_y,
                                            TensorLayout& dst) {
    auto errmsg = [&]() {
        return megdnn_layout_msg(src) + ", " + megdnn_layout_msg(filter_x) +
               ", " + megdnn_layout_msg(dst) + ", " +
               megdnn_mangle("borderMode=") +
               std::to_string((int)(param().borderMode)) + ", " +
               megdnn_mangle("ksize_h=") + std::to_string(param().ksize_h) +
               ", " + megdnn_mangle("ksize_w=") +
               std::to_string(param().ksize_w) + ", " +
               megdnn_mangle("anchor_h=") + std::to_string(param().anchor_h) +
               ", " + megdnn_mangle("anchor_w=") +
               std::to_string(param().anchor_w);
    };
    MEGDNN_MARK_USED_VAR(errmsg);
    megdnn_assert_contiguous(src);
    megdnn_assert_contiguous(filter_x);
    megdnn_assert_contiguous(filter_y);
    megdnn_assert(src.ndim == 4_z, "%s", errmsg().c_str());
    megdnn_assert(param().format == Param::Format::NHWC,
                  "Only NHWC was supported by now");
    size_t n = src[0];
    size_t ih = src[1];
    size_t iw = src[2];
    size_t ic = src[3];
    dst = TensorLayout(TensorShape({n, ih, iw, ic}), src.dtype);
}

void SeparableFilterBase::check_layout_fwd(const TensorLayout& src,
                                           const TensorLayout& filter_x,
                                           const TensorLayout& filter_y,
                                           const TensorLayout& dst) {
    TensorLayout dst_expected;
    megdnn_assert_eq_layout(src, dst);
    deduce_layout_fwd(src, filter_x, filter_y, dst_expected);
    megdnn_assert_eq_layout(dst_expected, dst);
}

void SeparableFilterForward::deduce_layout(const TensorLayout& src,
                                           const TensorLayout& filter_x,
                                           const TensorLayout& filter_y,
                                           TensorLayout& dst) {
    deduce_layout_fwd(src, filter_x, filter_y, dst);
}

void SeparableFilterForward::check_exec(const TensorLayout& src,
                                        const TensorLayout& filter_x,
                                        const TensorLayout& filter_y,
                                        const TensorLayout& dst,
                                        size_t workspace_in_bytes) {
    megdnn_assert(param().ksize_h > 0 && (param().ksize_h & 1));
    megdnn_assert(param().ksize_w > 0 && (param().ksize_w & 1));
    check_layout_fwd(src, filter_x, filter_y, dst);
    auto required_workspace_in_bytes =
            get_workspace_in_bytes(src, filter_x, filter_y, dst);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
}

} // namespace megdnn

// vim: syntax=cpp.doxygen
