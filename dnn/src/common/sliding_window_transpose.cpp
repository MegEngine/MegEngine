/**
 * \file dnn/src/common/sliding_window_transpose.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "megdnn/oprs.h"

#include "src/common/utils.h"

namespace megdnn {

void SlidingWindowTransposeBase::deduce_layout_fwd(const TensorLayout &src,
        TensorLayout &dst)
{
    auto errmsg = [&]() {
        return megdnn_layout_msg(src) + ", " +
               "out_h=" + std::to_string(param().out_h) + ", " +
               "out_w=" + std::to_string(param().out_w) + ", " +
               "pad_h=" + std::to_string(param().pad_h) + ", " +
               "pad_w=" + std::to_string(param().pad_w) + ", " +
               "stride_h=" + std::to_string(param().stride_h) + ", " +
               "stride_w=" + std::to_string(param().stride_w) + ", " +
               "window_h=" + std::to_string(param().window_h) + ", " +
               "window_w=" + std::to_string(param().window_w);
    };
    MEGDNN_MARK_USED_VAR(errmsg);
    megdnn_assert_contiguous(src);
    megdnn_assert(src.ndim == 6_z, "%s", errmsg().c_str());
    size_t n = src[0], ic = src[1];
    size_t oh = this->param().out_h;
    size_t ow = this->param().out_w;

    dst = TensorLayout(TensorShape({n, ic, oh, ow}), src.dtype);
}

void SlidingWindowTransposeBase::check_layout_fwd(const TensorLayout &src,
        const TensorLayout &dst)
{
    TensorLayout dst_expected;
    deduce_layout_fwd(src, dst_expected);
    megdnn_assert_eq_layout(dst_expected, dst);
}

void SlidingWindowTransposeForward::deduce_layout(const TensorLayout &src,
        TensorLayout &dst)
{
    deduce_layout_fwd(src, dst);
}

void SlidingWindowTransposeForward::check_exec(const TensorLayout &src,
        const TensorLayout &dst,
        size_t workspace_in_bytes)
{
    check_layout_fwd(src, dst);
    auto required_workspace_in_bytes = get_workspace_in_bytes(src, dst);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
}

void SlidingWindowTransposeBackward::check_exec(const TensorLayout &diff,
        const TensorLayout &grad,
        size_t workspace_in_bytes)
{
    check_layout_fwd(grad, diff);
    auto required_workspace_in_bytes = get_workspace_in_bytes(grad, diff);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
}

} // namespace megdnn

// vim: syntax=cpp.doxygen
