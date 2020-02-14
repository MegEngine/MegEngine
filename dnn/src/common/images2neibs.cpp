/**
 * \file dnn/src/common/images2neibs.cpp
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

void Images2NeibsBase::deduce_layout_fwd(const TensorLayout &src,
        TensorLayout &dst)
{
    auto errmsg = [&]() {
        return megdnn_layout_msg(src) + ", " +
            megdnn_mangle("pad_h=") + std::to_string(param().pad_h) + ", " +
            megdnn_mangle("pad_w=") + std::to_string(param().pad_w) + ", " +
            megdnn_mangle("stride_h=") +
            std::to_string(param().stride_h) + ", " +
            megdnn_mangle("stride_w=") +
            std::to_string(param().stride_w) + ", " +
            megdnn_mangle("window_h=") +
            std::to_string(param().window_h) + ", " +
            megdnn_mangle("window_w=") +
            std::to_string(param().window_w);
    };
    MEGDNN_MARK_USED_VAR(errmsg);
    megdnn_assert_contiguous(src);
    megdnn_assert(src.ndim == 4_z, "%s", errmsg().c_str());
    size_t n = src[0], ic = src[1], ih = src[2], iw = src[3];
    size_t ph = this->param().pad_h;
    size_t pw = this->param().pad_w;
    size_t sh = this->param().stride_h;
    size_t sw = this->param().stride_w;
    size_t wh = this->param().window_h;
    size_t ww = this->param().window_w;
    size_t oh, ow;

    infer_conv_shape2d(ih, iw, wh, ww, sh, sw, ph, pw, oh, ow);
    dst = TensorLayout(TensorShape({n, ic, oh, ow, wh, ww}), src.dtype);
}

void Images2NeibsBase::check_layout_fwd(const TensorLayout &src,
        const TensorLayout &dst)
{
    TensorLayout dst_expected;
    deduce_layout_fwd(src, dst_expected);
    megdnn_assert_eq_layout(dst_expected, dst);
}

void Images2NeibsForward::deduce_layout(const TensorLayout &src,
        TensorLayout &dst)
{
    deduce_layout_fwd(src, dst);
}

void Images2NeibsForward::check_exec(const TensorLayout &src,
        const TensorLayout &dst,
        size_t workspace_in_bytes)
{
    check_layout_fwd(src, dst);
    auto required_workspace_in_bytes = get_workspace_in_bytes(src, dst);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
}

void Images2NeibsBackward::check_exec(const TensorLayout &diff,
        const TensorLayout &grad,
        size_t workspace_in_bytes)
{
    check_layout_fwd(grad, diff);
    auto required_workspace_in_bytes = get_workspace_in_bytes(grad, diff);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
}

} // namespace megdnn

// vim: syntax=cpp.doxygen
