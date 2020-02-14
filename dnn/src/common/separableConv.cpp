/**
 * \file dnn/src/common/separableConv.cpp
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

void SeparableConvBase::deduce_layout_fwd(const TensorLayout &src,
        const TensorLayout &filter_x, 
        const TensorLayout &filter_y,
        TensorLayout &dst)
{
    auto errmsg = [&]() {
        return megdnn_layout_msg(src) + ", "
            + megdnn_layout_msg(filter_x) + ", "
            + megdnn_layout_msg(dst) + ", "
            + megdnn_mangle("is_xcorr=")
            + megdnn_mangle("borderMode=")
            + std::to_string((param().mode == Mode::CROSS_CORRELATION)) + ", "
            + std::to_string((int)(param().borderMode)) + ", "
            + megdnn_mangle("pad_h=") + std::to_string(param().pad_h) + ", "
            + megdnn_mangle("pad_w=") + std::to_string(param().pad_w) + ", "
            + megdnn_mangle("stride_h=") + std::to_string(param().stride_h) + ", "
            + megdnn_mangle("stride_w=") + std::to_string(param().stride_w);
    };
    MEGDNN_MARK_USED_VAR(errmsg);
    megdnn_assert_contiguous(src);
    megdnn_assert_contiguous(filter_x);
    megdnn_assert(src.ndim == 4_z, "%s", errmsg().c_str());
    megdnn_assert(filter_x.ndim == 4_z, "%s", errmsg().c_str());
    size_t n = src[0];
    size_t ic = src[1];
    size_t ih = src[2];
    size_t iw = src[3];
    size_t oc = filter_x[0];
    megdnn_assert_eq_layout(filter_x, filter_y);
    megdnn_assert(filter_x[1] == ic, "%s", errmsg().c_str());
    size_t fw = filter_x[3];
    size_t fh = fw;
    size_t sh = this->param().stride_h;
    size_t sw = this->param().stride_w;
    size_t ph = this->param().pad_h;
    size_t pw = this->param().pad_w;
    size_t oh, ow;
    infer_conv_shape2d(ih, iw, fh, fw, sh, sw, ph, pw, oh, ow);
    dst = TensorLayout(TensorShape({n, oc, oh, ow}), src.dtype);
}

void SeparableConvBase::check_layout_fwd(const TensorLayout &src,
        const TensorLayout &filter_x, 
        const TensorLayout &filter_y,
        const TensorLayout &dst)
{
    TensorLayout dst_expected;
    megdnn_assert_eq_dtype(src, filter_x);
    megdnn_assert_eq_dtype(src, filter_y);
    megdnn_assert_eq_layout(filter_x, filter_y);
    megdnn_assert_eq_dtype(src, dst);
    deduce_layout_fwd(src, filter_x, filter_y, dst_expected);
    megdnn_assert_eq_layout(dst_expected, dst);
}

void SeparableConvForward::deduce_layout(const TensorLayout &src,
        const TensorLayout &filter_x,
        const TensorLayout &filter_y,
        TensorLayout &dst)
{
    deduce_layout_fwd(src, filter_x, filter_y, dst);
}

void SeparableConvForward::check_exec(const TensorLayout &src,
        const TensorLayout &filter_x,
        const TensorLayout &filter_y,
        const TensorLayout &dst,
        size_t workspace_in_bytes)
{
    check_layout_fwd(src, filter_x, filter_y, dst);
    auto required_workspace_in_bytes = get_workspace_in_bytes(src, filter_x, filter_y, dst);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
}

} // namespace megdnn

// vim: syntax=cpp.doxygen
