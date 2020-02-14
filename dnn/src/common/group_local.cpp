/**
 * \file dnn/src/common/group_local.cpp
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

void GroupLocalBase::deduce_layout_fwd(const TensorLayout &src,
        const TensorLayout &filter,
        TensorLayout &dst)
{
    auto errmsg = [&]() {
        return megdnn_layout_msg(src) + ", "
            + megdnn_layout_msg(filter) + ", "
            + megdnn_layout_msg(dst) + ", "
            + megdnn_mangle("pad_h=") + std::to_string(param().pad_h) + ", "
            + megdnn_mangle("pad_w=") + std::to_string(param().pad_w) + ", "
            + megdnn_mangle("stride_h=") + std::to_string(param().stride_h) + ", "
            + megdnn_mangle("stride_w=") + std::to_string(param().stride_w);
    };
    MEGDNN_MARK_USED_VAR(errmsg);
    megdnn_assert_contiguous(src);
    megdnn_assert_contiguous(filter);
    megdnn_assert(param().mode == Mode::CROSS_CORRELATION,
            "only CROSS_CORRELATION mode is supported for glocal.");

    megdnn_assert(param().sparse == Param::Sparse::DENSE &&
            param().dilate_h == 1 && param().dilate_w == 1 &&
            src.dtype.category() == DTypeCategory::FLOAT &&
            src.dtype == dst.dtype,
            "unsupported conv param for Local opr");
    megdnn_assert(src.ndim == 4_z, "%s", errmsg().c_str());
    megdnn_assert(filter.ndim == 7_z, "%s", errmsg().c_str());
    size_t group = filter[0];
    size_t n = src[0];
    size_t ic = src[1];
    size_t ih = src[2];
    size_t iw = src[3];
    size_t oc = filter[6]*group;
    size_t oh = filter[1], ow = filter[2];
    megdnn_assert_eq_size_t(filter[0], group);
    megdnn_assert_eq_size_t(filter[3]*group, ic);
    size_t fh = filter[4], fw = filter[5];
    // (group, oh, ow, ic/group, fh, fw, oc/group)
    infer_conv_shape2d(ih, iw, fh, fw,
            param().stride_h, param().stride_w,
            param().pad_h, param().pad_w, oh, ow);
    dst = TensorLayout(TensorShape({n, oc, oh, ow}), src.dtype);
}

void GroupLocalBase::check_layout_fwd(const TensorLayout &src,
        const TensorLayout &filter,
        const TensorLayout &dst)
{
    TensorLayout dst_expected{dst.dtype};
    megdnn_assert_eq_dtype(src, filter);
    megdnn_assert_eq_dtype(src, dst);
    deduce_layout_fwd(src, filter, dst_expected);
    megdnn_assert_eq_layout(dst_expected, dst);
    megdnn_assert(src.dtype == dtype::Float32() || MEGDNN_FLOAT16_SELECT(src.dtype == dtype::Float16(), true));
}

void GroupLocalForward::check_exec(const TensorLayout &src,
        const TensorLayout &filter,
        const TensorLayout &dst,
        size_t workspace_in_bytes)
{
    check_layout_fwd(src, filter, dst);
    auto required_workspace_in_bytes = get_workspace_in_bytes(src, filter, dst);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
}

void GroupLocalBackwardData::check_exec(const TensorLayout &filter,
        const TensorLayout &diff,
        const TensorLayout &grad,
        size_t workspace_in_bytes)
{
    check_layout_fwd(grad, filter, diff);
    auto required_workspace_in_bytes = get_workspace_in_bytes(filter, diff, grad);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
}

void GroupLocalBackwardFilter::check_exec(const TensorLayout &src,
        const TensorLayout &diff,
        const TensorLayout &grad,
        size_t workspace_in_bytes)
{
    check_layout_fwd(src, grad, diff);
    auto required_workspace_in_bytes = get_workspace_in_bytes(src, diff, grad);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
}

} // namespace megdnn
// vim: syntax=cpp.doxygen
