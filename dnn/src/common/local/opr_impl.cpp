/**
 * \file dnn/src/common/local/opr_impl.cpp
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

void LocalBase::deduce_layout_fwd(const TensorLayout &src,
        const TensorLayout &filter, TensorLayout &dst)
{
    auto errmsg = megdnn_layout_msg(src) + ", "
        + megdnn_layout_msg(filter) + ", "
        + megdnn_layout_msg(dst) + ", "
        + megdnn_mangle("is_xcorr=")
        + std::to_string((param().mode == Mode::CROSS_CORRELATION)) + ", "
        + megdnn_mangle("pad_h=") + std::to_string(param().pad_h) + ", "
        + megdnn_mangle("pad_w=") + std::to_string(param().pad_w) + ", "
        + megdnn_mangle("stride_h=") + std::to_string(param().stride_h) + ", "
        + megdnn_mangle("stride_w=") + std::to_string(param().stride_w) ;
    auto errmsg_c = errmsg.c_str();
    MEGDNN_MARK_USED_VAR(errmsg_c);

    //! in batch dim we don't need contiguous
    TensorLayout src_contig = src;
    src_contig.init_contiguous_stride();
    src_contig.stride[0] = src.stride[0];
    megdnn_assert_eq_layout(src_contig, src);
    megdnn_assert_contiguous(filter);
    megdnn_assert(src.ndim == 4_z, "%s", errmsg_c);
    megdnn_assert(filter.ndim == 6_z, "%s", errmsg_c);
    megdnn_assert(param().dilate_h == 1 && param().dilate_w == 1,
            "dilation in local not supported");

    megdnn_assert(param().sparse == Param::Sparse::DENSE &&
            param().dilate_h == 1 && param().dilate_w == 1 &&
            src.dtype.category() == DTypeCategory::FLOAT &&
            dst.dtype == src.dtype &&
            "unsupported conv param for Local opr");

    size_t n = src[0];
    size_t ic = src[1];
    size_t ih = src[2];
    size_t iw = src[3];
    megdnn_assert_eq_size_t(filter[2], ic);
    size_t fh = filter[3];
    size_t fw = filter[4];
    size_t oc = filter[5];
    size_t sh = param().stride_h;
    size_t sw = param().stride_w;
    size_t ph = param().pad_h;
    size_t pw = param().pad_w;
    size_t oh, ow;
    infer_conv_shape2d(ih, iw, fh, fw, sh, sw, ph, pw, oh, ow);
    dst = TensorLayout(TensorShape({n, oc, oh, ow}), src.dtype);
}

void LocalBase::check_layout_fwd(const TensorLayout &src,
        const TensorLayout &filter,
        const TensorLayout &dst)
{
    TensorLayout dst_expected{dst.dtype};
    megdnn_assert_eq_dtype(src, filter);
    megdnn_assert_eq_dtype(src, dst);
    deduce_layout_fwd(src, filter, dst_expected);
    //! in batch dim we don't need contiguous
    dst_expected.stride[0] = dst.stride[0];
    megdnn_assert_eq_layout(dst_expected, dst);

    megdnn_assert(src.dtype == filter.dtype && src.dtype == dst.dtype);
    megdnn_assert(src.dtype == dtype::Float32() ||
                  MEGDNN_FLOAT16_SELECT(src.dtype == dtype::Float16(), true));
}

void LocalForward::deduce_layout(const TensorLayout &src,
        const TensorLayout &filter,
        TensorLayout &dst)
{
    deduce_layout_fwd(src, filter, dst);
}

void LocalForward::check_exec(const TensorLayout &src,
        const TensorLayout &filter,
        const TensorLayout &dst,
        size_t workspace_in_bytes)
{
    check_layout_fwd(src, filter, dst);
    auto required_workspace_in_bytes = get_workspace_in_bytes(src, filter, dst);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
}

void LocalBackwardData::check_exec(const TensorLayout &filter,
        const TensorLayout &diff,
        const TensorLayout &grad,
        size_t workspace_in_bytes)
{
    check_layout_fwd(grad, filter, diff);
    auto required_workspace_in_bytes = get_workspace_in_bytes(filter,
            diff, grad);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
}

void LocalBackwardFilter::check_exec(const TensorLayout &src,
        const TensorLayout &diff,
        const TensorLayout &grad,
        size_t workspace_in_bytes)
{
    check_layout_fwd(src, grad, diff);
    auto required_workspace_in_bytes = get_workspace_in_bytes(src,
            diff, grad);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
}

} // namespace megdnn

// vim: syntax=cpp.doxygen
