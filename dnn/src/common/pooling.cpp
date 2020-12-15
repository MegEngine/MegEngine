/**
 * \file dnn/src/common/pooling.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#include "megdnn/oprs.h"

#include "src/common/utils.h"

namespace megdnn {

void PoolingBase::deduce_layout_fwd(const TensorLayout& src,
                                    TensorLayout& dst) {
    auto errmsg =
            megdnn_layout_msg(src) + ", " + megdnn_layout_msg(dst) + ", " +
            megdnn_mangle("pad_h=") + std::to_string(param().pad_h) + ", " +
            megdnn_mangle("pad_w=") + std::to_string(param().pad_w) + ", " +
            megdnn_mangle("stride_h=") + std::to_string(param().stride_h) +
            ", " + megdnn_mangle("stride_w=") +
            std::to_string(param().stride_w) + ", " +
            megdnn_mangle("window_h=") + std::to_string(param().window_h) +
            ", " + megdnn_mangle("window_w=") +
            std::to_string(param().window_w) + ", " + megdnn_mangle("is_max=") +
            std::to_string(param().mode == Mode::MAX) + ", " +
            megdnn_mangle("is_nhwc=") +
            std::to_string(param().format == Param::Format::NHWC) + ", " +
            megdnn_mangle("is_nhwcd4=") +
            std::to_string(param().format == Param::Format::NHWCD4);
    auto errmsg_c = errmsg.c_str();

    MEGDNN_MARK_USED_VAR(errmsg_c);
    megdnn_assert_contiguous(src);
    size_t spatial_pos, c_pos, batch_pos = 0;
    if (param().format == Param::Format::NCHW) {
        megdnn_assert(src.ndim == 4_z, "%s", errmsg_c);

        spatial_pos = 2;
        c_pos = 1;
    } else if (param().format == Param::Format::NHWC) {
        megdnn_assert(src.ndim == 4_z, "%s", errmsg_c);

        spatial_pos = 1;
        c_pos = 3;
    } else if (param().format == Param::Format::NCHW4 ||
               param().format == Param::Format::NCHW44 ||
               param().format == Param::Format::NCHW88 ||
               param().format == Param::Format::NCHW32) {
        megdnn_assert(src.ndim == 5_z, "%s", errmsg_c);

        spatial_pos = 2;
        c_pos = 1;
    } else if (param().format == Param::Format::CHWN4) {
        spatial_pos = 1;
        c_pos = 0;
        batch_pos = 3;
    } else {
        megdnn_assert(
                param().format == Param::Format::NHWCD4 && src.ndim == 5_z,
                "%s", errmsg_c);
        spatial_pos = 1;
        c_pos = 2;
    }
    size_t n = src[batch_pos];
    size_t c = src[c_pos];
    size_t ih = src[spatial_pos];
    size_t iw = src[spatial_pos + 1];
    if (param().format == Param::Format::NHWCD4) {
        c *= 4;
        iw = src[spatial_pos + 2];
    }
    if (param().format == Param::Format::NCHW4 ||
        param().format == Param::Format::NCHW44 ||
        param().format == Param::Format::CHWN4) {
        c *= 4;
    }
    if (param().format == Param::Format::NCHW88) {
        c *= 8;
    }
    if (param().format == Param::Format::NCHW32) {
        c *= 32;
    }
    size_t oh, ow;
    size_t fh = this->param().window_h;
    size_t fw = this->param().window_w;
    size_t sh = this->param().stride_h;
    size_t sw = this->param().stride_w;
    size_t ph = this->param().pad_h;
    size_t pw = this->param().pad_w;
    megdnn_assert(ph < fh && pw < fw,
                  "pooling padding size (%zu %zu) should not be bigger than "
                  "window size (%zu %zu)",
                  pw, ph, fw, fh);
    infer_conv_shape2d(ih, iw, fh, fw, sh, sw, ph, pw, oh, ow);
    if (param().format == Param::Format::NCHW) {
        dst = TensorLayout(TensorShape({n, c, oh, ow}), src.dtype);
    } else if (param().format == Param::Format::NHWC) {
        megdnn_assert(param().format == Param::Format::NHWC,
                      "invalid pooling format");
        dst = TensorLayout({n, oh, ow, c}, src.dtype, src.format);
    } else if (param().format == Param::Format::NCHW4 ||
               param().format == Param::Format::NCHW44) {
        dst = TensorLayout{{n, c / 4, oh, ow, 4}, src.dtype, src.format};
    } else if (param().format == Param::Format::NCHW88) {
        dst = TensorLayout{{n, c / 8, oh, ow, 8}, src.dtype, src.format};
    } else if (param().format == Param::Format::NCHW32) {
        dst = TensorLayout{{n, c / 32, oh, ow, 32}, src.dtype, src.format};
    } else if (param().format == Param::Format::CHWN4) {
        dst = TensorLayout{{c / 4, oh, ow, n, 4}, src.dtype, src.format};
    } else {
        megdnn_assert(param().format == Param::Format::NHWCD4,
                      "invalid pooling format");
        dst = TensorLayout{{n, oh, c / 4, ow, 4}, src.dtype, src.format};
    }
}

void PoolingBase::check_layout_fwd(const TensorLayout& src,
                                   const TensorLayout& dst) {
    TensorLayout dst_expected;
    megdnn_assert_eq_dtype(src, dst);
    deduce_layout_fwd(src, dst_expected);
    megdnn_assert_eq_layout(dst_expected, dst);
    megdnn_assert(src.dtype == dst.dtype);
    megdnn_assert(src.dtype.category() == DTypeCategory::FLOAT ||
                  src.dtype == dtype::Int8() ||
                  src.dtype.category() == DTypeCategory::QUANTIZED);
}

void PoolingForward::deduce_layout(const TensorLayout& src, TensorLayout& dst) {
    deduce_layout_fwd(src, dst);
}

void PoolingForward::check_exec(const TensorLayout& src,
                                const TensorLayout& dst,
                                size_t workspace_in_bytes) {
    check_layout_fwd(src, dst);
    auto required_workspace_in_bytes = get_workspace_in_bytes(src, dst);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
}

void PoolingBackward::check_exec(const TensorLayout& src,
                                 const TensorLayout& dst,
                                 const TensorLayout& diff,
                                 const TensorLayout& grad,
                                 size_t workspace_in_bytes) {
    check_layout_fwd(src, dst);
    megdnn_assert_eq_layout(src, grad);
    megdnn_assert_eq_layout(dst, diff);
    auto required_workspace_in_bytes =
            get_workspace_in_bytes(src, dst, diff, grad);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
}

}  // namespace megdnn

// vim: syntax=cpp.doxygen
