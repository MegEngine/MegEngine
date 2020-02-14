/**
 * \file dnn/src/common/local_share/opr_impl.cpp
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

void LocalShareBase::deduce_layout_fwd(const TensorLayout& src,
                                       const TensorLayout& filter,
                                       TensorLayout& dst) {
    using Mode = LocalShare::Param::Mode;
    auto errmsg =
            megdnn_layout_msg(src) + ", " + megdnn_layout_msg(filter) + ", " +
            megdnn_layout_msg(dst) + ", " + megdnn_mangle("is_xcorr=") +
            std::to_string((param().mode == Mode::CROSS_CORRELATION)) + ", " +
            megdnn_mangle("pad_h=") + std::to_string(param().pad_h) + ", " +
            megdnn_mangle("pad_w=") + std::to_string(param().pad_w) + ", " +
            megdnn_mangle("stride_h=") + std::to_string(param().stride_h) +
            ", " + megdnn_mangle("stride_w=") +
            std::to_string(param().stride_w) + ", " +
            megdnn_mangle("dilate_h=") + std::to_string(param().dilate_h) +
            ", " + megdnn_mangle("dilate_w=") +
            std::to_string(param().dilate_w) + ", " +
            megdnn_mangle("spatial_groups_h=") +
            std::to_string(param().spatial_groups_h) + ", " +
            megdnn_mangle("spatial_groups_w=") +
            std::to_string(param().spatial_groups_w);
    auto errmsg_c = errmsg.c_str();
    MEGDNN_MARK_USED_VAR(errmsg_c);

    megdnn_assert_contiguous(src);
    megdnn_assert_contiguous(filter);
    using Param = LocalShare::Param;
    using Sparse = Param::Sparse;
    using Format = Param::Format;
    using ComputeMode = Param::ComputeMode;
    megdnn_assert(param().format == Format::NCHW,
                  "local shared only support NCHW format");
    megdnn_assert(src.ndim == 4_z, "%s", errmsg_c);
    megdnn_assert(
            (filter.ndim == 6_z && param().sparse == Sparse::DENSE) ||
                    (filter.ndim == 7_z && param().sparse == Sparse::GROUP),
            "%s", errmsg_c);
    megdnn_assert(param().dilate_h == 1 && param().dilate_w == 1,
                  "dilated local shared is not supported");
    megdnn_assert(src.dtype == dtype::Float32() &&
                          param().computeMode == ComputeMode::DEFAULT,
                  "local shared only support float32");

    size_t n = src[0], ci = src[1], hi = src[2], wi = src[3];
    size_t sgh = param().spatial_groups_h, sgw = param().spatial_groups_w;
    size_t groups = 1;
    size_t weights_shp_pos = 0;
    if (param().sparse == Sparse::GROUP) {
        groups = filter[0];
        weights_shp_pos = 1;
    }
    megdnn_assert(sgh == filter[weights_shp_pos] &&
                          sgw == filter[weights_shp_pos + 1],
                  "spatial groups in filter tensor mismatch with those "
                  "provided in parameter %s",
                  errmsg_c);
    size_t fh = filter[weights_shp_pos + 3], fw = filter[weights_shp_pos + 4],
           co = filter[weights_shp_pos + 5] * groups;
    megdnn_assert(filter[weights_shp_pos + 2] * groups == ci,
                  "input channels of src and filter mismatch %s", errmsg_c);
    size_t sh = param().stride_h;
    size_t sw = param().stride_w;
    size_t ph = param().pad_h;
    size_t pw = param().pad_w;
    size_t ho = infer_conv_shape(hi, fh, sh, ph),
           wo = infer_conv_shape(wi, fw, sw, pw);
    megdnn_assert(
            ho % sgh == 0 && wo % sgw == 0,
            "height and width of output cannot be divided by spatial groups %s",
            errmsg_c);
    dst = TensorLayout{{n, co, ho, wo}, src.dtype};
}

void LocalShareBase::check_layout_fwd(const TensorLayout& src,
                                      const TensorLayout& filter,
                                      const TensorLayout& dst) {
    TensorLayout dst_expected;
    megdnn_assert_eq_dtype(src, filter);
    megdnn_assert_eq_dtype(src, dst);
    deduce_layout_fwd(src, filter, dst_expected);
    megdnn_assert_eq_layout(dst_expected, dst);

    megdnn_assert(src.dtype == dtype::Float32());
}

void LocalShareForward::deduce_layout(const TensorLayout& src,
                                      const TensorLayout& filter,
                                      TensorLayout& dst) {
    deduce_layout_fwd(src, filter, dst);
}

void LocalShareForward::check_exec(const TensorLayout& src,
                                   const TensorLayout& filter,
                                   const TensorLayout& dst,
                                   size_t workspace_in_bytes) {
    check_layout_fwd(src, filter, dst);
    auto required_workspace_in_bytes = get_workspace_in_bytes(src, filter, dst);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
}

void LocalShareBackwardData::deduce_layout(const TensorLayout& filter,
                                           const TensorLayout& diff,
                                           TensorLayout& grad) {
    using Mode = LocalShare::Param::Mode;
    auto errmsg =
            megdnn_layout_msg(filter) + ", " + megdnn_layout_msg(diff) + ", " +
            megdnn_layout_msg(grad) + ", " + megdnn_mangle("is_xcorr=") +
            std::to_string((param().mode == Mode::CROSS_CORRELATION)) + ", " +
            megdnn_mangle("pad_h=") + std::to_string(param().pad_h) + ", " +
            megdnn_mangle("pad_w=") + std::to_string(param().pad_w) + ", " +
            megdnn_mangle("stride_h=") + std::to_string(param().stride_h) +
            ", " + megdnn_mangle("stride_w=") +
            std::to_string(param().stride_w) + ", " +
            megdnn_mangle("dilate_h=") + std::to_string(param().dilate_h) +
            ", " + megdnn_mangle("dilate_w=") +
            std::to_string(param().dilate_w) + ", " +
            megdnn_mangle("spatial_groups_h=") +
            std::to_string(param().spatial_groups_h) + ", " +
            megdnn_mangle("spatial_groups_w=") +
            std::to_string(param().spatial_groups_w);
    auto errmsg_c = errmsg.c_str();
    MEGDNN_MARK_USED_VAR(errmsg_c);

    megdnn_assert_contiguous(filter);
    megdnn_assert_contiguous(diff);
    using Param = LocalShare::Param;
    using Sparse = Param::Sparse;
    using Format = Param::Format;
    using ComputeMode = Param::ComputeMode;
    megdnn_assert(param().format == Format::NCHW,
                  "local shared only support NCHW format");
    megdnn_assert(
            (filter.ndim == 6_z && param().sparse == Sparse::DENSE) ||
                    (filter.ndim == 7_z && param().sparse == Sparse::GROUP),
            "%s", errmsg_c);
    megdnn_assert(diff.ndim == 4_z, "%s", errmsg_c);
    megdnn_assert(param().dilate_h == 1 && param().dilate_w == 1,
                  "dilated local shared is not supported");
    megdnn_assert(diff.dtype == dtype::Float32() &&
                          param().computeMode == ComputeMode::DEFAULT,
                  "local shared only support float32");

    size_t n = diff[0], co = diff[1], ho = diff[2], wo = diff[3];
    size_t sgh = param().spatial_groups_h, sgw = param().spatial_groups_w;
    megdnn_assert(
            ho % sgh == 0 && wo % sgw == 0,
            "height and width of output cannot be divided by spatial groups %s",
            errmsg_c);
    size_t groups = 1;
    size_t weights_shp_pos = 0;
    if (param().sparse == Sparse::GROUP) {
        groups = filter[0];
        weights_shp_pos = 1;
    }
    megdnn_assert(sgh == filter[weights_shp_pos] &&
                          sgw == filter[weights_shp_pos + 1],
                  "spatial groups in filter tensor mismatch with those "
                  "provided in parameter %s",
                  errmsg_c);
    size_t ci = filter[weights_shp_pos + 2] * groups,
           fh = filter[weights_shp_pos + 3], fw = filter[weights_shp_pos + 4];
    megdnn_assert(filter[weights_shp_pos + 5] * groups == co,
                  "input channels of src and filter mismatch %s", errmsg_c);
    size_t sh = param().stride_h;
    size_t sw = param().stride_w;
    size_t ph = param().pad_h;
    size_t pw = param().pad_w;

    auto deduce = [&errmsg_c](size_t out, size_t filter, size_t stride,
                              size_t pad) {
        MEGDNN_MARK_USED_VAR(errmsg_c);
        auto i = (out - 1) * stride + filter;
        megdnn_assert(i > pad * 2, "%s", errmsg_c);
        return i - pad * 2;
    };
    grad.ndim = diff.ndim;
    grad[0] = n;
    grad[1] = ci;
    grad[2] = deduce(ho, fh, sh, ph);
    grad[3] = deduce(wo, fw, sw, pw);
    grad.init_contiguous_stride();
    grad.dtype = diff.dtype;
}

void LocalShareBackwardData::check_exec(const TensorLayout& filter,
                                   const TensorLayout& diff,
                                   const TensorLayout& grad,
                                   size_t workspace_in_bytes) {
    auto filter_dtype = filter.dtype, diff_dtype = diff.dtype,
         grad_dtype = grad.dtype;
    megdnn_assert(filter_dtype == dtype::Float32() &&
                  filter_dtype == diff_dtype && filter_dtype == grad_dtype);
    check_layout_fwd(grad, filter, diff);
    auto required_workspace_in_bytes =
            get_workspace_in_bytes(filter, diff, grad);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
}

void LocalShareBackwardFilter::check_exec(const TensorLayout& src,
                                          const TensorLayout& diff,
                                          const TensorLayout& grad,
                                          size_t workspace_in_bytes) {
    auto src_dtype = src.dtype, diff_dtype = diff.dtype,
         grad_dtype = grad.dtype;
    megdnn_assert(src_dtype == dtype::Float32() && src_dtype == diff_dtype &&
                  src_dtype == grad_dtype);
    check_layout_fwd(src, grad, diff);
    auto required_workspace_in_bytes = get_workspace_in_bytes(src, diff, grad);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
}

}  // namespace megdnn

// vim: syntax=cpp.doxygen
