/**
 * \file dnn/src/common/winograd_filter_preprocess.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "megdnn/oprs.h"

#include <numeric>
#include "src/common/utils.h"

using namespace megdnn;
void WinogradFilterPreprocess::deduce_layout(const TensorLayout& src,
                                             TensorLayout& dst) {
    auto errmsg = [&]() {
        return "invalid filter layout:" + megdnn_layout_msg(src);
    };
    MEGDNN_MARK_USED_VAR(errmsg);
    //! NCHW88 weight layout include
    //! dense{oc/8, ic/8, fh, fw, 8, 8}; group {g, oc/8, ic/8, fh, fw, 8, 8};
    //! channel wise{g/8, 1, 1, fh, fw, 8}
    megdnn_assert(
            src.ndim == 4 || src.ndim == 5 || src.ndim == 6 || src.ndim == 7,
            "%s", errmsg().c_str());
    //! nchw88 channel wise conv
    megdnn_assert(!(src.ndim == 6 && src[1] == 1 && src[2] == 1),
                  "chennel wise nchw88 can not use winograd ");
    //! nchw88 group conv
    size_t flt_start = 0;
    size_t pack_c_size = 1;
    size_t group = 1;
    //! group conv
    if (src.ndim == 5) {
        flt_start = 1;
        group = src[0];
        //! nchw88 dense conv
    } else if (src.ndim == 6) {
        pack_c_size = src[5];
        //! nchw88 group conv
    } else if (src.ndim == 7) {
        flt_start = 1;
        group = src[0];
        pack_c_size = src[6];
    }
    size_t OC = src[flt_start] * pack_c_size,
           IC = src[flt_start + 1] * pack_c_size, FH = src[flt_start + 2],
           FW = src[flt_start + 3];
    size_t m = param().output_block_size;
    megdnn_assert(FH == FW, "%s", errmsg().c_str());

    size_t alpha = FH + m - 1;
    DType dst_type = src.dtype;
    if (src.dtype.category() == DTypeCategory::QUANTIZED) {
        megdnn_assert(src.dtype.enumv() == DTypeEnum::QuantizedS8);
        if (param().compute_mode ==
            param::ConvBias::ComputeMode::DEFAULT) {
            //! input int8 compute short
            dst_type = dtype::QuantizedS16(
                    src.dtype.param<dtype::QuantizedS8>().scale);
        } else {
            //! input int8 compute float32
            dst_type = dtype::QuantizedS32(
                    src.dtype.param<dtype::QuantizedS8>().scale);
        }
    }

    if (src.ndim == 4 || src.ndim == 6) {
        if (param().format == param::Winograd::Format::DEFAULT) {
            dst = TensorLayout({alpha, alpha, IC, OC}, dst_type);
        } else {
            megdnn_assert(param().format == param::Winograd::Format::MK4 ||
                          param().format == param::Winograd::Format::MK8);
            size_t pack_size = MatrixMulForward::pack_size(param().format);
            dst = TensorLayout({alpha, alpha, OC / pack_size, IC / pack_size,
                                pack_size, pack_size},
                               dst_type);
        }
    } else {
        megdnn_assert(src.ndim == 5 || src.ndim == 7);
        if (param().format == param::Winograd::Format::DEFAULT) {
            dst = TensorLayout({group, alpha, alpha, IC, OC}, dst_type);
        } else {
            megdnn_assert(param().format == param::Winograd::Format::MK4 ||
                          param().format == param::Winograd::Format::MK8);
            size_t pack_size = MatrixMulForward::pack_size(param().format);
            dst = TensorLayout({group, alpha, alpha, OC / pack_size,
                                IC / pack_size, pack_size, pack_size},
                               dst_type);
        }
    }
}

void WinogradFilterPreprocess::check_exec(const TensorLayout& src,
                                          const TensorLayout& dst,
                                          size_t workspace_in_bytes) {
    auto errmsg = [&]() {
        return megdnn_layout_msg(src) + ", " + megdnn_layout_msg(dst);
    };
    MEGDNN_MARK_USED_VAR(errmsg);
    megdnn_assert_contiguous(src);
    megdnn_assert_contiguous(dst);
    //! nchwxx now only support Format MKx
    if (param().format == param::Winograd::Format::DEFAULT) {
        megdnn_assert(src.ndim == dst.ndim && (src.ndim == 4 || src.ndim == 5),
                      "%s", errmsg().c_str());
    } else {
        megdnn_assert(
                (param().format == param::Winograd::Format::MK4 ||
                 param().format == param::Winograd::Format::MK8) &&
                        (src.ndim == dst.ndim - 2 || src.ndim == dst.ndim) &&
                        (src.ndim == 4 || src.ndim == 5 || src.ndim == 6 ||
                         src.ndim == 7),
                "%s", errmsg().c_str());
    }

    TensorLayout dst_expected;
    deduce_layout(src, dst_expected);
    megdnn_assert_eq_layout(dst_expected, dst);
    auto required_workspace_in_bytes = get_workspace_in_bytes(src, dst);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
}

size_t WinogradFilterPreprocess::get_workspace_in_bytes(
        const TensorLayout& src, const TensorLayout& dst) {
    MEGDNN_MARK_USED_VAR(dst);
    DType output_compute_dtype = src.dtype;
    if (src.dtype.category() == DTypeCategory::QUANTIZED) {
        megdnn_assert(src.dtype.enumv() == DTypeEnum::QuantizedS8 ||
                      src.dtype.enumv() == DTypeEnum::Quantized8Asymm);
        if (param().compute_mode ==
            param::ConvBias::ComputeMode::DEFAULT) {
            //! input int8 compute short
            output_compute_dtype = dtype::QuantizedS16(
                    src.dtype.param<dtype::QuantizedS8>().scale);
        } else {
            //! input int8 compute float32
            output_compute_dtype = dtype::QuantizedS32(
                    src.dtype.param<dtype::QuantizedS8>().scale);
        }
    }

    size_t FW = src[3];
    if (src.ndim == 5 || src.ndim == 7) {
        FW = src[4];
    }

    size_t pack_size = MatrixMulForward::pack_size(param().format);
    size_t alpha = param().output_block_size + FW - 1;
    return 2 * alpha * alpha * output_compute_dtype.size() * pack_size *
           pack_size;
}

// vim: syntax=cpp.doxygen
