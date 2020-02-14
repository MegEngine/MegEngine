/**
 * \file dnn/src/common/batch_conv_bias.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "megdnn/oprs.h"
#include "megdnn/oprs/nn_int.h"
#include "src/common/utils.h"

namespace megdnn {
void BatchConvBiasForward::deduce_dtype(DType src, DType filter,
                                        DType /* bias */, DType /* z */,
                                        DType& dst) {
    check_or_deduce_dtype_fwd(src, filter, dst);
}

void BatchConvBiasForward::deduce_layout(const TensorLayout& src,
                                         const TensorLayout& filter,
                                         const TensorLayout& /* bias */,
                                         const TensorLayout& /* z */,
                                         TensorLayout& dst) {
    TensorLayout non_batch_filter;
    non_batch_filter.ndim = filter.ndim - 1;
    non_batch_filter.dtype = filter.dtype;
    for (size_t i = 0; i < non_batch_filter.ndim; i++) {
        non_batch_filter[i] = filter[i + 1];
        non_batch_filter.stride[i] = filter.stride[i + 1];
    }
    non_batch_filter.format = filter.format;
    deduce_layout_fwd(src, non_batch_filter, dst);
}

BatchConvBiasForward::CanonizedFilterMeta BatchConvBiasForward::check_exec(
        const TensorLayout& src, const TensorLayout& filter,
        const TensorLayout& bias, const TensorLayout& z,
        const TensorLayout& dst, size_t workspace_in_bytes) {
    megdnn_assert(src.dtype.enumv() == filter.dtype.enumv() &&
                          src.dtype.enumv() == DTypeEnum::QuantizedS8,
                  "batch conv only support qint8");
    float scale_src = src.dtype.param<dtype::QuantizedS8>().scale;
    float scale_filter = filter.dtype.param<dtype::QuantizedS8>().scale;
    float scale_bias = bias.dtype.param<dtype::QuantizedS32>().scale;
    megdnn_assert(
            std::abs(scale_src * scale_filter - scale_bias) < 1e-6,
            "scale_bias is not equal to the product of scale_src and "
            "scale_filter (scale_src: %f scale_filter: %f scale_bias: %f).",
            scale_src, scale_filter, scale_bias);
    TensorLayout non_batch_filter;
    non_batch_filter.ndim = filter.ndim - 1;
    non_batch_filter.dtype = filter.dtype;
    for (size_t i = 0; i < non_batch_filter.ndim; i++) {
        non_batch_filter[i] = filter[i + 1];
        non_batch_filter.stride[i] = filter.stride[i + 1];
    }
    non_batch_filter.format = filter.format;
    auto ret = check_layout_fwd(src, non_batch_filter, dst);
    megdnn_assert_contiguous(bias);
    auto required_workspace_in_bytes =
            get_workspace_in_bytes(src, filter, bias, z, dst);
    megdnn_assert(workspace_in_bytes >= required_workspace_in_bytes);
    if (bias.ndim != 0) {
        //! bias.layout == dst.layout failed, no assert information
        auto check_eq = [](const TensorLayout& bias, const TensorLayout& dst) {
            if (dst.dtype.category() == DTypeCategory::QUANTIZED) {
                return bias.eq_shape(dst);
            } else {
                return bias.eq_layout(dst);
            }
        };
        if (check_eq(bias, dst))
            return ret;
        if (param().format == param::BatchConvBias::Format::NCHW4) {
            megdnn_assert(bias.shape[0] == 1);
            megdnn_assert(bias.shape[1] == dst.shape[1], "bias:%s, dst:%s",
                          bias.to_string().c_str(), dst.to_string().c_str());
            megdnn_assert(bias.shape[2] == 1);
            megdnn_assert(bias.shape[3] == 1);
            megdnn_assert(bias.shape[4] == 4);
        }
    }

    if (z.ndim != 0) {
        megdnn_assert(z.dtype.enumv() == dst.dtype.enumv());
        megdnn_assert(z.eq_shape(dst));
    }
    return ret;
}
}  // namespace megdnn

// vim: syntax=cpp.doxygen
