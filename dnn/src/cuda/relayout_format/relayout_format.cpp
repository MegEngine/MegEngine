/**
 * \file dnn/src/cuda/relayout_format/relayout_format.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/cuda/relayout_format/relayout_format.cuh"
#include "src/cuda/relayout_format/relayout_format.h"
using namespace megdnn;
using namespace cuda;

namespace {

inline void get_scale_zeropoint(const DType& tensor_dtype, float& scale,
                                uint8_t& zero_point) {
    if (tensor_dtype.enumv() == DTypeEnum::Quantized8Asymm) {
        zero_point = tensor_dtype.param<dtype::Quantized8Asymm>().zero_point;
        scale = tensor_dtype.param<dtype::Quantized8Asymm>().scale;
    } else if (tensor_dtype.enumv() == DTypeEnum::QuantizedS8) {
        scale = tensor_dtype.param<dtype::QuantizedS8>().scale;
    } else if (tensor_dtype.enumv() == DTypeEnum::QuantizedS4) {
        scale = tensor_dtype.param<dtype::QuantizedS4>().scale;
    } else if (tensor_dtype.enumv() == DTypeEnum::Quantized4Asymm) {
        zero_point = tensor_dtype.param<dtype::Quantized4Asymm>().zero_point;
        scale = tensor_dtype.param<dtype::Quantized4Asymm>().scale;
    }
}

}  // namespace

bool relayout_format::RelayoutFormatFast::usable(
        const TensorLayout& src_layout, const TensorLayout& dst_layout) {
    return relayout_format_cuda_usable(src_layout, dst_layout);
}

void relayout_format::RelayoutFormatFast::exec(const TensorND& src,
                                               const TensorND& dst,
                                               cudaStream_t stream,
                                               RelayoutFormat::Param::Mode mode,
                                               int group) {
    float src_scale = 1.f;
    float dst_scale = 1.f;
    uint8_t src_zero_point = 0;
    uint8_t dst_zero_point = 0;
    get_scale_zeropoint(src.layout.dtype, src_scale, src_zero_point);
    get_scale_zeropoint(dst.layout.dtype, dst_scale, dst_zero_point);
    if (src.layout.dtype.enumv() == DTypeEnum::Uint8) {
        src_zero_point = 128;
    }
    if (mode == RelayoutFormat::Param::Mode::NCHW_NCHW4 ||
        mode == RelayoutFormat::Param::Mode::NCHW_NCHW64) {
        return relayout_format_cuda_nchw_nchwx(src, dst, stream, src_scale,
                                               dst_scale, src_zero_point,
                                               dst_zero_point, group);
    } else if (mode == RelayoutFormat::Param::Mode::NCHW64_NCHW) {
        megdnn_assert(group == 1,
                      "RelayoutFormat kernel only support transforming NCHW64 "
                      "to NCHW with group = 1(group:%d)",
                      group);
        return relayout_format_cuda_nchwx_nchw(src, dst, stream, src_scale,
                                               dst_scale, src_zero_point,
                                               dst_zero_point);
    } else if (mode == RelayoutFormat::Param::Mode::NCHW_NHWC) {
#define CHECK(dt)                                             \
    megdnn_assert(dt.enumv() == DTypeEnum::Quantized4Asymm || \
                  dt.enumv() == DTypeEnum::QuantizedS4)
        CHECK(src.layout.dtype);
        CHECK(dst.layout.dtype);
        return relayout_format_cuda_nchw_nhwc(src, dst, stream, src_scale,
                                              dst_scale, src_zero_point,
                                              dst_zero_point);
    } else if (mode == RelayoutFormat::Param::Mode::NHWC_NCHW) {
        CHECK(src.layout.dtype);
        CHECK(dst.layout.dtype);
        return relayout_format_cuda_nhwc_nchw(src, dst, stream, src_scale,
                                              dst_scale, src_zero_point,
                                              dst_zero_point);
#undef CHECK
    } else if (mode == RelayoutFormat::Param::Mode::NCHW_NCHW4_WEIGHT) {
        return relayout_format_cuda_nchw_nchw4_weight(src, dst, stream);
    } else if (mode == RelayoutFormat::Param::Mode::NCHW4_NCHW) {
        return relayout_format_cuda_nchw4_nchw(src, dst, stream, group);
    } else {
        megdnn_throw(
                "only support nchw_nchw64/nchw64_nchw/nchw_nchw4/nchw4_nchw "
                "layout_format");
    }
}

// vim: ft=cpp syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
