/**
 * \file dnn/src/cuda/relayout_format/relayout_format.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
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
    }
}

}  // namespace

bool relayout_format::RelayoutFormatFast::usable(
        const TensorLayout& src_layout, const TensorLayout& dst_layout) {
    return relayout_format_cuda_usable(src_layout, dst_layout);
}

void relayout_format::RelayoutFormatFast::exec(const TensorND& src,
                                               const TensorND& dst,
                                               cudaStream_t stream) {
    size_t ih = src.layout[2];
    size_t iw = src.layout[3];
    size_t hw = ih * iw;
    float src_scale = 1.f;
    float dst_scale = 1.f;
    uint8_t src_zero_point = 0;
    uint8_t dst_zero_point = 0;
    get_scale_zeropoint(src.layout.dtype, src_scale, src_zero_point);
    get_scale_zeropoint(dst.layout.dtype, dst_scale, dst_zero_point);
    if (src.layout.dtype.enumv() == DTypeEnum::Uint8) {
        src_zero_point = 128;
    }
    if (hw % 4 == 0) {
        relayout_format_cuda_exec<4>(src, dst, stream, src_scale, dst_scale,
                                     src_zero_point, dst_zero_point);
    } else {
        relayout_format_cuda_exec<1>(src, dst, stream, src_scale, dst_scale,
                                     src_zero_point, dst_zero_point);
    }
}
