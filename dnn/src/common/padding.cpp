/**
 * \file dnn/src/common/padding.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "megdnn/oprs.h"
#include "megdnn/oprs/general.h"
#include "megdnn/thin/small_vector.h"
#include "src/common/opr_param_defs_enumv.cuh"
#include "src/common/utils.h"

namespace megdnn {

using padding_param = megdnn::param_enumv::Padding;

void PaddingForward::forward_check_exec(
        const TensorLayout& src, const TensorLayout& dst) {
    check_exec(src, dst);
    megdnn_assert(
            src.dtype.enumv() != DTypeEnum::Bool &&
                    src.dtype.enumv() != DTypeEnum::IntB1 &&
                    src.dtype.enumv() != DTypeEnum::IntB2 &&
                    src.dtype.enumv() != DTypeEnum::IntB4,
            "unsupported %s dtype for forward padding opr", src.dtype.name());
}

void PaddingForward::deduce_layout(const TensorLayout& src, TensorLayout& dst) {
    SmallVector<size_t> offsets(get_offsets());
    TensorShape dst_shape;
    switch (src.ndim) {
        case 1:
            dst_shape = {src.shape[0] + offsets[0] + offsets[1]};
            break;
        case 2:
            dst_shape = {
                    src.shape[0] + offsets[0] + offsets[1],
                    src.shape[1] + offsets[2] + offsets[3]};
            break;
        case 3:
            dst_shape = {
                    src.shape[0] + offsets[0] + offsets[1],
                    src.shape[1] + offsets[2] + offsets[3],
                    src.shape[2] + offsets[4] + offsets[5]};
            break;
        case 4:
            dst_shape = {
                    src.shape[0] + offsets[0] + offsets[1],
                    src.shape[1] + offsets[2] + offsets[3],
                    src.shape[2] + offsets[4] + offsets[5],
                    src.shape[3] + offsets[6] + offsets[7]};
            break;
        case 5:
            dst_shape = {
                    src.shape[0] + offsets[0] + offsets[1],
                    src.shape[1] + offsets[2] + offsets[3],
                    src.shape[2] + offsets[4] + offsets[5],
                    src.shape[3] + offsets[6] + offsets[7],
                    src.shape[4] + offsets[8] + offsets[9]};
            break;
        case 6:
            dst_shape = {src.shape[0] + offsets[0] + offsets[1],
                         src.shape[1] + offsets[2] + offsets[3],
                         src.shape[2] + offsets[4] + offsets[5],
                         src.shape[3] + offsets[6] + offsets[7],
                         src.shape[4] + offsets[8] + offsets[9],
                         src.shape[5] + offsets[10] + offsets[11]};
            break;
        case 7:
            dst_shape = {src.shape[0] + offsets[0] + offsets[1],
                         src.shape[1] + offsets[2] + offsets[3],
                         src.shape[2] + offsets[4] + offsets[5],
                         src.shape[3] + offsets[6] + offsets[7],
                         src.shape[4] + offsets[8] + offsets[9],
                         src.shape[5] + offsets[10] + offsets[11],
                         src.shape[6] + offsets[12] + offsets[13]};
            break;
        default:
            megdnn_assert(false, "invalid tensor ndim %zu", src.ndim);
            break;
    }
    dst = TensorLayout(dst_shape, src.dtype);
}

void PaddingBackward::backward_check_exec(
        const TensorLayout& src, const TensorLayout& dst) {
    check_exec(dst, src);
    megdnn_assert(
            src.dtype.enumv() == DTypeEnum::Float32 DNN_INC_FLOAT16(
                                         || src.dtype.enumv() == DTypeEnum::Float16 ||
                                         src.dtype.enumv() == DTypeEnum::BFloat16),
            "unsupported %s dtype for forward padding opr", src.dtype.name());
}

SmallVector<size_t> PaddingBase::get_offsets() {
    SmallVector<size_t> offsets = {param().front_offset_dim0, param().back_offset_dim0,
                                   param().front_offset_dim1, param().back_offset_dim1,
                                   param().front_offset_dim2, param().back_offset_dim2,
                                   param().front_offset_dim3, param().back_offset_dim3,
                                   param().front_offset_dim4, param().back_offset_dim4,
                                   param().front_offset_dim5, param().back_offset_dim5,
                                   param().front_offset_dim6, param().back_offset_dim6};
    return offsets;
}

void PaddingBase::check_exec(const TensorLayout& src, const TensorLayout& dst) {
    SmallVector<size_t> offsets(get_offsets());
    // make sure the src and dst tensor not empty
    megdnn_assert(src.ndim != 0 && dst.ndim != 0);
    // make sure src and dst is same dtype
    megdnn_assert_eq_dtype(src, dst);
    // make sure src and dst is same ndim
    megdnn_assert(
            src.ndim == dst.ndim, "the src.ndim = %zu the dst.ndim = %zu", src.ndim,
            dst.ndim);
    // make sure in every dimension dst is equal or greater than src
    for (size_t i = 0; i < src.ndim; ++i) {
        megdnn_assert(
                dst.shape[i] == src.shape[i] + offsets[i * 2] + offsets[i * 2 + 1]);
    }
    // check the padding mode is valid
    megdnn_assert(
            static_cast<uint32_t>(param().padding_mode) ==
                            padding_param::PaddingMode::REFLECT ||
                    static_cast<uint32_t>(param().padding_mode) ==
                            padding_param::PaddingMode::REPLICATE ||
                    static_cast<uint32_t>(param().padding_mode) ==
                            padding_param::PaddingMode::CONSTANT,
            "unsupported padding mode");
    // addition check for reflect padding, make sure the reflected index is
    // valid
    if (static_cast<uint32_t>(param().padding_mode) ==
        padding_param::PaddingMode::REFLECT) {
        for (size_t i = 0; i < src.ndim; ++i) {
            megdnn_assert(
                    offsets[i * 2] < src.shape[i] &&
                    dst.shape[i] - offsets[i * 2] - src.shape[i] < src.shape[i]);
        }
    }
}

}  // namespace megdnn
