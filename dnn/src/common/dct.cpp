/**
 * \file dnn/src/common/dct.cpp
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

void DctChannelSelectForward::deduce_layout_fwd(const TensorLayout& src,
                                                const TensorLayout& mask_offset,
                                                const TensorLayout& mask_val,
                                                TensorLayout& dst) {
    const size_t dct_block = param().dct_block_size;
    const size_t in = src.shape[0];
    const size_t ic = src.shape[1];
    const size_t ih = src.shape[2];
    const size_t iw = src.shape[3];
    check_layout_fwd(src, mask_offset, mask_val, dst);
    const size_t oh = ih / dct_block;
    const size_t ow = iw / dct_block;
    //! mask will be empty or (ic + 1) elements
    size_t oc = mask_offset.ndim > 0 && mask_offset[0] >= 2
                        ? mask_val.shape[0]
                        : ic * dct_block * dct_block;
    if (param().fastImpl == Param::FastImpl::FIX_32_MASK) {
        megdnn_assert(oc == 32,
                      "Param::FastImpl::FIX_32_MASK oc must be 32, but %zu",
                      oc);
    }
    if (param().format == Param::Format::NCHW) {
        dst = TensorLayout(TensorShape({in, oc, oh, ow}), dst.dtype);
    } else {
        megdnn_assert(param().format == Param::Format::NCHW4,
                      "dct format must be nchw or nchw4");
        megdnn_assert(oc % 4 == 0, "oc mod 4 == 0 in nchw4");
        dst = TensorLayout(TensorShape({in, oc / 4, oh, ow, 4}), dst.dtype);
    }
}

void DctChannelSelectForward::deduce_layout(const TensorLayout& src,
                                            const TensorLayout& mask_offset,
                                            const TensorLayout& mask_val,
                                            TensorLayout& dst) {
    deduce_layout_fwd(src, mask_offset, mask_val, dst);
}

void DctChannelSelectForward::check_layout_fwd(const TensorLayout& src,
                                               const TensorLayout& mask_offset,
                                               const TensorLayout& mask_val,
                                               const TensorLayout& dst) {
    const size_t dct_block = param().dct_block_size;
    const size_t ih = src.shape[2];
    const size_t iw = src.shape[3];

    megdnn_assert(mask_offset.ndim == 0 || (mask_offset.ndim == 1 &&
                                            (mask_offset.shape[0] == 0 ||
                                             mask_offset.shape[0] >= 2) &&
                                            mask_val.ndim == 1),
                  "mask only support one valid dim");
    megdnn_assert(mask_val.ndim <= 1, "only support one dim");
    megdnn_assert(src.dtype.enumv() == DTypeEnum::Uint8,
                  "src.dtype == dtype::Uint8");
    megdnn_assert(dst.dtype.enumv() == DTypeEnum::Float32 ||
                          dst.dtype.enumv() == DTypeEnum::QuantizedS8,
                  "dst.dtype == dtype::Float32 || dst.dtype.enumv() == "
                  "DTypeEnum::QuantizedS8");
    megdnn_assert(ih % dct_block == 0, "ih mod dctblock == 0");
    megdnn_assert(iw % dct_block == 0, "iw mod dctblock == 0");
}

}  // namespace megdnn

// vim: syntax=cpp.doxygen
