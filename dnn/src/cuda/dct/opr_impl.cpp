/**
 * \file dnn/src/naive/dct/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#include "src/common/utils.h"
#include "src/cuda/dct/dct_channel_select.cuh"
#include "src/cuda/dct/opr_impl.h"
#include "src/cuda/handle.h"
#include "src/cuda/utils.h"
namespace megdnn {
namespace cuda {

void DctChannelSelectForwardImpl::exec(_megdnn_tensor_in src,
                                       _megdnn_tensor_in mask_offset,
                                       _megdnn_tensor_in mask_val,
                                       _megdnn_tensor_out dst,
                                       _megdnn_workspace /*workspace*/) {
    auto stream = cuda_stream(this->handle());
    const int in = src.layout.shape[0];
    const int ic = src.layout.shape[1];
    const int ih = src.layout.shape[2];
    const int iw = src.layout.shape[3];
    int oc = dst.layout.shape[1];
    const bool with_fix_32_mask =
            param().fastImpl == Param::FastImpl::FIX_32_MASK;
    if (param().format == Param::Format::NCHW4) {
        megdnn_assert(dst.layout.ndim == 5 && dst.layout.shape[4] == 4,
                      "dst must be nchw4");
        oc = oc * 4;
    }
    megdnn_assert(!with_fix_32_mask || (with_fix_32_mask && oc == 32),
                  "only support specify mask");
    megdnn_assert(param().dct_block_size == 8, "only support dct block = 8");
    auto error_info =
            concrete_handle(this->handle())->megcore_context().error_info;
    constexpr int dct_block = 8;
    const int* mask_offset_ptr = nullptr;
    const int* mask_val_ptr = nullptr;
    if (mask_offset.layout.ndim == 1 && mask_offset.layout.shape[0] >= 2) {
        mask_offset_ptr = mask_offset.ptr<int32_t>();
        mask_val_ptr = mask_val.ptr<int32_t>();
    }
    if (dst.layout.dtype.enumv() == DTypeEnum::Float32) {
        megdnn_assert(param().format == Param::Format::NCHW,
                      "fp32 only support nchw");
        dct::call_kern_dct<dct_block, dct::DctLayoutFormat::NCHW>(
                src.ptr<uint8_t>(), dst.ptr<float>(), in, ic, ih, iw, oc,
                with_fix_32_mask, mask_offset_ptr, mask_val_ptr, stream,
                error_info, m_error_tracker);
    } else {
        megdnn_assert(dst.layout.dtype.enumv() == DTypeEnum::QuantizedS8,
                      "only support fp32 and qs8");
        megdnn_assert(param().format == Param::Format::NCHW4,
                      "qint8 only support nchw4");
        dct::call_kern_dct<dct_block, dct::DctLayoutFormat::NCHW4>(
                src.ptr<uint8_t>(), (int8_t*)dst.raw_ptr, in, ic, ih, iw, oc,
                with_fix_32_mask, mask_offset_ptr, mask_val_ptr, stream,
                error_info, m_error_tracker,
                dst.layout.dtype.param<::megdnn::dtype::QuantizedS8>().scale);
    }
}

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
