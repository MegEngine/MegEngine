/**
 * \file dnn/src/cuda/mask_conv/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/cuda/mask_conv/opr_impl.h"
#include "./mask_conv.cuh"
#include "src/cuda/utils.h"

namespace megdnn {
namespace cuda {

MaskConvForwardImpl::MaskConvForwardImpl(Handle* handle)
        : MaskConvForward(handle) {
    m_conv_opr = static_cast<HandleImpl*>(handle)
                         ->create_operator<ConvolutionForward>();
}

void MaskConvForwardImpl::exec(_megdnn_tensor_in src, _megdnn_tensor_in filter,
                               _megdnn_tensor_in mask, _megdnn_tensor_out dst,
                               _megdnn_workspace workspace) {
    megdnn_assert(dst.layout.dtype.enumv() == DTypeTrait<dtype::Float32>::enumv,
                  "Mask conv only support Float32 dtype.");
    m_conv_opr->exec(src, filter, dst, nullptr, workspace);
    auto stream = cuda_stream(handle());
#define cb(DType)                                                     \
    if (mask.layout.dtype == DType()) {                               \
        using ctype = typename DTypeTrait<DType>::ctype;              \
        mask_conv::set_zero_by_mask_proxy<ctype>(                     \
                dst.ptr<float>(), mask.ptr<ctype>(), dst.layout[0],   \
                dst.layout[1], dst.layout[2], dst.layout[3], stream); \
        return;                                                       \
    }

    MEGDNN_FOREACH_COMPUTING_DTYPE_INT(cb)
#undef cb
    megdnn_assert_internal(0);
}

void MaskPropagateImpl::exec(_megdnn_tensor_in src, _megdnn_tensor_out dst,
                             _megdnn_workspace) {
    auto stream = cuda_stream(handle());

#define cb(DType)                                                              \
    if (src.layout.dtype == DType()) {                                         \
        using ctype = typename DTypeTrait<DType>::ctype;                       \
        mask_conv::mask_propagate_exec_proxy<ctype>(                           \
                src.ptr<ctype>(), dst.ptr<ctype>(), src.layout[0],             \
                src.layout[1], dst.layout[0], dst.layout[1], param().kernel_h, \
                param().kernel_w, param().stride_h, param().stride_w,          \
                param().pad_h, param().pad_w, param().dilate_h,                \
                param().dilate_w, stream);                                     \
        return;                                                                \
    }

    MEGDNN_FOREACH_COMPUTING_DTYPE_INT(cb);
#undef cb
    megdnn_assert_internal(0);
}

}  // namespace cuda
}  // namespace megdnn
