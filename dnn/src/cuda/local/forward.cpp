/**
 * \file dnn/src/cuda/local/forward.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/cuda/local/opr_impl.h"

#include "src/cuda/local/local.cuh"
#include "src/cuda/utils.h"
#include "src/cuda/handle.h"

#include "src/common/utils.cuh"

namespace megdnn {
namespace cuda {

void LocalForwardImpl::exec(_megdnn_tensor_in src,
        _megdnn_tensor_in filter,
        _megdnn_tensor_out dst,
        _megdnn_workspace workspace)
{
    megdnn_assert(src.layout.dtype == dtype::Float32(),
                  "cuda do not support fp16 local operator");
    check_exec(src.layout, filter.layout, dst.layout, workspace.size);
    bool is_xcorr = param().mode == Mode::CROSS_CORRELATION;
    auto N = src.layout.shape[0],
         IC = src.layout.shape[1],
         IH = src.layout.shape[2],
         IW = src.layout.shape[3];
    auto OC = dst.layout.shape[1],
         OH = dst.layout.shape[2],
         OW = dst.layout.shape[3];
    auto FH = filter.layout.shape[3],
         FW = filter.layout.shape[4];
    auto handle = concrete_handle(this->handle());
    auto stream = cuda_stream(this->handle());
    auto cublas = cublas_handle(this->handle());
    auto one = handle->one_device();
    auto zero = handle->zero_device();
    size_t src_batch_strd = src.layout.stride[0];
    size_t dst_batch_strd = dst.layout.stride[0];
    if (use_cuda_convnet(src.layout, filter.layout, dst.layout)) {
        local::forward_proxy_convnet(src.ptr<dt_float32>(),
                filter.ptr<dt_float32>(),
                dst.ptr<dt_float32>(),
                reinterpret_cast<float *>(workspace.raw_ptr),
                N,
                IC, IH, IW,
                OC, OH, OW,
                FH, FW,
                src_batch_strd, dst_batch_strd,
                param().pad_h, param().pad_w,
                param().stride_h, param().stride_w,
                cublas, stream,
                one, zero);
    } else if (local::forward_proxy_default_share_mem_in_bytes(IH, IW) <=
               handle->device_prop().sharedMemPerBlock) {
        local::forward_proxy_default(src.ptr<dt_float32>(),
                filter.ptr<dt_float32>(),
                dst.ptr<dt_float32>(),
                N,
                IC, IH, IW,
                OC, OH, OW,
                FH, FW,
                src_batch_strd, dst_batch_strd,
                param().pad_h, param().pad_w,
                param().stride_h, param().stride_w,
                is_xcorr,
                stream);
    } else {
        megdnn_throw(ssprintf(
                "No usable kernel for local conv, src: %s filter: %s \n",
                src.layout.to_string().c_str(),
                filter.layout.to_string().c_str()));
    }
}

size_t LocalForwardImpl::get_workspace_in_bytes(const TensorLayout &src,
        const TensorLayout &filter,
        const TensorLayout &dst)
{
    size_t res = 0u;
    auto N = src.shape[0],
         IC = src.shape[1], IH = src.shape[2], IW = src.shape[3],
         OC = dst.shape[1], OH = dst.shape[2], OW = dst.shape[3],
         FH = filter.shape[3], FW = filter.shape[4];
    auto PH = param().pad_h, PW = param().pad_w,
         SH = param().stride_h, SW = param().stride_w;
    size_t src_batch_strd = src.stride[0];
    size_t dst_batch_strd = dst.stride[0];
    if (use_cuda_convnet(src, filter, dst)) {
        res = local::get_workspace_in_floats_forward_proxy_convnet(N,
                IC, IH, IW,
                OC, OH, OW,
                FH, FW,
                src_batch_strd, dst_batch_strd,
                PH, PW,
                SH, SW) * sizeof(dt_float32);
    } else {
        res = 0u;
    }
    return res;
}

bool LocalForwardImpl::use_cuda_convnet(const TensorLayout &src,
        const TensorLayout &filter,
        const TensorLayout &dst)
{
    auto N = src.shape[0],
         IC = src.shape[1], IH = src.shape[2], IW = src.shape[3],
         OC = dst.shape[1], OH = dst.shape[2], OW = dst.shape[3],
         FH = filter.shape[3], FW = filter.shape[4];
    auto PH = param().pad_h, PW = param().pad_w,
         SH = param().stride_h, SW = param().stride_w;
    return param().mode == Mode::CROSS_CORRELATION &&
        local::can_forward_proxy_convnet(N,
            IC, IH, IW,
            OC, OH, OW,
            FH, FW,
            IC*IH*IW, OC*OH*OW,
            PH, PW,
            SH, SW);
}

} // namespace cuda
} // namespace megdnn


// vim: syntax=cpp.doxygen
