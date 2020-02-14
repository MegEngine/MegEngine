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

namespace megdnn {
namespace cuda {
namespace local {

void check_input(size_t N,
        size_t IC, size_t IH, size_t IW,
        size_t OC, size_t OH, size_t OW,
        size_t FH, size_t FW,
        size_t INs, size_t ONs,
        size_t PH, size_t PW,
        size_t SH, size_t SW,
        bool is_xcorr)
{
    megdnn_ignore(N);
    megdnn_ignore(IC);
    megdnn_ignore(IH);
    megdnn_ignore(IW);
    megdnn_ignore(OC);
    megdnn_ignore(OH);
    megdnn_ignore(OW);
    megdnn_ignore(FH);
    megdnn_ignore(FW);
    megdnn_ignore(INs);
    megdnn_ignore(ONs);
    megdnn_ignore(PH);
    megdnn_ignore(PW);
    megdnn_ignore(SH);
    megdnn_ignore(SW);
    megdnn_ignore(is_xcorr);
    // shared memory constraint
    megdnn_assert(IH*IW <= 768, "spatial size should not be larger than 768.");
    // megdnn_assert(4 * 4 * 4 * IH * IW <= 49152);
}

} // namespace local
} // namespace cuda
} // namespace megdnn

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
    if (use_cuda_convnet(src.layout, filter.layout, dst.layout)) {
        local::forward_proxy_convnet(src.ptr<dt_float32>(),
                filter.ptr<dt_float32>(),
                dst.ptr<dt_float32>(),
                reinterpret_cast<float *>(workspace.raw_ptr),
                N,
                IC, IH, IW,
                OC, OH, OW,
                FH, FW,
                IC*IH*IW, OC*OH*OW,
                param().pad_h, param().pad_w,
                param().stride_h, param().stride_w,
                cublas, stream,
                one, zero);
    } else {
        local::check_input(N, IC, IH, IW, OC, OH, OW, FH, FW,
                IC*IH*IW, OC*OH*OW,
                param().pad_h, param().pad_w,
                param().stride_h, param().stride_w,
                is_xcorr);
        local::forward_proxy_weiming(src.ptr<dt_float32>(),
                filter.ptr<dt_float32>(),
                dst.ptr<dt_float32>(),
                N,
                IC, IH, IW,
                OC, OH, OW,
                FH, FW,
                IC*IH*IW, OC*OH*OW,
                param().pad_h, param().pad_w,
                param().stride_h, param().stride_w,
                is_xcorr,
                stream);
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
    if (use_cuda_convnet(src, filter, dst)) {
        res = local::get_workspace_in_floats_forward_proxy_convnet(N,
                IC, IH, IW,
                OC, OH, OW,
                FH, FW,
                IC*IH*IW, OC*OH*OW,
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
