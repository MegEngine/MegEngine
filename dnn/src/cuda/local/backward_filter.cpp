/**
 * \file dnn/src/cuda/local/backward_filter.cpp
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
#include "src/cuda/handle.h"
#include "src/cuda/utils.h"

namespace megdnn {
namespace cuda {
namespace local {

void boom_backward_filter()
{
    megdnn_throw("Local bad param: cannot do backward_filter by cuda_convnet");
}

} // namespace local
} // namespace cuda
} // namespace megdnn

namespace megdnn {
namespace cuda {

void LocalBackwardFilterImpl::exec(_megdnn_tensor_in src,
        _megdnn_tensor_in diff,
        _megdnn_tensor_out grad,
        _megdnn_workspace workspace)
{
    check_exec(src.layout, diff.layout, grad.layout, workspace.size);
    megdnn_assert(param().mode == Mode::CROSS_CORRELATION);
    auto N = src.layout.shape[0],
         IC = src.layout.shape[1],
         IH = src.layout.shape[2],
         IW = src.layout.shape[3];
    auto OC = diff.layout.shape[1],
         OH = diff.layout.shape[2],
         OW = diff.layout.shape[3];
    auto FH = grad.layout.shape[3],
         FW = grad.layout.shape[4];
    auto handle = concrete_handle(this->handle());
    auto stream = cuda_stream(this->handle());
    auto cublas = cublas_handle(this->handle());
    auto one = handle->one_device();
    auto zero = handle->zero_device();
    if (use_cuda_convnet(src.layout, diff.layout, grad.layout)) {
        local::backward_filter_proxy_convnet(src.ptr<dt_float32>(),
                diff.ptr<dt_float32>(),
                grad.ptr<dt_float32>(),
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
        local::boom_backward_filter();
    }
}

size_t LocalBackwardFilterImpl::get_workspace_in_bytes(const TensorLayout &src,
        const TensorLayout &diff,
        const TensorLayout &grad)
{
    auto N = src.shape[0],
         IC = src.shape[1], IH = src.shape[2], IW = src.shape[3],
         OC = diff.shape[1], OH = diff.shape[2], OW = diff.shape[3],
         FH = grad.shape[3], FW = grad.shape[4];
    auto SH = param().stride_h, SW = param().stride_w,
         PH = param().pad_h, PW = param().pad_w;
    size_t res = 0u;
    if (use_cuda_convnet(src, diff, grad)) {
        res = local::get_workspace_in_floats_backward_filter_proxy_convnet(N,
                IC, IH, IW,
                OC, OH, OW,
                FH, FW,
                IC*IH*IW, OC*OH*OW,
                SH, SW,
                PH, PW) * sizeof(dt_float32);
    } else {
        local::boom_backward_filter();
    }
    return res;
}

bool LocalBackwardFilterImpl::use_cuda_convnet(const TensorLayout &src,
        const TensorLayout &diff,
        const TensorLayout &grad)
{
    auto N = src.shape[0],
         IC = src.shape[1], IH = src.shape[2], IW = src.shape[3],
         OC = diff.shape[1], OH = diff.shape[2], OW = diff.shape[3],
         FH = grad.shape[3], FW = grad.shape[4];
    auto SH = param().stride_h, SW = param().stride_w,
         PH = param().pad_h, PW = param().pad_w;
    return local::can_backward_filter_proxy_convnet(N, IC, IH, IW,
            OC, OH, OW,
            FH, FW,
            IC*IH*IW, OC*OH*OW,
            PH, PW,
            SH, SW);
}

} // namespace cuda
} // namespace megdnn

// vim: syntax=cpp.doxygen
