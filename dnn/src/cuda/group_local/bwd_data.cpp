/**
 * \file dnn/src/cuda/group_local/bwd_data.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/cuda/group_local/opr_impl.h"

#include "src/common/utils.h"
#include "src/cuda/local/local.cuh"
#include "src/cuda/utils.h"

namespace megdnn {
namespace cuda {

void GroupLocalBackwardDataImpl::exec(_megdnn_tensor_in filter,
        _megdnn_tensor_in diff,
        _megdnn_tensor_out grad,
        _megdnn_workspace workspace)
{
    check_exec(filter.layout, diff.layout, grad.layout, workspace.size);

    auto G = filter.layout[0];
    auto N = grad.layout.shape[0], IC = grad.layout.shape[1]/G,
         IH = grad.layout.shape[2], IW = grad.layout.shape[3],
         OC = diff.layout.shape[1]/G,
         OH = diff.layout.shape[2], OW = diff.layout.shape[3];
    auto FH = filter.layout.shape[4], FW = filter.layout.shape[5];
    auto PH = param().pad_h, PW = param().pad_w;
    auto SH = param().stride_h, SW = param().stride_w;
    float *sptr = grad.ptr<dt_float32>();
    const float *fptr = filter.ptr<dt_float32>();
    const float *dptr = diff.ptr<dt_float32>();
    float *wptr = workspace.ptr<dt_float32>();
    auto handle = concrete_handle(this->handle());
    auto stream = cuda_stream(this->handle());
    auto cublas = cublas_handle(this->handle());
    auto one = handle->one_device();
    auto zero = handle->zero_device();
    megdnn_assert(local::can_backward_data_proxy_convnet(N, IC, IH, IW,
                OC, OH, OW,
                FH, FW,
                G*IC*IH*IW, G*OC*OH*OW,
                PH, PW,
                SH, SW),
            "Cannot do Group Local bwd data.");
    for (size_t g = 0; g < G; ++g) {
        local::backward_data_proxy_convnet(fptr + g*OH*OW*IC*FH*FW*OC,
                dptr + g*OC*OH*OW,
                sptr + g*IC*IH*IW,
                wptr,
                N, IC, IH, IW,
                OC, OH, OW,
                FH, FW,
                G*IC*IH*IW, G*OC*OH*OW,
                PH, PW,
                SH, SW,
                cublas, stream, one, zero);
    }
}

GroupLocalBackwardDataImpl::GroupLocalBackwardDataImpl(Handle *handle):
    GroupLocalBackwardData(handle)
{
}

size_t GroupLocalBackwardDataImpl::get_workspace_in_bytes(
        const TensorLayout &filter,
        const TensorLayout &diff,
        const TensorLayout &grad)
{
    auto G = filter[0];
    auto N = grad.shape[0], IC = grad.shape[1]/G,
         IH = grad.shape[2], IW = grad.shape[3],
         OC = diff.shape[1]/G,
         OH = diff.shape[2], OW = diff.shape[3];
    auto FH = filter.shape[4], FW = filter.shape[5];
    auto PH = param().pad_h, PW = param().pad_w;
    auto SH = param().stride_h, SW = param().stride_w;
    auto res = local::get_workspace_in_floats_backward_data_proxy_convnet(N,
            IC, IH, IW,
            OC, OH, OW,
            FH, FW,
            G*IC*IH*IW, G*OC*OH*OW,
            PH, PW,
            SH, SW) * sizeof(float);
    return res;
}

} // namespace cuda
} // namespace megdnn
// vim: syntax=cpp.doxygen
