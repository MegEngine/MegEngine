/**
 * \file dnn/src/cuda/group_local/fwd.cpp
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

#include "./cuda_interface.h"

namespace megdnn {
namespace cuda {

void GroupLocalForwardImpl::exec(_megdnn_tensor_in src,
        _megdnn_tensor_in filter,
        _megdnn_tensor_out dst,
        _megdnn_workspace workspace)
{
    megdnn_assert(src.layout.dtype == dtype::Float32(),
                  "cuda do not support fp16 group local operator");
    check_exec(src.layout, filter.layout, dst.layout, workspace.size);

    auto G = filter.layout[0];
    auto N = src.layout.shape[0], IC = src.layout.shape[1]/G,
         IH = src.layout.shape[2], IW = src.layout.shape[3],
         OC = dst.layout.shape[1]/G,
         OH = dst.layout.shape[2], OW = dst.layout.shape[3];
    auto FH = filter.layout.shape[4], FW = filter.layout.shape[5];
    auto PH = param().pad_h, PW = param().pad_w;
    auto SH = param().stride_h, SW = param().stride_w;
    const float *sptr = src.ptr<dt_float32>();
    const float *fptr = filter.ptr<dt_float32>();
    float *dptr = dst.ptr<dt_float32>();
    float *wptr = workspace.ptr<dt_float32>();
    auto handle = concrete_handle(this->handle());
    auto stream = cuda_stream(this->handle());
    auto cublas = cublas_handle(this->handle());
    auto one = handle->one_device();
    auto zero = handle->zero_device();
    if (prefer_inference_kernel(src.layout, filter.layout, dst.layout)) {
        run_inference_kernel(sptr, fptr, dptr, wptr,
                N, IC, IH, IW,
                OC, OH, OW,
                FH, FW,
                G,
                PH, PW,
                SH, SW,
                stream
                );
    } else if (local::can_forward_proxy_convnet(N, IC, IH, IW,
                OC, OH, OW,
                FH, FW,
                G*IC*IH*IW, G*OC*OH*OW,
                PH, PW,
                SH, SW))
    {
        // use convnet
        for (size_t g = 0; g < G; ++g) {
            local::forward_proxy_convnet(sptr + g*IC*IH*IW,
                    fptr + g*OH*OW*IC*FH*FW*OC,
                    dptr + g*OC*OH*OW,
                    wptr,
                    N, IC, IH, IW,
                    OC, OH, OW,
                    FH, FW,
                    G*IC*IH*IW, G*OC*OH*OW,
                    PH, PW,
                    SH, SW,
                    cublas, stream, one, zero);
        }
    } else {
        local::check_input(N, IC, IH, IW, OC, OH, OW, FH, FW,
                G*IC*IH*IW, G*OC*OH*OW,
                PH, PW,
                SH, SW,
                true);
        // do not use convnet
        for (size_t g = 0; g < G; ++g) {
            local::forward_proxy_weiming(sptr + g*IC*IH*IW,
                    fptr + g*OH*OW*IC*FH*FW*OC,
                    dptr + g*OC*OH*OW,
                    N, IC, IH, IW,
                    OC, OH, OW,
                    FH, FW,
                    G*IC*IH*IW, G*OC*OH*OW,
                    PH, PW,
                    SH, SW,
                    true, stream);
        }
    }
}

GroupLocalForwardImpl::GroupLocalForwardImpl(Handle *handle):
    GroupLocalForward(handle)
{
}

size_t GroupLocalForwardImpl::get_workspace_in_bytes(const TensorLayout &src,
        const TensorLayout &filter,
        const TensorLayout &dst)
{
    auto G = filter[0];
    auto N = src.shape[0], IC = src.shape[1]/G,
         IH = src.shape[2], IW = src.shape[3],
         OC = dst.shape[1]/G,
         OH = dst.shape[2], OW = dst.shape[3];
    auto FH = filter.shape[4], FW = filter.shape[5];
    auto PH = param().pad_h, PW = param().pad_w;
    auto SH = param().stride_h, SW = param().stride_w;
    if (prefer_inference_kernel(src, filter, dst)) {
        return 0;
    } else if (local::can_forward_proxy_convnet(N, IC, IH, IW,
            OC, OH, OW,
            FH, FW,
            G*IC*IH*IW, G*OC*OH*OW,
            PH, PW,
            SH, SW))
    {
        auto res = local::get_workspace_in_floats_forward_proxy_convnet(N,
                IC, IH, IW,
                OC, OH, OW,
                FH, FW,
                G*IC*IH*IW, G*OC*OH*OW,
                PH, PW,
                SH, SW) * sizeof(float);
        return res;
    } else {
        return 0;
    }
}

bool GroupLocalForwardImpl::prefer_inference_kernel(const TensorLayout &src,
        const TensorLayout &filter,
        const TensorLayout &dst)
{
    megdnn_ignore(filter);
    megdnn_ignore(dst);
    return src.shape[0] <= 8;
}

} // namespace cuda
} // namespace megdnn
// vim: syntax=cpp.doxygen
