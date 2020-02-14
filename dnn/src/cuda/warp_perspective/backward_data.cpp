/**
 * \file dnn/src/cuda/warp_perspective/backward_data.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/cuda/warp_perspective/opr_impl.h"

#include "src/cuda/utils.h"
#include "src/cuda/warp_perspective/common.h"
#include "src/cuda/warp_perspective/helper.h"

namespace megdnn {
namespace cuda {

void WarpPerspectiveBackwardDataImpl::exec(_megdnn_tensor_in mat,
        _megdnn_tensor_in diff,
        _megdnn_tensor_out grad,
        _megdnn_workspace workspace)
{
    check_exec(mat.layout, diff.layout, grad.layout, workspace.size);
    auto stream = cuda_stream(this->handle());
    auto N = grad.layout.shape[0],
         C = grad.layout.shape[1],
         IH = grad.layout.shape[2],
         IW = grad.layout.shape[3],
         OH = diff.layout.shape[2],
         OW = diff.layout.shape[3];
    auto bval = param().border_val;
    auto bmode = warp_perspective::get_bmode(param().bmode);

    size_t batch_x_channel_size = N * C;
    size_t max_batch_x_channel = max_batch_x_channel_size();
    if(batch_x_channel_size <= max_batch_x_channel) {
        warp_perspective::backward_data_proxy(
            mat.ptr<dt_float32>(),
            diff.ptr<dt_float32>(),
            grad.ptr<dt_float32>(),
            reinterpret_cast<float *>(workspace.raw_ptr),
            N, C, IH, IW, OH, OW, bval,
            bmode, stream);
    } else {
        dt_float32* mat_ptr = mat.ptr<dt_float32>();
        dt_float32* diff_ptr = diff.ptr<dt_float32>();
        dt_float32* grad_ptr = grad.ptr<dt_float32>();
        size_t max_batch_size = max_batch_x_channel / C;
        while (N > 0){
            size_t curr_batch_size = N > max_batch_size ? max_batch_size : N;
            warp_perspective::backward_data_proxy(
                mat_ptr, diff_ptr, grad_ptr,
                reinterpret_cast<float *>(workspace.raw_ptr),
                curr_batch_size, C, IH, IW, OH, OW, bval,
                bmode, stream);

            if( N <= max_batch_size) {
                break;
            }
            else {
                N -= max_batch_size;
                mat_ptr += curr_batch_size * mat.layout.stride[0];
                diff_ptr += curr_batch_size * diff.layout.stride[0];
                grad_ptr += curr_batch_size * grad.layout.stride[0];
            }
        }
    }
}

size_t WarpPerspectiveBackwardDataImpl::get_workspace_in_bytes(
        const TensorLayout & /* mat */,
        const TensorLayout &diff,
        const TensorLayout &grad)
{
    auto N = grad.shape[0], C = grad.shape[1],
         IH = grad.shape[2], IW = grad.shape[3];
    auto OH = diff.shape[2], OW = diff.shape[3];
    auto bmode = warp_perspective::get_bmode(param().bmode);

    size_t max_batch_size = N;
    size_t max_batch_x_channel = max_batch_x_channel_size();
    if(N * C > max_batch_x_channel) {
        max_batch_size = max_batch_x_channel / C;
    }

    auto res = warp_perspective::get_backward_data_workspace_in_bytes(
            max_batch_size, C, IH, IW, OH, OW, bmode);
    return res;
}

} // namespace cuda
} // namespace megdnn

// vim: syntax=cpp.doxygen
