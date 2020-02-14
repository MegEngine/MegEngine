/**
 * \file dnn/src/cuda/warp_perspective/backward_mat.cpp
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

void WarpPerspectiveBackwardMatImpl::exec(_megdnn_tensor_in src,
        _megdnn_tensor_in mat,
        _megdnn_tensor_in diff,
        _megdnn_tensor_out grad,
        _megdnn_workspace workspace)
{
    check_exec(src.layout, mat.layout, diff.layout, grad.layout,
            workspace.size);
    auto stream = cuda_stream(this->handle());
    auto N = src.layout.shape[0],
         C = src.layout.shape[1],
         IH = src.layout.shape[2],
         IW = src.layout.shape[3],
         OH = diff.layout.shape[2],
         OW = diff.layout.shape[3];
    auto bval = param().border_val;
    auto bmode = warp_perspective::get_bmode(param().bmode);

    size_t batch_x_channel_size = N * C;
    size_t max_batch_x_channel = max_batch_x_channel_size();
    if(batch_x_channel_size <= max_batch_x_channel) {
        warp_perspective::backward_mat_proxy(src.ptr<dt_float32>(),
            mat.ptr<dt_float32>(),
            diff.ptr<dt_float32>(),
            grad.ptr<dt_float32>(),
            N, C, IH, IW, OH, OW, bval,
            bmode, stream);
    } else {
        dt_float32* src_ptr = src.ptr<dt_float32>();
        dt_float32* mat_ptr = mat.ptr<dt_float32>();
        dt_float32* diff_ptr = diff.ptr<dt_float32>();
        dt_float32* grad_ptr = grad.ptr<dt_float32>();
        size_t max_batch_size = max_batch_x_channel / C;
        while (N > 0){
            size_t curr_batch_size = N > max_batch_size ? max_batch_size : N;
            warp_perspective::backward_mat_proxy(src_ptr,
                mat_ptr, diff_ptr, grad_ptr,
                curr_batch_size, C, IH, IW, OH, OW, bval,
                bmode, stream);

            if( N <= max_batch_size) {
                break;
            }
            else {
                N -= max_batch_size;
                src_ptr += curr_batch_size * src.layout.stride[0];
                mat_ptr += curr_batch_size * mat.layout.stride[0];
                diff_ptr += curr_batch_size * diff.layout.stride[0];
                grad_ptr += curr_batch_size * grad.layout.stride[0];
            }
        }
    }
}

} // namespace cuda
} // namespace megdnn

// vim: syntax=cpp.doxygen

