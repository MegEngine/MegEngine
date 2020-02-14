/**
 * \file dnn/src/cuda/resize/backward.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/cuda/resize/opr_impl.h"

#include "src/cuda/resize/common.h"
#include "src/cuda/resize/helper.h"
#include "src/cuda/utils.h"

namespace megdnn {
namespace cuda {

void ResizeBackwardImpl::exec(_megdnn_tensor_in diff, _megdnn_tensor_out grad,
                              _megdnn_workspace workspace) {
    check_exec(diff.layout, grad.layout, workspace.size);
    auto stream = cuda_stream(this->handle());
    auto N = grad.layout.shape[0], C = grad.layout.shape[1],
         IH = grad.layout.shape[2], IW = grad.layout.shape[3],
         OH = diff.layout.shape[2], OW = diff.layout.shape[3];
    size_t max_batch_x_channel = max_batch_x_channel_size();
    dt_float32* diff_ptr = diff.ptr<dt_float32>();
    dt_float32* grad_ptr = grad.ptr<dt_float32>();
    size_t max_batch_size = max_batch_x_channel / C;
    while (N > 0) {
        size_t curr_batch_size = N > max_batch_size ? max_batch_size : N;
        resize::backward_data_proxy(diff_ptr, grad_ptr, curr_batch_size, C, IH,
                                    IW, OH, OW, stream);

        if (N <= max_batch_size) {
            break;
        } else {
            N -= max_batch_size;
            diff_ptr += curr_batch_size * diff.layout.stride[0];
            grad_ptr += curr_batch_size * grad.layout.stride[0];
        }
    }
}

size_t ResizeBackwardImpl::get_workspace_in_bytes(const TensorLayout& diff,
                                                  const TensorLayout& grad) {
    MEGDNN_MARK_USED_VAR(diff);
    MEGDNN_MARK_USED_VAR(grad);
    return 0;
}

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
