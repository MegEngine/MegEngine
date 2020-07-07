/**
 * \file dnn/src/cuda/warp_perspective/backward_data.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#include "src/cuda/warp_perspective/opr_impl.h"

#include "src/cuda/utils.h"
#include "src/cuda/warp_perspective/common.h"
#include "src/cuda/warp_perspective/helper.h"

namespace megdnn {
namespace cuda {

WorkspaceBundle WarpPerspectiveBackwardDataImpl::get_workspace_bundle(
        void* ptr, const TensorLayout& mat, const TensorLayout& mat_idx,
        const TensorLayout& diff, const TensorLayout& grad) const {
    SmallVector<size_t> sizes;
    TensorLayout fmat = mat;
    TensorLayout fdiff = diff;
    TensorLayout fgrad = grad;
    auto get_workspace = [&sizes](TensorLayout& layout) {
        if (layout.dtype == dtype::BFloat16()) {
            layout.dtype = dtype::Float32();
            sizes.push_back(layout.span().dist_byte());
        }
    };
    get_workspace(fmat);
    get_workspace(fdiff);
    get_workspace(fgrad);
    sizes.push_back(
            get_float32_workspace_in_bytes(fmat, mat_idx, fdiff, fgrad));
    return {ptr, std::move(sizes)};
}

void WarpPerspectiveBackwardDataImpl::exec(_megdnn_tensor_in smat,
                                           _megdnn_tensor_in mat_idx,
                                           _megdnn_tensor_in sdiff,
                                           _megdnn_tensor_out sgrad,
                                           _megdnn_workspace sworkspace) {
    check_exec(smat.layout, mat_idx.layout, sdiff.layout, sgrad.layout,
               sworkspace.size);
    TensorND mat = smat;
    TensorND diff = sdiff;
    TensorND grad = sgrad;
    auto bundle =
            get_workspace_bundle(sworkspace.raw_ptr, smat.layout,
                                 mat_idx.layout, sdiff.layout, sgrad.layout);
    auto ctypecvt = CompTypeCvter<dtype::BFloat16, dtype::Float32>(
            concrete_handle(this->handle()), &bundle);
    if (sgrad.layout.dtype.enumv() == DTypeTrait<dtype::BFloat16>::enumv) {
        ctypecvt.src_to_comp_type(smat, mat)
                .src_to_comp_type(sdiff, diff)
                .src_to_comp_type(sgrad, grad);
    }
    {
        auto workspace = ctypecvt.workspace();
        auto stream = cuda_stream(this->handle());
        auto N = grad.layout.shape[0], C = grad.layout.shape[1],
             IH = grad.layout.shape[2], IW = grad.layout.shape[3],
             OH = diff.layout.shape[2], OW = diff.layout.shape[3];
        int* midx_ptr = nullptr;
        if (mat_idx.raw_ptr) {
            megdnn_assert(mat_idx.layout.ndim == 1);
            N = mat_idx.layout.shape[0];
            midx_ptr = mat_idx.ptr<int>();
        } else {
            megdnn_assert(mat_idx.layout.ndim == 0);
        }

        auto bval = param().border_val;
        auto bmode = warp_perspective::get_bmode(param().bmode);

        size_t batch_x_channel_size = N * C;
        size_t max_batch_x_channel = max_batch_x_channel_size();
        if (batch_x_channel_size <= max_batch_x_channel) {
            warp_perspective::backward_data_proxy(
                    mat.ptr<dt_float32>(), midx_ptr, diff.ptr<dt_float32>(),
                    grad.ptr<dt_float32>(),
                    reinterpret_cast<float*>(workspace.raw_ptr), N,
                    grad.layout.shape[0], C, IH, IW, OH, OW, bval, bmode,
                    stream);
        } else {
            dt_float32* mat_ptr = mat.ptr<dt_float32>();
            dt_float32* diff_ptr = diff.ptr<dt_float32>();
            dt_float32* grad_ptr = grad.ptr<dt_float32>();
            size_t max_batch_size = max_batch_x_channel / C;
            while (N > 0) {
                size_t curr_batch_size =
                        N > max_batch_size ? max_batch_size : N;
                warp_perspective::backward_data_proxy(
                        mat_ptr, midx_ptr, diff_ptr, grad_ptr,
                        reinterpret_cast<float*>(workspace.raw_ptr),
                        curr_batch_size, grad.layout.shape[0], C, IH, IW, OH,
                        OW, bval, bmode, stream);

                if (N <= max_batch_size) {
                    break;
                } else {
                    N -= max_batch_size;
                    mat_ptr += curr_batch_size * mat.layout.stride[0];
                    diff_ptr += curr_batch_size * diff.layout.stride[0];
                    if (midx_ptr == nullptr) {
                        grad_ptr += curr_batch_size * grad.layout.stride[0];
                    } else {
                        midx_ptr += curr_batch_size;
                    }
                }
            }
        }
    }
    if (sgrad.layout.dtype.enumv() == DTypeTrait<dtype::BFloat16>::enumv) {
        ctypecvt.comp_to_dst_type(grad, sgrad);
    }
}

size_t WarpPerspectiveBackwardDataImpl::get_float32_workspace_in_bytes(
        const TensorLayout& /* mat */, const TensorLayout& mat_idx,
        const TensorLayout& diff, const TensorLayout& grad) const {
    auto N = grad.shape[0], C = grad.shape[1], IH = grad.shape[2],
         IW = grad.shape[3];
    auto OH = diff.shape[2], OW = diff.shape[3];
    auto bmode = warp_perspective::get_bmode(param().bmode);

    size_t max_batch_size = N;
    size_t max_batch_x_channel = max_batch_x_channel_size();
    if (N * C > max_batch_x_channel) {
        /* when batch size is too large, the workspace only contains part of grad,
           this will cause out of range with mat idx */
        megdnn_assert(mat_idx.ndim == 0, "batch size is too large, it's unsupported with mat idx backward.");
        max_batch_size = max_batch_x_channel / C;
    }

    auto res = warp_perspective::get_backward_data_workspace_in_bytes(
            max_batch_size, C, IH, IW, OH, OW, bmode);
    return res;
}

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
