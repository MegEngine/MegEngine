/**
 * \file dnn/src/cuda/warp_perspective/backward_mat.cpp
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

WorkspaceBundle WarpPerspectiveBackwardMatImpl::get_workspace_bundle(
        void* ptr, const TensorLayout& src, const TensorLayout& mat,
        const TensorLayout& diff, const TensorLayout& grad) const {
    SmallVector<size_t> sizes;
    TensorLayout fsrc = src;
    TensorLayout fmat = mat;
    TensorLayout fdiff = diff;
    TensorLayout fgrad = grad;
    auto get_workspace = [&sizes](TensorLayout& layout) {
        if (layout.dtype == dtype::BFloat16()) {
            layout.dtype = dtype::Float32();
            sizes.push_back(layout.span().dist_byte());
        }
    };
    get_workspace(fsrc);
    get_workspace(fmat);
    get_workspace(fdiff);
    get_workspace(fgrad);
    return {ptr, std::move(sizes)};
}

void WarpPerspectiveBackwardMatImpl::exec(_megdnn_tensor_in ssrc,
                                          _megdnn_tensor_in smat,
                                          _megdnn_tensor_in smat_idx,
                                          _megdnn_tensor_in sdiff,
                                          _megdnn_tensor_out sgrad,
                                          _megdnn_workspace sworkspace) {
    check_exec(ssrc.layout, smat.layout, smat_idx.layout, sdiff.layout,
               sgrad.layout, sworkspace.size);
    TensorND src = ssrc;
    TensorND mat = smat;
    TensorND diff = sdiff;
    TensorND grad = sgrad;
    TensorND mat_idx = smat_idx;
    auto bundle = get_workspace_bundle(sworkspace.raw_ptr, ssrc.layout,
                                       smat.layout, sdiff.layout, sgrad.layout);
    auto ctypecvt = CompTypeCvter<dtype::BFloat16, dtype::Float32>(
            concrete_handle(this->handle()), &bundle);
    if (ssrc.layout.dtype.enumv() == DTypeTrait<dtype::BFloat16>::enumv) {
        ctypecvt.src_to_comp_type(ssrc, src)
                .src_to_comp_type(smat, mat)
                .src_to_comp_type(sdiff, diff)
                .src_to_comp_type(sgrad, grad);
    }
    {
        auto stream = cuda_stream(this->handle());
        auto N = src.layout.shape[0], C = src.layout.shape[1],
             IH = src.layout.shape[2], IW = src.layout.shape[3],
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
            warp_perspective::backward_mat_proxy(
                    src.ptr<dt_float32>(), mat.ptr<dt_float32>(), midx_ptr,
                    diff.ptr<dt_float32>(), grad.ptr<dt_float32>(), N, C, IH,
                    IW, OH, OW, bval, bmode, stream);
        } else {
            dt_float32* src_ptr = src.ptr<dt_float32>();
            dt_float32* mat_ptr = mat.ptr<dt_float32>();
            dt_float32* diff_ptr = diff.ptr<dt_float32>();
            dt_float32* grad_ptr = grad.ptr<dt_float32>();
            size_t max_batch_size = max_batch_x_channel / C;
            while (N > 0) {
                size_t curr_batch_size =
                        N > max_batch_size ? max_batch_size : N;
                warp_perspective::backward_mat_proxy(
                        src_ptr, mat_ptr, midx_ptr, diff_ptr, grad_ptr,
                        curr_batch_size, C, IH, IW, OH, OW, bval, bmode,
                        stream);

                if (N <= max_batch_size) {
                    break;
                } else {
                    N -= max_batch_size;
                    if (midx_ptr == nullptr) {
                        src_ptr += curr_batch_size * src.layout.stride[0];
                    } else {
                        midx_ptr += curr_batch_size;
                    }
                    mat_ptr += curr_batch_size * mat.layout.stride[0];
                    diff_ptr += curr_batch_size * diff.layout.stride[0];
                    grad_ptr += curr_batch_size * grad.layout.stride[0];
                }
            }
        }
    }

    if (ssrc.layout.dtype.enumv() == DTypeTrait<dtype::BFloat16>::enumv) {
        ctypecvt.comp_to_dst_type(grad, sgrad);
    }
}

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
