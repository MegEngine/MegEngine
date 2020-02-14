/**
 * \file dnn/src/cuda/lrn/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/cuda/lrn/opr_impl.h"

#include "src/cuda/utils.h"

namespace megdnn {
namespace cuda {

void LRNForwardImpl::setup_descs(const TensorLayout &src,
        const TensorLayout &dst)
{
    src_desc.set(src);
    dst_desc.set(dst);
    lrn_desc.set(this->param());
}

void LRNForwardImpl::exec(_megdnn_tensor_in src,
        _megdnn_tensor_out dst,
        _megdnn_workspace workspace)
{
    check_exec(src.layout, dst.layout, workspace.size);
    auto handle = cudnn_handle(this->handle());
    setup_descs(src.layout, dst.layout);
    float alpha = 1.0f, beta = 0.0f;
    cudnn_check(cudnnLRNCrossChannelForward(handle,
                lrn_desc.desc,
                CUDNN_LRN_CROSS_CHANNEL_DIM1,
                &alpha, src_desc.desc, src.raw_ptr,
                &beta, dst_desc.desc, dst.raw_ptr));
}

void LRNBackwardImpl::setup_descs(const TensorLayout &src,
        const TensorLayout &dst,
        const TensorLayout &diff,
        const TensorLayout &grad)
{
    src_desc.set(src);
    dst_desc.set(dst);
    diff_desc.set(diff);
    grad_desc.set(grad);
    lrn_desc.set(this->param());
}

void LRNBackwardImpl::exec(_megdnn_tensor_in src,
        _megdnn_tensor_in dst,
        _megdnn_tensor_in diff,
        _megdnn_tensor_out grad,
        _megdnn_workspace workspace)
{
    check_exec(src.layout, dst.layout, diff.layout, grad.layout,
            workspace.size);
    auto handle = cudnn_handle(this->handle());
    setup_descs(src.layout, dst.layout, diff.layout, grad.layout);
    float alpha = 1.0f, beta = 0.0f;
    cudnn_check(cudnnLRNCrossChannelBackward(handle,
                lrn_desc.desc,
                CUDNN_LRN_CROSS_CHANNEL_DIM1,
                &alpha,
                dst_desc.desc, dst.raw_ptr,
                diff_desc.desc, diff.raw_ptr,
                src_desc.desc, src.raw_ptr,
                &beta,
                grad_desc.desc, grad.raw_ptr));
}

} // namespace cuda
} // namespace megdnn

// vim: syntax=cpp.doxygen

