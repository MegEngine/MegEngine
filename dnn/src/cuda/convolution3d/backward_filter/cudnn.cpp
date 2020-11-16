/**
 * \file dnn/src/cuda/convolution3d/backward_filter/cudnn.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./algo.h"

#include "src/cuda/convolution3d/helper.h"
#include "src/cuda/cudnn_wrapper.h"
#include "src/cuda/utils.h"

using namespace megdnn;
using namespace cuda;
using namespace convolution3d;

bool Convolution3DBackwardFilterImpl::AlgoCUDNN::is_available(
        const SizeArgs& args) const {
    CUDNNBwdFilterDescs D;

    if (!is_cudnn_supported(args.as_fwd_args()))
        return false;

    args.init_desc(D);
    size_t workspace_size;
    auto status = cudnnGetConvolutionBackwardFilterWorkspaceSize(
            args.handle->cudnn_handle(), D.src_desc.desc, D.diff_desc.desc,
            D.conv_desc.desc, D.grad_desc.desc, m_cudnn_enum, &workspace_size);
    return status == CUDNN_STATUS_SUCCESS;
}

size_t Convolution3DBackwardFilterImpl::AlgoCUDNN::get_workspace_in_bytes(
        const SizeArgs& args) const {
    CUDNNBwdFilterDescs D;

    args.init_desc(D);
    size_t workspace_size;
    auto status = cudnnGetConvolutionBackwardFilterWorkspaceSize(
            args.handle->cudnn_handle(), D.src_desc.desc, D.diff_desc.desc,
            D.conv_desc.desc, D.grad_desc.desc, m_cudnn_enum, &workspace_size);
    megdnn_assert(status == CUDNN_STATUS_SUCCESS,
                  "conv bwd_filter get workspace failed: %s; info: %s",
                  cudnnGetErrorString(status), args.to_string().c_str());
    return workspace_size;
}

void Convolution3DBackwardFilterImpl::AlgoCUDNN::exec(
        const ExecArgs& args) const {
    CUDNNBwdFilterDescs D;
    args.init_desc(D);
    float alpha = 1.0f, beta = 0.0f;
    auto status = cudnnConvolutionBackwardFilter(
            args.handle->cudnn_handle(), &alpha, D.src_desc.desc,
            args.src_tensor->raw_ptr, D.diff_desc.desc,
            args.diff_tensor->raw_ptr, D.conv_desc.desc, m_cudnn_enum,
            args.workspace.raw_ptr, args.workspace.size, &beta,
            D.grad_desc.desc, args.grad_tensor->raw_ptr);
    megdnn_assert(status == CUDNN_STATUS_SUCCESS,
                  "conv bwd_data failed: %s; info: %s",
                  cudnnGetErrorString(status), args.to_string().c_str());
}

void Convolution3DBackwardFilterImpl::AlgoPack::fill_cudnn_algos() {
    for (auto&& algo : CudnnAlgoPack::conv3d_bwd_flt_algos()) {
        cudnn.push_back(algo.first);
    }
}

// vim: syntax=cpp.doxygen
