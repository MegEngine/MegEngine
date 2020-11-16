/**
 * \file dnn/src/cuda/convolution3d/backward_data/cudnn.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./algo.h"

#include "src/cuda/utils.h"
#include "src/cuda/cudnn_wrapper.h"
#include "src/cuda/convolution3d/helper.h"

using namespace megdnn;
using namespace cuda;
using namespace convolution3d;

bool Convolution3DBackwardDataImpl::AlgoCUDNN::is_available(
        const SizeArgs &args) const {
    CUDNNBwdDataDescs D;

    if (!is_cudnn_supported(args.as_fwd_args()))
        return false;

    args.init_desc(D);
    size_t workspace_size;
    auto status = cudnnGetConvolutionBackwardDataWorkspaceSize(
            args.handle->cudnn_handle(),
            D.filter_desc.desc,
            D.diff_desc.desc,
            D.conv_desc.desc,
            D.grad_desc.desc,
            m_cudnn_enum,
            &workspace_size);
    return status == CUDNN_STATUS_SUCCESS;
}

size_t Convolution3DBackwardDataImpl::AlgoCUDNN::get_workspace_in_bytes(
        const SizeArgs &args) const {
    CUDNNBwdDataDescs D;
    args.init_desc(D);
    size_t workspace_size;
    auto status = cudnnGetConvolutionBackwardDataWorkspaceSize(
            args.handle->cudnn_handle(),
            D.filter_desc.desc,
            D.diff_desc.desc,
            D.conv_desc.desc,
            D.grad_desc.desc,
            m_cudnn_enum,
            &workspace_size);
    megdnn_assert(status == CUDNN_STATUS_SUCCESS,
            "conv bwd_data get workspace failed: %s; info: %s",
            cudnnGetErrorString(status), args.to_string().c_str());
    return workspace_size;
}

void Convolution3DBackwardDataImpl::AlgoCUDNN::exec(
        const ExecArgs &args) const {
    CUDNNBwdDataDescs D;
    args.init_desc(D);
    float alpha = 1.0f, beta = 0.0f;
    auto status = cudnnConvolutionBackwardData(args.handle->cudnn_handle(),
                &alpha,
                D.filter_desc.desc, args.filter_tensor->raw_ptr,
                D.diff_desc.desc, args.diff_tensor->raw_ptr,
                D.conv_desc.desc,
                m_cudnn_enum,
                args.workspace.raw_ptr,
                args.workspace.size,
                &beta,
                D.grad_desc.desc,
                args.grad_tensor->raw_ptr);
    megdnn_assert(status == CUDNN_STATUS_SUCCESS,
            "conv bwd_data failed: %s; info: %s",
            cudnnGetErrorString(status), args.to_string().c_str());
}

void Convolution3DBackwardDataImpl::AlgoPack::fill_cudnn_algos() {
    for (auto&& algo : CudnnAlgoPack::conv3d_bwd_data_algos()) {
        cudnn.push_back(algo.first);
    }
}

// vim: syntax=cpp.doxygen
