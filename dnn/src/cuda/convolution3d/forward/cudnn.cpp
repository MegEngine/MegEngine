/**
 * \file dnn/src/cuda/convolution3d/forward/cudnn.cpp
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

bool Convolution3DForwardImpl::AlgoCUDNN::is_available(
        const SizeArgs &args) const {
    CUDNNForwardDescs D;

    if (!is_cudnn_supported(args))
        return false;

    args.init_desc(D);
    size_t workspace_size;
    auto status = cudnnGetConvolutionForwardWorkspaceSize(
            args.handle->cudnn_handle(),
            D.src_desc.desc,
            D.filter_desc.desc,
            D.conv_desc.desc,
            D.dst_desc.desc,
            m_cudnn_enum,
            &workspace_size);
    return status == CUDNN_STATUS_SUCCESS;
}

size_t Convolution3DForwardImpl::AlgoCUDNN::get_workspace_in_bytes(
        const SizeArgs &args) const {
    CUDNNForwardDescs D;
    args.init_desc(D);
    size_t workspace_size;
    auto status = cudnnGetConvolutionForwardWorkspaceSize(
            args.handle->cudnn_handle(),
            D.src_desc.desc,
            D.filter_desc.desc,
            D.conv_desc.desc,
            D.dst_desc.desc,
            m_cudnn_enum,
            &workspace_size);
    megdnn_assert(status == CUDNN_STATUS_SUCCESS,
            "conv fwd get workspace failed: %s; info: %s",
            cudnnGetErrorString(status), args.to_string().c_str());
    return workspace_size;
}

void Convolution3DForwardImpl::AlgoCUDNN::exec(
        const ExecArgs &args) const {
    CUDNNForwardDescs D;
    args.init_desc(D);
    float alpha = 1.0f, beta = 0.0f;
    auto status = cudnnConvolutionForward(args.handle->cudnn_handle(),
                &alpha,
                D.src_desc.desc, args.src_tensor->raw_ptr,
                D.filter_desc.desc, args.filter_tensor->raw_ptr,
                D.conv_desc.desc,
                m_cudnn_enum,
                args.workspace.raw_ptr,
                args.workspace.size,
                &beta,
                D.dst_desc.desc,
                args.dst_tensor->raw_ptr);
    megdnn_assert(status == CUDNN_STATUS_SUCCESS,
            "conv fwd failed: %s; info: %s",
            cudnnGetErrorString(status), args.to_string().c_str());
}

void Convolution3DForwardImpl::AlgoPack::fill_cudnn_algos() {
    for (auto&& algo : CudnnAlgoPack::conv3d_fwd_algos()) {
        cudnn.push_back(algo.first);
    }
}

// vim: syntax=cpp.doxygen
