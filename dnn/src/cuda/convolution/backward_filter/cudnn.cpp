/**
 * \file dnn/src/cuda/convolution/backward_filter/cudnn.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./algo.h"

#include "src/cuda/utils.h"
#include "src/cuda/cudnn_wrapper.h"
#include "src/cuda/convolution/helper.h"
#include "src/cuda/conv_bias/helper.h"

using namespace megdnn;
using namespace cuda;
using namespace convolution;

bool ConvolutionBackwardFilterImpl::AlgoCUDNN::is_available(
        const SizeArgs &args) const {
    if (args.grad_filter_meta.format != Param::Format::NCHW &&
        args.grad_filter_meta.format != Param::Format::NHWC) {
        if (!args.grad_layout->is_contiguous() ||
            !args.diff_layout->is_contiguous()) {
            return false;
        }
    }
    auto& cudnn = args.handle->cudnn();
    CUDNNBwdFilterDescs D;

    TensorLayout bias_layout, z_layout;
    conv_bias::CanonizedFilterMeta meta;
    meta.copy_from(args.grad_filter_meta);
    conv_bias::BiasForwardSizeArgs bias_args{args.handle,
        args.src_layout, args.grad_layout, &bias_layout,
        &z_layout, meta, args.diff_layout, param::ConvBias::NonlineMode::IDENTITY,
    };
    if (!conv_bias::is_cudnn_supported(bias_args))
        return false;

    args.init_desc(D);
    size_t workspace_size;
    auto status = cudnn.GetConvolutionBackwardFilterWorkspaceSize(
            args.handle->cudnn_handle(),
            D.src_desc.desc,
            D.diff_desc.desc,
            D.conv_desc.desc,
            D.grad_desc.desc,
            m_cudnn_enum,
            &workspace_size);
    return status == CUDNN_STATUS_SUCCESS;
}

size_t ConvolutionBackwardFilterImpl::AlgoCUDNN::get_workspace_in_bytes(
        const SizeArgs &args) const {
    auto& cudnn = args.handle->cudnn();
    CUDNNBwdFilterDescs D;
    args.init_desc(D);
    size_t workspace_size;
    auto status = cudnn.GetConvolutionBackwardFilterWorkspaceSize(
            args.handle->cudnn_handle(),
            D.src_desc.desc,
            D.diff_desc.desc,
            D.conv_desc.desc,
            D.grad_desc.desc,
            m_cudnn_enum,
            &workspace_size);
    megdnn_assert(status == CUDNN_STATUS_SUCCESS,
            "conv bwd_filter get workspace failed: %s; info: %s",
            cudnnGetErrorString(status), args.to_string().c_str());
    return workspace_size;
}

void ConvolutionBackwardFilterImpl::AlgoCUDNN::exec(
        const ExecArgs &args) const {
    CUDNNBwdFilterDescs D;
    args.init_desc(D);
    float alpha = 1.0f, beta = 0.0f;
    auto status = cudnnConvolutionBackwardFilter(args.handle->cudnn_handle(),
                &alpha,
                D.src_desc.desc, args.src_tensor->raw_ptr,
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

void ConvolutionBackwardFilterImpl::AlgoPack::fill_cudnn_algos() {
    for(auto&& algo : CudnnAlgoPack::conv_bwd_flt_algos()) {
        cudnn.push_back(algo.first);
    }
}

// vim: syntax=cpp.doxygen
