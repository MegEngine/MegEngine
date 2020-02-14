/**
 * \file dnn/src/cuda/conv_bias/cudnn_conv.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/cuda/conv_bias/algo.h"
#include "src/cuda/cudnn_wrapper.h"
#include "src/cuda/utils.h"
#include "src/common/conv_bias.h"

using namespace megdnn;
using namespace cuda;
using namespace conv_bias;

bool ConvBiasForwardImpl::AlgoCUDNNConv::is_available(
        const SizeArgs& args) const {
    if (args.z_layout->ndim > 0)
        return false;

    auto dst_layout = *args.dst_layout;
    if (dst_layout.dtype.enumv() != args.bias_layout->dtype.enumv()) {
        dst_layout.dtype = DType();
        args.opr->check_or_deduce_dtype_fwd(args.src_layout->dtype,
                                            args.filter_layout->dtype,
                                            dst_layout.dtype);
    }
    SizeArgs conv_args = args;
    conv_args.dst_layout = &dst_layout;

    if (!is_cudnn_supported(conv_args))
        return false;
    CUDNNForwardDescs D;
    conv_args.init_conv_desc(D);

    size_t workspace_size;
    auto status = cudnnGetConvolutionForwardWorkspaceSize(
            conv_args.handle->cudnn_handle(), D.src_desc.desc,
            D.filter_desc.desc, D.conv_desc.conv_desc, D.dst_desc.desc,
            m_cudnn_enum, &workspace_size);
    return status == CUDNN_STATUS_SUCCESS;
}

WorkspaceBundle ConvBiasForwardImpl::AlgoCUDNNConv::get_workspace_bundle(
        void* ptr, const SizeArgs& args) const {
    auto dst_layout = *args.dst_layout;
    SmallVector<size_t> sizes;
    if (dst_layout.dtype.enumv() != args.bias_layout->dtype.enumv()) {
        dst_layout.dtype = DType();
        args.opr->check_or_deduce_dtype_fwd(args.src_layout->dtype,
                                            args.filter_layout->dtype,
                                            dst_layout.dtype);
        sizes.push_back(dst_layout.span().dist_byte());
    }

    SizeArgs conv_args = args;
    conv_args.dst_layout = &dst_layout;

    CUDNNForwardDescs D;
    conv_args.init_conv_desc(D);

    size_t conv_workspace_size;
    auto status = cudnnGetConvolutionForwardWorkspaceSize(
            conv_args.handle->cudnn_handle(), D.src_desc.desc,
            D.filter_desc.desc, D.conv_desc.conv_desc, D.dst_desc.desc,
            m_cudnn_enum, &conv_workspace_size);
    megdnn_assert(status == CUDNN_STATUS_SUCCESS,
                  "conv fwd get workspace failed: %s; info: %s",
                  cudnnGetErrorString(status), args.to_string().c_str());
    sizes.insert(sizes.begin(), conv_workspace_size);
    return {ptr, std::move(sizes)};
}

size_t ConvBiasForwardImpl::AlgoCUDNNConv::get_workspace_in_bytes(
        const SizeArgs& args) const {
    return get_workspace_bundle(nullptr, args).total_size_in_bytes();
}

void ConvBiasForwardImpl::AlgoCUDNNConv::exec(const ExecArgs& args) const {
    auto bundle = get_workspace_bundle(args.workspace.raw_ptr, args);
    auto conv_dst_tensor = *args.dst_tensor;
    if (args.dst_layout->dtype.enumv() != args.bias_layout->dtype.enumv()) {
        conv_dst_tensor.raw_ptr = bundle.get(1);
        conv_dst_tensor.layout.dtype = DType();
        args.opr->check_or_deduce_dtype_fwd(args.src_layout->dtype,
                                            args.filter_layout->dtype,
                                            conv_dst_tensor.layout.dtype);
    }

    ExecArgs conv_args = args;
    conv_args.dst_tensor = &conv_dst_tensor;
    conv_args.dst_layout = &conv_dst_tensor.layout;

    {
        CUDNNForwardDescs D;
        conv_args.init_conv_desc(D);
        auto conv_workspace = bundle.get_workspace(0);
        float alpha = 1.0f, beta = 0.0f;
        auto status = cudnnConvolutionForward(
                conv_args.handle->cudnn_handle(), &alpha, D.src_desc.desc,
                conv_args.src_tensor->raw_ptr, D.filter_desc.desc,
                conv_args.filter_tensor->raw_ptr, D.conv_desc.conv_desc,
                m_cudnn_enum, conv_workspace.raw_ptr, conv_workspace.size,
                &beta, D.dst_desc.desc, conv_args.dst_tensor->raw_ptr);
        megdnn_assert(status == CUDNN_STATUS_SUCCESS,
                      "conv fwd failed: %s; info: %s", cudnnGetErrorString(status),
                      conv_args.to_string().c_str());
    }

    handle_bias_and_nonlinear(args.handle, args.nonlinear_mode,
                              &conv_dst_tensor, args.dst_tensor,
                              args.bias_tensor);
}

// vim: syntax=cpp.doxygen
