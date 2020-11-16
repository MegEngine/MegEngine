/**
 * \file dnn/src/cuda/convolution/backward_data/cudnn.cpp
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
#include "src/cuda/convolution/helper.h"

using namespace megdnn;
using namespace cuda;
using namespace convolution;

bool ConvolutionBackwardDataImpl::AlgoCUDNN::is_available(
        const SizeArgs &args) const {
    CUDNNBwdDataDescs D;

    if (!is_cudnn_supported(args.as_fwd_args()))
        return false;

#if CUDNN_VERSION >= 7500
    // As in cuda10.0 and cudnn7.5, algo CUDNN_CONVOLUTION_BWD_DATA_ALGO_1 with
    // TensorCore operations produces incorrect result. So we disable
    // this algo. Please remove the following code, when
    // nvidia has fixed this issue.
    // incorrect case:
    // inp={2x8x18x18}, kern={8x8x2x2}, pad_h=pad_w=2, stride_h=stride_w=2,
    // dtype=float16
    if (args.filter_meta.dtype == dtype::Float16()) {
        const char* algo_1 = "CUDNN_CONVOLUTION_BWD_DATA_ALGO_1";
        auto cmp_len = strlen(algo_1);
        if (is_compute_capability_required(7, 0) &&
            strncmp(name(), algo_1, cmp_len) == 0) {
            return false;
        }
    }
#endif

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

size_t ConvolutionBackwardDataImpl::AlgoCUDNN::get_workspace_in_bytes(
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

void ConvolutionBackwardDataImpl::AlgoCUDNN::exec(
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

void ConvolutionBackwardDataImpl::AlgoPack::fill_cudnn_algos() {
    for (auto&& algo : CudnnAlgoPack::conv_bwd_data_algos()) {
        cudnn.push_back(algo.first);
    }
}

// vim: syntax=cpp.doxygen
