#include "./algo.h"

#include "src/cuda/conv_bias/helper.h"
#include "src/cuda/convolution/helper.h"
#include "src/cuda/cudnn_wrapper.h"
#include "src/cuda/utils.h"

using namespace megdnn;
using namespace cuda;
using namespace convolution;

bool ConvolutionBackwardDataImpl::AlgoCUDNN::is_available(const SizeArgs& args) const {
    if (args.filter_meta.format != Param::Format::NCHW &&
        args.filter_meta.format != Param::Format::NHWC) {
        if (!args.grad_layout->is_contiguous() || !args.diff_layout->is_contiguous()) {
            return false;
        }
    }

    CUDNNBwdDataDescs D;

    TensorLayout bias_layout, z_layout;
    conv_bias::CanonizedFilterMeta meta;
    meta.copy_from(args.filter_meta);
    conv_bias::BiasForwardSizeArgs bias_args{
            args.handle,        args.grad_layout,
            args.filter_layout, &bias_layout,
            &z_layout,          meta,
            args.diff_layout,   param::ConvBias::NonlineMode::IDENTITY,
    };
    if (!conv_bias::is_cudnn_supported(bias_args))
        return false;

    args.init_desc(D);
    size_t workspace_size;
    auto status = cudnnGetConvolutionBackwardDataWorkspaceSize(
            args.handle->cudnn_handle(), D.filter_desc.desc, D.diff_desc.desc,
            D.conv_desc.desc, D.grad_desc.desc, m_cudnn_enum, &workspace_size);
    return status == CUDNN_STATUS_SUCCESS;
}

size_t ConvolutionBackwardDataImpl::AlgoCUDNN::get_workspace_in_bytes(
        const SizeArgs& args) const {
    CUDNNBwdDataDescs D;
    args.init_desc(D);
    size_t workspace_size;
    auto status = cudnnGetConvolutionBackwardDataWorkspaceSize(
            args.handle->cudnn_handle(), D.filter_desc.desc, D.diff_desc.desc,
            D.conv_desc.desc, D.grad_desc.desc, m_cudnn_enum, &workspace_size);
    megdnn_assert(
            status == CUDNN_STATUS_SUCCESS,
            "conv bwd_data get workspace failed: %s; info: %s",
            cudnnGetErrorString(status), args.to_string().c_str());
    return workspace_size;
}

void ConvolutionBackwardDataImpl::AlgoCUDNN::exec(const ExecArgs& args) const {
    CUDNNBwdDataDescs D;
    args.init_desc(D);
    float alpha = 1.0f, beta = 0.0f;
    auto status = cudnnConvolutionBackwardData(
            args.handle->cudnn_handle(), &alpha, D.filter_desc.desc,
            args.filter_tensor->raw_ptr(), D.diff_desc.desc,
            args.diff_tensor->raw_ptr(), D.conv_desc.desc, m_cudnn_enum,
            args.workspace.raw_ptr, args.workspace.size, &beta, D.grad_desc.desc,
            args.grad_tensor->raw_ptr());
    megdnn_assert(
            status == CUDNN_STATUS_SUCCESS, "conv bwd_data failed: %s; info: %s",
            cudnnGetErrorString(status), args.to_string().c_str());
}

void ConvolutionBackwardDataImpl::AlgoPack::fill_cudnn_algos() {
    for (auto&& algo : CudnnAlgoPack::conv_bwd_data_algos()) {
        cudnn.push_back(algo.first);
    }
}

// vim: syntax=cpp.doxygen
