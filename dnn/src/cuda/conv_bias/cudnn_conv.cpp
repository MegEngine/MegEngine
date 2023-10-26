#include "src/common/conv_bias.h"
#include "src/cuda/conv_bias/algo.h"
#include "src/cuda/cudnn_wrapper.h"
#include "src/cuda/utils.h"

using namespace megdnn;
using namespace cuda;
using namespace conv_bias;

bool ConvBiasForwardImpl::AlgoCUDNNConv::is_available(const SizeArgs& args) const {
    if (args.filter_meta.format != Param::Format::NCHW &&
        args.filter_meta.format != Param::Format::NHWC) {
        if (!args.src_layout->is_contiguous() || !args.dst_layout->is_contiguous()) {
            return false;
        }
    }

    if (args.dst_layout->dtype.enumv() == DTypeEnum::QuantizedS4 ||
        args.dst_layout->dtype.enumv() == DTypeEnum::Quantized4Asymm) {
        return false;
    }

    // FIXME: cudnn cannot handle the case when the initial value of dst tensor
    // contains nan and beta is zero, because the result of 0.f * nan is still
    // nan
    if (args.src_layout->dtype.enumv() == DTypeEnum::QuantizedS8 &&
        args.dst_layout->dtype.enumv() == DTypeEnum::Float32 &&
        args.opr->param().format == param::ConvBias::Format::NCHW) {
        return false;
    }

    if (args.src_layout->dtype.enumv() == DTypeEnum::Int8 &&
        args.src_layout->dtype.enumv() == DTypeEnum::Int8 &&
        args.dst_layout->dtype.enumv() == DTypeEnum::Int32) {
        return false;
    }

    // In conv_args.init_conv_desc will call cudnnSetTensor4dDescriptorEx(),which can't
    // been supported when total_nr_elems() > 2 ^ 31
    if (args.src_layout->total_nr_elems() > INT_MAX ||
        args.dst_layout->total_nr_elems() > INT_MAX) {
        return false;
    }
    auto dst_layout = *args.dst_layout;
    if (dst_layout.dtype.enumv() != args.bias_layout->dtype.enumv()) {
        dst_layout.dtype = DType();
        args.opr->check_or_deduce_dtype_fwd(
                args.src_layout->dtype, args.filter_layout->dtype, dst_layout.dtype);
    }
    SizeArgs conv_args = args;
    conv_args.dst_layout = &dst_layout;

    if (!is_cudnn_supported(conv_args))
        return false;
    CUDNNForwardDescs D;
    conv_args.init_conv_desc(D);

    size_t workspace_size;
    auto status = cudnnGetConvolutionForwardWorkspaceSize(
            conv_args.handle->cudnn_handle(), D.src_desc.desc, D.filter_desc.desc,
            D.conv_desc.conv_desc, D.dst_desc.desc, m_cudnn_enum, &workspace_size);
    return status == CUDNN_STATUS_SUCCESS;
}

size_t ConvBiasForwardImpl::AlgoCUDNNConv::cudnn_get_workspace_in_bytes(
        const SizeArgs& args) const {
    CUDNNForwardDescs D;
    args.init_conv_desc(D);

    size_t conv_workspace_size;
    cudnn_check(cudnnGetConvolutionForwardWorkspaceSize(
            args.handle->cudnn_handle(), D.src_desc.desc, D.filter_desc.desc,
            D.conv_desc.conv_desc, D.dst_desc.desc, m_cudnn_enum,
            &conv_workspace_size));
    return conv_workspace_size;
}

void ConvBiasForwardImpl::AlgoCUDNNConv::cudnn_execute(
        const ExecArgs& args, const Workspace& workspace) const {
    CUDNNForwardDescs D;
    args.init_conv_desc(D);
    float alpha = 1.0f, beta = 0.0f;
    auto status = cudnnConvolutionForward(
            args.handle->cudnn_handle(), &alpha, D.src_desc.desc,
            args.src_tensor->raw_ptr(), D.filter_desc.desc,
            args.filter_tensor->raw_ptr(), D.conv_desc.conv_desc, m_cudnn_enum,
            workspace.raw_ptr, workspace.size, &beta, D.dst_desc.desc,
            args.dst_tensor->raw_ptr());
    megdnn_assert(
            status == CUDNN_STATUS_SUCCESS, "conv fwd failed: %s; info: %s",
            cudnnGetErrorString(status), args.to_string().c_str());
}

// vim: syntax=cpp.doxygen
