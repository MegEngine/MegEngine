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

WorkspaceBundle ConvBiasForwardImpl::AlgoCUDNNConv::get_workspace_bundle(
        void* ptr, const SizeArgs& args) const {
    auto dst_layout = *args.dst_layout;
    SmallVector<size_t> sizes;
    if (dst_layout.dtype.enumv() != args.bias_layout->dtype.enumv()) {
        dst_layout.dtype = DType();
        args.opr->check_or_deduce_dtype_fwd(
                args.src_layout->dtype, args.filter_layout->dtype, dst_layout.dtype);
        sizes.push_back(dst_layout.span().dist_byte());
    }

    if (args.z_layout->ndim > 0 &&
        args.z_layout->dtype.enumv() != args.bias_layout->dtype.enumv()) {
        auto z_layout = *args.z_layout;
        z_layout.dtype = DType();
        args.opr->check_or_deduce_dtype_fwd(
                args.src_layout->dtype, args.filter_layout->dtype, z_layout.dtype);
        sizes.push_back(z_layout.span().dist_byte());
    }

    SizeArgs conv_args = args;
    conv_args.dst_layout = &dst_layout;

    CUDNNForwardDescs D;
    conv_args.init_conv_desc(D);

    size_t conv_workspace_size;
    auto status = cudnnGetConvolutionForwardWorkspaceSize(
            conv_args.handle->cudnn_handle(), D.src_desc.desc, D.filter_desc.desc,
            D.conv_desc.conv_desc, D.dst_desc.desc, m_cudnn_enum, &conv_workspace_size);
    megdnn_assert(
            status == CUDNN_STATUS_SUCCESS,
            "conv fwd get workspace failed: %s; info: %s", cudnnGetErrorString(status),
            args.to_string().c_str());
    sizes.insert(sizes.begin(), conv_workspace_size);
    return {ptr, std::move(sizes)};
}

size_t ConvBiasForwardImpl::AlgoCUDNNConv::get_workspace_in_bytes(
        const SizeArgs& args) const {
    return get_workspace_bundle(nullptr, args).total_size_in_bytes();
}

void ConvBiasForwardImpl::AlgoCUDNNConv::exec(const ExecArgs& args) const {
    auto bundle = get_workspace_bundle(args.workspace.raw_ptr, args);
    TensorND conv_dst_tensor = *args.dst_tensor;
    if (args.dst_layout->dtype.enumv() != args.bias_layout->dtype.enumv()) {
        conv_dst_tensor = TensorND{bundle.get(1), args.dst_tensor->layout};
        conv_dst_tensor.layout.dtype = DType();
        args.opr->check_or_deduce_dtype_fwd(
                args.src_layout->dtype, args.filter_layout->dtype,
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
                conv_args.src_tensor->raw_ptr(), D.filter_desc.desc,
                conv_args.filter_tensor->raw_ptr(), D.conv_desc.conv_desc, m_cudnn_enum,
                conv_workspace.raw_ptr, conv_workspace.size, &beta, D.dst_desc.desc,
                conv_args.dst_tensor->raw_ptr());
        megdnn_assert(
                status == CUDNN_STATUS_SUCCESS, "conv fwd failed: %s; info: %s",
                cudnnGetErrorString(status), conv_args.to_string().c_str());
    }

    if (args.z_layout->ndim > 0) {
        auto z_tensor = *args.z_tensor;
        if (args.z_layout->dtype.enumv() != args.bias_layout->dtype.enumv()) {
            z_tensor = TensorND{bundle.get(2), args.z_tensor->layout};
            z_tensor.layout.dtype = DType();
            args.opr->check_or_deduce_dtype_fwd(
                    args.src_layout->dtype, args.filter_layout->dtype,
                    z_tensor.layout.dtype);
            auto typecvt = args.handle->create_operator<TypeCvt>();
            typecvt->exec(*args.z_tensor, z_tensor);
        }
        auto add = args.handle->create_operator<ElemwiseForward>();
        add->param().mode = Elemwise::Param::Mode::ADD;
        add->exec({conv_dst_tensor, z_tensor}, conv_dst_tensor);
    }

    handle_bias_and_nonlinear(
            args.handle, args.nonlinear_mode, &conv_dst_tensor, args.dst_tensor,
            args.bias_tensor);
}

// vim: syntax=cpp.doxygen
