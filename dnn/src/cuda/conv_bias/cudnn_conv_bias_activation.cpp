#include "megdnn/oprs/general.h"

#include "./algo.h"

#include "src/common/conv_bias.h"
#include "src/cuda/conv_bias/helper.h"
#include "src/cuda/cudnn_wrapper.h"
#include "src/cuda/utils.h"

using namespace megdnn;
using namespace cuda;
using namespace conv_bias;

bool ConvBiasForwardImpl::AlgoCUDNNConvBiasActivation::is_available(
        const SizeArgs& args) const {
    if (args.filter_meta.format != Param::Format::NCHW &&
        args.filter_meta.format != Param::Format::NHWC) {
        if (!args.src_layout->is_contiguous() || !args.dst_layout->is_contiguous()) {
            return false;
        }
    }
    if (args.src_layout->dtype.enumv() == DTypeEnum::QuantizedS1)
        return false;
    if ((args.src_layout->dtype.enumv() == DTypeEnum::QuantizedS4 ||
         args.src_layout->dtype.enumv() == DTypeEnum::Quantized4Asymm) &&
        args.filter_layout->dtype.enumv() == DTypeEnum::QuantizedS4)
        return false;
    if (args.dst_layout->dtype.enumv() == DTypeEnum::QuantizedS4 ||
        args.dst_layout->dtype.enumv() == DTypeEnum::Quantized4Asymm)
        return false;
    if (args.src_layout->dtype == args.filter_layout->dtype &&
        args.src_layout->dtype == dtype::BFloat16()) {
        return false;
    }

    if (args.bias_layout->ndim == 0 ||
        !check_bias_share_in_channel(*(args.bias_layout), args.opr->param().format)) {
        return false;
    }
    auto&& param = args.opr->param();

#if (CUDNN_MAJOR == 8 && CUDNN_MINOR < 2)
    if (m_cudnn_enum == CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM &&
        (param.format == param::ConvBias::Format::NCHW4
#if (CUDNN_VERSION == 8004)
         || param.format == param::ConvBias::Format::NCHW32
#endif
         ) &&
        args.filter_meta.group * args.filter_meta.ocpg > 256 &&
        args.src_layout->dtype.enumv() == DTypeEnum::QuantizedS8 &&
        args.filter_layout->dtype.enumv() == DTypeEnum::QuantizedS8) {
        return false;
    }
#endif

    // FIXME: cudnn cannot handle the case when the initial value of dst tensor
    // contains nan and beta is zero, because the result of 0.f * nan is still
    // nan
    if (args.src_layout->dtype.enumv() == DTypeEnum::QuantizedS8 &&
        args.dst_layout->dtype.enumv() == DTypeEnum::Float32 &&
        param.format == param::ConvBias::Format::NCHW) {
        return false;
    }

#if CUDNN_VERSION < 7605
    if (args.src_layout->dtype.enumv() == DTypeEnum::Float16 &&
        args.dst_layout->dtype.enumv() == DTypeEnum::Float16) {
        return false;
    }
#endif

#if CUDNN_MAJOR < 8
    if (m_cudnn_enum == CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM &&
        param.format == param::ConvBias::Format::NCHW4_NCHW)
        return false;
#endif
    if (param.format == param::ConvBias::Format::NCHW4_NCHW32 ||
        param.format == param::ConvBias::Format::NCHW32_NCHW4)
        return false;
    if (param.format == param::ConvBias::Format::NCHW &&
        (param.dilate_h != 1 || param.dilate_w != 1) &&
        m_cudnn_enum == CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM) {
        auto&& device_prop = current_device_prop();
        // Dilated convbias in NCHW format produces wrong result on Pascal
        // Architecture, so we disable the algo here.
        if (device_prop.major == 6) {
            return false;
        }
    }

    if (param.format == param::ConvBias::Format::NCHW8 ||
        param.format == param::ConvBias::Format::NCHW64 ||
        param.format == param::ConvBias::Format::CHWN4)
        return false;
    if (param.format == param::ConvBias::Format::NCHW32) {
        auto&& filter_meta = args.filter_meta;
        // NCHW32 layout only support group = 1
        if (filter_meta.group != 1)
            return false;
        // The data type (CUDNN_DATA_INT8x32) can only be used with algo
        // "CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM", for details, see
        // https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html
        if (m_cudnn_enum != CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM)
            return false;
        // check cudnn version
        if (CUDNN_VERSION < 7500)
            return false;
        // sm version
        auto&& device_prop = current_device_prop();
        if (device_prop.major < 7 || (device_prop.major == 7 && device_prop.minor < 5))
            return false;
    }

    CUDNNForwardDescs D;

    if (CUDNN_VERSION < 7401)
        return false;

    args.init_conv_bias_desc(D);
    switch (args.nonlinear_mode) {
        case param::ConvBias::NonlineMode::RELU:
            break;
        case param::ConvBias::NonlineMode::SIGMOID:
            // forbits sigmoid for quantized
            if (args.src_layout->dtype.category() == DTypeCategory::QUANTIZED)
                return false;
            MEGDNN_FALLTHRU;  // XXX: why?
        case param::ConvBias::NonlineMode::IDENTITY:
            if (args.src_layout->dtype.category() == DTypeCategory::QUANTIZED)
                break;
            if (m_cudnn_enum != CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM) {
                // cudnn require algo to
                // CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM
                // when activation if IDENTITY
                return false;
            }
            break;
        case param::ConvBias::NonlineMode::H_SWISH:
            if (args.src_layout->dtype.category() == DTypeCategory::QUANTIZED)
                break;
            return false;
        default:
            megdnn_throw("unsupported NonlineMode");
    }
    size_t workspace_size;
    auto status = cudnnGetConvolutionForwardWorkspaceSize(
            args.handle->cudnn_handle(), D.src_desc.desc, D.filter_desc.desc,
            D.conv_desc.conv_desc, D.dst_desc.desc, m_cudnn_enum, &workspace_size);
    return status == CUDNN_STATUS_SUCCESS;
}

size_t ConvBiasForwardImpl::AlgoCUDNNConvBiasActivation::cudnn_get_workspace_in_bytes(
        const SizeArgs& args) const {
    CUDNNForwardDescs D;

    args.init_conv_bias_desc(D);
    size_t workspace_size;
    auto status = cudnnGetConvolutionForwardWorkspaceSize(
            args.handle->cudnn_handle(), D.src_desc.desc, D.filter_desc.desc,
            D.conv_desc.conv_desc, D.dst_desc.desc, m_cudnn_enum, &workspace_size);
    megdnn_assert(
            status == CUDNN_STATUS_SUCCESS,
            "conv fwd get workspace failed: %s; info: %s", cudnnGetErrorString(status),
            args.to_string().c_str());
    return workspace_size;
}

void ConvBiasForwardImpl::AlgoCUDNNConvBiasActivation::cudnn_execute(
        const ExecArgs& args, const Workspace& workspace, float alpha,
        float beta) const {
#if CUDNN_MAJOR < 7
    megdnn_throw("ConvBias require cudnn 7.0 or higher");
#else
    megdnn_assert(cudnnGetVersion() >= 7401);
    CUDNNForwardDescs D;
    args.init_conv_bias_desc(D);

    cudnnStatus_t status;
    if (args.z_layout->ndim == 0) {
        status = cudnnConvolutionBiasActivationForward(
                args.handle->cudnn_handle(), &alpha, D.src_desc.desc,
                args.src_tensor->raw_ptr(), D.filter_desc.desc,
                args.filter_tensor->raw_ptr(), D.conv_desc.conv_desc, m_cudnn_enum,
                workspace.raw_ptr, workspace.size, &beta, D.dst_desc.desc,
                args.dst_tensor->raw_ptr(), D.bias_desc.desc,
                args.bias_tensor->raw_ptr(), D.conv_desc.act_desc, D.dst_desc.desc,
                args.dst_tensor->raw_ptr());
    } else {
        status = cudnnConvolutionBiasActivationForward(
                args.handle->cudnn_handle(), &alpha, D.src_desc.desc,
                args.src_tensor->raw_ptr(), D.filter_desc.desc,
                args.filter_tensor->raw_ptr(), D.conv_desc.conv_desc, m_cudnn_enum,
                workspace.raw_ptr, workspace.size, &beta, D.z_desc.desc,
                args.z_tensor->raw_ptr(), D.bias_desc.desc, args.bias_tensor->raw_ptr(),
                D.conv_desc.act_desc, D.dst_desc.desc, args.dst_tensor->raw_ptr());
    }

    megdnn_assert(
            status == CUDNN_STATUS_SUCCESS, "conv fwd failed: %s; info: %s, algo %s",
            cudnnGetErrorString(status), args.to_string().c_str(), name());
#endif
}

// vim: syntax=cpp.doxygen
