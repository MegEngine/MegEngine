#include "megdnn/oprs/general.h"

#include "./algo.h"

#include "src/common/conv_bias.h"
#include "src/cuda/cudnn_wrapper_v8.h"
#include "src/cuda/utils.h"

#if CUDNN_VERSION >= 8020
using namespace megdnn;
using namespace cuda;
using namespace conv_bias;

namespace {
TensorLayout canonical_bias_layout(
        const TensorLayout& bias_layout, const param::ConvBias::Format format) {
    int64_t vector_count, vector_dimension;
    std::tie(vector_count, vector_dimension) = get_vector_count_and_dimension(format);
    size_t channel = bias_layout[vector_dimension] * vector_count;
    if (bias_layout.dtype.category() != DTypeCategory::FLOAT) {
        return TensorLayout{{1, channel, 1, 1}, dtype::Float32()};
    }
    return TensorLayout{{1, channel, 1, 1}, bias_layout.dtype};
}
}  // namespace

bool ConvBiasForwardImpl::AlgoCUDNNConvBiasActivationV8::is_available(
        const SizeArgs& args) const {
    auto&& param = args.opr->param();
    if (param.format == param::ConvBias::Format::NCHW4_NCHW32 ||
        param.format == param::ConvBias::Format::NCHW32_NCHW4 ||
        param.format == param::ConvBias::Format::NCHW4_NCHW ||
        param.format == param::ConvBias::Format::NCHW8 ||
        param.format == param::ConvBias::Format::NCHW64 ||
        param.format == param::ConvBias::Format::CHWN4)
        return false;
    if (param.format != Param::Format::NCHW && param.format != Param::Format::NHWC) {
        if (!args.src_layout->is_contiguous() || !args.dst_layout->is_contiguous()) {
            return false;
        }
    }
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
        !check_bias_share_in_channel(*(args.bias_layout), param.format)) {
        return false;
    }

    // FIXME: cudnn cannot handle the case when the initial value of dst tensor
    // contains nan and beta is zero, because the result of 0.f * nan is still
    // nan
    if (args.src_layout->dtype.enumv() == DTypeEnum::QuantizedS8 &&
        args.dst_layout->dtype.enumv() == DTypeEnum::Float32 &&
        param.format == param::ConvBias::Format::NCHW) {
        return false;
    }

    if (param.format == param::ConvBias::Format::NCHW32) {
        // sm version
        auto&& device_prop = current_device_prop();
        if (device_prop.major < 7 || (device_prop.major == 7 && device_prop.minor < 5))
            return false;
    }

    switch (args.nonlinear_mode) {
        case param::ConvBias::NonlineMode::RELU:
        case param::ConvBias::NonlineMode::IDENTITY:
            break;
        case param::ConvBias::NonlineMode::SIGMOID:
            // forbits sigmoid for quantized
            if (args.src_layout->dtype.category() == DTypeCategory::QUANTIZED)
                return false;
            break;
        case param::ConvBias::NonlineMode::H_SWISH:
            if (args.src_layout->dtype.category() == DTypeCategory::QUANTIZED)
                break;
            return false;
        default:
            megdnn_throw("unsupported NonlineMode");
    }

    auto bias_layout =
            canonical_bias_layout(*args.bias_layout, args.opr->param().format);
    auto plan = get_heuristic_plan_from_opr(
            static_cast<const ConvBiasForward*>(args.opr), *args.src_layout,
            *args.dst_layout, *args.filter_layout, bias_layout, *args.z_layout,
            args.filter_meta);
    return plan != nullptr;
}

size_t ConvBiasForwardImpl::AlgoCUDNNConvBiasActivationV8::cudnn_get_workspace_in_bytes(
        const SizeArgs& args) const {
    auto bias_layout =
            canonical_bias_layout(*args.bias_layout, args.opr->param().format);
    auto plan = get_heuristic_plan_from_opr(
            static_cast<const ConvBiasForward*>(args.opr), *args.src_layout,
            *args.dst_layout, *args.filter_layout, bias_layout, *args.z_layout,
            args.filter_meta);
    megdnn_assert(
            plan != nullptr, "algo(%s) cannot find execution from heuristics", name());
    return plan->getWorkspaceSize();
}

void ConvBiasForwardImpl::AlgoCUDNNConvBiasActivationV8::cudnn_execute(
        const ExecArgs& args, const Workspace& workspace, float alpha,
        float beta) const {
    auto&& bias_layout =
            canonical_bias_layout(args.bias_tensor->layout, args.opr->param().format);
    auto plan = get_heuristic_plan_from_opr(
            static_cast<const ConvBiasForward*>(args.opr), args.src_tensor->layout,
            args.dst_tensor->layout, args.filter_tensor->layout, bias_layout,
            args.z_tensor->layout, args.filter_meta);
    megdnn_assert(
            plan != nullptr, "algo(%s) cannot find execution from heuristics", name());
    auto&& handle = cudnn_handle(args.handle);
    TensorND bias_tensor{args.bias_tensor->raw_ptr(), bias_layout};
    run_conv_bias_act_with_plan(
            handle, *plan, *args.src_tensor, *args.dst_tensor, *args.filter_tensor,
            bias_tensor, *args.z_tensor, workspace);
}

#endif

// vim: syntax=cpp.doxygen
