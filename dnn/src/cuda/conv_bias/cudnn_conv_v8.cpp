/**
 * \file dnn/src/cuda/conv_bias/cudnn_conv_v8.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/common/conv_bias.h"
#include "src/cuda/conv_bias/algo.h"
#include "src/cuda/cudnn_wrapper_v8.h"
#include "src/cuda/utils.h"

#if CUDNN_VERSION >= 8004
using namespace megdnn;
using namespace cuda;
using namespace conv_bias;

bool ConvBiasForwardImpl::AlgoCUDNNConvV8::is_available(const SizeArgs& args) const {
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

    auto conv_opr = args.handle->create_operator<ConvolutionForward>();
    conv_opr->param() = get_param_convolution(args);
    ConvolutionForward::CanonizedFilterMeta fm;
    fm.copy_from(args.filter_meta);
    auto plan = get_heuristic_plan_from_opr(
            conv_opr.get(), *conv_args.src_layout, *conv_args.dst_layout,
            *conv_args.filter_layout, {}, {}, fm);
    return plan != nullptr;
}

size_t ConvBiasForwardImpl::AlgoCUDNNConvV8::cudnn_get_workspace_in_bytes(
        const SizeArgs& args) const {
    auto conv_opr = args.handle->create_operator<ConvolutionForward>();
    conv_opr->param() = get_param_convolution(args);
    ConvolutionForward::CanonizedFilterMeta fm;
    fm.copy_from(args.filter_meta);
    auto plan = get_heuristic_plan_from_opr(
            conv_opr.get(), *args.src_layout, *args.dst_layout, *args.filter_layout, {},
            {}, fm);
    megdnn_assert(
            plan != nullptr, "algo(%s) cannot find execution from heuristics", name());
    return plan->getWorkspaceSize();
}

void ConvBiasForwardImpl::AlgoCUDNNConvV8::cudnn_execute(
        const ExecArgs& args, const Workspace& workspace) const {
    auto conv_opr = args.handle->create_operator<ConvolutionForward>();
    conv_opr->param() = get_param_convolution(args);
    ConvolutionForward::CanonizedFilterMeta fm;
    fm.copy_from(args.filter_meta);
    auto plan = get_heuristic_plan_from_opr(
            conv_opr.get(), args.src_tensor->layout, args.dst_tensor->layout,
            args.filter_tensor->layout, {}, {}, fm);
    megdnn_assert(
            plan != nullptr, "algo(%s) cannot find execution from heuristics", name());
    auto&& handle = cudnn_handle(args.handle);
    run_single_conv_with_plan(
            handle, *plan, *args.src_tensor, *args.dst_tensor, *args.filter_tensor,
            workspace);
}
#endif

// vim: syntax=cpp.doxygen
