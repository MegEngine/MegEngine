/**
 * \file dnn/src/cuda/conv_bias/cudnn_conv_bias_activation_base.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megdnn/oprs/general.h"

#include "./algo.h"

#include "src/common/conv_bias.h"
#include "src/cuda/conv_bias/helper.h"
#include "src/cuda/cudnn_wrapper.h"
#include "src/cuda/utils.h"

using namespace megdnn;
using namespace cuda;
using namespace conv_bias;

size_t ConvBiasForwardImpl::AlgoCUDNNConvBiasActivationBase::get_workspace_in_bytes(
        const SizeArgs& args) const {
    auto workspace_size = cudnn_get_workspace_in_bytes(args);

    auto&& param = args.opr->param();
    if (args.preprocessed_filter == nullptr) {
        if (args.bias_layout && args.bias_layout->dtype != dtype::Float32() &&
            args.src_layout->dtype.category() != DTypeCategory::FLOAT) {
            // cudnn require bias to be float when executing CONFIG_INT
            // convert bias to float if bias is not float at first
            workspace_size += sizeof(float) * args.bias_layout->span().dist_elem();
        }
        if (param.format == param::ConvBias::Format::NCHW32) {
            workspace_size += args.filter_layout->span().dist_byte() +
                              args.bias_layout->span().dist_byte();
        }
    }
    return workspace_size;
}

void ConvBiasForwardImpl::AlgoCUDNNConvBiasActivationBase::exec(
        const ExecArgs& args) const {
    float alpha, beta;
    std::tie(alpha, beta) = cudnn_get_conv_bias_act_scale_param(
            args.src_tensor->layout, args.dst_tensor->layout,
            args.filter_tensor->layout, args.bias_tensor->layout,
            args.z_tensor->layout);

    auto workspace_ptr = args.workspace.raw_ptr;
    auto workspace_size = args.workspace.size;
    auto bias_ptr = args.bias_tensor->raw_ptr();
    TensorND filter_tensor;
    TensorND bias_tensor;

    auto&& param = args.opr->param();
    if (args.preprocessed_filter != nullptr) {
        bias_tensor = TensorND{
                args.bias_tensor->layout,
                args.preprocessed_filter->tensors[0].raw_ptr()};
        if (param.format == Param::Format::NCHW32) {
            megdnn_assert(args.preprocessed_filter->tensors.size() == 2);
            filter_tensor = TensorND{
                    args.filter_tensor->layout,
                    args.preprocessed_filter->tensors[1].raw_ptr()};
        } else {
            filter_tensor = *args.filter_tensor;
        }
    } else {
        if (args.bias_layout && args.bias_layout->dtype != dtype::Float32() &&
            args.src_layout->dtype.category() != DTypeCategory::FLOAT) {
            auto cvt = args.handle->create_operator<TypeCvt>();
            auto float_bias_layout = *args.bias_layout;
            auto converted_bias_layout = *args.bias_layout;
            converted_bias_layout.dtype = dtype::QuantizedS32(alpha);
            float_bias_layout.dtype = dtype::Float32();
            auto bias_size_in_bytes = float_bias_layout.span().dist_byte();
            megdnn_assert(args.workspace.size >= bias_size_in_bytes);
            cvt->exec(
                    {args.bias_tensor->raw_ptr(), converted_bias_layout},
                    TensorND{workspace_ptr, float_bias_layout});

            bias_ptr = workspace_ptr;
            workspace_ptr += bias_size_in_bytes;
            workspace_size -= bias_size_in_bytes;
        }
        if (param.format == Param::Format::NCHW32) {
            size_t reorder_workspace_size =
                    args.filter_tensor->layout.span().dist_byte() +
                    args.bias_tensor->layout.span().dist_byte();
            auto reorder_filter_ptr = workspace_ptr;
            auto reorder_bias_ptr =
                    workspace_ptr + args.filter_tensor->layout.span().dist_byte();
            cudnn_reorder_filer_and_bias_nchw32(
                    cudnn_handle(args.opr->handle()), args.filter_tensor->raw_ptr(),
                    args.filter_meta, bias_ptr, reorder_filter_ptr, reorder_bias_ptr);
            filter_tensor = TensorND(args.filter_tensor->layout, reorder_filter_ptr);
            bias_ptr = reorder_bias_ptr;
            workspace_ptr += reorder_workspace_size;
            workspace_size -= reorder_workspace_size;
        } else {
            filter_tensor = *args.filter_tensor;
        }
    }

    bias_tensor = TensorND{args.bias_tensor->layout, bias_ptr};
    ExecArgs exec_args{
            const_cast<ConvBiasForwardImpl*>(args.opr),
            *args.src_tensor,
            filter_tensor,
            bias_tensor,
            *args.z_tensor,
            *args.dst_tensor,
            args.workspace};
    Workspace cudnn_workspace{workspace_ptr, workspace_size};
    cudnn_execute(exec_args, cudnn_workspace, alpha, beta);

    // Noline
    switch (args.nonlinear_mode) {
        case param::ConvBias::NonlineMode::RELU:
            break;
        case param::ConvBias::NonlineMode::SIGMOID: {
            megdnn_assert(
                    args.dst_layout->dtype.category() != DTypeCategory::QUANTIZED);
            auto&& elem_opr = args.handle->create_operator<ElemwiseForward>();
            elem_opr->param().mode = Elemwise::Param::Mode::SIGMOID;
            elem_opr->exec({*(args.dst_tensor)}, *(args.dst_tensor));
            break;
        }
        case param::ConvBias::NonlineMode::IDENTITY:
            break;
        case param::ConvBias::NonlineMode::H_SWISH: {
            megdnn_assert(
                    args.dst_layout->dtype.category() == DTypeCategory::QUANTIZED ||
                    (args.dst_layout->dtype.category() == DTypeCategory::FLOAT &&
                     args.opr->param().format == param::ConvBias::Format::NCHW4_NCHW));
            if (args.dst_layout->dtype.category() == DTypeCategory::QUANTIZED) {
                auto&& elem_opr = args.handle->create_operator<ElemwiseMultiType>();
                elem_opr->param().mode = ElemwiseMultiType::Param::Mode::QH_SWISH;
                elem_opr->exec({*(args.dst_tensor)}, *(args.dst_tensor));
            } else {
                auto&& elem_opr = args.handle->create_operator<ElemwiseForward>();
                elem_opr->param().mode = ElemwiseForward::Param::Mode::H_SWISH;
                elem_opr->exec({*(args.dst_tensor)}, *(args.dst_tensor));
            }
            break;
        }
        default:
            megdnn_throw("unsupported NonlineMode");
    }
}

size_t ConvBiasForwardImpl::AlgoCUDNNConvBiasActivationBase::
        get_preprocess_workspace_in_bytes(const SizeArgs& args) const {
    auto&& param = args.opr->param();
    if (param.format == Param::Format::NCHW32) {
        return args.bias_layout->span().dist_byte();
    }
    return 0_z;
}

SmallVector<TensorLayout> ConvBiasForwardImpl::AlgoCUDNNConvBiasActivationBase::
        deduce_preprocessed_filter_layout(const SizeArgs& args) const {
    auto&& param = args.opr->param();
    if (param.format == Param::Format::NCHW32) {
        return {args.bias_layout->collapse_contiguous(),
                args.filter_layout->collapse_contiguous()};
    } else {
        return {args.bias_layout->collapse_contiguous()};
    }
}

void ConvBiasForwardImpl::AlgoCUDNNConvBiasActivationBase::exec_preprocess(
        const ExecArgs& args) const {
    float alpha, beta;
    std::tie(alpha, beta) = cudnn_get_conv_bias_act_scale_param(
            args.src_tensor->layout, args.dst_tensor->layout,
            args.filter_tensor->layout, args.bias_tensor->layout,
            args.z_tensor->layout);
    MEGDNN_MARK_USED_VAR(beta);

    auto workspace_ptr = args.workspace.raw_ptr;
    auto workspace_size = args.workspace.size;
    auto bias_ptr = workspace_size > 0 ? workspace_ptr
                                       : args.preprocessed_filter->tensors[0].raw_ptr();
    if (args.bias_layout && args.bias_layout->dtype != dtype::Float32() &&
        args.src_layout->dtype.category() != DTypeCategory::FLOAT) {
        auto cvt = args.handle->create_operator<TypeCvt>();
        auto float_bias_layout = *args.bias_layout;
        auto converted_bias_layout = *args.bias_layout;
        converted_bias_layout.dtype = dtype::QuantizedS32(alpha);
        float_bias_layout.dtype = dtype::Float32();

        cvt->exec(
                {args.bias_tensor->raw_ptr(), converted_bias_layout},
                TensorND{bias_ptr, float_bias_layout});
    }
    if (args.opr->param().format == Param::Format::NCHW32) {
        auto reorder_filter_ptr = args.preprocessed_filter->tensors[1].raw_ptr();
        auto reorder_bias_ptr = args.preprocessed_filter->tensors[0].raw_ptr();
        cudnn_reorder_filer_and_bias_nchw32(
                cudnn_handle(args.opr->handle()), args.filter_tensor->raw_ptr(),
                args.filter_meta, bias_ptr, reorder_filter_ptr, reorder_bias_ptr);
    }
}

// vim: syntax=cpp.doxygen
