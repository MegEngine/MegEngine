/**
 * \file dnn/src/cuda/conv_bias/cudnn_conv_bias_activation.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megdnn/oprs/general.h"

#include "./algo.h"

#include "src/cuda/conv_bias/helper.h"
#include "src/cuda/cudnn_wrapper.h"
#include "src/cuda/utils.h"

using namespace megdnn;
using namespace cuda;
using namespace conv_bias;

bool ConvBiasForwardImpl::AlgoCUDNNConvBiasActivation::is_available(
        const SizeArgs& args) const {
    if (args.src_layout->dtype == args.filter_layout->dtype &&
        args.src_layout->dtype == dtype::BFloat16()) {
        return false;
    }
    if (args.bias_layout->ndim == 0 ||
        args.bias_layout->eq_shape(*args.dst_layout))
        return false;
    auto&& param = args.opr->param();
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
        if (device_prop.major < 7 ||
            (device_prop.major == 7 && device_prop.minor < 5))
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
            MEGDNN_FALLTHRU  // XXX: why?
        case param::ConvBias::NonlineMode::IDENTITY:
            if (args.src_layout->dtype.category() == DTypeCategory::QUANTIZED)
                break;
            if (m_cudnn_enum !=
                    CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM) {
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
            megdnn_throw(megdnn_mangle("unsupported NonlineMode"));
    }
    size_t workspace_size;
    auto status = cudnnGetConvolutionForwardWorkspaceSize(
            args.handle->cudnn_handle(), D.src_desc.desc, D.filter_desc.desc,
            D.conv_desc.conv_desc, D.dst_desc.desc, m_cudnn_enum,
            &workspace_size);
    return status == CUDNN_STATUS_SUCCESS;
}

size_t ConvBiasForwardImpl::AlgoCUDNNConvBiasActivation::get_workspace_in_bytes(
        const SizeArgs& args) const {
    CUDNNForwardDescs D;

    args.init_conv_bias_desc(D);
    size_t workspace_size;
    auto status = cudnnGetConvolutionForwardWorkspaceSize(
            args.handle->cudnn_handle(), D.src_desc.desc, D.filter_desc.desc,
            D.conv_desc.conv_desc, D.dst_desc.desc, m_cudnn_enum,
            &workspace_size);
    megdnn_assert(status == CUDNN_STATUS_SUCCESS,
                  "conv fwd get workspace failed: %s; info: %s",
                  cudnnGetErrorString(status), args.to_string().c_str());
    if (args.bias_layout && args.bias_layout->dtype != dtype::Float32() &&
        args.src_layout->dtype.category() != DTypeCategory::FLOAT) {
        // cudnn require bias to be float when executing CONFIG_INT
        // convert bias to float if bias is not float at first
        workspace_size += sizeof(float) * args.bias_layout->span().dist_elem();
    }
    return workspace_size;
}

void ConvBiasForwardImpl::AlgoCUDNNConvBiasActivation::exec(
        const ExecArgs& args) const {
#if CUDNN_MAJOR < 7
    megdnn_throw(megdnn_mangle("ConvBias require cudnn 7.0 or higher"));
#else
    megdnn_assert(cudnnGetVersion() >= 7401);
    CUDNNForwardDescs D;
    args.init_conv_bias_desc(D);
    float alpha = 1.0f, beta = 0.0f;
    if (args.z_layout->ndim > 0)
        beta = 1.0f;

    auto get_scale = [](const DType& dtype) -> float {
        megdnn_assert(dtype.category() == DTypeCategory::QUANTIZED);
        switch (dtype.enumv()) {
#define cb(_dt)                  \
    case DTypeTrait<_dt>::enumv: \
        return dtype.param<_dt>().scale;
            MEGDNN_FOREACH_QUANTIZED_DTYPE(cb)
#undef cb
            default:
                megdnn_assert_internal(0);
        }
    };

    auto src_dtype = args.src_layout->dtype,
         filter_dtype = args.filter_layout->dtype,
         dst_dtype = args.dst_layout->dtype;
    megdnn_assert(
            (src_dtype.category() == dst_dtype.category()) ||
            (args.opr->param().format == param::ConvBias::Format::NCHW4_NCHW &&
             src_dtype.enumv() == DTypeEnum::QuantizedS8 &&
             dst_dtype.enumv() == DTypeEnum::Float32));
    megdnn_assert(src_dtype.category() == filter_dtype.category());

    if (args.src_layout->dtype.category() == DTypeCategory::QUANTIZED) {
        auto expected_bias_scale = get_scale(args.src_layout->dtype) *
                                   get_scale(args.filter_layout->dtype);
        alpha = expected_bias_scale;
        if (args.dst_layout->dtype.category() == DTypeCategory::QUANTIZED)
            alpha /= get_scale(args.dst_layout->dtype);
        if (args.z_layout->ndim > 0 &&
            args.z_layout->dtype.category() == DTypeCategory::QUANTIZED) {
            beta = get_scale(args.z_layout->dtype) /
                   get_scale(args.dst_layout->dtype);
        }
        if (args.bias_layout->dtype.category() == DTypeCategory::QUANTIZED) {
            megdnn_assert(fabs(expected_bias_scale -
                               get_scale(args.bias_layout->dtype)) < 1e-4);
        }
    }

    auto workspace_ptr = args.workspace.raw_ptr;
    auto workspace_size = args.workspace.size;
    auto bias_ptr = args.bias_tensor->raw_ptr;
    if (args.bias_layout && args.bias_layout->dtype != dtype::Float32() &&
        args.src_layout->dtype.category() != DTypeCategory::FLOAT) {
        auto cvt = args.handle->create_operator<TypeCvt>();
        auto float_bias_layout = *args.bias_layout;
        auto converted_bias_layout = *args.bias_layout;
        converted_bias_layout.dtype = dtype::QuantizedS32(alpha);
        float_bias_layout.dtype = dtype::Float32();
        auto bias_size_in_bytes = float_bias_layout.span().dist_byte();
        megdnn_assert(args.workspace.size >= bias_size_in_bytes);
        cvt->exec({args.bias_tensor->raw_ptr, converted_bias_layout},
                  TensorND{workspace_ptr, float_bias_layout});

        bias_ptr = workspace_ptr;
        workspace_ptr += bias_size_in_bytes;
        workspace_size -= bias_size_in_bytes;
    }

    cudnnStatus_t status;
    if (args.z_layout->ndim == 0) {
        status = cudnnConvolutionBiasActivationForward(
                args.handle->cudnn_handle(), &alpha, D.src_desc.desc,
                args.src_tensor->raw_ptr, D.filter_desc.desc,
                args.filter_tensor->raw_ptr, D.conv_desc.conv_desc,
                m_cudnn_enum, workspace_ptr, workspace_size, &beta,
                D.dst_desc.desc, args.dst_tensor->raw_ptr, D.bias_desc.desc,
                bias_ptr, D.conv_desc.act_desc, D.dst_desc.desc,
                args.dst_tensor->raw_ptr);
    } else {
        status = cudnnConvolutionBiasActivationForward(
                args.handle->cudnn_handle(), &alpha, D.src_desc.desc,
                args.src_tensor->raw_ptr, D.filter_desc.desc,
                args.filter_tensor->raw_ptr, D.conv_desc.conv_desc,
                m_cudnn_enum, workspace_ptr, workspace_size, &beta,
                D.z_desc.desc, args.z_tensor->raw_ptr, D.bias_desc.desc,
                bias_ptr, D.conv_desc.act_desc, D.dst_desc.desc,
                args.dst_tensor->raw_ptr);
    }

    megdnn_assert(status == CUDNN_STATUS_SUCCESS,
                  "conv fwd failed: %s; info: %s, algo %s",
                  cudnnGetErrorString(status), args.to_string().c_str(),
                  name());
    // Noline
    switch (args.nonlinear_mode) {
        case param::ConvBias::NonlineMode::RELU:
            break;
        case param::ConvBias::NonlineMode::SIGMOID: {
            megdnn_assert(args.dst_layout->dtype.category() !=
                          DTypeCategory::QUANTIZED);
            auto&& elem_opr = args.handle->create_operator<ElemwiseForward>();
            elem_opr->param().mode = Elemwise::Param::Mode::SIGMOID;
            elem_opr->exec({*(args.dst_tensor)}, *(args.dst_tensor));
            break;
        }
        case param::ConvBias::NonlineMode::IDENTITY:
            break;
        case param::ConvBias::NonlineMode::H_SWISH: {
            megdnn_assert(args.dst_layout->dtype.category() ==
                                  DTypeCategory::QUANTIZED ||
                          (args.dst_layout->dtype.category() ==
                                   DTypeCategory::FLOAT &&
                           args.opr->param().format ==
                                   param::ConvBias::Format::NCHW4_NCHW));
            if (args.dst_layout->dtype.category() == DTypeCategory::QUANTIZED) {
                auto&& elem_opr =
                        args.handle->create_operator<ElemwiseMultiType>();
                elem_opr->param().mode =
                        ElemwiseMultiType::Param::Mode::QH_SWISH;
                elem_opr->exec({*(args.dst_tensor)}, *(args.dst_tensor));
            } else {
                auto&& elem_opr =
                        args.handle->create_operator<ElemwiseForward>();
                elem_opr->param().mode = ElemwiseForward::Param::Mode::H_SWISH;
                elem_opr->exec({*(args.dst_tensor)}, *(args.dst_tensor));
            }
            break;
        }
        default:
            megdnn_throw(megdnn_mangle("unsupported NonlineMode"));
    }
#endif
}

// vim: syntax=cpp.doxygen
