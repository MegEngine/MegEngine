/**
 * \file dnn/src/cuda/conv_bias/helper.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/cuda/conv_bias/helper.h"

#include "src/cuda/utils.h"

namespace megdnn {
namespace cuda {

ConvBiasDesc::ConvBiasDesc() {
    cudnn_check(cudnnCreateActivationDescriptor(&act_desc));
    cudnn_check(cudnnCreateConvolutionDescriptor(&conv_desc));
#if CUDNN_VERSION >= 7000
    cudnn_check(cudnnSetConvolutionMathType(conv_desc, CUDNN_TENSOR_OP_MATH));
#endif
}

ConvBiasDesc::~ConvBiasDesc() {
    cudnn_check(cudnnDestroyConvolutionDescriptor(conv_desc));
    cudnn_check(cudnnDestroyActivationDescriptor(act_desc));
}

void ConvBiasDesc::set_conv_bias(DType data_type, const param::ConvBias& param,
                                 size_t nr_group) {
#if CUDNN_VERSION < 7100
    megdnn_throw(megdnn_mangle(
            "ConvBias(CUDNN_ACTIVATION_IDENTITY) require cudnn 7.1 or higher"));
#else
    cudnnConvolutionMode_t mode;
    using Param = param::ConvBias;
    switch (param.mode) {
        case Param::Mode::CROSS_CORRELATION:
            mode = CUDNN_CROSS_CORRELATION;
            break;
        case Param::Mode::CONVOLUTION:
            mode = CUDNN_CONVOLUTION;
            break;
        default:
            megdnn_throw(megdnn_mangle("conv mode must be conv or xcorr."));
    }
    cudnn_check(cudnnSetConvolutionGroupCount(conv_desc, nr_group));
    cudnnDataType_t compute_type;
    switch (data_type.category()) {
        case DTypeCategory::FLOAT:
            compute_type = CUDNN_DATA_FLOAT;
            break;
        case DTypeCategory::INT:
        case DTypeCategory::QUANTIZED:
            compute_type = CUDNN_DATA_INT32;
            break;
        default:
            megdnn_throw(megdnn_mangle("unspport data type for conv bias"));
    }
    if (data_type.enumv() == DTypeEnum::Float16) {
        auto comp_mode = param.compute_mode;
        compute_type = get_compute_type_fp16(comp_mode);
    }
    cudnn_check(cudnnSetConvolution2dDescriptor(
            conv_desc, param.pad_h, param.pad_w, param.stride_h, param.stride_w,
            param.dilate_h, param.dilate_w, mode, compute_type));

    switch (param.nonlineMode) {
        case Param::NonlineMode::IDENTITY:
        case Param::NonlineMode::SIGMOID:
        case Param::NonlineMode::H_SWISH:
            cudnn_check(cudnnSetActivationDescriptor(
                    act_desc, CUDNN_ACTIVATION_IDENTITY,
                    CUDNN_NOT_PROPAGATE_NAN, 0));
            break;
        case Param::NonlineMode::RELU:
            cudnn_check(cudnnSetActivationDescriptor(
                    act_desc, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN,
                    0));
            break;
        default:
            megdnn_throw(megdnn_mangle("unsupported non linear mode"));
    }
#endif
}

void ConvBiasDesc::set_conv(DType data_type, const param::ConvBias& param,
                            const size_t nr_group) {
    using Param = param::ConvBias;
    cudnnConvolutionMode_t mode;
    switch (param.mode) {
        case Param::Mode::CROSS_CORRELATION:
            mode = CUDNN_CROSS_CORRELATION;
            break;
        case Param::Mode::CONVOLUTION:
            mode = CUDNN_CONVOLUTION;
            break;
        default:
            megdnn_throw(megdnn_mangle("conv mode must be conv or xcorr."));
    }
    cudnnDataType_t compute_type;
    MEGDNN_MARK_USED_VAR(compute_type);
    if (data_type.enumv() == DTypeEnum::Float32) {
        // FLOAT_CONFIG
        compute_type = CUDNN_DATA_FLOAT;
    } else if (data_type.enumv() == DTypeEnum::Float16) {
        auto comp_mode = param.compute_mode;
        compute_type = get_compute_type_fp16(comp_mode);
#if CUDNN_MAJOR >= 7
    } else if (data_type.category() == DTypeCategory::INT ||
               data_type.category() == DTypeCategory::QUANTIZED) {
        compute_type = CUDNN_DATA_INT32;
#endif
    } else {
        megdnn_throw(megdnn_mangle("unspport data type for conv bias"));
    }
#if CUDNN_MAJOR >= 7
    cudnn_check(cudnnSetConvolutionGroupCount(conv_desc, nr_group));
#else
    megdnn_assert(nr_group == 1);
#endif

#if CUDNN_MAJOR >= 6
    cudnn_check(cudnnSetConvolution2dDescriptor(
            conv_desc, param.pad_h, param.pad_w, param.stride_h, param.stride_w,
            param.dilate_h, param.dilate_w, mode, compute_type));
#else
    cudnn_check(cudnnSetConvolution2dDescriptor(
            conv_desc, param.pad_h, param.pad_w, param.stride_h, param.stride_w,
            param.dilate_h, param.dilate_w, mode));
#endif
}

namespace conv_bias {

bool is_cudnn_supported(const BiasForwardSizeArgs& args) {
    if (args.src_layout->dtype == args.filter_layout->dtype &&
        args.src_layout->dtype == dtype::BFloat16()) {
        return false;
    }

    // CUDNN_STATUS_EXECUTION_FAILED on Tegra K1, so disable CUDNN
    // on Tegra K1.
    if (args.handle->is_tegra_k1())
        return false;

    // TODO: We only support NCHW format now. It seems cuDNN provides support
    // for NHWC as well.
    if (args.filter_meta.format == param::Convolution::Format::NCHW4) {
        if (args.dst_layout->dtype.enumv() != DTypeEnum::Int8 &&
            args.dst_layout->dtype.enumv() != DTypeEnum::QuantizedS8) {
            return false;
        }
    } else if (args.filter_meta.format != param::Convolution::Format::NCHW) {
        return false;
    }
    auto& fm = args.filter_meta;
    bool supported = true;
    supported &= (fm.spatial_ndim == 2);
#if CUDNN_VERSION < 7000
    supported &= (fm.group == 1);
#endif
#if CUDNN_VERSION < 7500
    supported &= (fm.dilation[0] == 1 && fm.dilation[1] == 1);
#endif
    return supported;
}

bool check_bias_share_in_channel(const TensorLayout& bias,
                                 const param::ConvBias::Format format) {
    bool share_in_channel = false;
    if (format == param::ConvBias::Format::NCHW ||
        format == param::ConvBias::Format::NCHW4_NCHW) {
        share_in_channel = (bias.ndim == 4 && bias[0] == 1 && bias[2] == 1 &&
                            bias[3] == 1);
    } else if (format == param::ConvBias::Format::NHWC) {
        share_in_channel = (bias.ndim == 4 && bias[0] == 1 && bias[1] == 1 &&
                            bias[2] == 1);
    } else if (format == param::ConvBias::Format::NCHW4 ||
               format == param::ConvBias::Format::NCHW8 ||
               format == param::ConvBias::Format::NCHW32 ||
               format == param::ConvBias::Format::NCHW4_NCHW32 ||
               format == param::ConvBias::Format::NCHW32_NCHW4) {
        share_in_channel = (bias.ndim == 5 && bias[0] == 1 && bias[2] == 1 &&
                            bias[3] == 1);
    } else if (format == param::ConvBias::Format::NHWCD4) {
        share_in_channel = (bias.ndim == 5 && bias[0] == 1 && bias[1] == 1 &&
                            bias[3] == 1);
    } else {
        megdnn_assert(format == param::ConvBias::Format::CHWN4);
        share_in_channel = (bias.ndim == 5 && bias[1] == 1 && bias[2] == 1 &&
                            bias[3] == 1);
    }
    return share_in_channel;
}

WorkspaceBundle matmul_get_workspace_bundle(const BiasForwardSizeArgs& args) {
    auto dtype = args.src_layout->dtype;
    auto&& fm = args.filter_meta;
    megdnn_assert(fm.group == 1);
    auto N = args.src_layout->shape[0];
    auto OC = fm.ocpg, IC = fm.icpg, FH = fm.spatial[0], FW = fm.spatial[1];
    auto OH = args.dst_layout->shape[2], OW = args.dst_layout->shape[3];
    SmallVector<size_t> sizes{dtype.size() * args.dst_layout->total_nr_elems(),
                              dtype.size() * IC * FH * FW * OH * OW * N};
    if (args.filter_meta.should_flip) {
        sizes.push_back(dtype.size() * OC * IC * FH * FW);
    }
    return {nullptr, std::move(sizes)};
}

void flip_filter(const BiasForwardSizeArgs& args, const Workspace& workspace,
                 void*& raw_ptr) {
    auto&& fm = args.filter_meta;
    megdnn_assert(fm.group == 1 && fm.spatial_ndim == 2);
    auto OC = fm.ocpg, IC = fm.icpg, FH = fm.spatial[0], FW = fm.spatial[1];
    auto dtype = fm.dtype;
    megdnn_assert(workspace.size >= dtype.size() * OC * IC * FH * FW);

    TensorND src{raw_ptr, {{OC, IC, FH, FW}, dtype}},
            dst{workspace.raw_ptr + (FH * FW - 1) * dtype.size(), src.layout};
    dst.layout.stride[2] = -dst.layout.stride[2];
    dst.layout.stride[3] = -dst.layout.stride[3];
    args.handle->relayout_opr()->exec(src, dst);
    raw_ptr = workspace.raw_ptr;
}

} // conv_bias

} // cuda
} // megdnn

// vim: syntax=cpp.doxygen
