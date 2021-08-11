/**
 * \file dnn/src/cuda/conv_bias/cutlass_convolution_base.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/cuda/conv_bias/algo.h"
#include "src/cuda/cutlass/singleton.h"

namespace megdnn {
namespace cuda {

using namespace cutlass::library;
using namespace cutlass::epilogue;

ConvBiasForwardImpl::AlgoCutlassConvolutionBase::AlgoParam::AlgoParam(
        int threadblock_m_, int threadblock_n_, int threadblock_k_, int warp_m_,
        int warp_n_, int warp_k_, int instruction_m_, int instruction_n_,
        int instruction_k_, int stage_, int access_size_)
        : threadblock_m(threadblock_m_),
          threadblock_n(threadblock_n_),
          threadblock_k(threadblock_k_),
          warp_m(warp_m_),
          warp_n(warp_n_),
          warp_k(warp_k_),
          instruction_m(instruction_m_),
          instruction_n(instruction_m_),
          instruction_k(instruction_k_),
          stage(stage_),
          access_size(access_size_) {}

std::string
ConvBiasForwardImpl::AlgoCutlassConvolutionBase::AlgoParam::to_string() const {
    /// default algorithm
    if (threadblock_m == 128 && threadblock_n == 128 && threadblock_k == 32 &&
        warp_m == 32 && warp_n == 64 && warp_k == 32 && stage == 2) {
        return "";
    }
    return ssprintf("_%dX%dX%d_%dX%dX%d_%dstage", threadblock_m, threadblock_n,
                    threadblock_k, warp_m, warp_n, warp_k, stage);
}

namespace {

using Base = ConvBiasForwardImpl::AlgoCutlassConvolutionBase;

cutlass::conv::Operator convert_conv_op(Base::ConvOperator conv_op) {
    switch (conv_op) {
        case Base::ConvOperator::kFprop:
            return cutlass::conv::Operator::kFprop;
        case Base::ConvOperator::kDgrad:
            return cutlass::conv::Operator::kDgrad;
        case Base::ConvOperator::kWgrad:
            return cutlass::conv::Operator::kWgrad;
        default:
            megdnn_assert(0, "invalid conv op");
    }
}

cutlass::conv::ConvType convert_conv_type(Base::ConvType conv_type) {
    switch (conv_type) {
        case Base::ConvType::kConvolution:
            return cutlass::conv::ConvType::kConvolution;
        case Base::ConvType::kBatchConvolution:
            return cutlass::conv::ConvType::kBatchConvolution;
        case Base::ConvType::kLocal:
            return cutlass::conv::ConvType::kLocal;
        case Base::ConvType::kLocalShare:
            return cutlass::conv::ConvType::kLocalShare;
        default:
            megdnn_assert(0, "invalid conv type");
    }
}

NumericTypeID convert_dtype(DTypeEnum dtype) {
    switch (dtype) {
        case DTypeEnum::Float32:
            return NumericTypeID::kF32;
        case DTypeEnum::Float16:
            return NumericTypeID::kF16;
        case DTypeEnum::Int8:
            return NumericTypeID::kS8;
        case DTypeEnum::QuantizedS32:
            return NumericTypeID::kS32;
        case DTypeEnum::QuantizedS8:
            return NumericTypeID::kS8;
        case DTypeEnum::QuantizedS4:
            return NumericTypeID::kS4;
        case DTypeEnum::Quantized4Asymm:
            return NumericTypeID::kU4;
        default:
            megdnn_assert(0, "invalid dtype");
    }
}

struct LayoutPack {
    LayoutTypeID src;
    LayoutTypeID filter;
    LayoutTypeID dst;
    LayoutTypeID bias;
};

LayoutPack get_layout_pack(const param::ConvBias::Format format,
                           int access_type) {
    using Format = param::ConvBias::Format;

    switch (format) {
        case Format::NCHW4:
            return {LayoutTypeID::kTensorNC4HW4, LayoutTypeID::kTensorC4RSK4,
                    LayoutTypeID::kTensorNC4HW4, LayoutTypeID::kTensorNC4HW4};
        case Format::NCHW4_NCHW:
            return {LayoutTypeID::kTensorNC4HW4, LayoutTypeID::kTensorC4RSK4,
                    LayoutTypeID::kTensorNCHW, LayoutTypeID::kTensorNCHW};
        case Format::NCHW4_NHWC:
            return {LayoutTypeID::kTensorNC4HW4, LayoutTypeID::kTensorC4RSK4,
                    LayoutTypeID::kTensorNHWC, LayoutTypeID::kTensorNHWC};
        case Format::NCHW4_NCHW32:
            return {LayoutTypeID::kTensorNC4HW4, LayoutTypeID::kTensorC4RSK4,
                    LayoutTypeID::kTensorNC32HW32,
                    LayoutTypeID::kTensorNC32HW32};
        case Format::NCHW32:
            return {LayoutTypeID::kTensorNC32HW32,
                    LayoutTypeID::kTensorC32RSK32,
                    LayoutTypeID::kTensorNC32HW32,
                    LayoutTypeID::kTensorNC32HW32};
        case Format::NCHW32_NCHW4:
            return {LayoutTypeID::kTensorNC32HW32,
                    LayoutTypeID::kTensorC32RSK32, LayoutTypeID::kTensorNC4HW4,
                    LayoutTypeID::kTensorNC4HW4};
        case Format::NCHW64:
            return {LayoutTypeID::kTensorNC64HW64,
                    LayoutTypeID::kTensorC64RSK64,
                    LayoutTypeID::kTensorNC64HW64,
                    LayoutTypeID::kTensorNC64HW64};
        case Format::NHWC:
            switch (access_type) {
                case 4:
                    return {LayoutTypeID::kTensorNHWC,
                            LayoutTypeID::kTensorNC4HW4,
                            LayoutTypeID::kTensorNHWC,
                            LayoutTypeID::kTensorNHWC};
                case 8:
                    return {LayoutTypeID::kTensorNHWC,
                            LayoutTypeID::kTensorNC8HW8,
                            LayoutTypeID::kTensorNHWC,
                            LayoutTypeID::kTensorNHWC};
                case 16:
                    return {LayoutTypeID::kTensorNHWC,
                            LayoutTypeID::kTensorNC16HW16,
                            LayoutTypeID::kTensorNHWC,
                            LayoutTypeID::kTensorNHWC};
                case 32:
                    return {LayoutTypeID::kTensorNHWC,
                            LayoutTypeID::kTensorNC32HW32,
                            LayoutTypeID::kTensorNHWC,
                            LayoutTypeID::kTensorNHWC};
                default:
                    megdnn_assert(0, "invalid access_type");
            }
        default:
            megdnn_assert(0, "invalid format");
    }
}

EpilogueType get_epilogue_type(const param::ConvBias::NonlineMode mode,
                               bool clamp) {
    using NonlineMode = param::ConvBias::NonlineMode;

    if (clamp) {
        if (mode == NonlineMode::IDENTITY) {
            return EpilogueType::kBiasAddLinearCombinationClamp;
        } else if (mode == NonlineMode::RELU) {
            return EpilogueType::kBiasAddLinearCombinationReluClamp;
        } else if (mode == NonlineMode::H_SWISH) {
            return EpilogueType::kBiasAddLinearCombinationHSwishClamp;
        }
    } else {
        if (mode == NonlineMode::IDENTITY) {
            return EpilogueType::kBiasAddLinearCombination;
        } else if (mode == NonlineMode::RELU) {
            return EpilogueType::kBiasAddLinearCombinationRelu;
        } else if (mode == NonlineMode::H_SWISH) {
            return EpilogueType::kBiasAddLinearCombinationHSwish;
        }
    }
    megdnn_assert(0, "invalid nonlinear mode");
}

}  // namespace

const Operation*
ConvBiasForwardImpl::AlgoCutlassConvolutionBase::get_cutlass_conv_op(
        const SizeArgs& args, ConvOperator conv_op, ConvType conv_type,
        bool use_conv_filter_unity_opt, bool without_shared_load) const {
    auto&& param = args.opr->param();
    auto layouts = get_layout_pack(param.format, m_algo_param.access_size);
    auto epilogue_type = get_epilogue_type(
            param.nonlineMode,
            args.dst_layout->dtype.enumv() != DTypeEnum::Float32);

    cutlass::conv::SpecialOptimizeDesc special_optimization =
            (use_conv_filter_unity_opt)
                    ? cutlass::conv::SpecialOptimizeDesc::CONV_FILTER_UNITY
                    : cutlass::conv::SpecialOptimizeDesc::NONE;

    ConvolutionKey key{convert_conv_op(conv_op),
                       convert_dtype(args.src_layout->dtype.enumv()),
                       layouts.src,
                       convert_dtype(args.filter_layout->dtype.enumv()),
                       layouts.filter,
                       convert_dtype(args.dst_layout->dtype.enumv()),
                       layouts.dst,
                       convert_dtype(args.bias_layout->dtype.enumv()),
                       layouts.bias,
                       convert_conv_type(conv_type),
                       m_algo_param.threadblock_m,
                       m_algo_param.threadblock_n,
                       m_algo_param.threadblock_k,
                       m_algo_param.warp_m,
                       m_algo_param.warp_n,
                       m_algo_param.warp_k,
                       m_algo_param.instruction_m,
                       m_algo_param.instruction_n,
                       m_algo_param.instruction_k,
                       epilogue_type,
                       m_algo_param.stage,
                       special_optimization,
                       without_shared_load};

    return Singleton::get().operation_table.find_op(key);
}

void ConvBiasForwardImpl::AlgoCutlassConvolutionBase::execute_cutlass_conv_op(
        const Operation* op, const void* src, const void* filter,
        const void* bias, const void* z, void* dst, void* workspace, size_t n,
        size_t hi, size_t wi, size_t ci, size_t co, size_t fh, size_t fw,
        size_t ho, size_t wo, size_t ph, size_t pw, size_t sh, size_t sw,
        size_t dh, size_t dw, const void* alpha, const void* beta,
        const void* gamma, const void* delta, const void* theta,
        const void* threshold, const void* dst_scale, cudaStream_t stream,
        const void* extra_param) const {
    // gcc prints warnings when size_t values are implicitly narrowed to int
    cutlass::conv::Conv2dProblemSize problem_size{
            int(n),  int(hi), int(wi), int(ci),
            int(co), int(fh), int(fw), int(ho),
            int(wo), int(ph), int(pw), int(sh),
            int(sw), int(dh), int(dw), cutlass::conv::Mode::kCrossCorrelation};

    ConvolutionArguments conv_args{
            problem_size, src,       filter,    bias,       z,
            dst,          alpha,     beta,      gamma,      delta,
            theta,        threshold, dst_scale, extra_param};

    cutlass_check(op->run(&conv_args, workspace, stream));
}

}  // namespace cuda
}  // namespace megdnn
