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

std::string ConvBiasForwardImpl::AlgoCutlassConvolutionBase::AlgoParam::to_string()
        const {
    /// default algorithm
    if (threadblock_m == 128 && threadblock_n == 128 && threadblock_k == 32 &&
        warp_m == 32 && warp_n == 64 && warp_k == 32 && stage == 2) {
        return "";
    }
    return ssprintf(
            "_%dX%dX%d_%dX%dX%d_%dstage", threadblock_m, threadblock_n, threadblock_k,
            warp_m, warp_n, warp_k, stage);
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
        case Base::ConvType::kDepthwiseConvolution:
            return cutlass::conv::ConvType::kDepthwiseConvolution;
        default:
            megdnn_assert(0, "invalid conv type");
    }
}

NumericTypeID convert_dtype(DType dtype) {
    // just make convolution with no bias happy
    if (!dtype.valid())
        return NumericTypeID::kF32;
    switch (dtype.enumv()) {
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

NumericTypeID get_accumulator_dtype(
        DType dtype, const param::ConvBias::ComputeMode comp_mode) {
    if (dtype.category() == DTypeCategory::QUANTIZED) {
        return NumericTypeID::kS32;
    } else {
        megdnn_assert(dtype.category() == DTypeCategory::FLOAT);
        if (comp_mode == param::ConvBias::ComputeMode::DEFAULT) {
            return convert_dtype(dtype);
        } else {
            megdnn_assert(comp_mode == param::ConvBias::ComputeMode::FLOAT32);
            return NumericTypeID::kF32;
        }
    }
}

struct LayoutPack {
    LayoutTypeID src;
    LayoutTypeID filter;
    LayoutTypeID dst;
    LayoutTypeID bias;
};

LayoutPack get_layout_pack(const param::ConvBias::Format format, int access_type) {
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
                    LayoutTypeID::kTensorNC32HW32, LayoutTypeID::kTensorNC32HW32};
        case Format::NCHW32:
            return {LayoutTypeID::kTensorNC32HW32, LayoutTypeID::kTensorC32RSK32,
                    LayoutTypeID::kTensorNC32HW32, LayoutTypeID::kTensorNC32HW32};
        case Format::NCHW32_NCHW4:
            return {LayoutTypeID::kTensorNC32HW32, LayoutTypeID::kTensorC32RSK32,
                    LayoutTypeID::kTensorNC4HW4, LayoutTypeID::kTensorNC4HW4};
        case Format::NCHW64:
            return {LayoutTypeID::kTensorNC64HW64, LayoutTypeID::kTensorC64RSK64,
                    LayoutTypeID::kTensorNC64HW64, LayoutTypeID::kTensorNC64HW64};
        case Format::NHWC:
            switch (access_type) {
                case 4:
                    return {LayoutTypeID::kTensorNHWC, LayoutTypeID::kTensorNC4HW4,
                            LayoutTypeID::kTensorNHWC, LayoutTypeID::kTensorNHWC};
                case 8:
                    return {LayoutTypeID::kTensorNHWC, LayoutTypeID::kTensorNC8HW8,
                            LayoutTypeID::kTensorNHWC, LayoutTypeID::kTensorNHWC};
                case 16:
                    return {LayoutTypeID::kTensorNHWC, LayoutTypeID::kTensorNC16HW16,
                            LayoutTypeID::kTensorNHWC, LayoutTypeID::kTensorNHWC};
                case 32:
                    return {LayoutTypeID::kTensorNHWC, LayoutTypeID::kTensorNC32HW32,
                            LayoutTypeID::kTensorNHWC, LayoutTypeID::kTensorNHWC};
                default:
                    megdnn_assert(0, "invalid access_type");
            }
        case Format::NCHW:
            return {LayoutTypeID::kTensorNCHW, LayoutTypeID::kTensorNCHW,
                    LayoutTypeID::kTensorNCHW, LayoutTypeID::kTensorNCHW};
        default:
            megdnn_assert(0, "invalid format");
    }
}

EpilogueType get_epilogue_type(const param::ConvBias::NonlineMode mode, bool clamp) {
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

std::pair<int, int> get_tensor_alignment(
        const param::ConvBias::Format format, const TensorLayout& src,
        const TensorLayout& filter, const Base::AlgoParam& algo_param,
        bool is_chanwise) {
    int alignment_src = 0;
    int alignment_filter = 0;

    using Format = param::ConvBias::Format;

    // get tensor alignment for tensor op operations
    // for tensor op operations, the alignment is determined by the size of a vector
    auto get_tensor_alignment_tensor_op = [&]() {
        switch (format) {
            /// case int8
            case Format::NCHW32:
            case Format::NCHW32_NCHW4:
                alignment_src = 16;
                alignment_filter = 16;
                break;
            /// case int4 or uint4
            case Format::NCHW64:
                alignment_src = 32;
                alignment_filter = 32;
                break;
            case Format::NHWC:
                alignment_src = alignment_filter = algo_param.access_size;
                break;
            default:
                megdnn_throw("invalid format");
        };
    };

    // get tensor alignment for dot product operations
    // for integer dot product operations, alignment src is always 4
    // and the alignment filter is determined by the threadblock shape
    auto get_tensor_alignment_dp4a = [&]() {
        megdnn_assert(
                format == Format::NCHW4 || format == Format::NCHW4_NCHW ||
                format == Format::NCHW4_NHWC || format == Format::NCHW4_NCHW32);
        alignment_src = 4;
        // determine alignment filter
        constexpr int warp_size = 32;
        int threads = warp_size * algo_param.threadblock_m * algo_param.threadblock_n *
                      algo_param.threadblock_k /
                      (algo_param.warp_m * algo_param.warp_n * algo_param.warp_k);
        int threadblock_loads = filter.dtype.size(
                algo_param.threadblock_m * algo_param.threadblock_n *
                algo_param.threadblock_k);
        int load_per_thread = threadblock_loads / threads;
        if (load_per_thread >= 16)
            alignment_filter = 16;
        else if (load_per_thread >= 8)
            alignment_filter = 8;
        else {
            megdnn_assert(load_per_thread >= 4);
            alignment_filter = 4;
        }
    };

    // get tensor alignment for depthwise convolution
    auto get_tensor_alignment_dwconv2d_nchw = [&]() {
        alignment_filter = 1;
        size_t wi = src.dtype.size(src[3]);  // width extent in bytes
        for (size_t candidate : {16, 4, 2}) {
            if (wi % candidate == 0) {
                alignment_src = candidate;
                break;
            }
        }
        alignment_src /= src.dtype.size(1);
    };

    /// TODO: need a better way to check whether tensor core instruction is used
    if (format == Format::NCHW32 || format == Format::NCHW32_NCHW4 ||
        format == Format::NCHW64 || format == Format::NCHW64 ||
        format == Format::NHWC) {
        get_tensor_alignment_tensor_op();
    } else if (
            format == Format::NCHW4 || format == Format::NCHW4_NCHW ||
            format == Format::NCHW4_NHWC || format == Format::NCHW4_NCHW32) {
        get_tensor_alignment_dp4a();
    } else {
        /// the following is used for depthwise convolution
        megdnn_assert(format == Format::NCHW && is_chanwise);
        get_tensor_alignment_dwconv2d_nchw();
    }
    megdnn_assert(alignment_src >= 1 && alignment_filter >= 1);
    return {alignment_src, alignment_filter};
}
}  // namespace

const Operation* ConvBiasForwardImpl::AlgoCutlassConvolutionBase::get_cutlass_conv_op(
        const SizeArgs& args, ConvOperator conv_op, ConvType conv_type,
        bool use_conv_filter_unity_opt, bool without_shared_load) const {
    auto&& param = args.opr->param();
    auto layouts = get_layout_pack(param.format, m_algo_param.access_size);
    auto epilogue_type = get_epilogue_type(
            param.nonlineMode,
            args.dst_layout->dtype.category() != DTypeCategory::FLOAT);

    cutlass::conv::SpecialOptimizeDesc special_optimization =
            (use_conv_filter_unity_opt)
                    ? cutlass::conv::SpecialOptimizeDesc::CONV_FILTER_UNITY
                    : cutlass::conv::SpecialOptimizeDesc::NONE;

    int alignment_src, alignment_filter;
    auto&& fm = args.filter_meta;
    bool is_chanwise = param.sparse == param::ConvBias::Sparse::GROUP && fm.icpg == 1 &&
                       fm.ocpg == 1;
    std::tie(alignment_src, alignment_filter) = get_tensor_alignment(
            param.format, *args.src_layout, *args.filter_layout, m_algo_param,
            is_chanwise);

    auto accumulator_dtype =
            get_accumulator_dtype(args.src_layout->dtype, param.compute_mode);

    ConvolutionKey key{
            convert_conv_op(conv_op),
            convert_dtype(args.src_layout->dtype),
            layouts.src,
            convert_dtype(args.filter_layout->dtype),
            layouts.filter,
            convert_dtype(args.dst_layout->dtype),
            layouts.dst,
            convert_dtype(args.bias_layout->dtype),
            layouts.bias,
            accumulator_dtype,
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
            alignment_src,
            alignment_filter,
            without_shared_load};

    return Singleton::get().operation_table.find_op(key);
}

void ConvBiasForwardImpl::AlgoCutlassConvolutionBase::execute_cutlass_conv_op(
        const Operation* op, const void* src, const void* filter, const void* bias,
        const void* z, void* dst, void* workspace, size_t n, size_t hi, size_t wi,
        size_t ci, size_t co, size_t fh, size_t fw, size_t ho, size_t wo, size_t ph,
        size_t pw, size_t sh, size_t sw, size_t dh, size_t dw, const void* alpha,
        const void* beta, const void* gamma, const void* delta, const void* theta,
        const void* threshold, const void* dst_scale, cudaStream_t stream,
        const void* extra_param, size_t groups) const {
    // gcc prints warnings when size_t values are implicitly narrowed to int
    cutlass::conv::Conv2dProblemSize problem_size{
            int(n),      int(hi), int(wi), int(ci),
            int(co),     int(fh), int(fw), int(ho),
            int(wo),     int(ph), int(pw), int(sh),
            int(sw),     int(dh), int(dw), cutlass::conv::Mode::kCrossCorrelation,
            1,            // split k slices, always 1
            int(groups),  // groups
    };

    ConvolutionArguments conv_args{
            problem_size, src,   filter, bias,  z,         dst,       alpha,
            beta,         gamma, delta,  theta, threshold, dst_scale, extra_param};

    cutlass_check(op->run(&conv_args, workspace, stream));
}

}  // namespace cuda
}  // namespace megdnn
