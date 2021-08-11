/**
 * \file dnn/src/cuda/conv_bias/implicit_gemm_int8_nhwc_imma.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/common/conv_bias.h"
#include "src/cuda/conv_bias/algo.h"
#include "src/cuda/conv_bias/cutlass_reorder_filter.cuh"
#include "src/cuda/convolution_helper/parameter.cuh"
#include "src/cuda/utils.h"

using namespace megdnn;
using namespace cuda;
using namespace convolution;

#if CUDA_VERSION >= 10020
bool ConvBiasForwardImpl::AlgoInt8NHWCIMMAImplicitGemm::is_available(
        const SizeArgs& args) const {
    if (args.bias_layout->ndim <= 0)
        return false;

    using Param = param::ConvBias;
    using Format = Param::Format;
    using Sparse = Param::Sparse;
    using Mode = Param::Mode;
    using NonlineMode = megdnn::param::ConvBias::NonlineMode;

    auto&& param = args.opr->param();

    if (!check_bias_share_in_channel(*(args.bias_layout), param.format))
        return false;

    if (param.format != Format::NHWC || param.sparse != Sparse::DENSE ||
        param.mode != Mode::CROSS_CORRELATION)
        return false;

    if (param.nonlineMode != NonlineMode::IDENTITY &&
        param.nonlineMode != NonlineMode::RELU &&
        param.nonlineMode != NonlineMode::H_SWISH)
        return false;

    if (args.src_layout->dtype.enumv() != DTypeEnum::QuantizedS8 ||
        args.filter_layout->dtype.enumv() != DTypeEnum::QuantizedS8)
        return false;

    auto dst_dtype = args.dst_layout->dtype.enumv();

    if (!(dst_dtype == DTypeEnum::QuantizedS8 ||
          dst_dtype == DTypeEnum::QuantizedS4 ||
          dst_dtype == DTypeEnum::Quantized4Asymm ||
          dst_dtype == DTypeEnum::Float32))
        return false;

    if (!(args.bias_layout->dtype.enumv() == DTypeEnum::QuantizedS32 ||
          (args.bias_layout->dtype.enumv() == DTypeEnum::Float32 &&
           dst_dtype == DTypeEnum::Float32)))
        return false;

    if (!is_compute_capability_required(7, 5))
        return false;

    size_t co = args.filter_layout->operator[](0),
           ci = args.filter_layout->operator[](3),
           fh = args.filter_layout->operator[](1),
           fw = args.filter_layout->operator[](2);

    // param buffer size is 4K, use 3.4K to store precomputed offset
    size_t kMaxFilterPixels =
            848 / (m_algo_param.warp_k / m_algo_param.access_size) - 1;
    if (fh * fw > kMaxFilterPixels)
        return false;
    // co should be aligned with 4, and ci should be aligned with
    // algo_param.access_size
    if ((co % 4 != 0) || (ci % m_algo_param.access_size != 0))
        return false;

    bool use_conv_filter_unity_opt = (fh == 1 && fw == 1);
    bool without_shared_load = ((co % m_algo_param.threadblock_n == 0) &&
                                (m_algo_param.threadblock_n == 16 ||
                                 (m_algo_param.threadblock_n == 32 &&
                                  dst_dtype != DTypeEnum::Float32)));
    const auto* op = get_cutlass_conv_op(
            args, ConvOperator::kFprop, ConvType::kConvolution,
            use_conv_filter_unity_opt, without_shared_load);
    if (op == nullptr)
        return false;

    return true;
}

size_t
ConvBiasForwardImpl::AlgoInt8NHWCIMMAImplicitGemm::get_workspace_in_bytes(
        const SizeArgs& args) const {
    if (args.preprocessed_filter) {
        return 0;
    } else {
        return args.filter_layout->span().dist_byte();
    }
}

size_t ConvBiasForwardImpl::AlgoInt8NHWCIMMAImplicitGemm::
        get_preprocess_workspace_in_bytes(const SizeArgs& args) const {
    return 0;
}

SmallVector<TensorLayout> ConvBiasForwardImpl::AlgoInt8NHWCIMMAImplicitGemm::
        deduce_preprocessed_filter_layout(const SizeArgs& args) const {
    return {args.filter_layout->collapse_contiguous()};
}

void ConvBiasForwardImpl::AlgoInt8NHWCIMMAImplicitGemm::exec_preprocess(
        const ExecArgs& args) const {
    void* filter_ptr = args.preprocessed_filter->tensors[0].raw_ptr;
    reorder_filter(args, m_algo_param.access_size, filter_ptr);
}

std::tuple<float, float, float, float, float>
ConvBiasForwardImpl::AlgoInt8NHWCIMMAImplicitGemm::get_constants(
        const ExecArgs& args) const {
    float src_scale = args.src_layout->dtype.param<dtype::QuantizedS8>().scale,
          filter_scale =
                  args.filter_layout->dtype.param<dtype::QuantizedS8>().scale,
          bias_scale = 1.f, dst_scale;

    if (args.bias_layout->dtype.enumv() == DTypeEnum::QuantizedS32) {
        bias_scale = args.bias_layout->dtype.param<dtype::QuantizedS32>().scale;
    }

    uint8_t dst_zero = 0;

    if (args.dst_layout->dtype.enumv() == DTypeEnum::QuantizedS8) {
        dst_scale = args.dst_layout->dtype.param<dtype::QuantizedS8>().scale;
    } else if (args.dst_layout->dtype.enumv() == DTypeEnum::QuantizedS4) {
        dst_scale = args.dst_layout->dtype.param<dtype::QuantizedS4>().scale;
    } else if (args.dst_layout->dtype.enumv() == DTypeEnum::Quantized4Asymm) {
        dst_scale =
                args.dst_layout->dtype.param<dtype::Quantized4Asymm>().scale;
        dst_zero = args.dst_layout->dtype.param<dtype::Quantized4Asymm>()
                           .zero_point;
    } else {  // DTypeEnum::Float32
        megdnn_assert(args.dst_layout->dtype.enumv() == DTypeEnum::Float32);
        dst_scale = 1.f;
    }

    float alpha = src_scale * filter_scale / dst_scale,
          beta = bias_scale / dst_scale, gamma = 0.f, delta = 0.f,
          theta = dst_zero;

    if (args.z_layout->ndim > 0) {
        float z_scale;
        if (args.z_layout->dtype.enumv() == DTypeEnum::QuantizedS8) {
            z_scale = args.z_layout->dtype.param<dtype::QuantizedS8>().scale;
            gamma = z_scale / dst_scale;
        } else if (args.z_layout->dtype.enumv() == DTypeEnum::QuantizedS4) {
            z_scale = args.z_layout->dtype.param<dtype::QuantizedS4>().scale;
            gamma = z_scale / dst_scale;
        } else if (args.z_layout->dtype.enumv() == DTypeEnum::Quantized4Asymm) {
            z_scale =
                    args.z_layout->dtype.param<dtype::Quantized4Asymm>().scale;
            uint8_t z_zero =
                    args.z_layout->dtype.param<dtype::Quantized4Asymm>()
                            .zero_point;
            gamma = z_scale / dst_scale;
            delta = -z_zero * gamma;
        } else {  // DTypeEnum::Float32
            megdnn_assert(args.z_layout->dtype.enumv() == DTypeEnum::Float32);
            gamma = 1.f;
        }
    }

    if (args.opr->param().nonlineMode ==
        param::ConvBias::NonlineMode::IDENTITY) {
        delta += theta;
        theta = 0.f;
    }

    return {alpha, beta, gamma, delta, theta};
}

void ConvBiasForwardImpl::AlgoInt8NHWCIMMAImplicitGemm::exec(
        const ExecArgs& args) const {
    auto&& param = args.opr->param();
    auto&& fm = args.filter_meta;
    size_t n = args.src_layout->operator[](0),
           ci = args.src_layout->operator[](3),
           hi = args.src_layout->operator[](1),
           wi = args.src_layout->operator[](2);
    size_t co = args.dst_layout->operator[](3),
           ho = args.dst_layout->operator[](1),
           wo = args.dst_layout->operator[](2);
    UNPACK_CONV_PARAMETER(fm, param);
    MARK_USED_VAR

    void* filter_ptr = nullptr;
    void* bias_ptr = nullptr;
    void* z_ptr = nullptr;

    if (args.preprocessed_filter) {
        filter_ptr = args.preprocessed_filter->tensors[0].raw_ptr;
    } else {
        filter_ptr = reinterpret_cast<void*>(args.workspace.raw_ptr);
        reorder_filter(args, m_algo_param.access_size, filter_ptr);
    }
    bias_ptr = args.bias_tensor->raw_ptr;

    if (args.z_layout->ndim > 0)
        z_ptr = args.z_tensor->raw_ptr;

    // \note these constants of cutlass epilogue will be passed to method
    // `execute_cutlass_conv_op` by pointer and interpreted as ElementCompute*,
    // a different dtype here results in undefined epilogue behaviors
    float alpha, beta, gamma, delta, theta;
    std::tie(alpha, beta, gamma, delta, theta) = get_constants(args);

    float dst_scale = 1.f;
    float threshold = 0.f;
    bool use_conv_filter_unity_opt = (fh == 1 && fw == 1);

    auto dst_dtype = args.dst_layout->dtype.enumv();

    bool without_shared_load = ((co % m_algo_param.threadblock_n == 0) &&
                                (m_algo_param.threadblock_n == 16 ||
                                 (m_algo_param.threadblock_n == 32 &&
                                  dst_dtype != DTypeEnum::Float32)));

    if (dst_dtype == DTypeEnum::QuantizedS8) {  // DTypeEnum::QuantizedS8
        dst_scale = args.dst_layout->dtype.param<dtype::QuantizedS8>().scale;
    } else if (dst_dtype == DTypeEnum::QuantizedS4) {
        dst_scale = args.dst_layout->dtype.param<dtype::QuantizedS4>().scale;
    } else if (dst_dtype == DTypeEnum::Quantized4Asymm) {
        dst_scale =
                args.dst_layout->dtype.param<dtype::Quantized4Asymm>().scale;
    } else {  // DTypeEnum::Float32
        dst_scale = 1.f;
    }

    cudaStream_t stream = cuda_stream(args.opr->handle());

    const auto* op = get_cutlass_conv_op(
            args, ConvOperator::kFprop, ConvType::kConvolution,
            use_conv_filter_unity_opt, without_shared_load);

    execute_cutlass_conv_op(op, args.src_tensor->raw_ptr, filter_ptr, bias_ptr,
                            z_ptr, args.dst_tensor->raw_ptr, nullptr, n, hi, wi,
                            ci, co, fh, fw, ho, wo, ph, pw, sh, sw, dh, dw,
                            &alpha, &beta, &gamma, &delta, &theta, &threshold,
                            &dst_scale, stream);

    after_kernel_launch();
}

std::string ConvBiasForwardImpl::AlgoInt8NHWCIMMAImplicitGemm::to_string(
        AlgoParam algo_param) {
    return ssprintf("%dX%dX%d_%dX%dX%d_%d_%d", algo_param.threadblock_m,
                    algo_param.threadblock_n, algo_param.threadblock_k,
                    algo_param.warp_m, algo_param.warp_n, algo_param.warp_k,
                    algo_param.stage, algo_param.access_size);
}

void ConvBiasForwardImpl::AlgoInt8NHWCIMMAImplicitGemm::reorder_filter(
        const ExecArgs& args, const int iterleaved,
        void* reordered_filter) const {
    size_t co = args.filter_layout->operator[](0),
           ci = args.filter_layout->operator[](3),
           fh = args.filter_layout->operator[](1),
           fw = args.filter_layout->operator[](2);

    cudaStream_t stream = cuda_stream(args.opr->handle());

    // reformat filter from nhwc to ncxhwx and reorder oc
    // use trans_oc threadblock_n must be 16 or 32 and src dtype == dest dtype
    bool trans_oc = ((co % m_algo_param.threadblock_n == 0) &&
                     (m_algo_param.threadblock_n == 16 ||
                      (m_algo_param.threadblock_n == 32 &&
                       args.dst_layout->dtype.enumv() != DTypeEnum::Float32)));
    uint32_t oc_iterleaved = (m_algo_param.threadblock_n == 32) ? 32 : 16;

    uint32_t alignbits = iterleaved * 8;

    cutlass_wrapper::reorder_nhwc_imma_filter<8>(
            reinterpret_cast<int8_t*>(reordered_filter),
            reinterpret_cast<int8_t*>(args.filter_tensor->raw_ptr), co, ci, fh,
            fw, trans_oc, alignbits, oc_iterleaved, stream);
}
#endif

// vim: syntax=cpp.doxygen
