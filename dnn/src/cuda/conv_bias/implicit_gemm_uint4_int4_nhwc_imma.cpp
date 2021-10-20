/**
 * \file dnn/src/cuda/conv_bias/implicit_gemm_uint4_int4_nhwc_imma.cpp
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
#include "src/cuda/conv_bias/reduce_filter.cuh"
#include "src/cuda/utils.h"

using namespace megdnn;
using namespace cuda;
using namespace convolution;

#if CUDA_VERSION >= 10020
size_t ConvBiasForwardImpl::AlgoUInt4Int4NHWCIMMAImplicitGemm::get_workspace_in_bytes(
        const SizeArgs& args) const {
    if (args.preprocessed_filter) {
        return 0;
    } else {
        size_t ws_filter = args.filter_layout->span().dist_byte(),
               ws_bias = args.bias_layout->span().dist_byte(),
               ws_reduce_filter = get_preprocess_workspace_in_bytes(args);
        return ws_filter + ws_bias + ws_reduce_filter;
    }
}

size_t ConvBiasForwardImpl::AlgoUInt4Int4NHWCIMMAImplicitGemm::
        get_preprocess_workspace_in_bytes(const SizeArgs& args) const {
    size_t co = args.filter_layout->operator[](0),
           ci = args.filter_layout->operator[](3),
           fh = args.filter_layout->operator[](1),
           fw = args.filter_layout->operator[](2);
    size_t ws_size_reduce_filter = co * sizeof(int32_t);
    size_t A = co, B = ci * fh * fw / 8, C = 1;
    ws_size_reduce_filter += do_dispatch_reduce_workspace_in_bytes(A, B, C);
    return ws_size_reduce_filter;
}

SmallVector<TensorLayout> ConvBiasForwardImpl::AlgoUInt4Int4NHWCIMMAImplicitGemm::
        deduce_preprocessed_filter_layout(const SizeArgs& args) const {
    return {args.filter_layout->collapse_contiguous(),
            args.bias_layout->collapse_contiguous()};
}

void ConvBiasForwardImpl::AlgoUInt4Int4NHWCIMMAImplicitGemm::exec_preprocess(
        const ExecArgs& args) const {
    megdnn_assert(args.preprocessed_filter->tensors.size() == 2);
    void* filter_ptr = args.preprocessed_filter->tensors[0].raw_ptr;
    void* bias_ptr = args.preprocessed_filter->tensors[1].raw_ptr;
    void* reduce_filter_ptr = reinterpret_cast<void*>(args.workspace.raw_ptr);
    void* reduce_workspace = reinterpret_cast<void*>(
            args.workspace.raw_ptr + args.bias_layout->span().dist_byte());
    reorder_filter(args, m_algo_param.access_size, filter_ptr);
    update_bias(args, bias_ptr, reduce_filter_ptr, reduce_workspace);
}

std::tuple<void*, void*> ConvBiasForwardImpl::AlgoUInt4Int4NHWCIMMAImplicitGemm::
        prepare_filter_bias(const ExecArgs& args) const {
    void* filter_ptr = nullptr;
    void* bias_ptr = nullptr;
    if (args.preprocessed_filter) {
        megdnn_assert(args.preprocessed_filter->tensors.size() == 2);
        filter_ptr = args.preprocessed_filter->tensors[0].raw_ptr;
        bias_ptr = args.preprocessed_filter->tensors[1].raw_ptr;
        return {filter_ptr, bias_ptr};
    } else {
        filter_ptr = reinterpret_cast<void*>(args.workspace.raw_ptr);
        bias_ptr = reinterpret_cast<void*>(
                args.workspace.raw_ptr + args.filter_layout->span().dist_byte());
        void* reduce_filter_ptr = reinterpret_cast<void*>(
                args.workspace.raw_ptr + args.filter_layout->span().dist_byte() +
                args.bias_layout->span().dist_byte());
        void* reduce_workspace = reinterpret_cast<void*>(
                args.workspace.raw_ptr + args.filter_layout->span().dist_byte() +
                args.bias_layout->span().dist_byte() +
                args.bias_layout->span().dist_byte());
        reorder_filter(args, m_algo_param.access_size, filter_ptr);
        update_bias(args, bias_ptr, reduce_filter_ptr, reduce_workspace);
    }
    return {filter_ptr, bias_ptr};
}

std::tuple<float, float, float, float, float> ConvBiasForwardImpl::
        AlgoUInt4Int4NHWCIMMAImplicitGemm::get_constants(const ExecArgs& args) const {
    float src_scale = args.src_layout->dtype.param<dtype::Quantized4Asymm>().scale,
          filter_scale = args.filter_layout->dtype.param<dtype::QuantizedS4>().scale,
          bias_scale = args.bias_layout->dtype.param<dtype::QuantizedS32>().scale,
          dst_scale;

    uint8_t dst_zero = 0;

    if (args.dst_layout->dtype.enumv() == DTypeEnum::Quantized4Asymm) {
        dst_scale = args.dst_layout->dtype.param<dtype::Quantized4Asymm>().scale;

        dst_zero = args.dst_layout->dtype.param<dtype::Quantized4Asymm>().zero_point;
    } else {  // DTypeEnum::QuantizedS8
        megdnn_assert(args.dst_layout->dtype.enumv() == DTypeEnum::QuantizedS8);
        dst_scale = args.dst_layout->dtype.param<dtype::QuantizedS8>().scale;
    }

    float alpha = src_scale * filter_scale / dst_scale, beta = bias_scale / dst_scale,
          gamma = 0.f, delta = 0.f, theta = dst_zero;

    if (args.z_layout->ndim > 0) {
        float z_scale;
        if (args.z_layout->dtype.enumv() == DTypeEnum::Quantized4Asymm) {
            z_scale = args.z_layout->dtype.param<dtype::Quantized4Asymm>().scale;
            uint8_t z_zero =
                    args.z_layout->dtype.param<dtype::Quantized4Asymm>().zero_point;
            gamma = z_scale / dst_scale;
            delta = -z_zero * gamma;
        } else {  // DTypeEnum::QuantizedS8
            megdnn_assert(args.z_layout->dtype.enumv() == DTypeEnum::QuantizedS8);
            z_scale = args.z_layout->dtype.param<dtype::QuantizedS8>().scale;
            gamma = z_scale / dst_scale;
        }
    }

    // identity epilogue has no theta:
    // alpha * accumulator + beta * bias + gamma * source + delta
    if (args.opr->param().nonlineMode == param::ConvBias::NonlineMode::IDENTITY) {
        delta += theta;
        theta = 0.f;
    }

    return {alpha, beta, gamma, delta, theta};
}

void ConvBiasForwardImpl::AlgoUInt4Int4NHWCIMMAImplicitGemm::update_bias(
        const ExecArgs& args, void* updated_bias, void* reduce_filter_ptr,
        void* reduce_workspace) const {
    size_t co = args.filter_layout->operator[](0),
           ci = args.filter_layout->operator[](3),
           fh = args.filter_layout->operator[](1),
           fw = args.filter_layout->operator[](2);

    auto&& stream = cuda_stream(args.opr->handle());

    int src_zero_point =
            args.src_tensor->layout.dtype.param<dtype::Quantized4Asymm>().zero_point;
    do_dispatch_reduce_filter_and_update_bias_4bit<true>(
            reinterpret_cast<uint8_t*>(args.filter_tensor->raw_ptr),
            args.bias_tensor->compatible_ptr<int32_t>(), co, ci * fh * fw / 8,
            reinterpret_cast<int32_t*>(updated_bias),
            reinterpret_cast<int32_t*>(reduce_workspace), src_zero_point, stream);
}
#endif

// vim: syntax=cpp.doxygen
