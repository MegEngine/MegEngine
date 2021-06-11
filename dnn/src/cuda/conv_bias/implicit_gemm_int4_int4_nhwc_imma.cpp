/**
 * \file dnn/src/cuda/conv_bias/implicit_gemm_int4_int4_nhwc_imma.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "./algo.h"
#include "src/cuda/conv_bias/cutlass_convolution_wrapper.cuh"

using namespace megdnn;
using namespace cuda;
using namespace convolution;

#if CUDA_VERSION >= 10020
size_t
ConvBiasForwardImpl::AlgoInt4Int4NHWCIMMAImplicitGemm::get_workspace_in_bytes(
        const SizeArgs& args) const {
    if (args.preprocessed_filter) {
        return 0;
    } else {
        return args.filter_layout->span().dist_byte();
    }
}

size_t ConvBiasForwardImpl::AlgoInt4Int4NHWCIMMAImplicitGemm::
        get_preprocess_workspace_in_bytes(const SizeArgs& args) const {
    return 0;
}

SmallVector<TensorLayout> ConvBiasForwardImpl::
        AlgoInt4Int4NHWCIMMAImplicitGemm::deduce_preprocessed_filter_layout(
                const SizeArgs& args) const {
    return {args.filter_layout->collapse_contiguous()};
}

void ConvBiasForwardImpl::AlgoInt4Int4NHWCIMMAImplicitGemm::exec_preprocess(
        const ExecArgs& args) const {
    megdnn_assert(args.preprocessed_filter->tensors.size() == 1);
    void* filter_ptr = args.preprocessed_filter->tensors[0].raw_ptr;
    reorder_filter(args, m_algo_param.access_size, filter_ptr);
}

std::tuple<void*, void*>
ConvBiasForwardImpl::AlgoInt4Int4NHWCIMMAImplicitGemm::prepare_filter_bias(
        const ExecArgs& args) const {
    void* filter_ptr = nullptr;
    if (args.preprocessed_filter) {
        megdnn_assert(args.preprocessed_filter->tensors.size() == 1);
        filter_ptr = args.preprocessed_filter->tensors[0].raw_ptr;
    } else {
        filter_ptr = reinterpret_cast<void*>(args.workspace.raw_ptr);
        reorder_filter(args, m_algo_param.access_size, filter_ptr);
    }
    void* bias_ptr = args.bias_tensor->raw_ptr;
    return {filter_ptr, bias_ptr};
}

std::tuple<float, float, float, float, float>
ConvBiasForwardImpl::AlgoInt4Int4NHWCIMMAImplicitGemm::get_constants(
        const ExecArgs& args) const {
    float src_scale = args.src_layout->dtype.param<dtype::QuantizedS4>().scale,
          filter_scale =
                  args.filter_layout->dtype.param<dtype::QuantizedS4>().scale,
          bias_scale =
                  args.bias_layout->dtype.param<dtype::QuantizedS32>().scale,
          dst_scale = args.dst_layout->dtype.param<dtype::QuantizedS4>().scale;

    float alpha = src_scale * filter_scale / dst_scale,
          beta = bias_scale / dst_scale, gamma = 0.f, delta = 0.f, theta = 0.f;

    if (args.z_layout->ndim > 0) {
        float z_scale = args.z_layout->dtype.param<dtype::QuantizedS4>().scale;
        gamma = z_scale / dst_scale;
    }

    return {alpha, beta, gamma, delta, theta};
}

void ConvBiasForwardImpl::AlgoInt4Int4NHWCIMMAImplicitGemm::do_exec(
        const ExecArgs& args, void* filter_ptr, void* bias_ptr, void* z_ptr,
        ConvParam kern_param, uint32_t nonlinear_mode, float alpha, float beta,
        float gamma, float delta, float theta, cudaStream_t stream) const {
    float dst_scale = args.dst_layout->dtype.param<dtype::QuantizedS4>().scale;

    cutlass_wrapper::GemmCoord threadblock_shape{m_algo_param.threadblock_m,
                                                 m_algo_param.threadblock_n,
                                                 m_algo_param.threadblock_k};

    cutlass_wrapper::GemmCoord warp_shape{
            m_algo_param.warp_m, m_algo_param.warp_n, m_algo_param.warp_k};

    if (kern_param.fh == 1 && kern_param.fw == 1) {
        cutlass_wrapper::do_conv_bias_int4_int4_implicit_gemm_imma_nhwc<false>(
                reinterpret_cast<int8_t*>(args.src_tensor->raw_ptr),
                reinterpret_cast<int8_t*>(filter_ptr),
                reinterpret_cast<int32_t*>(bias_ptr),
                reinterpret_cast<int8_t*>(z_ptr),
                reinterpret_cast<int8_t*>(args.dst_tensor->raw_ptr), nullptr,
                kern_param, nonlinear_mode, alpha, beta, gamma, dst_scale,
                threadblock_shape, warp_shape, m_algo_param.access_size,
                stream);
    } else {
        cutlass_wrapper::do_conv_bias_int4_int4_implicit_gemm_imma_nhwc<true>(
                reinterpret_cast<int8_t*>(args.src_tensor->raw_ptr),
                reinterpret_cast<int8_t*>(filter_ptr),
                reinterpret_cast<int32_t*>(bias_ptr),
                reinterpret_cast<int8_t*>(z_ptr),
                reinterpret_cast<int8_t*>(args.dst_tensor->raw_ptr), nullptr,
                kern_param, nonlinear_mode, alpha, beta, gamma, dst_scale,
                threadblock_shape, warp_shape, m_algo_param.access_size,
                stream);
    }
}
#endif

// vim: syntax=cpp.doxygen
