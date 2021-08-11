/**
 * \file dnn/src/cuda/conv_bias/implicit_gemm_int4_nchw64_imma_base.cpp
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
#include "src/cuda/conv_bias/reduce_filter.cuh"
#include "src/cuda/convolution_helper/parameter.cuh"
#include "src/cuda/utils.h"

using namespace megdnn;
using namespace cuda;
using namespace convolution;

#if CUDA_VERSION >= 10020
std::string ConvBiasForwardImpl::AlgoInt4NCHW64IMMAImplicitGemmBase::param()
        const {
    std::string ret;
    serialize_write_pod(m_algo_param, ret);
    return ret;
}

bool ConvBiasForwardImpl::AlgoInt4NCHW64IMMAImplicitGemmBase::is_available(
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

    if (param.format != Format::NCHW64 || param.sparse != Sparse::DENSE ||
        param.mode != Mode::CROSS_CORRELATION)
        return false;

    if (param.nonlineMode != NonlineMode::IDENTITY &&
        param.nonlineMode != NonlineMode::RELU &&
        param.nonlineMode != NonlineMode::H_SWISH)
        return false;

    if (args.src_layout->dtype.enumv() != src_dtype() ||
        args.filter_layout->dtype.enumv() != DTypeEnum::QuantizedS4 ||
        args.bias_layout->dtype.enumv() != DTypeEnum::QuantizedS32 ||
        args.dst_layout->dtype.enumv() != src_dtype())
        return false;

    // uint4 do not support H_SWISH activition
    if (src_dtype() == DTypeEnum::Quantized4Asymm &&
        param.nonlineMode == NonlineMode::H_SWISH)
        return false;

    if (!is_compute_capability_required(7, 5))
        return false;

    size_t fh = args.filter_layout->operator[](1),
           fw = args.filter_layout->operator[](2);

    // param buffer size is 4K, use 3.4K to store precomputed offset
    size_t kMaxFilterPixels = 848 / (2 * m_algo_param.warp_k / 64) - 2;
    if (fh * fw > kMaxFilterPixels)
        return false;

    bool use_conv_filter_unity_opt = (fh == 1 && fw == 1);
    bool without_shared_load = true;
    const auto* op = get_cutlass_conv_op(
            args, ConvOperator::kFprop, ConvType::kConvolution,
            use_conv_filter_unity_opt, without_shared_load);
    if (op == nullptr)
        return false;

    return true;
}

void ConvBiasForwardImpl::AlgoInt4NCHW64IMMAImplicitGemmBase::exec(
        const ExecArgs& args) const {
    auto&& param = args.opr->param();
    auto&& fm = args.filter_meta;
    size_t n = args.src_layout->operator[](0),
           ci = args.src_layout->operator[](1) * 64,
           hi = args.src_layout->operator[](2),
           wi = args.src_layout->operator[](3);
    size_t co = args.dst_layout->operator[](1) * 64,
           ho = args.dst_layout->operator[](2),
           wo = args.dst_layout->operator[](3);
    UNPACK_CONV_PARAMETER(fm, param);
    MARK_USED_VAR

    void* filter_ptr = nullptr;
    void* bias_ptr = nullptr;
    void* z_ptr = nullptr;

    std::tie(filter_ptr, bias_ptr) = prepare_filter_bias(args);
    if (args.z_layout->ndim > 0)
        z_ptr = args.z_tensor->raw_ptr;

    // \note these constants of cutlass epilogue will be passed to method
    // `execute_cutlass_conv_op` by pointer and interpreted as ElementCompute*,
    // a different dtype here results in undefined epilogue behaviors
    float alpha, beta, gamma, delta, theta;

    std::tie(alpha, beta, gamma, delta, theta) = get_constants(args);
    float dst_scale = 0.f;
    float threshold = 0.f;
    uint8_t src_zero = 0;
    bool use_conv_filter_unity_opt = (fh == 1 && fw == 1);
    bool without_shared_load = true;

    if (args.dst_layout->dtype.enumv() == DTypeEnum::Quantized4Asymm) {
        dst_scale =
                args.dst_layout->dtype.param<dtype::Quantized4Asymm>().scale;
        src_zero = args.src_layout->dtype.param<dtype::Quantized4Asymm>()
                           .zero_point;
    } else {  // DTypeEnum::QuantizedS4
        dst_scale = args.dst_layout->dtype.param<dtype::QuantizedS4>().scale;
    }

    cudaStream_t stream = cuda_stream(args.opr->handle());

    const auto* op = get_cutlass_conv_op(args, ConvOperator::kFprop,
                                         ConvType::kConvolution,
                                         use_conv_filter_unity_opt, without_shared_load);

    execute_cutlass_conv_op(op, args.src_tensor->raw_ptr, filter_ptr, bias_ptr,
                            z_ptr, args.dst_tensor->raw_ptr, nullptr, n, hi, wi,
                            ci, co, fh, fw, ho, wo, ph, pw, sh, sw, dh, dw,
                            &alpha, &beta, &gamma, &delta, &theta, &threshold,
                            &dst_scale, stream, &src_zero);

    after_kernel_launch();
}

std::string ConvBiasForwardImpl::AlgoInt4NCHW64IMMAImplicitGemmBase::to_string(
        AlgoParam algo_param) {
    return ssprintf("%dX%dX%d_%dX%dX%d_%d", algo_param.threadblock_m,
                    algo_param.threadblock_n, algo_param.threadblock_k,
                    algo_param.warp_m, algo_param.warp_n, algo_param.warp_k,
                    algo_param.stage);
}

void ConvBiasForwardImpl::AlgoInt4NCHW64IMMAImplicitGemmBase::reorder_filter(
        const ExecArgs& args, void* reordered_filter) const {
    size_t ci = args.src_layout->operator[](1) * 64;
    size_t co = args.dst_layout->operator[](1) * 64;
    auto&& fm = args.filter_meta;
    size_t fh = fm.spatial[0], fw = fm.spatial[1];

    cudaStream_t stream = cuda_stream(args.opr->handle());

    // filter: KCRS64 => CRSK64 and reorder oc
    cutlass_wrapper::reorder_ncxhwx_imma_filter<4, 64>(
            reinterpret_cast<int8_t*>(reordered_filter),
            reinterpret_cast<int8_t*>(args.filter_tensor->raw_ptr), co, ci, fh,
            fw, true, stream);
}
#endif

// vim: syntax=cpp.doxygen
