/**
 * \file dnn/src/cuda/conv_bias/implicit_gemm_int4_int4_nchw64_imma.cpp
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
#include "src/common/conv_bias.h"
#include "src/cuda/conv_bias/cutlass_convolution_wrapper.cuh"
#include "src/cuda/convolution_helper/parameter.cuh"
#include "src/cuda/utils.h"

using namespace megdnn;
using namespace cuda;
using namespace convolution;

#if CUDA_VERSION >= 10020
bool ConvBiasForwardImpl::AlgoInt4Int4NCHW64IMMAImplicitGemm::is_available(
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

    if (args.src_layout->dtype.enumv() != DTypeEnum::QuantizedS4 ||
        args.filter_layout->dtype.enumv() != DTypeEnum::QuantizedS4 ||
        args.bias_layout->dtype.enumv() != DTypeEnum::QuantizedS32 ||
        args.dst_layout->dtype.enumv() != DTypeEnum::QuantizedS4)
        return false;

    if (!is_compute_capability_required(7, 5))
        return false;

    return true;
}

WorkspaceBundle
ConvBiasForwardImpl::AlgoInt4Int4NCHW64IMMAImplicitGemm::get_workspace_bundle(
        dt_byte* raw_ptr, const SizeArgs& args) const {
    if (args.preprocessed_filter) {
        return WorkspaceBundle{raw_ptr, {}};
    } else {
        size_t ws_filter = args.filter_layout->span().dist_byte();
        return WorkspaceBundle{raw_ptr, {ws_filter}};
    }
}

size_t
ConvBiasForwardImpl::AlgoInt4Int4NCHW64IMMAImplicitGemm::get_workspace_in_bytes(
        const SizeArgs& args) const {
    return get_workspace_bundle(nullptr, args).total_size_in_bytes();
}

void ConvBiasForwardImpl::AlgoInt4Int4NCHW64IMMAImplicitGemm::exec(
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
    auto&& stream = cuda_stream(args.opr->handle());

    int8_t* filter_ptr = nullptr;
    if (args.preprocessed_filter == nullptr) {
        filter_ptr = reinterpret_cast<int8_t*>(args.workspace.raw_ptr);
        // reformat filter from nchw64 to chwn64
        TensorLayout src{{co, ci / 64, fh, fw, 64}, dtype::QuantizedS4()};
        src.init_contiguous_stride();
        TensorLayout dst = src;
        dst.stride[0] = 64;
        dst.stride[1] = co * fh * fw * 64;
        dst.stride[2] = co * fw * 64;
        dst.stride[3] = co * 64;
        dst.stride[4] = 1;
        TensorND ts_src, ts_dst;
        ts_src.raw_ptr = args.filter_tensor->raw_ptr;
        ts_src.layout = src;
        ts_dst.raw_ptr = args.workspace.raw_ptr;
        ts_dst.layout = dst;
        auto&& transpose =
                args.opr->handle()->create_operator<RelayoutForward>();
        transpose->exec(ts_src, ts_dst);
    } else {
        filter_ptr = reinterpret_cast<int8_t*>(
                args.preprocessed_filter->tensors[0].raw_ptr);
    }

    ConvParam kern_param;
    kern_param.n = n, kern_param.co = co, kern_param.ci = ci,
    kern_param.hi = hi, kern_param.wi = wi, kern_param.ho = ho,
    kern_param.wo = wo, kern_param.ph = ph, kern_param.pw = pw,
    kern_param.sh = sh, kern_param.sw = sw, kern_param.fh = fh,
    kern_param.fw = fw;

    float src_scale = args.src_layout->dtype.param<dtype::QuantizedS4>().scale,
          filter_scale =
                  args.filter_layout->dtype.param<dtype::QuantizedS4>().scale,
          bias_scale =
                  args.bias_layout->dtype.param<dtype::QuantizedS32>().scale,
          dst_scale = args.dst_layout->dtype.param<dtype::QuantizedS4>().scale;

    float alpha = src_scale * filter_scale / dst_scale,
          beta = bias_scale / dst_scale;

    int8_t* z_dev_ptr = nullptr;
    float gamma = 0.f;
    if (args.z_layout->ndim > 0) {
        z_dev_ptr = reinterpret_cast<int8_t*>(args.z_tensor->raw_ptr);
        float z_scale = args.z_layout->dtype.param<dtype::QuantizedS4>().scale;
        gamma = z_scale / dst_scale;
    }

    uint32_t nonlinear_mode = static_cast<uint32_t>(param.nonlineMode);

    cutlass_wrapper::do_conv_bias_int4_int4_implicit_gemm_imma_ncdiv64hw64<
            true>(
            reinterpret_cast<int8_t*>(args.src_tensor->raw_ptr), filter_ptr,
            args.bias_tensor->compatible_ptr<int32_t>(), z_dev_ptr,
            reinterpret_cast<int8_t*>(args.dst_tensor->raw_ptr), nullptr,
            kern_param, nonlinear_mode, alpha, beta, gamma, dst_scale,
            cutlass_wrapper::GemmCoord{m_algo_param.threadblock_m,
                                       m_algo_param.threadblock_n,
                                       m_algo_param.threadblock_k},
            cutlass_wrapper::GemmCoord{m_algo_param.warp_m, m_algo_param.warp_n,
                                       m_algo_param.warp_k},
            stream);
}

std::string ConvBiasForwardImpl::AlgoInt4Int4NCHW64IMMAImplicitGemm::to_string(
        AlgoParam algo_param) {
    return ssprintf("%uX%uX%u_%uX%uX%u", algo_param.threadblock_m,
                    algo_param.threadblock_n, algo_param.threadblock_k,
                    algo_param.warp_m, algo_param.warp_n, algo_param.warp_k);
}

size_t ConvBiasForwardImpl::AlgoInt4Int4NCHW64IMMAImplicitGemm::
        get_preprocess_workspace_in_bytes(const SizeArgs& args) const {
    return 0_z;
}

SmallVector<TensorLayout> ConvBiasForwardImpl::
        AlgoInt4Int4NCHW64IMMAImplicitGemm::deduce_preprocessed_filter_layout(
                const SizeArgs& args) const {
    return {args.filter_layout->collapse_contiguous()};
}

void ConvBiasForwardImpl::AlgoInt4Int4NCHW64IMMAImplicitGemm::exec_preprocess(
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
    TensorLayout src{{co, ci / 64, fh, fw, 64}, dtype::QuantizedS4()};
    src.init_contiguous_stride();
    TensorLayout dst = src;
    dst.stride[0] = 64;
    dst.stride[1] = co * fh * fw * 64;
    dst.stride[2] = co * fw * 64;
    dst.stride[3] = co * 64;
    dst.stride[4] = 1;
    TensorND ts_src, ts_dst;
    ts_src.raw_ptr = args.filter_tensor->raw_ptr;
    ts_src.layout = src;
    ts_dst.raw_ptr = args.preprocessed_filter->tensors[0].raw_ptr;
    ts_dst.layout = dst;
    auto&& transpose = args.opr->handle()->create_operator<RelayoutForward>();
    transpose->exec(ts_src, ts_dst);
}
#endif

// vim: syntax=cpp.doxygen
