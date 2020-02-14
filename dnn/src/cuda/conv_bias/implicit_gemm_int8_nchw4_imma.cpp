/**
 * \file dnn/src/cuda/conv_bias/implicit_gemm_int8_nchw4_imma.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./algo.h"
#include "src/cuda/utils.h"
#include "src/cuda/convolution_helper/bias_visitor.cuh"

using namespace megdnn;
using namespace cuda;

#if CUDA_VERSION >= 10000
bool ConvBiasForwardImpl::AlgoInt8NCHW4IMMAImplicitGemm::is_available(
        const SizeArgs& args) const {
    if (args.bias_layout->ndim <= 0)
        return false;

    using Param = param::ConvBias;
    using Format = Param::Format;
    using Sparse = Param::Sparse;
    using Mode = Param::Mode;
    bool available = true;
    auto&& param = args.opr->param();
    auto&& fm = args.filter_meta;
    if (!conv_bias::check_bias_share_in_channel(*(args.bias_layout),
                                                param.format))
        return false;
    if (param.format != Format::NCHW4)
        return false;
    UNPACK_CONV_BIAS_NCHW4_PARAM(*(args.src_layout), fm, *(args.dst_layout),
                                 param);
    // TODO support group conv
    available &= param.sparse == Sparse::DENSE;
    // mode must be cross correlation
    available &= param.mode == Mode::CROSS_CORRELATION;
    // check data type
    auto src_dtype = args.src_layout->dtype,
         filter_dtype = args.filter_layout->dtype,
         bias_dtype = args.bias_layout->dtype,
         dst_dtype = args.dst_layout->dtype;
    available &= (src_dtype.enumv() == DTypeEnum::QuantizedS8 &&
                  filter_dtype.enumv() == DTypeEnum::QuantizedS8 &&
                  bias_dtype.enumv() == DTypeEnum::QuantizedS32 &&
                  dst_dtype.enumv() == DTypeEnum::QuantizedS8);
    // check layout
    available &= (ci % 16 == 0);
    // TODO: support dialtion
    available &= dh == 1 && dw == 1;
    // only support sm_75 or later, platform should have tensorcore int8
    // support
    available &= is_compute_capability_required(7, 5);
    return available;
}

WorkspaceBundle
ConvBiasForwardImpl::AlgoInt8NCHW4IMMAImplicitGemm::get_workspace_bundle(
        dt_byte* raw_ptr, const SizeArgs& args) const {
    size_t ws_size_src = args.src_layout->span().dist_byte();
    size_t ws_size_filter = args.filter_layout->span().dist_byte();
    size_t ws_size_dst = args.dst_layout->span().dist_byte();
    if (args.z_layout->ndim > 0) {
        size_t ws_size_z = args.z_layout->span().dist_byte();
        return WorkspaceBundle{
                raw_ptr, {ws_size_src, ws_size_filter, ws_size_dst, ws_size_z}};
    }
    return WorkspaceBundle{raw_ptr, {ws_size_src, ws_size_filter, ws_size_dst}};
}

size_t
ConvBiasForwardImpl::AlgoInt8NCHW4IMMAImplicitGemm::get_workspace_in_bytes(
        const SizeArgs& args) const {
    return get_workspace_bundle(nullptr, args).total_size_in_bytes();
}

void ConvBiasForwardImpl::AlgoInt8NCHW4IMMAImplicitGemm::exec(
        const ExecArgs& args) const {
    using Format = Param::Format;
    auto&& param = args.opr->param();
    auto&& fm = args.filter_meta;
    UNPACK_CONV_BIAS_NCHW4_PARAM(*(args.src_layout), fm, *(args.dst_layout),
                                 param);
    auto ws = get_workspace_bundle(args.workspace.raw_ptr, args);
    auto ws_src = ws.get(0);
    auto ws_filter = ws.get(1);
    auto ws_dst = ws.get(2);
    auto&& stream = cuda_stream(args.opr->handle());

    // reformat src from nchw4 to chwn4
    {
        TensorLayout src{{n, ci / 4 * hi * wi}, dtype::Int32()};
        src.init_contiguous_stride();
        TensorLayout dst = src;
        dst.stride[0] = 1, dst.stride[1] = dst[0];
        TensorND ts_src, ts_dst;
        ts_src.raw_ptr = args.src_tensor->raw_ptr;
        ts_src.layout = src;
        ts_dst.raw_ptr = ws_src;
        ts_dst.layout = dst;
        auto&& transpose =
                args.opr->handle()->create_operator<RelayoutForward>();
        transpose->exec(ts_src, ts_dst);
    }
    
    // reformat filter from nchw4 to chwn4
    {
        TensorLayout src{{co, ci / 4 * fh * fw}, dtype::Int32()};
        src.init_contiguous_stride();
        TensorLayout dst = src;
        dst.stride[0] = 1, dst.stride[1] = dst[0];
        TensorND ts_src, ts_dst;
        ts_src.raw_ptr = args.filter_tensor->raw_ptr;
        ts_src.layout = src;
        ts_dst.raw_ptr = ws_filter;
        ts_dst.layout = dst;
        auto&& transpose =
                args.opr->handle()->create_operator<RelayoutForward>();
        transpose->exec(ts_src, ts_dst);
    }

    convolution::ConvParam kern_param;
    kern_param.n = n, kern_param.co = co, kern_param.ci = ci,
    kern_param.hi = hi, kern_param.wi = wi, kern_param.ho = ho,
    kern_param.wo = wo, kern_param.ph = ph, kern_param.pw = pw,
    kern_param.sh = sh, kern_param.sw = sw, kern_param.fh = fh,
    kern_param.fw = fw;

    float src_scale = args.src_layout->dtype.param<dtype::QuantizedS8>().scale,
          filter_scale =
                  args.filter_layout->dtype.param<dtype::QuantizedS8>().scale,
          bias_scale =
                  args.bias_layout->dtype.param<dtype::QuantizedS32>().scale,
          dst_scale = args.dst_layout->dtype.param<dtype::QuantizedS8>().scale;
    float alpha = src_scale * filter_scale / dst_scale,
          beta = bias_scale / dst_scale;

    // process z
    int8_t* z_dev_ptr = nullptr;
    float gamma = 1.f;
    if (args.z_layout->ndim > 0) {
        auto ws_z = ws.get(3);

        TensorLayout src{{n, co / 4 * ho * wo}, dtype::Int32()};
        src.init_contiguous_stride();
        TensorLayout dst = src;
        dst.stride[0] = 1, dst.stride[1] = dst[0];
        TensorND ts_src, ts_dst;
        ts_src.raw_ptr = args.z_tensor->raw_ptr;
        ts_src.layout = src;
        ts_dst.raw_ptr = ws_z;
        ts_dst.layout = dst;
        auto&& transpose =
                args.opr->handle()->create_operator<RelayoutForward>();
        transpose->exec(ts_src, ts_dst);
        z_dev_ptr = reinterpret_cast<int8_t*>(ws_z);
        float z_scale = args.z_layout->dtype.param<dtype::QuantizedS8>().scale;
        gamma = z_scale / dst_scale;
    }

    convolution::PerChannelBiasVisitor bias_visitor;
    bias_visitor.bias = args.bias_tensor->compatible_ptr<int32_t>();
    ConvBiasForwardImpl::AlgoInt8CHWN4IMMAImplicitGemm::dispatch_nonlinear_mode<
            convolution::PerChannelBiasVisitor>(
            reinterpret_cast<int8_t*>(ws_src),
            reinterpret_cast<int8_t*>(ws_filter), bias_visitor, z_dev_ptr,
            reinterpret_cast<int8_t*>(ws_dst), kern_param, alpha, beta, gamma,
            dst_scale, stream, param.nonlineMode, m_mma_tile_size);

    // reformat chwn4 to nchw4
    {
        TensorLayout src{{co / 4 * ho * wo, n}, dtype::Int32()};
        src.init_contiguous_stride();
        TensorLayout dst = src;
        dst.stride[0] = 1, dst.stride[1] = dst[0];
        TensorND ts_src, ts_dst;
        ts_src.raw_ptr = ws_dst;
        ts_src.layout = src;
        ts_dst.raw_ptr = args.dst_tensor->raw_ptr;
        ts_dst.layout = dst;
        auto&& transpose =
                args.opr->handle()->create_operator<RelayoutForward>();
        transpose->exec(ts_src, ts_dst);
    }
}
#endif

// vim: syntax=cpp.doxygen
