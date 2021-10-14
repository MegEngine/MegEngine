/**
 * \file dnn/src/cuda/conv_bias/implicit_gemm_int8_nchw4_dp4a.cpp
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
#include "src/cuda/convolution_helper/parameter.cuh"
#include "src/cuda/utils.h"

using namespace megdnn;
using namespace cuda;

bool ConvBiasForwardImpl::AlgoInt8NCHW4DotProdImplicitGemm::is_available(
        const SizeArgs& args) const {
    if (!args.src_layout->is_contiguous() || !args.dst_layout->is_contiguous()) {
        return false;
    }
    if (args.bias_layout->ndim <= 0)
        return false;

    using Param = param::ConvBias;
    using Format = Param::Format;
    using Sparse = Param::Sparse;
    using Mode = Param::Mode;
    bool available = true;
    auto&& param = args.opr->param();
    auto&& fm = args.filter_meta;
    if (!check_bias_share_in_channel(*(args.bias_layout), param.format))
        return false;
    bool valid_format = param.format == Format::NCHW4_NCHW32 &&
                        m_algo_param.threadblock_m % 32 == 0;
    valid_format |= param.format == Format::NCHW4_NCHW &&
                    args.bias_layout->dtype.enumv() == DTypeEnum::Float32 &&
                    args.dst_layout->dtype.enumv() == DTypeEnum::Float32;
    valid_format |= param.format == Format::NCHW4_NHWC &&
                    args.bias_layout->dtype.enumv() == DTypeEnum::QuantizedS32 &&
                    (args.dst_layout->dtype.enumv() == DTypeEnum::QuantizedS4 ||
                     args.dst_layout->dtype.enumv() == DTypeEnum::Quantized4Asymm);
    valid_format |= param.format == Format::NCHW4;
    if (!valid_format)
        return false;
    size_t n = args.src_layout->operator[](0), ci = args.src_layout->operator[](1) * 4,
           hi = args.src_layout->operator[](2), wi = args.src_layout->operator[](3);
    size_t co;
    size_t dst_spatial_pos;
    if (param.format == Format::NCHW4) {
        co = args.dst_layout->operator[](1) * 4;
        dst_spatial_pos = 2;
    } else if (param.format == Format::NCHW4_NCHW) {
        co = args.dst_layout->operator[](1);
        dst_spatial_pos = 2;
    } else if (param.format == Format::NCHW4_NHWC) {
        co = args.dst_layout->operator[](3);
        dst_spatial_pos = 1;
    } else {
        megdnn_assert(param.format == Format::NCHW4_NCHW32);
        dst_spatial_pos = 2;
        co = args.dst_layout->operator[](1) * 32;
    }
    size_t ho = args.dst_layout->operator[](dst_spatial_pos),
           wo = args.dst_layout->operator[](dst_spatial_pos + 1);
    UNPACK_CONV_PARAMETER(fm, param);
    MARK_USED_VAR
    // TODO support group conv
    available &= param.sparse == Sparse::DENSE;
    // mode must be cross correlation
    available &= param.mode == Mode::CROSS_CORRELATION;
    // check data type
    auto src_dtype = args.src_layout->dtype, filter_dtype = args.filter_layout->dtype,
         bias_dtype = args.bias_layout->dtype, dst_dtype = args.dst_layout->dtype;
    available &=
            (src_dtype.enumv() == DTypeEnum::QuantizedS8 &&
             filter_dtype.enumv() == DTypeEnum::QuantizedS8);
    available &= (bias_dtype.enumv() == DTypeEnum::QuantizedS32 &&
                  (dst_dtype.enumv() == DTypeEnum::QuantizedS8 ||
                   dst_dtype.enumv() == DTypeEnum::QuantizedS4 ||
                   dst_dtype.enumv() == DTypeEnum::Quantized4Asymm)) ||
                 (bias_dtype.enumv() == DTypeEnum::Float32 &&
                  dst_dtype.enumv() == DTypeEnum::Float32);
    // TODO: support dialtion
    available &= dh == 1 && dw == 1;
    // only support sm_61 or later, platform should have fast native int8
    // support
    available &= is_compute_capability_required(6, 1);
    // FIXME: too large filter size is not supported now
    size_t kMaxFilterPixels = 848 / (2 * m_algo_param.warp_k / 4) - 2;
    available &= fh * fw <= kMaxFilterPixels;

    bool use_conv_filter_unity_opt = (fh == 1 && fw == 1);
    bool without_shared_load = false;
    const auto* op = get_cutlass_conv_op(
            args, ConvOperator::kFprop, ConvType::kConvolution,
            use_conv_filter_unity_opt, without_shared_load);
    available &= (op != nullptr);

    return available;
}

WorkspaceBundle ConvBiasForwardImpl::AlgoInt8NCHW4DotProdImplicitGemm::
        get_workspace_bundle(dt_byte* raw_ptr, const SizeArgs& args) const {
    if (args.preprocessed_filter) {
        return WorkspaceBundle{raw_ptr, {}};
    } else {
        size_t ws_filter = args.filter_layout->span().dist_byte();
        return WorkspaceBundle{raw_ptr, {ws_filter}};
    }
}

size_t ConvBiasForwardImpl::AlgoInt8NCHW4DotProdImplicitGemm::get_workspace_in_bytes(
        const SizeArgs& args) const {
    return get_workspace_bundle(nullptr, args).total_size_in_bytes();
}

void ConvBiasForwardImpl::AlgoInt8NCHW4DotProdImplicitGemm::exec(
        const ExecArgs& args) const {
    using Format = Param::Format;
    auto&& param = args.opr->param();
    auto&& fm = args.filter_meta;
    size_t n = args.src_layout->operator[](0), ci = args.src_layout->operator[](1) * 4,
           hi = args.src_layout->operator[](2), wi = args.src_layout->operator[](3);
    size_t co, dst_spatial_pos;
    if (param.format == Format::NCHW4) {
        co = args.dst_layout->operator[](1) * 4;
        dst_spatial_pos = 2;
    } else if (param.format == Format::NCHW4_NCHW) {
        co = args.dst_layout->operator[](1);
        dst_spatial_pos = 2;
    } else if (param.format == Format::NCHW4_NHWC) {
        co = args.dst_layout->operator[](3);
        dst_spatial_pos = 1;
    } else {
        megdnn_assert(param.format == Format::NCHW4_NCHW32);
        dst_spatial_pos = 2;
        co = args.dst_layout->operator[](1) * 32;
    }
    size_t ho = args.dst_layout->operator[](dst_spatial_pos),
           wo = args.dst_layout->operator[](dst_spatial_pos + 1);
    UNPACK_CONV_PARAMETER(fm, param);
    MARK_USED_VAR
    auto&& stream = cuda_stream(args.opr->handle());

    int8_t* filter_ptr = nullptr;
    if (args.preprocessed_filter == nullptr) {
        filter_ptr = reinterpret_cast<int8_t*>(args.workspace.raw_ptr);
        // reformat filter from nchw4 to chwn4
        TensorLayout src{{co, ci / 4 * fh * fw}, dtype::Int32()};
        src.init_contiguous_stride();
        TensorLayout dst = src;
        dst.stride[0] = 1, dst.stride[1] = dst[0];
        TensorND ts_src, ts_dst;
        ts_src.raw_ptr = args.filter_tensor->raw_ptr;
        ts_src.layout = src;
        ts_dst.raw_ptr = args.workspace.raw_ptr;
        ts_dst.layout = dst;
        auto&& transpose = args.opr->handle()->create_operator<RelayoutForward>();
        transpose->exec(ts_src, ts_dst);
    } else {
        filter_ptr =
                reinterpret_cast<int8_t*>(args.preprocessed_filter->tensors[0].raw_ptr);
    }

    float src_scale = args.src_layout->dtype.param<dtype::QuantizedS8>().scale,
          filter_scale = args.filter_layout->dtype.param<dtype::QuantizedS8>().scale;

    // \note these constants of cutlass epilogue will be passed to method
    // `execute_cutlass_conv_op` by pointer and interpreted as ElementCompute*,
    // a different dtype here results in undefined epilogue behaviors
    float alpha = src_scale * filter_scale;
    float beta = 1.f;
    float dst_scale = 1.f;
    float gamma = 0.f;
    float theta = 0.f;
    if (args.dst_layout->dtype.enumv() == DTypeEnum::Quantized4Asymm) {
        theta = args.dst_layout->dtype.param<dtype::Quantized4Asymm>().zero_point;
    }
    if (args.bias_layout->dtype.enumv() == DTypeEnum::QuantizedS32) {
        megdnn_assert(args.dst_layout->dtype.category() == DTypeCategory::QUANTIZED);
        float bias_scale = args.bias_layout->dtype.param<dtype::QuantizedS32>().scale;
        dst_scale = get_scale(args.dst_layout->dtype);
        alpha /= dst_scale, beta = bias_scale / dst_scale;
    }
    float delta = 0.f;
    void* z_ptr = nullptr;
    if (args.z_layout->ndim > 0) {
        z_ptr = args.z_tensor->raw_ptr;
        gamma = 1.f;
        if (args.z_layout->dtype.category() == DTypeCategory::QUANTIZED) {
            megdnn_assert(
                    args.dst_layout->dtype.category() == DTypeCategory::QUANTIZED);
            float z_scale = get_scale(args.z_layout->dtype);
            gamma = z_scale / dst_scale;
        }
        if (args.z_layout->dtype.enumv() == DTypeEnum::Quantized4Asymm) {
            uint8_t z_zero =
                    args.z_layout->dtype.param<dtype::Quantized4Asymm>().zero_point;
            delta = -z_zero * gamma;
        }
    }
    float threshold = 0.f;
    bool use_conv_filter_unity_opt = (fh == 1 && fw == 1);
    bool without_shared_load = false;

    const auto* op = get_cutlass_conv_op(
            args, ConvOperator::kFprop, ConvType::kConvolution,
            use_conv_filter_unity_opt, without_shared_load);

    execute_cutlass_conv_op(
            op, args.src_tensor->raw_ptr, filter_ptr, args.bias_tensor->raw_ptr, z_ptr,
            args.dst_tensor->raw_ptr, nullptr, n, hi, wi, ci, co, fh, fw, ho, wo, ph,
            pw, sh, sw, dh, dw, &alpha, &beta, &gamma, &delta, &theta, &threshold,
            &dst_scale, stream);

    after_kernel_launch();
}

size_t ConvBiasForwardImpl::AlgoInt8NCHW4DotProdImplicitGemm::
        get_preprocess_workspace_in_bytes(const SizeArgs& args) const {
    return 0_z;
}

SmallVector<TensorLayout> ConvBiasForwardImpl::AlgoInt8NCHW4DotProdImplicitGemm::
        deduce_preprocessed_filter_layout(const SizeArgs& args) const {
    return {args.filter_layout->collapse_contiguous()};
}

void ConvBiasForwardImpl::AlgoInt8NCHW4DotProdImplicitGemm::exec_preprocess(
        const ExecArgs& args) const {
    using Format = Param::Format;
    auto&& param = args.opr->param();
    auto&& fm = args.filter_meta;
    size_t n = args.src_layout->operator[](0), ci = args.src_layout->operator[](1) * 4,
           hi = args.src_layout->operator[](2), wi = args.src_layout->operator[](3);
    size_t co, dst_spatial_pos;
    if (param.format == Format::NCHW4) {
        co = args.dst_layout->operator[](1) * 4;
        dst_spatial_pos = 2;
    } else if (param.format == Format::NCHW4_NCHW) {
        co = args.dst_layout->operator[](1);
        dst_spatial_pos = 2;
    } else if (param.format == Format::NCHW4_NHWC) {
        co = args.dst_layout->operator[](3);
        dst_spatial_pos = 1;
    } else {
        megdnn_assert(param.format == Format::NCHW4_NCHW32);
        dst_spatial_pos = 2;
        co = args.dst_layout->operator[](1) * 32;
    }
    size_t ho = args.dst_layout->operator[](dst_spatial_pos),
           wo = args.dst_layout->operator[](dst_spatial_pos + 1);
    UNPACK_CONV_PARAMETER(fm, param);
    MARK_USED_VAR
    TensorLayout src{{co, ci / 4 * fh * fw}, dtype::Int32()};
    src.init_contiguous_stride();
    TensorLayout dst = src;
    dst.stride[0] = 1, dst.stride[1] = dst[0];
    TensorND ts_src, ts_dst;
    ts_src.raw_ptr = args.filter_tensor->raw_ptr;
    ts_src.layout = src;
    ts_dst.raw_ptr = args.preprocessed_filter->tensors[0].raw_ptr;
    ts_dst.layout = dst;
    auto&& transpose = args.opr->handle()->create_operator<RelayoutForward>();
    transpose->exec(ts_src, ts_dst);
}

// vim: syntax=cpp.doxygen
