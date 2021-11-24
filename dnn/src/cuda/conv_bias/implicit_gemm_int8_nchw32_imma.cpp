/**
 * \file dnn/src/cuda/conv_bias/implicit_gemm_int8_nchw32_imma.cpp
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
bool ConvBiasForwardImpl::AlgoInt8NCHW32IMMAImplicitGemm::is_available(
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
    if (param.format != Format::NCHW32 && param.format != Format::NCHW32_NCHW4)
        return false;
    size_t n = args.src_layout->operator[](0), ci = args.src_layout->operator[](1) * 32,
           hi = args.src_layout->operator[](2), wi = args.src_layout->operator[](3);
    size_t ho = args.dst_layout->operator[](2), wo = args.dst_layout->operator[](3);
    size_t co;
    if (param.format == Format::NCHW32) {
        co = args.dst_layout->operator[](1) * 32;
    } else {
        megdnn_assert(param.format == Format::NCHW32_NCHW4);
        co = args.dst_layout->operator[](1) * 4;
    }
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
             filter_dtype.enumv() == DTypeEnum::QuantizedS8 &&
             bias_dtype.enumv() == DTypeEnum::QuantizedS32 &&
             dst_dtype.enumv() == DTypeEnum::QuantizedS8);
    // TODO: support dialtion
    available &= dh == 1 && dw == 1;
    // only support sm_75 or later, platform should have tensorcore int8
    // support
    available &= is_compute_capability_required(7, 5);
    // FIXME: too large filter size is not supported now
    size_t kMaxFilterPixels = 848 / (2 * m_algo_param.warp_k / 32) - 2;
    available &= fh * fw <= kMaxFilterPixels;

    bool use_conv_filter_unity_opt = (fh == 1 && fw == 1);
    bool without_shared_load = (param.format == Format::NCHW32);
    const auto* op = get_cutlass_conv_op(
            args, ConvOperator::kFprop, ConvType::kConvolution,
            use_conv_filter_unity_opt, without_shared_load);
    available &= (op != nullptr);

    return available;
}

WorkspaceBundle ConvBiasForwardImpl::AlgoInt8NCHW32IMMAImplicitGemm::
        get_workspace_bundle(dt_byte* raw_ptr, const SizeArgs& args) const {
    if (args.preprocessed_filter) {
        return WorkspaceBundle{raw_ptr, {}};
    } else {
        size_t ws_filter = args.filter_layout->span().dist_byte();
        return WorkspaceBundle{raw_ptr, {ws_filter}};
    }
}

size_t ConvBiasForwardImpl::AlgoInt8NCHW32IMMAImplicitGemm::get_workspace_in_bytes(
        const SizeArgs& args) const {
    return get_workspace_bundle(nullptr, args).total_size_in_bytes();
}

void ConvBiasForwardImpl::AlgoInt8NCHW32IMMAImplicitGemm::exec(
        const ExecArgs& args) const {
    using Format = Param::Format;
    auto&& param = args.opr->param();
    auto&& fm = args.filter_meta;
    size_t n = args.src_layout->operator[](0), ci = args.src_layout->operator[](1) * 32,
           hi = args.src_layout->operator[](2), wi = args.src_layout->operator[](3);
    size_t ho = args.dst_layout->operator[](2), wo = args.dst_layout->operator[](3);
    size_t co;
    bool trans_oc;
    if (param.format == Format::NCHW32) {
        co = args.dst_layout->operator[](1) * 32;
        trans_oc = true;
    } else {
        megdnn_assert(param.format == Format::NCHW32_NCHW4);
        co = args.dst_layout->operator[](1) * 4;
        trans_oc = false;
    }
    UNPACK_CONV_PARAMETER(fm, param);
    MARK_USED_VAR
    auto&& stream = cuda_stream(args.opr->handle());

    int8_t* filter_ptr = nullptr;
    if (args.preprocessed_filter == nullptr) {
        filter_ptr = reinterpret_cast<int8_t*>(args.workspace.raw_ptr);
        // filter: KCRS32 => CRSK32 and reorder oc
        cutlass_wrapper::reorder_ncxhwx_imma_filter<8, 32>(
                filter_ptr, reinterpret_cast<int8_t*>(args.filter_tensor->raw_ptr()),
                co, ci, fh, fw, trans_oc, stream);
    } else {
        filter_ptr = reinterpret_cast<int8_t*>(
                args.preprocessed_filter->tensors[0].raw_ptr());
    }

    float src_scale = args.src_layout->dtype.param<dtype::QuantizedS8>().scale,
          filter_scale = args.filter_layout->dtype.param<dtype::QuantizedS8>().scale,
          bias_scale = args.bias_layout->dtype.param<dtype::QuantizedS32>().scale,
          dst_scale = args.dst_layout->dtype.param<dtype::QuantizedS8>().scale;

    // \note these constants of cutlass epilogue will be passed to method
    // `execute_cutlass_conv_op` by pointer and interpreted as ElementCompute*,
    // a different dtype here results in undefined epilogue behaviors
    float alpha = src_scale * filter_scale / dst_scale, beta = bias_scale / dst_scale;
    int8_t* z_dev_ptr = nullptr;
    float gamma = 0.0;
    if (args.z_layout->ndim > 0) {
        z_dev_ptr = args.z_tensor->compatible_ptr<int8_t>();
        float z_scale = args.z_layout->dtype.param<dtype::QuantizedS8>().scale;
        gamma = z_scale / dst_scale;
    }
    float delta = 0.f, theta = 0.f, threshold = 0.f;
    bool use_conv_filter_unity_opt = (fh == 1 && fw == 1);
    bool without_shared_load = (param.format == Format::NCHW32);

    const auto* op = get_cutlass_conv_op(
            args, ConvOperator::kFprop, ConvType::kConvolution,
            use_conv_filter_unity_opt, without_shared_load);

    execute_cutlass_conv_op(
            op, args.src_tensor->raw_ptr(), filter_ptr, args.bias_tensor->raw_ptr(),
            z_dev_ptr, args.dst_tensor->raw_ptr(), nullptr, n, hi, wi, ci, co, fh, fw,
            ho, wo, ph, pw, sh, sw, dh, dw, &alpha, &beta, &gamma, &delta, &theta,
            &threshold, &dst_scale, stream);

    after_kernel_launch();
}

std::string ConvBiasForwardImpl::AlgoInt8NCHW32IMMAImplicitGemm::to_string(
        AlgoParam algo_param) {
    return ssprintf(
            "%uX%uX%u_%uX%uX%u_%u", algo_param.threadblock_m, algo_param.threadblock_n,
            algo_param.threadblock_k, algo_param.warp_m, algo_param.warp_n,
            algo_param.warp_k, algo_param.stage);
}

size_t ConvBiasForwardImpl::AlgoInt8NCHW32IMMAImplicitGemm::
        get_preprocess_workspace_in_bytes(const SizeArgs& args) const {
    return 0_z;
}

SmallVector<TensorLayout> ConvBiasForwardImpl::AlgoInt8NCHW32IMMAImplicitGemm::
        deduce_preprocessed_filter_layout(const SizeArgs& args) const {
    return {args.filter_layout->collapse_contiguous()};
}

void ConvBiasForwardImpl::AlgoInt8NCHW32IMMAImplicitGemm::exec_preprocess(
        const ExecArgs& args) const {
    using Format = Param::Format;
    auto&& param = args.opr->param();
    auto&& fm = args.filter_meta;
    size_t ci = args.src_layout->operator[](1) * 32;
    size_t co;
    bool trans_oc;
    if (param.format == Format::NCHW32) {
        co = args.dst_layout->operator[](1) * 32;
        trans_oc = true;
    } else {
        megdnn_assert(param.format == Format::NCHW32_NCHW4);
        co = args.dst_layout->operator[](1) * 4;
        trans_oc = false;
    }
    size_t fh = fm.spatial[0], fw = fm.spatial[1];

    cudaStream_t stream = cuda_stream(args.opr->handle());
    // filter: KCRS32 => CRSK32 and reorder oc
    cutlass_wrapper::reorder_ncxhwx_imma_filter<8, 32>(
            reinterpret_cast<int8_t*>(args.preprocessed_filter->tensors[0].raw_ptr()),
            reinterpret_cast<int8_t*>(args.filter_tensor->raw_ptr()), co, ci, fh, fw,
            trans_oc, stream);
}
#endif

// vim: syntax=cpp.doxygen
