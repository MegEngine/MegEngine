/**
 * \file dnn/src/cuda/conv_bias/implicit_batched_gemm_float16_nchw_hmma.cpp
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

bool ConvBiasForwardImpl::AlgoFloat16NCHWHMMAImplicitBatchedGemm::is_available(
        const SizeArgs& args) const {
#define RETURN_IF_FALSE(stmt_) \
    if (!(stmt_))              \
        return false;
    RETURN_IF_FALSE(is_compute_capability_required(7, 0));
    RETURN_IF_FALSE(
            args.src_layout->is_contiguous() && args.dst_layout->is_contiguous());
    using Param = param::ConvBias;
    using Format = Param::Format;
    using Sparse = Param::Sparse;
    using Mode = Param::Mode;
    auto&& param = args.opr->param();
    auto&& fm = args.filter_meta;
    RETURN_IF_FALSE(
            param.format == Format::NCHW &&
            args.src_layout->dtype.enumv() == DTypeEnum::Float16 &&
            args.filter_layout->dtype.enumv() == DTypeEnum::Float16 &&
            args.dst_layout->dtype.enumv() == DTypeEnum::Float16);
    RETURN_IF_FALSE(
            args.bias_layout->ndim <= 0 ||
            (args.bias_layout->dtype.enumv() == DTypeEnum::Float16 &&
             check_bias_share_in_channel(*args.bias_layout, param.format)));
    RETURN_IF_FALSE(
            args.z_layout->ndim <= 0 ||
            args.z_layout->dtype.enumv() == DTypeEnum::Float16);
    RETURN_IF_FALSE(param.sparse == Sparse::GROUP);
    RETURN_IF_FALSE(param.mode == Mode::CROSS_CORRELATION);
    // check if channelwise convolution
    RETURN_IF_FALSE(fm.icpg == 1 && fm.ocpg == 1);
    const auto* op = get_cutlass_conv_op(
            args, ConvOperator::kFprop, ConvType::kDepthwiseConvolution, false, false);
    RETURN_IF_FALSE(op != nullptr);
    return true;
#undef RETURN_IF_FALSE
}

void ConvBiasForwardImpl::AlgoFloat16NCHWHMMAImplicitBatchedGemm::exec(
        const ExecArgs& args) const {
    auto&& param = args.opr->param();
    auto&& fm = args.filter_meta;
    size_t n = args.src_layout->operator[](0), hi = args.src_layout->operator[](2),
           wi = args.src_layout->operator[](3);
    size_t ho = args.dst_layout->operator[](2), wo = args.dst_layout->operator[](3);
    size_t co = fm.group;
    size_t ci = co;
    // check if channelwise convolution
    megdnn_assert(fm.icpg == 1 && fm.ocpg == 1);
    auto&& stream = cuda_stream(args.opr->handle());

    float alpha = 1.f;
    float beta = args.bias_layout->ndim > 0 ? 1.f : 0.f;
    void* bias_ptr = args.bias_layout->ndim > 0 ? args.bias_tensor->raw_ptr() : nullptr;
    float gamma = args.z_layout->ndim > 0 ? 1.f : 0.f;
    void* z_ptr = args.z_layout->ndim > 0 ? args.z_tensor->raw_ptr() : nullptr;

    // dummy parameters, used for quantization cases
    float theta = 0.f;
    float delta = 0.f;
    float threshold = 0.f;

    const auto* op = get_cutlass_conv_op(
            args, ConvOperator::kFprop, ConvType::kDepthwiseConvolution, false, false);

    UNPACK_CONV_PARAMETER(fm, param);
    MARK_USED_VAR
    execute_cutlass_conv_op(
            op, args.src_tensor->raw_ptr(), args.filter_tensor->raw_ptr(), bias_ptr,
            z_ptr, args.dst_tensor->raw_ptr(), nullptr, n, hi, wi, ci, co, fh, fw, ho,
            wo, ph, pw, sh, sw, dh, dw, &alpha, &beta, &gamma, &delta, &theta,
            &threshold, nullptr, stream, nullptr, fm.group);

    after_kernel_launch();
}

// vim: syntax=cpp.doxygen
