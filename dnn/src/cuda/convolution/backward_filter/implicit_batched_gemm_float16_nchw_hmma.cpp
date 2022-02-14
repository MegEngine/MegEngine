/**
 * \file
 * dnn/src/cuda/convolution/backward_filter/implicit_batched_gemm_float16_nchw_hmma.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/cuda/convolution/backward_filter/algo.h"
#include "src/cuda/cutlass/singleton.h"
#include "src/cuda/utils.h"

using namespace megdnn;
using namespace cuda;
using namespace cutlass::library;

const void* ConvolutionBackwardFilterImpl::AlgoFloat16NCHWHMMAImplicitBatchedGemm::
        get_available_op(const SizeArgs& args) const {
    auto get_alignment = [](const TensorLayout& layout) {
        int alignment = 0;
        int width = layout.dtype.size(layout[3]);
        for (int candidate : {16, 4, 2}) {
            if (width % candidate == 0) {
                alignment = candidate;
                break;
            }
        }
        alignment /= layout.dtype.size(1);
        return alignment;
    };
    int alignment_src = get_alignment(*args.src_layout);
    int alignment_diff = get_alignment(*args.diff_layout);
    megdnn_assert(alignment_src >= 1 && alignment_diff >= 1);
    NumericTypeID accumulator_dtype =
            args.opr->param().compute_mode == param::Convolution::ComputeMode::DEFAULT
                    ? NumericTypeID::kF16
                    : NumericTypeID::kF32;
    ConvolutionKey key{
            cutlass::conv::Operator::kWgrad,
            NumericTypeID::kF16,        // src tensor data type
            LayoutTypeID::kTensorNCHW,  // src tensor layout
            NumericTypeID::kF16,        // diff tensor data type
            LayoutTypeID::kTensorNCHW,  // diff tensor layout
            NumericTypeID::kF32,        // grad tensor data type
            LayoutTypeID::kTensorNCHW,  // grad tensor layout
            NumericTypeID::kF32,        // dummy argument, not used.
            LayoutTypeID::kTensorNCHW,  // dummy argument, not used
            accumulator_dtype,
            cutlass::conv::ConvType::kDepthwiseConvolution,
            m_algo_param.threadblock_m,
            m_algo_param.threadblock_n,
            m_algo_param.threadblock_k,
            m_algo_param.warp_m,
            m_algo_param.warp_n,
            m_algo_param.warp_k,
            m_algo_param.instruction_m,
            m_algo_param.instruction_n,
            m_algo_param.instruction_k,
            cutlass::epilogue::EpilogueType::kLinearCombination,  // no bias
            m_algo_param.stage,
            cutlass::conv::SpecialOptimizeDesc::NONE,
            alignment_src,
            alignment_diff,
            true};
    return (void*)Singleton::get().operation_table.find_op(key);
}

bool ConvolutionBackwardFilterImpl::AlgoFloat16NCHWHMMAImplicitBatchedGemm::
        is_available(const SizeArgs& args) const {
#define RETURN_IF_FALSE(stmt_) \
    if (!(stmt_))              \
        return false;
    RETURN_IF_FALSE(is_compute_capability_required(7, 0));
    RETURN_IF_FALSE(
            args.src_layout->is_contiguous() && args.diff_layout->is_contiguous() &&
            args.grad_layout->is_contiguous());
    using Param = param::Convolution;
    using Format = Param::Format;
    using Sparse = Param::Sparse;
    using Mode = Param::Mode;
    using ComputeMode = Param::ComputeMode;
    auto&& param = args.opr->param();
    auto&& fm = args.grad_filter_meta;
    RETURN_IF_FALSE(param.compute_mode == ComputeMode::FLOAT32);
    RETURN_IF_FALSE(
            param.format == Format::NCHW &&
            args.src_layout->dtype.enumv() == DTypeEnum::Float16 &&
            args.diff_layout->dtype.enumv() == DTypeEnum::Float16 &&
            args.grad_layout->dtype.enumv() == DTypeEnum::Float16);
    RETURN_IF_FALSE(param.sparse == Sparse::GROUP);
    RETURN_IF_FALSE(param.mode == Mode::CROSS_CORRELATION);
    // check if channelwise convolution
    RETURN_IF_FALSE(fm.icpg == 1 && fm.ocpg == 1);
    RETURN_IF_FALSE(param.dilate_h == 1 && param.dilate_w == 1);
    const auto* op = get_available_op(args);
    RETURN_IF_FALSE(op != nullptr);
    return true;
#undef RETURN_IF_FALSE
}

size_t ConvolutionBackwardFilterImpl::AlgoFloat16NCHWHMMAImplicitBatchedGemm::
        get_workspace_in_bytes(const SizeArgs& args) const {
    auto layout = *args.grad_layout;
    // modify data type
    layout.modify_dtype_inplace(dtype::Float32());
    return layout.span().dist_byte();
}

void ConvolutionBackwardFilterImpl::AlgoFloat16NCHWHMMAImplicitBatchedGemm::exec(
        const ExecArgs& args) const {
    auto&& param = args.opr->param();
    auto&& fm = args.grad_filter_meta;
    int hi = args.src_layout->operator[](2), wi = args.src_layout->operator[](3);
    int n = args.diff_layout->operator[](0), ho = args.diff_layout->operator[](2),
        wo = args.diff_layout->operator[](3);
    int co = fm.group, ci = co, groups = co;
    int fh = fm.spatial[0], fw = fm.spatial[1];
    int sh = fm.stride[0], sw = fm.stride[1];
    int ph = fm.padding[0], pw = fm.padding[1];
    int dh = param.dilate_h, dw = param.dilate_w;

    // check if channelwise convolution
    megdnn_assert(fm.icpg == 1 && fm.ocpg == 1);
    auto&& stream = cuda_stream(args.opr->handle());

    float alpha = 1.f;
    float beta = 0.f;

    const Operation* op = (const Operation*)get_available_op(args);

    cutlass::conv::Conv2dProblemSize problem_size{
            n,      hi, wi, ci, co, fh, fw, ho,
            wo,     ph, pw, sh, sw, dh, dw, cutlass::conv::Mode::kCrossCorrelation,
            1,       // split k slices, always 1
            groups,  // groups
    };

    cutlass::library::ConvolutionArguments conv_args{
            problem_size,
            args.src_tensor->raw_ptr(),
            args.diff_tensor->raw_ptr(),
            nullptr,
            nullptr,
            args.workspace.raw_ptr,
            &alpha,
            &beta,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr,
            nullptr};

    cutlass_check(op->run(&conv_args, nullptr, stream));

    after_kernel_launch();

    auto&& typecvt = args.opr->handle()->create_operator<TypeCvt>();
    auto f32_grad_layout = *args.grad_layout;
    // modify data type
    f32_grad_layout.modify_dtype_inplace(dtype::Float32());
    TensorND src{args.workspace.raw_ptr, f32_grad_layout},
            dst{args.grad_tensor->raw_ptr(), *args.grad_layout};
    typecvt->exec(src, dst);
}

// vim: syntax=cpp.doxygen
