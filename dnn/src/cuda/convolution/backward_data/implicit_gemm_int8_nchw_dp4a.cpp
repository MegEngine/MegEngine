/**
 * \file
 * dnn/src/cuda/convolution/backward_data/implicit_gemm_int8_nchw4_dp4a.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/cuda/convolution/backward_data/algo.h"
#include "src/cuda/convolution_helper/parameter.cuh"
#include "src/cuda/cutlass/singleton.h"
#include "src/cuda/utils.h"

using namespace megdnn;
using namespace cuda;

const void* ConvolutionBackwardDataImpl::AlgoInt8NCHWDotProdImplicitGemm::
        get_available_op(const SizeArgs& args) const {
    using namespace cutlass::library;
    auto&& fm = args.filter_meta;
    size_t sh = fm.stride[0], sw = fm.stride[1];
    cutlass::conv::SpecialOptimizeDesc special_optimization =
            (sh == 2 && sw == 2)
                    ? cutlass::conv::SpecialOptimizeDesc::DECONV_DOUBLE_UPSAMPLING
                    : cutlass::conv::SpecialOptimizeDesc::NONE;
    // only use 16x64x8_16x64x8_2stages impl
    ConvolutionKey key{
            cutlass::conv::Operator::kDgrad,
            NumericTypeID::kS8,
            LayoutTypeID::kTensorNC4HW4,
            NumericTypeID::kS8,
            LayoutTypeID::kTensorK4RSC4,
            NumericTypeID::kS8,
            LayoutTypeID::kTensorNC4HW4,
            NumericTypeID::kS32,
            LayoutTypeID::kTensorNC4HW4,
            cutlass::conv::ConvType::kConvolution,
            16,
            64,
            8,
            16,
            64,
            8,
            1,
            1,
            4,
            cutlass::epilogue::EpilogueType::kBiasAddLinearCombinationClamp,
            2,
            special_optimization,
            false};
    return (void*)Singleton::get().operation_table.find_op(key);
}

bool ConvolutionBackwardDataImpl::AlgoInt8NCHWDotProdImplicitGemm::is_available(
        const SizeArgs& args) const {
    auto&& fm = args.filter_meta;
    if (fm.format != Param::Format::NCHW)
        return false;

    if (!args.grad_layout->is_contiguous() || !args.diff_layout->is_contiguous()) {
        return false;
    }

    bool available = true;

    auto src_dtype = args.diff_layout->dtype, filter_dtype = args.filter_layout->dtype,
         dst_dtype = args.grad_layout->dtype;

    available &=
            (src_dtype.enumv() == DTypeEnum::QuantizedS8 &&
             filter_dtype.enumv() == DTypeEnum::QuantizedS8 &&
             dst_dtype.enumv() == DTypeEnum::QuantizedS8);
    // TODO support group deconv int8
    available &= (fm.group == 1);
    // ic and oc must be multiples of 4
    available &= ((fm.group * fm.icpg) % 4 == 0 && (fm.group * fm.ocpg) % 4 == 0);
    // mode must be cross correlation
    available &= !fm.should_flip;
    // mode must be 2D
    available &= fm.spatial_ndim == 2;
    // TODO: support dialtion
    available &= (fm.dilation[0] == 1 && fm.dilation[1] == 1);
    // FIXME: too large filter size is not supported now
    available &= fm.spatial[0] * fm.spatial[1] <= (848 / (2 * 8 / 4) - 2);

    available &= (get_available_op(args) != nullptr);

    // only support sm_61 or later, platform should have fast native int8
    // support
    available &= is_compute_capability_required(6, 1);

    return available;
}

WorkspaceBundle ConvolutionBackwardDataImpl::AlgoInt8NCHWDotProdImplicitGemm::
        get_workspace_bundle(dt_byte* raw_ptr, const SizeArgs& args) const {
    size_t ws_filter = args.filter_layout->span().dist_byte();
    size_t ws_diff = args.diff_layout->span().dist_byte();
    size_t ws_grad = args.grad_layout->span().dist_byte();
    return WorkspaceBundle{raw_ptr, {ws_filter, ws_diff, ws_grad}};
}

size_t ConvolutionBackwardDataImpl::AlgoInt8NCHWDotProdImplicitGemm::
        get_workspace_in_bytes(const SizeArgs& args) const {
    return get_workspace_bundle(nullptr, args).total_size_in_bytes();
}

void ConvolutionBackwardDataImpl::AlgoInt8NCHWDotProdImplicitGemm::exec(
        const ExecArgs& args) const {
    auto&& param = args.opr->param();
    auto&& fm = args.filter_meta;
    size_t n = args.diff_layout->operator[](0), co = args.diff_layout->operator[](1),
           ho = args.diff_layout->operator[](2), wo = args.diff_layout->operator[](3);
    size_t ci = args.grad_layout->operator[](1), hi = args.grad_layout->operator[](2),
           wi = args.grad_layout->operator[](3);
    size_t fh = fm.spatial[0], fw = fm.spatial[1];
    size_t sh = fm.stride[0], sw = fm.stride[1];
    size_t ph = fm.padding[0], pw = fm.padding[1];
    size_t dh = param.dilate_h, dw = param.dilate_w;

    auto&& stream = cuda_stream(args.opr->handle());

    auto bundle = get_workspace_bundle(args.workspace.raw_ptr, args);

    int8_t* inner_filter_ptr = nullptr;
    int8_t* inner_diff_ptr = nullptr;
    // TODO: weight preprocess
    {
        inner_filter_ptr = reinterpret_cast<int8_t*>(bundle.get(0));
        // reformat filter from nchw to n4hwc4
        TensorLayout exec_src{{co / 4, 4, ci, fh, fw}, dtype::Int8()};
        TensorLayout exec_dst{{co / 4, fh, fw, ci, 4}, dtype::Int8()};

        exec_src = exec_src.dimshuffle({0, 3, 4, 2, 1});

        auto&& relayout = args.opr->handle()->create_operator<RelayoutForward>();
        relayout->exec(
                {args.filter_tensor->raw_ptr, exec_src}, {inner_filter_ptr, exec_dst});
    }
    {
        inner_diff_ptr = reinterpret_cast<int8_t*>(bundle.get(1));
        // reformat diff from nchw to nchw4
        TensorLayout exec_src{{n, co / 4, 4, ho, wo}, dtype::Int8()};
        TensorLayout exec_dst{{n, co / 4, ho, wo, 4}, dtype::Int8()};

        exec_src = exec_src.dimshuffle({0, 1, 3, 4, 2});

        auto&& relayout = args.opr->handle()->create_operator<RelayoutForward>();
        relayout->exec(
                {args.diff_tensor->raw_ptr, exec_src}, {inner_diff_ptr, exec_dst});
    }
    int8_t* inner_grad_ptr = reinterpret_cast<int8_t*>(bundle.get(2));

    float diff_scale = args.diff_layout->dtype.param<dtype::QuantizedS8>().scale,
          filter_scale = args.filter_layout->dtype.param<dtype::QuantizedS8>().scale,
          grad_scale = args.grad_layout->dtype.param<dtype::QuantizedS8>().scale;

    // \note these constants of cutlass epilogue will be passed to struct
    // `ConvolutionArguments` by pointer and interpreted as ElementCompute*, a
    // different dtype here results in undefined epilogue behaviors
    float alpha = diff_scale * filter_scale / grad_scale, beta = 0.f, gamma = 0.f,
          delta = 0.f;

    using namespace cutlass::library;

    const Operation* op = (const Operation*)get_available_op(args);

    // gcc prints warnings when size_t values are implicitly narrowed to int
    cutlass::conv::Conv2dProblemSize problem_size{
            int(n),  int(hi), int(wi), int(ci),
            int(co), int(fh), int(fw), int(ho),
            int(wo), int(ph), int(pw), int(sh),
            int(sw), int(dh), int(dw), cutlass::conv::Mode::kCrossCorrelation};

    cutlass::library::ConvolutionArguments conv_args{
            problem_size, inner_diff_ptr, inner_filter_ptr, nullptr,
            nullptr,      inner_grad_ptr, &alpha,           &beta,
            &gamma,       &delta,         nullptr,          nullptr,
            nullptr,      nullptr};

    cutlass_check(op->run(&conv_args, nullptr, stream));

    after_kernel_launch();

    {
        // reformat grad from nchw4 to nchw
        TensorLayout exec_src{{n, ci / 4, hi, wi, 4}, dtype::Int8()};
        TensorLayout exec_dst{{n, ci / 4, 4, hi, wi}, dtype::Int8()};

        exec_src = exec_src.dimshuffle({0, 1, 4, 2, 3});

        auto&& relayout = args.opr->handle()->create_operator<RelayoutForward>();
        relayout->exec(
                {inner_grad_ptr, exec_src}, {args.grad_tensor->raw_ptr, exec_dst});
    }
}
// vim: syntax=cpp.doxygen
