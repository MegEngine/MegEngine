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
#include "src/cuda/convolution/backward_data/deconv_int8_helper.cuh"
#include "src/cuda/convolution_helper/parameter.cuh"
#include "src/cuda/cutlass/singleton.h"
#include "src/cuda/utils.h"

using namespace megdnn;
using namespace cuda;

const void*
ConvolutionBackwardDataImpl::AlgoInt8NCHW4DotProdImplicitGemm::get_available_op(
        const SizeArgs& args) const {
    using namespace cutlass::library;
    auto&& fm = args.filter_meta;
    size_t sh = fm.stride[0], sw = fm.stride[1];
    cutlass::conv::SpecialOptimizeDesc special_optimization =
            (sh == 2 && sw == 2) ? cutlass::conv::SpecialOptimizeDesc::
                                           DECONV_DOUBLE_UPSAMPLING
                                 : cutlass::conv::SpecialOptimizeDesc::NONE;
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
            m_algo_param.threadblock_m,
            m_algo_param.threadblock_n,
            m_algo_param.threadblock_k,
            m_algo_param.warp_m,
            m_algo_param.warp_n,
            m_algo_param.warp_k,
            1,
            1,
            4,
            cutlass::epilogue::EpilogueType::kBiasAddLinearCombinationClamp,
            m_algo_param.stage,
            special_optimization,
            false};
    return (void*)Singleton::get().operation_table.find_op(key);
}

bool ConvolutionBackwardDataImpl::AlgoInt8NCHW4DotProdImplicitGemm::
        is_available(const SizeArgs& args) const {
    auto&& fm = args.filter_meta;
    if (fm.format != Param::Format::NCHW4)
        return false;

    if (!args.grad_layout->is_contiguous() ||
        !args.diff_layout->is_contiguous()) {
        return false;
    }

    bool available = true;

    auto src_dtype = args.diff_layout->dtype,
         filter_dtype = args.filter_layout->dtype,
         dst_dtype = args.grad_layout->dtype;

    available &= (src_dtype.enumv() == DTypeEnum::QuantizedS8 &&
                  filter_dtype.enumv() == DTypeEnum::QuantizedS8 &&
                  dst_dtype.enumv() == DTypeEnum::QuantizedS8);
    // TODO support group deconv int8
    available &= (fm.group == 1);
    // mode must be cross correlation
    available &= !fm.should_flip;
    // mode must be 2D
    available &= fm.spatial_ndim == 2;
    // TODO: support dialtion
    available &= (fm.dilation[0] == 1 && fm.dilation[1] == 1);
    // FIXME: too large filter size is not supported now
    available &= fm.spatial[0] * fm.spatial[1] <=
                 (uint32_t)(848 / (2 * m_algo_param.warp_k / 4) - 2);
    available &= (get_available_op(args) != nullptr);
    // only support sm_61 or later, platform should have fast native int8
    // support
    available &= is_compute_capability_required(6, 1);

    return available;
}

WorkspaceBundle ConvolutionBackwardDataImpl::AlgoInt8NCHW4DotProdImplicitGemm::
        get_workspace_bundle(dt_byte* raw_ptr, const SizeArgs& args) const {
    size_t ws_filter = args.filter_layout->span().dist_byte();
    return WorkspaceBundle{raw_ptr, {ws_filter}};
}

size_t ConvolutionBackwardDataImpl::AlgoInt8NCHW4DotProdImplicitGemm::
        get_workspace_in_bytes(const SizeArgs& args) const {
    return get_workspace_bundle(nullptr, args).total_size_in_bytes();
}

void ConvolutionBackwardDataImpl::AlgoInt8NCHW4DotProdImplicitGemm::exec(
        const ExecArgs& args) const {
    auto&& param = args.opr->param();
    auto&& fm = args.filter_meta;
    size_t n = args.diff_layout->operator[](0),
           co = args.diff_layout->operator[](1) * 4,
           ho = args.diff_layout->operator[](2),
           wo = args.diff_layout->operator[](3);
    size_t ci = args.grad_layout->operator[](1) * 4,
           hi = args.grad_layout->operator[](2),
           wi = args.grad_layout->operator[](3);
    size_t fh = fm.spatial[0], fw = fm.spatial[1];
    size_t sh = fm.stride[0], sw = fm.stride[1];
    size_t ph = fm.padding[0], pw = fm.padding[1];
    size_t dh = param.dilate_h, dw = param.dilate_w;

    auto&& stream = cuda_stream(args.opr->handle());

    int8_t* filter_ptr = nullptr;
    // TODO: weight preprocess
    {
        filter_ptr = reinterpret_cast<int8_t*>(args.workspace.raw_ptr);
        // reformat filter from nc4hw4 to n4hwc4
        megdnn::cuda::deconv::reorder_filter_nc4hw4_to_n4hwc4(
                filter_ptr, args.filter_tensor->compatible_ptr<int8_t>(), co,
                ci, fh, fw, stream);
    }

    float diff_scale =
                  args.diff_layout->dtype.param<dtype::QuantizedS8>().scale,
          filter_scale =
                  args.filter_layout->dtype.param<dtype::QuantizedS8>().scale,
          grad_scale =
                  args.grad_layout->dtype.param<dtype::QuantizedS8>().scale;

    // \note these constants of cutlass epilogue will be passed to struct
    // `ConvolutionArguments` by pointer and interpreted as ElementCompute*,
    // a different dtype here results in undefined epilogue behaviors
    float alpha = diff_scale * filter_scale / grad_scale, beta = 0.f,
          gamma = 0.f, delta = 0.f;

    using namespace cutlass::library;

    const Operation* op = (const Operation*)get_available_op(args);

    // gcc prints warnings when size_t values are implicitly narrowed to int
    cutlass::conv::Conv2dProblemSize problem_size{
            int(n),  int(hi), int(wi), int(ci),
            int(co), int(fh), int(fw), int(ho),
            int(wo), int(ph), int(pw), int(sh),
            int(sw), int(dh), int(dw), cutlass::conv::Mode::kCrossCorrelation};

    cutlass::library::ConvolutionArguments conv_args{
            problem_size, args.diff_tensor->compatible_ptr<int8_t>(),
            filter_ptr,   nullptr,
            nullptr,      args.grad_tensor->compatible_ptr<int8_t>(),
            &alpha,       &beta,
            &gamma,       &delta,
            nullptr,      nullptr,
            nullptr,      nullptr};

    cutlass_check(op->run(&conv_args, nullptr, stream));

    after_kernel_launch();
}

void ConvolutionBackwardDataImpl::AlgoPack::fill_int8_dp4a_algos() {
    using AlgoParam = AlgoInt8NCHW4DotProdImplicitGemm::AlgoParam;
    int8_nchw4_dotprod.emplace_back(AlgoParam{16, 64, 8, 16, 64, 8, 2});
    int8_nchw4_dotprod.emplace_back(AlgoParam{16, 128, 16, 16, 64, 16, 2});
    int8_nchw4_dotprod.emplace_back(AlgoParam{16, 128, 16, 16, 128, 16, 1});
    int8_nchw4_dotprod.emplace_back(AlgoParam{32, 128, 32, 32, 64, 32, 2});
}

// vim: syntax=cpp.doxygen
