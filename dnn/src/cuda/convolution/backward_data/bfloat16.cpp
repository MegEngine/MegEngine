/**
 * \file src/cuda/convolution/backward_data/bfloat16.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./algo.h"
#include "src/cuda/convolution/chanwise/kern.cuh"
#include "src/cuda/utils.h"

using namespace megdnn;
using namespace cuda;
using namespace convolution;

ConvolutionBackwardDataImpl::AlgoBFloat16::AlgoBFloat16(
        ConvolutionBackwardDataImpl::AlgoBase* algorithm)
        : m_algorithm(algorithm) {
    megdnn_assert_internal(algorithm);
    m_name = ssprintf("CONVOLUTION_BACKWARD_DATD_BFLOAT16:%s",
                      m_algorithm->name());
}

ConvolutionBackwardDataImpl::AlgoBase::SizeArgs
ConvolutionBackwardDataImpl::AlgoBFloat16::float_args(
        const SizeArgs& args, ConvolutionBackwardDataImpl* opr,
        TensorLayout& ffilter, TensorLayout& fdiff, TensorLayout& fgrad) const {
    ffilter = *args.filter_layout;
    fdiff = *args.diff_layout;
    fgrad = *args.grad_layout;
    auto change_dtype = [](TensorLayout& layout) {
        if (layout.dtype == dtype::BFloat16()) {
            layout.dtype = dtype::Float32();
        }
    };
    change_dtype(ffilter);
    change_dtype(fdiff);
    change_dtype(fgrad);
    opr->param() = args.opr->param();
    opr->param().compute_mode = Param::ComputeMode::DEFAULT;
    opr->execution_policy() = {m_algorithm->info()};
    return SizeArgs(opr, ffilter, fdiff, fgrad);
}

bool ConvolutionBackwardDataImpl::AlgoBFloat16::is_available(
        const SizeArgs& args) const {
    TensorLayout ffilter, fdiff, fgrad;
    auto conv_back_data_opr =
            args.handle->create_operator<ConvolutionBackwardData>();
    SizeArgs fargs = float_args(
            args,
            static_cast<ConvolutionBackwardDataImpl*>(conv_back_data_opr.get()),
            ffilter, fdiff, fgrad);
    return args.diff_layout->dtype == args.filter_layout->dtype &&
           args.diff_layout->dtype == dtype::BFloat16() &&
           m_algorithm->is_available(fargs);
}

WorkspaceBundle ConvolutionBackwardDataImpl::AlgoBFloat16::get_workspace_bundle(
        void* ptr, const SizeArgs& args) const {
    TensorLayout ffilter, fdiff, fgrad;
    auto conv_back_data_opr =
            args.handle->create_operator<ConvolutionBackwardData>();
    SizeArgs fargs = float_args(
            args,
            static_cast<ConvolutionBackwardDataImpl*>(conv_back_data_opr.get()),
            ffilter, fdiff, fgrad);
    SmallVector<size_t> sizes;
    auto get_workspace = [&sizes](const TensorLayout& src,
                                  const TensorLayout& dst) {
        if (src.dtype != dst.dtype) {
            sizes.push_back(dst.span().dist_byte());
        }
    };
    get_workspace(*args.filter_layout, ffilter);
    get_workspace(*args.diff_layout, fdiff);
    get_workspace(*args.grad_layout, fgrad);
    sizes.push_back(m_algorithm->get_workspace_in_bytes(fargs));
    return {ptr, std::move(sizes)};
}

size_t ConvolutionBackwardDataImpl::AlgoBFloat16::get_workspace_in_bytes(
        const SizeArgs& args) const {
    return get_workspace_bundle(nullptr, args).total_size_in_bytes();
}

void ConvolutionBackwardDataImpl::AlgoBFloat16::exec(
        const ExecArgs& args) const {
    TensorND ffilter_tensor = *args.filter_tensor;
    TensorND fdiff_tensor = *args.diff_tensor;
    TensorND fgrad_tensor = *args.grad_tensor;
    auto bundle = get_workspace_bundle(args.workspace.raw_ptr, args);
    CompTypeCvter<dtype::BFloat16, dtype::Float32> cvter(args.handle, &bundle);
    {
        cvter.src_to_comp_type(*args.filter_tensor, ffilter_tensor)
                .src_to_comp_type(*args.diff_tensor, fdiff_tensor)
                .src_to_comp_type(*args.grad_tensor, fgrad_tensor);
    }
    {
        auto conv_back_data_opr =
                args.handle->create_operator<ConvolutionBackwardData>();
        conv_back_data_opr->param() = args.opr->param();
        conv_back_data_opr->param().compute_mode = Param::ComputeMode::DEFAULT;
        conv_back_data_opr->execution_policy() = {m_algorithm->info()};
        conv_back_data_opr->exec(ffilter_tensor, fdiff_tensor, fgrad_tensor,
                                 cvter.workspace());
    }
    { cvter.comp_to_dst_type(fgrad_tensor, *args.grad_tensor); }
}

// vim: syntax=cpp.doxygen
