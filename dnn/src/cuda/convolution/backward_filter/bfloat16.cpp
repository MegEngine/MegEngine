/**
 * \file src/cuda/convolution/backward_filter/bfloat16.cpp
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
#include "src/cuda/convolution/chanwise/kern.cuh"
#include "src/cuda/utils.h"

using namespace megdnn;
using namespace cuda;
using namespace convolution;

namespace {
std::pair<TensorLayoutArray, ConvolutionBackwardFilterImpl::Param>
sub_opr_config(const TensorLayoutArray& layouts,
               const ConvolutionBackwardFilterImpl* opr) {
    megdnn_assert(layouts.size() >= 3);
    std::pair<TensorLayoutArray, ConvolutionBackwardFilterImpl::Param> ret;
    ret.first = layouts;
    auto change_dtype = [](TensorLayout& layout) {
        if (layout.dtype == dtype::BFloat16()) {
            layout.dtype = dtype::Float32();
        }
    };
    change_dtype(ret.first[0]);
    change_dtype(ret.first[1]);
    change_dtype(ret.first[2]);

    ret.second = opr->param();
    ret.second.compute_mode =
            ConvolutionBackwardFilter::Param::ComputeMode::DEFAULT;
    return ret;
}
}  // namespace

std::vector<Algorithm::SearchItem>
ConvolutionBackwardFilterImpl::AlgoBFloat16::get_subopr_list(
        const TensorLayoutArray& layouts, const OperatorBase* opr) const {
    auto&& config = sub_opr_config(
            layouts, static_cast<const ConvolutionBackwardFilterImpl*>(opr));

    std::string param_str;
    Algorithm::serialize_write_pod(config.second, param_str);
    return {{Algorithm::OprType::CONVOLUTION_BACKWARD_FILTER, param_str,
             config.first}};
}

bool ConvolutionBackwardFilterImpl::AlgoBFloat16::is_available(
        const SizeArgs& args) const {
    TensorLayout fsrc, fdiff, fgrad;
    auto conv_back_filter_opr =
            args.handle->create_operator<ConvolutionBackwardFilter>();

    auto&& config = sub_opr_config(
            {*args.src_layout, *args.diff_layout, *args.grad_layout},
            args.opr);
    conv_back_filter_opr->param() =  config.second;
    return args.src_layout->dtype == args.diff_layout->dtype &&
           args.src_layout->dtype == dtype::BFloat16() &&
           get_algorithm(static_cast<ConvolutionBackwardFilterImpl*>(
                                 conv_back_filter_opr.get()),
                         config.first[0], config.first[1], config.first[2]);
}

WorkspaceBundle
ConvolutionBackwardFilterImpl::AlgoBFloat16::get_workspace_bundle(
        void* ptr, const SizeArgs& args) const {
    auto conv_back_filter_opr =
            args.handle->create_operator<ConvolutionBackwardFilter>();
    if (args.opr->execution_policy().algo.valid()) {
        megdnn_assert(args.opr->execution_policy().sub_policy.size() == 1);
        conv_back_filter_opr->execution_policy() =
                args.opr->execution_policy().sub_policy[0];
    }
    auto&& config = sub_opr_config(
            {*args.src_layout, *args.diff_layout, *args.grad_layout},
            args.opr);

    conv_back_filter_opr->param() = config.second;
    SmallVector<size_t> sizes;
    auto get_workspace = [&sizes](const TensorLayout& src,
                                  const TensorLayout& dst) {
        if (src.dtype != dst.dtype) {
            sizes.push_back(dst.span().dist_byte());
        }
    };

    get_workspace(*args.src_layout, config.first[0]);
    get_workspace(*args.diff_layout, config.first[1]);
    get_workspace(*args.grad_layout, config.first[2]);
    sizes.push_back(conv_back_filter_opr->get_workspace_in_bytes(
            config.first[0], config.first[1], config.first[2]));
    auto ret = WorkspaceBundle{ptr, std::move(sizes)};
    return ret;
}

size_t ConvolutionBackwardFilterImpl::AlgoBFloat16::get_workspace_in_bytes(
        const SizeArgs& args) const {
    return get_workspace_bundle(nullptr, args).total_size_in_bytes();
}

void ConvolutionBackwardFilterImpl::AlgoBFloat16::exec(
        const ExecArgs& args) const {
    TensorND fsrc_tensor = *args.src_tensor;
    TensorND fdiff_tensor = *args.diff_tensor;
    TensorND fgrad_tensor = *args.grad_tensor;
    auto bundle = get_workspace_bundle(args.workspace.raw_ptr, args);
    CompTypeCvter<dtype::BFloat16, dtype::Float32> cvter(args.handle, &bundle);
    {
        cvter.src_to_comp_type(*args.src_tensor, fsrc_tensor)
                .src_to_comp_type(*args.diff_tensor, fdiff_tensor)
                .src_to_comp_type(*args.grad_tensor, fgrad_tensor);
    }
    {
        auto conv_back_filter_opr =
                args.handle->create_operator<ConvolutionBackwardFilter>();
        conv_back_filter_opr->param() = args.opr->param();
        conv_back_filter_opr->param().compute_mode =
                Param::ComputeMode::DEFAULT;

        if (args.opr->execution_policy().algo.valid()) {
            megdnn_assert(args.opr->execution_policy().sub_policy.size() == 1);
            conv_back_filter_opr->execution_policy() =
                    args.opr->execution_policy().sub_policy[0];
        }
        conv_back_filter_opr->exec(fsrc_tensor, fdiff_tensor, fgrad_tensor,
                                   cvter.workspace());
    }
    { cvter.comp_to_dst_type(fgrad_tensor, *args.grad_tensor); }
}

// vim: syntax=cpp.doxygen
