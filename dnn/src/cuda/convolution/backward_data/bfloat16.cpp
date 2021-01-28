/**
 * \file src/cuda/convolution/backward_data/bfloat16.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./algo.h"
#include "src/common/algo_base.h"
#include "src/cuda/convolution/chanwise/kern.cuh"
#include "src/cuda/utils.h"

using namespace megdnn;
using namespace cuda;
using namespace convolution;

namespace {
std::pair<TensorLayoutArray, ConvolutionBackwardDataImpl::Param> sub_opr_config(
        const TensorLayoutArray& layouts,
        const ConvolutionBackwardDataImpl* opr) {
    megdnn_assert(layouts.size() >= 3);
    std::pair<TensorLayoutArray, ConvolutionBackwardDataImpl::Param> ret;
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
            ConvolutionBackwardData::Param::ComputeMode::DEFAULT;
    return ret;
}

std::pair<TensorLayoutArray, std::unique_ptr<ConvolutionBackwardData>>
prepare_sub_opr(const ConvolutionBackwardDataImpl::AlgoBase::SizeArgs& args) {
    auto conv_back_data_opr =
            args.handle->create_operator<ConvolutionBackwardData>();
    auto&& config = sub_opr_config(
            {*args.filter_layout, *args.diff_layout, *args.grad_layout},
            args.opr);
    conv_back_data_opr->param() = config.second;

    return {config.first, std::move(conv_back_data_opr)};
}
}  // namespace

std::vector<Algorithm::SearchItem>
ConvolutionBackwardDataImpl::AlgoBFloat16::get_subopr_list(
        const TensorLayoutArray& layouts, const OperatorBase* opr) const {
    auto&& config = sub_opr_config(
            layouts, static_cast<const ConvolutionBackwardDataImpl*>(opr));

    std::string param_str;
    Algorithm::serialize_write_pod(config.second, param_str);
    return {{Algorithm::OprType::CONVOLUTION_BACKWARD_DATA, param_str,
             config.first}};
}

bool ConvolutionBackwardDataImpl::AlgoBFloat16::is_available(
        const SizeArgs& args) const {
    auto config = prepare_sub_opr(args);
    return args.diff_layout->dtype == args.filter_layout->dtype &&
           args.diff_layout->dtype == dtype::BFloat16() &&
           get_algorithm(static_cast<ConvolutionBackwardDataImpl*>(
                                 config.second.get()),
                         config.first[0], config.first[1], config.first[2]);
}

WorkspaceBundle ConvolutionBackwardDataImpl::AlgoBFloat16::get_workspace_bundle(
        void* ptr, const SizeArgs& args) const {
    auto config = prepare_sub_opr(args);
    SmallVector<size_t> sizes;
    auto get_workspace = [&sizes](const TensorLayout& src,
                                  const TensorLayout& dst) {
        if (src.dtype != dst.dtype) {
            sizes.push_back(dst.span().dist_byte());
        }
    };
    get_workspace(*args.filter_layout, config.first[0]);
    get_workspace(*args.diff_layout, config.first[1]);
    get_workspace(*args.grad_layout, config.first[2]);

    sizes.push_back(config.second->get_workspace_in_bytes(
            config.first[0], config.first[1], config.first[2]));
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
        auto config = prepare_sub_opr(args);
        config.second->exec(ffilter_tensor, fdiff_tensor, fgrad_tensor,
                            cvter.workspace());
    }
    { cvter.comp_to_dst_type(fgrad_tensor, *args.grad_tensor); }
}

// vim: syntax=cpp.doxygen
