/**
 * \file dnn/src/cuda/conv_bias/bfloat16.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/cuda/conv_bias/algo.h"
#include "src/cuda/handle.h"
#include "src/cuda/utils.cuh"
#include "src/cuda/utils.h"
#include "src/common/algo_base.h"

using namespace megdnn;
using namespace cuda;
using namespace conv_bias;

namespace {
std::pair<TensorLayoutArray, ConvBiasForwardImpl::Param> sub_opr_config(
        const TensorLayoutArray& layouts, const ConvBiasForwardImpl* opr) {
    megdnn_assert(layouts.size() >= 3);
    std::pair<TensorLayoutArray, ConvBiasForwardImpl::Param> ret;
    ret.first = layouts;
    auto change_dtype = [](TensorLayout& layout) {
        if (layout.dtype == dtype::BFloat16()) {
            layout.dtype = dtype::Float32();
        }
    };
    change_dtype(ret.first[0]);
    change_dtype(ret.first[1]);
    change_dtype(ret.first[2]);
    change_dtype(ret.first[3]);
    change_dtype(ret.first[4]);

    ret.second = opr->param();
    ret.second.compute_mode = ConvBiasForwardImpl::Param::ComputeMode::DEFAULT;
    return ret;
}

std::pair<TensorLayoutArray, std::unique_ptr<ConvBiasForward>> prepare_sub_opr(
        const ConvBiasForwardImpl::AlgoBase::SizeArgs& args) {
    auto convbias_opr = args.handle->create_operator<ConvBias>();
    auto&& config = sub_opr_config(
            {*args.src_layout, *args.filter_layout, *args.bias_layout,
             *args.z_layout, *args.dst_layout},
            args.opr);
    convbias_opr->param() = config.second;

    return {config.first, std::move(convbias_opr)};
}
}  // namespace

std::vector<Algorithm::SearchItem>
ConvBiasForwardImpl::AlgoBFloat16::get_subopr_list(
        const TensorLayoutArray& layouts, const OperatorBase* opr) const {
    auto&& config = sub_opr_config(
            layouts, static_cast<const ConvBiasForwardImpl*>(opr));

    std::string param_str;
    Algorithm::serialize_write_pod(config.second, param_str);
    return {{Algorithm::OprType::CONVBIAS_FORWARD, param_str, config.first}};
}

bool ConvBiasForwardImpl::AlgoBFloat16::is_available(
        const SizeArgs& args) const {
    auto config = prepare_sub_opr(args);

    return args.src_layout->dtype == args.filter_layout->dtype &&
           args.src_layout->dtype == dtype::BFloat16() &&
           get_algorithm(static_cast<ConvBiasForwardImpl*>(config.second.get()),
                         config.first[0], config.first[1], config.first[2],
                         config.first[3], config.first[4]);
}

WorkspaceBundle ConvBiasForwardImpl::AlgoBFloat16::get_workspace_bundle(
        void* ptr, const SizeArgs& args) const {
    auto config = prepare_sub_opr(args);

    SmallVector<size_t> sizes;
    auto get_workspace = [&sizes](const TensorLayout& src,
                                  const TensorLayout& dst) {
        if (src.dtype != dst.dtype) {
            sizes.push_back(dst.span().dist_byte());
        }
    };
    get_workspace(*args.src_layout, config.first[0]);
    get_workspace(*args.filter_layout, config.first[1]);
    get_workspace(*args.bias_layout, config.first[2]);
    get_workspace(*args.z_layout, config.first[3]);
    get_workspace(*args.dst_layout, config.first[4]);
    sizes.push_back(config.second->get_workspace_in_bytes(
            config.first[0], config.first[1], config.first[2], config.first[3],
            config.first[4], nullptr));

    return {ptr, std::move(sizes)};
}

size_t ConvBiasForwardImpl::AlgoBFloat16::get_workspace_in_bytes(
        const SizeArgs& args) const {
    return get_workspace_bundle(nullptr, args).total_size_in_bytes();
}

void ConvBiasForwardImpl::AlgoBFloat16::exec(const ExecArgs& args) const {
    TensorND fsrc_tensor = *args.src_tensor;
    TensorND ffilter_tensor = *args.filter_tensor;
    TensorND fbias_tensor = *args.bias_tensor;
    TensorND fz_tensor = *args.z_tensor;
    TensorND fdst_tensor = *args.dst_tensor;
    auto bundle = get_workspace_bundle(args.workspace.raw_ptr, args);
    CompTypeCvter<dtype::BFloat16, dtype::Float32> cvter(args.handle, &bundle);
    {
        cvter.src_to_comp_type(*args.src_tensor, fsrc_tensor)
                .src_to_comp_type(*args.filter_tensor, ffilter_tensor)
                .src_to_comp_type(*args.bias_tensor, fbias_tensor)
                .src_to_comp_type(*args.z_tensor, fz_tensor)
                .src_to_comp_type(*args.dst_tensor, fdst_tensor);
    }
    {
        auto config = prepare_sub_opr(args);

        config.second->exec(fsrc_tensor, ffilter_tensor, fbias_tensor,
                            fz_tensor, fdst_tensor, nullptr, cvter.workspace());
    }
    { cvter.comp_to_dst_type(fdst_tensor, *args.dst_tensor); }
}

// vim: syntax=cpp.doxygen
