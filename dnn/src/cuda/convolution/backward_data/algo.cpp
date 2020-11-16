/**
 * \file dnn/src/cuda/convolution/backward_data/algo.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./algo.h"
#include "src/cuda/utils.h"

using namespace megdnn;
using namespace cuda;

ConvolutionBackwardDataImpl::AlgoPack::AlgoPack() {
    non_cudnn_algos.push_back(&chanwise);
    non_cudnn_algos.push_back(&chanwise_small);
    non_cudnn_algos.push_back(&matmul);

    all_algos.push_back(&chanwise); // prefer chanwise
    all_algos.push_back(&chanwise_small); // prefer small chanwise

    fill_cudnn_algos();
    for (auto &&i: cudnn) {
        all_algos.push_back(&i);
    }
    all_algos.push_back(&matmul);

    all_algos.reserve(all_algos.size() * 2);

    // add gconv algos by AlgoGroupConvGeneral
    auto all_algos_data = all_algos.data();
    size_t group_algo_start = 2;
    for (size_t i = group_algo_start; i < all_algos.size(); ++ i) {
        gconv.push_back({all_algos[i]});
    }
    for (size_t i = group_algo_start; i < all_algos.size(); ++ i) {
        algo2gconv[all_algos[i]] = &gconv[i - group_algo_start];
    }
    for (auto &&i: gconv) {
        all_algos.push_back(&i);
    }
    megdnn_assert(all_algos_data == all_algos.data());

    non_cudnn_algos.push_back(all_algos.rbegin()[0]);   // group matmul
    size_t algo_size = all_algos.size();
    for (size_t i=0; i<algo_size; ++i) {
        bfloat16_refhold.emplace_back(new AlgoBFloat16(all_algos[i]));
        all_algos.push_back(bfloat16_refhold.back().get());
        bfloat16_algos.push_back(bfloat16_refhold.back().get());
    }

    for (auto&& algo : all_algos) {
        m_all_algos_map.emplace(algo->info().desc, algo);
    }
}

MEGDNN_DEF_GET_ALGO_FROM_DESC(ConvolutionBackwardDataImpl)

ConvolutionBackwardDataImpl::AlgoCUDNN*
ConvolutionBackwardDataImpl::AlgoPack::cudnn_from_enum(
        cudnnConvolutionBwdDataAlgo_t algo) {
    for (auto &&i: cudnn) {
        if (i.cudnn_enum() == algo)
            return &i;
    }
    megdnn_throw(megdnn_mangle(ssprintf(
                    "can not find cudnn bwd_data algorithm %d",
                    static_cast<int>(algo))));
}

ConvolutionBackwardDataImpl::AlgoPack ConvolutionBackwardDataImpl::sm_algo_pack;

ConvolutionBackwardDataImpl::AlgoBase::SizeArgs::SizeArgs(
        ConvolutionBackwardDataImpl *o,
        const TensorLayout &filter, const TensorLayout &diff,
        const TensorLayout &grad):
    SizeArgs(o, filter, o->check_layout_fwd(grad, filter, diff), diff, grad)
{
}

ConvolutionBackwardDataImpl::AlgoBase::SizeArgs::SizeArgs(
        ConvolutionBackwardDataImpl *o, const TensorLayout& filter,
        const CanonizedFilterMeta &filter_meta, const TensorLayout &diff,
        const TensorLayout &grad):
    handle{concrete_handle(o->handle())},
    filter_meta{filter_meta},
    diff_layout{&diff},
    grad_layout{&grad},
    filter_layout{&filter},
    opr{o}
{
}

ConvolutionBackwardDataImpl::AlgoBase::ExecArgs::ExecArgs(
        ConvolutionBackwardDataImpl *opr,
        _megdnn_tensor_in filter,
        _megdnn_tensor_in diff,
        _megdnn_tensor_out grad,
        _megdnn_workspace workspace):
    SizeArgs(opr, filter.layout, diff.layout, grad.layout),
    filter_tensor{&filter}, diff_tensor{&diff}, grad_tensor{&grad},
    workspace{workspace}
{
}

std::string ConvolutionBackwardDataImpl::AlgoBase::SizeArgs::to_string() const {
    auto &&fm = filter_meta;
    MEGDNN_MARK_USED_VAR(fm);
    return megdnn_mangle(ssprintf(
                "filter=%u{%u,%u,%u,%u}, diff=%s, grad=%s, "
                "pad=%ux%u, stride=%ux%u, dilate=%ux%u, xcorr=%d, dtype=%s,%s",
                fm.group, fm.ocpg, fm.icpg, fm.spatial[0], fm.spatial[1],
                diff_layout->to_string().c_str(),
                grad_layout->to_string().c_str(),
                fm.padding[0], fm.padding[1], fm.stride[0], fm.stride[1],
                fm.dilation[0], fm.dilation[1],
                !fm.should_flip,
                diff_layout->dtype.name(), grad_layout->dtype.name()));
}

// vim: syntax=cpp.doxygen
