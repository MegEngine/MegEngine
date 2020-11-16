/**
 * \file dnn/src/cuda/convolution3d/backward_data/algo.cpp
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

Convolution3DBackwardDataImpl::AlgoPack::AlgoPack() {
    non_cudnn_algos.push_back(&chanwise);

    all_algos.push_back(&chanwise); // prefer chanwise

    fill_cudnn_algos();
    for (auto &&i: cudnn) {
        all_algos.push_back(&i);
    }

    all_algos.reserve(all_algos.size() * 2);

    // add gconv algos by AlgoGroupConvGeneral
    auto all_algos_data = all_algos.data();
    for (size_t i = 1; i < all_algos.size(); ++ i) {
        gconv.push_back({all_algos[i]});
    }
    for (size_t i = 1; i < all_algos.size(); ++ i) {
        algo2gconv[all_algos[i]] = &gconv[i - 1];
    }
    for (auto &&i: gconv) {
        all_algos.push_back(&i);
    }
    megdnn_assert(all_algos_data == all_algos.data());

    for (auto&& algo : all_algos) {
        m_all_algos_map.emplace(algo->info().desc, algo);
    }
}

MEGDNN_DEF_GET_ALGO_FROM_DESC(Convolution3DBackwardDataImpl)

Convolution3DBackwardDataImpl::AlgoCUDNN*
Convolution3DBackwardDataImpl::AlgoPack::cudnn_from_enum(
        cudnnConvolutionBwdDataAlgo_t algo) {
    for (auto &&i: cudnn) {
        if (i.cudnn_enum() == algo)
            return &i;
    }
    megdnn_throw(megdnn_mangle(ssprintf(
                    "can not find cudnn bwd_data algorithm %d",
                    static_cast<int>(algo))));
}

Convolution3DBackwardDataImpl::AlgoPack Convolution3DBackwardDataImpl::sm_algo_pack;

Convolution3DBackwardDataImpl::AlgoBase::SizeArgs::SizeArgs(
        Convolution3DBackwardDataImpl *o,
        const TensorLayout &filter, const TensorLayout &diff,
        const TensorLayout &grad):
    SizeArgs(o, o->check_layout_fwd(grad, filter, diff), diff, grad)
{
}

Convolution3DBackwardDataImpl::AlgoBase::SizeArgs::SizeArgs(
        Convolution3DBackwardDataImpl *o,
        const CanonizedFilterMeta &filter, const TensorLayout &diff,
        const TensorLayout &grad):
    handle{concrete_handle(o->handle())},
    filter_meta{filter},
    diff_layout{&diff},
    grad_layout{&grad},
    opr{o}
{
}

Convolution3DBackwardDataImpl::AlgoBase::ExecArgs::ExecArgs(
        Convolution3DBackwardDataImpl *opr,
        _megdnn_tensor_in filter,
        _megdnn_tensor_in diff,
        _megdnn_tensor_out grad,
        _megdnn_workspace workspace):
    SizeArgs(opr, filter.layout, diff.layout, grad.layout),
    filter_tensor{&filter}, diff_tensor{&diff}, grad_tensor{&grad},
    workspace{workspace}
{
}

std::string Convolution3DBackwardDataImpl::AlgoBase::SizeArgs::to_string() const {
    auto &&fm = filter_meta;
    MEGDNN_MARK_USED_VAR(fm);
    return megdnn_mangle(ssprintf(
                "filter=%u{%u,%u,%u,%u,%u}, diff=%s, grad=%s, "
                "pad=%ux%ux%u, stride=%ux%ux%u, dilate=%ux%ux%u, xcorr=%d, dtype=%s,%s",
                fm.group, fm.ocpg, fm.icpg, fm.spatial[0], fm.spatial[1], fm.spatial[2],
                diff_layout->to_string().c_str(),
                grad_layout->to_string().c_str(),
                fm.padding[0], fm.padding[1], fm.padding[2],
                fm.stride[0], fm.stride[1], fm.stride[2],
                fm.dilation[0], fm.dilation[1] ,fm.dilation[2],
                !fm.should_flip,
                diff_layout->dtype.name(), grad_layout->dtype.name()));
}

// vim: syntax=cpp.doxygen
