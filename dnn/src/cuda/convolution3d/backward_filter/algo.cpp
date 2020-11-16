/**
 * \file dnn/src/cuda/convolution3d/backward_filter/algo.cpp
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

Convolution3DBackwardFilterImpl::AlgoPack::AlgoPack() {
    non_cudnn_algos.push_back(&chanwise);
    non_cudnn_algos.push_back(&inplace_matmul);
    all_algos.push_back(&chanwise); // prefer chanwise

    fill_cudnn_algos();
    for (auto &&i: cudnn) {
        all_algos.push_back(&i);
    }

    all_algos.push_back(&inplace_matmul);
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
    non_cudnn_algos.push_back(all_algos.rbegin()[0]); //group inplace_matmul

    for (auto&& algo : all_algos) {
        m_all_algos_map.emplace(algo->info().desc, algo);
    }
}

MEGDNN_DEF_GET_ALGO_FROM_DESC(Convolution3DBackwardFilterImpl)

Convolution3DBackwardFilterImpl::AlgoCUDNN*
Convolution3DBackwardFilterImpl::AlgoPack::cudnn_from_enum(
        cudnnConvolutionBwdFilterAlgo_t algo) {
    for (auto &&i: cudnn) {
        if (i.cudnn_enum() == algo)
            return &i;
    }
    megdnn_throw(megdnn_mangle(ssprintf(
                    "can not find cudnn bwd_filter algorithm %d",
                    static_cast<int>(algo))));
}

Convolution3DBackwardFilterImpl::AlgoPack
Convolution3DBackwardFilterImpl::sm_algo_pack;

Convolution3DBackwardFilterImpl::AlgoBase::SizeArgs::SizeArgs(
        Convolution3DBackwardFilterImpl *o,
        const TensorLayout &src, const TensorLayout &diff,
        const TensorLayout &grad):
    SizeArgs(o, src, diff, o->check_layout_fwd(src, grad, diff))
{
}

Convolution3DBackwardFilterImpl::AlgoBase::SizeArgs::SizeArgs(
        Convolution3DBackwardFilterImpl *o,
        const TensorLayout &src, const TensorLayout &diff,
        const CanonizedFilterMeta &grad):
    handle{concrete_handle(o->handle())},
    src_layout{&src},
    diff_layout{&diff},
    grad_filter_meta{grad},
    opr{o}
{
}

Convolution3DBackwardFilterImpl::AlgoBase::ExecArgs::ExecArgs(
        Convolution3DBackwardFilterImpl *opr,
        _megdnn_tensor_in src,
        _megdnn_tensor_in diff,
        _megdnn_tensor_out grad,
        _megdnn_workspace workspace):
    SizeArgs(opr, src.layout, diff.layout, grad.layout),
    src_tensor{&src}, diff_tensor{&diff}, grad_tensor{&grad},
    workspace{workspace}
{
}

std::string
Convolution3DBackwardFilterImpl::AlgoBase::SizeArgs::to_string() const {
    auto &&fm = grad_filter_meta;
    MEGDNN_MARK_USED_VAR(fm);
    return megdnn_mangle(ssprintf(
                "src=%s diff=%s grad_filter=%u{%u,%u,%u,%u,%u}, "
                "pad=%ux%ux%u, stride=%ux%ux%u, dilate=%ux%ux%u, xcorr=%d, dtype=%s,%s",
                src_layout->to_string().c_str(),
                diff_layout->to_string().c_str(),
                fm.group, fm.ocpg, fm.icpg,
                fm.spatial[0], fm.spatial[1], fm.spatial[2],
                fm.padding[0], fm.padding[1], fm.padding[2],
                fm.stride[0], fm.stride[1], fm.stride[2],
                fm.dilation[0], fm.dilation[1], fm.dilation[2],
                !fm.should_flip,
                src_layout->dtype.name(), diff_layout->dtype.name()));
}

// vim: syntax=cpp.doxygen
