/**
 * \file dnn/src/rocm/convolution/backward_filter/algo.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "hcc_detail/hcc_defs_prologue.h"

#include "./algo.h"
#include "src/rocm/utils.h"

using namespace megdnn;
using namespace rocm;

ConvolutionBackwardFilterImpl::AlgoPack::AlgoPack() {
    all_algos.push_back(&miopen);
    all_algos.push_back(&matmul);
    all_algos.push_back(&chanwise);
    non_miopen_algos.push_back(&matmul);
    non_miopen_algos.push_back(&chanwise);
    non_miopen_algos.push_back(all_algos.back());
    miopen_algos.push_back(&miopen);

    for (auto&& algo : all_algos) {
        m_all_algos_map.emplace(algo->info().desc, algo);
    }
}

MEGDNN_DEF_GET_ALGO_FROM_DESC(ConvolutionBackwardFilterImpl)
ConvolutionBackwardFilterImpl::AlgoPack
        ConvolutionBackwardFilterImpl::sm_algo_pack;

ConvolutionBackwardFilterImpl::AlgoBase::SizeArgs::SizeArgs(
        ConvolutionBackwardFilterImpl* o, const TensorLayout& src,
        const TensorLayout& diff, const TensorLayout& grad)
        : SizeArgs(o, src, diff, o->check_layout_fwd(src, grad, diff)) {}

ConvolutionBackwardFilterImpl::AlgoBase::SizeArgs::SizeArgs(
        ConvolutionBackwardFilterImpl* o, const TensorLayout& src,
        const TensorLayout& diff, const CanonizedFilterMeta& grad)
        : handle{concrete_handle(o->handle())},
          src_layout{&src},
          diff_layout{&diff},
          grad_filter_meta{grad},
          opr{o} {}

ConvolutionBackwardFilterImpl::AlgoBase::ExecArgs::ExecArgs(
        ConvolutionBackwardFilterImpl* opr, _megdnn_tensor_in src,
        _megdnn_tensor_in diff, _megdnn_tensor_out grad,
        _megdnn_workspace workspace)
        : SizeArgs(opr, src.layout, diff.layout, grad.layout),
          src_tensor{&src},
          diff_tensor{&diff},
          grad_tensor{&grad},
          workspace{workspace} {}

std::string ConvolutionBackwardFilterImpl::AlgoBase::SizeArgs::to_string()
        const {
    auto&& fm = grad_filter_meta;
    MEGDNN_MARK_USED_VAR(fm);
    return megdnn_mangle(ssprintf(
            "src=%s diff=%s grad_filter=%u{%u,%u,%u,%u}, "
            "pad=%ux%u, stride=%ux%u, dilate=%ux%u, xcorr=%d, dtype=%s,%s",
            src_layout->to_string().c_str(), diff_layout->to_string().c_str(),
            fm.group, fm.ocpg, fm.icpg, fm.spatial[0], fm.spatial[1],
            fm.padding[0], fm.padding[1], fm.stride[0], fm.stride[1],
            fm.dilation[0], fm.dilation[1], !fm.should_flip,
            src_layout->dtype.name(), diff_layout->dtype.name()));
}

convolution::MIOpenCacheKey
ConvolutionBackwardFilterImpl::AlgoBase::SizeArgs::to_miopen_algo_cache_key()
        const {
    convolution::MIOpenCacheKey res;
    res.miopen_handle = reinterpret_cast<intptr_t>(handle->miopen_handle());
    res.batch = src_layout->operator[](0);
    res.IC = src_layout->operator[](1);
    res.IH = src_layout->operator[](2);
    res.IW = src_layout->operator[](3);
    res.OH = diff_layout->operator[](2);
    res.OW = diff_layout->operator[](3);
    res.FH = grad_filter_meta.spatial[0];
    res.FW = grad_filter_meta.spatial[1];
    res.SH = grad_filter_meta.stride[0];
    res.SW = grad_filter_meta.stride[1];
    res.PH = grad_filter_meta.padding[0];
    res.PW = grad_filter_meta.padding[1];
    res.DH = grad_filter_meta.dilation[0];
    res.DW = grad_filter_meta.dilation[1];
    res.group = grad_filter_meta.group;
    res.ocpg = grad_filter_meta.ocpg;
    res.icpg = grad_filter_meta.icpg;
    res.dtype_enum = static_cast<uint32_t>(src_layout->dtype.enumv());
    res.exhaustive_search =
            static_cast<int32_t>(handle->enable_miopen_algo_search());
    res.OC = res.group * res.ocpg;
    return res;
}
// vim: syntax=cpp.doxygen
