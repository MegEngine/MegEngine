/**
 * \file dnn/src/rocm/convolution/backward_data/algo.cpp
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

ConvolutionBackwardDataImpl::AlgoPack::AlgoPack() {
    all_algos.push_back(&miopen);
    all_algos.push_back(&matmul);
    all_algos.push_back(&chanwise);
    non_miopen_algos.push_back(&matmul);
    non_miopen_algos.push_back(&chanwise);
    miopen_algos.push_back(&miopen);

    for (auto&& algo : all_algos) {
        m_all_algos_map.emplace(algo->info().desc, algo);
    }
}

MEGDNN_DEF_GET_ALGO_FROM_DESC(ConvolutionBackwardDataImpl)
ConvolutionBackwardDataImpl::AlgoPack ConvolutionBackwardDataImpl::sm_algo_pack;

ConvolutionBackwardDataImpl::AlgoBase::SizeArgs::SizeArgs(
        ConvolutionBackwardDataImpl* o, const TensorLayout& filter,
        const TensorLayout& diff, const TensorLayout& grad)
        : SizeArgs(o, o->check_layout_fwd(grad, filter, diff), diff, grad) {}

ConvolutionBackwardDataImpl::AlgoBase::SizeArgs::SizeArgs(
        ConvolutionBackwardDataImpl* o, const CanonizedFilterMeta& filter,
        const TensorLayout& diff, const TensorLayout& grad)
        : handle{concrete_handle(o->handle())},
          filter_meta{filter},
          diff_layout{&diff},
          grad_layout{&grad},
          opr{o} {}

ConvolutionBackwardDataImpl::AlgoBase::ExecArgs::ExecArgs(
        ConvolutionBackwardDataImpl* opr, _megdnn_tensor_in filter,
        _megdnn_tensor_in diff, _megdnn_tensor_out grad,
        _megdnn_workspace workspace)
        : SizeArgs(opr, filter.layout, diff.layout, grad.layout),
          filter_tensor{&filter},
          diff_tensor{&diff},
          grad_tensor{&grad},
          workspace{workspace} {}

std::string ConvolutionBackwardDataImpl::AlgoBase::SizeArgs::to_string() const {
    auto&& fm = filter_meta;
    MEGDNN_MARK_USED_VAR(fm);
    return megdnn_mangle(ssprintf(
            "filter=%u{%u,%u,%u,%u}, diff=%s, grad=%s, "
            "pad=%ux%u, stride=%ux%u, dilate=%ux%u, xcorr=%d, dtype=%s,%s",
            fm.group, fm.ocpg, fm.icpg, fm.spatial[0], fm.spatial[1],
            diff_layout->to_string().c_str(), grad_layout->to_string().c_str(),
            fm.padding[0], fm.padding[1], fm.stride[0], fm.stride[1],
            fm.dilation[0], fm.dilation[1], !fm.should_flip,
            diff_layout->dtype.name(), grad_layout->dtype.name()));
}

convolution::MIOpenCacheKey
ConvolutionBackwardDataImpl::AlgoBase::SizeArgs::to_miopen_algo_cache_key()
        const {
    convolution::MIOpenCacheKey res;
    res.miopen_handle = reinterpret_cast<intptr_t>(handle->miopen_handle());
    res.batch = grad_layout->operator[](0);
    res.IC = grad_layout->operator[](1);
    res.IH = grad_layout->operator[](2);
    res.IW = grad_layout->operator[](3);
    res.OH = diff_layout->operator[](2);
    res.OW = diff_layout->operator[](3);
    res.FH = filter_meta.spatial[0];
    res.FW = filter_meta.spatial[1];
    res.SH = filter_meta.stride[0];
    res.SW = filter_meta.stride[1];
    res.PH = filter_meta.padding[0];
    res.PW = filter_meta.padding[1];
    res.DH = filter_meta.dilation[0];
    res.DW = filter_meta.dilation[1];
    res.group = filter_meta.group;
    res.ocpg = filter_meta.ocpg;
    res.icpg = filter_meta.icpg;
    res.dtype_enum = static_cast<uint32_t>(diff_layout->dtype.enumv());
    res.exhaustive_search =
            static_cast<int32_t>(handle->enable_miopen_algo_search());
    res.OC = res.group * res.ocpg;
    return res;
}
// vim: syntax=cpp.doxygen
