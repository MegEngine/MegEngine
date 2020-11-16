/**
 * \file dnn/src/rocm/convolution/forward/algo.cpp
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

ConvolutionForwardImpl::AlgoPack::AlgoPack() {
    miopen_algos.push_back(&miopen);
    non_miopen_algos.push_back(&matmul);
    non_miopen_algos.push_back(&inplace_matmul);
    non_miopen_algos.push_back(&a1x1);
    non_miopen_algos.push_back(&batched_matrix_mul);
    non_miopen_algos.push_back(&chanwise);

    all_algos.push_back(&matmul);
    all_algos.push_back(&inplace_matmul);
    all_algos.push_back(&a1x1);
    all_algos.push_back(&batched_matrix_mul);
    all_algos.push_back(&chanwise);
    all_algos.push_back(&miopen);

    for (auto&& algo : all_algos) {
        m_all_algos_map.emplace(algo->info().desc, algo);
    }
}

MEGDNN_DEF_GET_ALGO_FROM_DESC(ConvolutionForwardImpl)

ConvolutionForwardImpl::AlgoPack ConvolutionForwardImpl::sm_algo_pack;

ConvolutionForwardImpl::AlgoBase::SizeArgs::SizeArgs(ConvolutionForwardImpl* o,
                                                     const TensorLayout& src,
                                                     const TensorLayout& filter,
                                                     const TensorLayout& dst)
        : SizeArgs(o, src, o->check_layout_fwd(src, filter, dst), dst) {}

ConvolutionForwardImpl::AlgoBase::SizeArgs::SizeArgs(
        ConvolutionForwardImpl* o, const TensorLayout& src,
        const CanonizedFilterMeta& filter, const TensorLayout& dst)
        : ForwardSizeArgs{concrete_handle(o->handle()), &src, filter, &dst},
          opr{o} {}

ConvolutionForwardImpl::AlgoBase::ExecArgs::ExecArgs(
        ConvolutionForwardImpl* opr, _megdnn_tensor_in src,
        _megdnn_tensor_in filter, _megdnn_tensor_out dst,
        _megdnn_workspace workspace)
        : SizeArgs(opr, src.layout, filter.layout, dst.layout),
          src_tensor{&src},
          filter_tensor{&filter},
          dst_tensor{&dst},
          workspace{workspace} {}

std::string ConvolutionForwardImpl::AlgoBase::SizeArgs::to_string() const {
    auto&& fm = filter_meta;
    MEGDNN_MARK_USED_VAR(fm);
    return megdnn_mangle(ssprintf(
            "src=%s, filter=%u{%u,%u,%u,%u}, dst=%s, "
            "pad=%ux%u, stride=%ux%u, dilate=%ux%u, xcorr=%d, dtype=%s,%s",
            src_layout->to_string().c_str(), fm.group, fm.ocpg, fm.icpg,
            fm.spatial[0], fm.spatial[1], dst_layout->to_string().c_str(),
            fm.padding[0], fm.padding[1], fm.stride[0], fm.stride[1],
            fm.dilation[0], fm.dilation[1], !fm.should_flip,
            src_layout->dtype.name(), dst_layout->dtype.name()));
}

convolution::MIOpenCacheKey
ConvolutionForwardImpl::AlgoBase::SizeArgs::to_miopen_algo_cache_key() const {
    convolution::MIOpenCacheKey res;
    res.miopen_handle = reinterpret_cast<intptr_t>(handle->miopen_handle());
    res.batch = src_layout->operator[](0);
    res.IC = src_layout->operator[](1);
    res.IH = src_layout->operator[](2);
    res.IW = src_layout->operator[](3);
    res.OH = dst_layout->operator[](2);
    res.OW = dst_layout->operator[](3);
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
    res.dtype_enum = static_cast<uint32_t>(src_layout->dtype.enumv());
    res.exhaustive_search =
            static_cast<int32_t>(handle->enable_miopen_algo_search());
    res.OC = res.group * res.ocpg;
    return res;
}

// vim: syntax=cpp.doxygen
