/**
 * \file dnn/src/cuda/convolution3d/forward/algo.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./algo.h"
#include "src/cuda/utils.h"

using namespace megdnn;
using namespace cuda;

Convolution3DForwardImpl::AlgoPack::AlgoPack() {
    non_cudnn_algos.push_back(&chanwise);
    non_cudnn_algos.push_back(&inplace_matmul);
    non_cudnn_algos.push_back(&a1x1x1);

    all_algos.push_back(&chanwise);

    fill_cudnn_algos();
    for (auto&& i : cudnn) {
        all_algos.push_back(&i);
    }
    all_algos.push_back(&inplace_matmul);
    all_algos.push_back(&a1x1x1);
    all_algos.push_back(&group);

    for (auto&& algo : all_algos) {
        m_all_algos_map.emplace(algo->info().desc, algo);
    }
}

MEGDNN_DEF_GET_ALGO_FROM_DESC(Convolution3DForwardImpl)

Convolution3DForwardImpl::AlgoCUDNN* Convolution3DForwardImpl::AlgoPack::
        cudnn_from_enum(cudnnConvolutionFwdAlgo_t algo) {
    for (auto&& i : cudnn) {
        if (i.cudnn_enum() == algo)
            return &i;
    }
    megdnn_throw(
            ssprintf("can not find cudnn fwd algorithm %d", static_cast<int>(algo)));
}

Convolution3DForwardImpl::AlgoPack Convolution3DForwardImpl::sm_algo_pack;

Convolution3DForwardImpl::AlgoBase::SizeArgs::SizeArgs(
        const Convolution3DForwardImpl* o, const TensorLayout& src,
        const TensorLayout& filter, const TensorLayout& dst)
        : SizeArgs(
                  o, src, filter, o->make_canonized_filter_meta(src.ndim, filter),
                  dst) {}

Convolution3DForwardImpl::AlgoBase::SizeArgs::SizeArgs(
        const Convolution3DForwardImpl* o, const TensorLayout& src,
        const TensorLayout& filter, const CanonizedFilterMeta& filter_meta,
        const TensorLayout& dst)
        : ForwardSizeArgs{concrete_handle(o->handle()),
                          &src,
                          &filter,
                          filter_meta,
                          &dst,
                          o->param().data_type},
          opr{o} {}

Convolution3DForwardImpl::AlgoBase::ExecArgs::ExecArgs(
        const Convolution3DForwardImpl* opr, _megdnn_tensor_in src,
        _megdnn_tensor_in filter, _megdnn_tensor_out dst, _megdnn_workspace workspace)
        : SizeArgs(opr, src.layout, filter.layout, dst.layout),
          src_tensor{&src},
          filter_tensor{&filter},
          dst_tensor{&dst},
          workspace{workspace} {}

std::string Convolution3DForwardImpl::AlgoBase::SizeArgs::to_string() const {
    auto&& fm = filter_meta;
    MEGDNN_MARK_USED_VAR(fm);
    return ssprintf(
            "src=%s, filter=%u{%u,%u,%u,%u,%u}, dst=%s, "
            "pad=%ux%ux%u, stride=%ux%ux%u, dilate=%ux%ux%u, xcorr=%d, "
            "dtype=%s,%s",
            src_layout->to_string().c_str(), fm.group, fm.ocpg, fm.icpg, fm.spatial[0],
            fm.spatial[1], fm.spatial[2], dst_layout->to_string().c_str(),
            fm.padding[0], fm.padding[1], fm.padding[2], fm.stride[0], fm.stride[1],
            fm.stride[2], fm.dilation[0], fm.dilation[1], fm.dilation[2],
            !fm.should_flip, src_layout->dtype.name(), dst_layout->dtype.name());
}

// vim: syntax=cpp.doxygen
