/**
 * \file dnn/src/cuda/local_share/forward/algo.cpp
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

LocalShareForwardImpl::AlgoPack::AlgoPack() {
    all_algos.push_back(&batch_size_aware_chwn_small_image);
    all_algos.push_back(&batch_size_aware_chwn);
    all_algos.push_back(&batched_matmul);

    for (auto&& algo : all_algos) {
        m_all_algos_map.emplace(algo->info().desc, algo);
    }
}

MEGDNN_DEF_GET_ALGO_FROM_DESC(LocalShareForwardImpl)

LocalShareForwardImpl::AlgoPack LocalShareForwardImpl::sm_algo_pack;

LocalShareForwardImpl::AlgoBase::SizeArgs::SizeArgs(LocalShareForwardImpl* o,
                                                    const TensorLayout& src,
                                                    const TensorLayout& filter,
                                                    const TensorLayout& dst)
        : opr{o}, src_layout{src}, filter_layout{filter}, dst_layout{dst} {}

LocalShareForwardImpl::AlgoBase::ExecArgs::ExecArgs(LocalShareForwardImpl* opr,
                                                    _megdnn_tensor_in src,
                                                    _megdnn_tensor_in filter,
                                                    _megdnn_tensor_out dst,
                                                    _megdnn_workspace workspace)
        : SizeArgs(opr, src.layout, filter.layout, dst.layout),
          src_tensor{&src},
          filter_tensor{&filter},
          dst_tensor{&dst},
          workspace{workspace} {}

std::string LocalShareForwardImpl::AlgoBase::SizeArgs::to_string() const {
    auto&& param = opr->param();
    MEGDNN_MARK_USED_VAR(param);
    return megdnn_mangle(ssprintf(
            "src=%s, filter=%s, dst=%s, "
            "pad=%ux%u, stride=%ux%u, dilate=%ux%u, xcorr=%d, dtype=%s,%s",
            src_layout.to_string().c_str(), filter_layout.to_string().c_str(),
            dst_layout.to_string().c_str(), param.pad_h, param.pad_w,
            param.stride_h, param.stride_w, param.dilate_h, param.dilate_w,
            static_cast<int>(param.mode), src_layout.dtype.name(),
            dst_layout.dtype.name()));
}

// vim: syntax=cpp.doxygen
