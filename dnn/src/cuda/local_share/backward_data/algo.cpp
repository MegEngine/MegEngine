/**
 * \file dnn/src/cuda/local_share/backward_data/algo.cpp
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

LocalShareBackwardDataImpl::AlgoPack::AlgoPack() {
    all_algos.push_back(&implicit_gemm);
    all_algos.push_back(&batched_matmul);

    for (auto&& algo : all_algos) {
        m_all_algos_map.emplace(algo->info().desc, algo);
    }
}

MEGDNN_DEF_GET_ALGO_FROM_DESC(LocalShareBackwardDataImpl)

LocalShareBackwardDataImpl::AlgoPack LocalShareBackwardDataImpl::sm_algo_pack;

LocalShareBackwardDataImpl::AlgoBase::SizeArgs::SizeArgs(
        LocalShareBackwardDataImpl* o, const TensorLayout& filter,
        const TensorLayout& diff, const TensorLayout& grad)
        : opr{o}, filter_layout{filter}, diff_layout{diff}, grad_layout{grad} {}

LocalShareBackwardDataImpl::AlgoBase::ExecArgs::ExecArgs(LocalShareBackwardDataImpl* opr,
                                                    _megdnn_tensor_in filter,
                                                    _megdnn_tensor_in diff,
                                                    _megdnn_tensor_out grad,
                                                    _megdnn_workspace workspace)
        : SizeArgs(opr, filter.layout, diff.layout, grad.layout),
          filter_tensor{&filter},
          diff_tensor{&diff},
          grad_tensor{&grad},
          workspace{workspace} {}

std::string LocalShareBackwardDataImpl::AlgoBase::SizeArgs::to_string() const {
    auto&& param = opr->param();
    MEGDNN_MARK_USED_VAR(param);
    return megdnn_mangle(ssprintf(
            "filter=%s, diff=%s, grad=%s, "
            "pad=%ux%u, stride=%ux%u, dilate=%ux%u, xcorr=%d, dtype=%s,%s->%s",
            filter_layout.to_string().c_str(), diff_layout.to_string().c_str(),
            grad_layout.to_string().c_str(), param.pad_h, param.pad_w,
            param.stride_h, param.stride_w, param.dilate_h, param.dilate_w,
            static_cast<int>(param.mode), filter_layout.dtype.name(),
            diff_layout.dtype.name(), grad_layout.dtype.name()));
}

// vim: syntax=cpp.doxygen
