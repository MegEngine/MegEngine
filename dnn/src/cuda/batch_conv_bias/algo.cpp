/**
 * \file dnn/src/cuda/batch_conv_bias/algo.cpp
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

BatchConvBiasForwardImpl::AlgoPack::AlgoPack() {
    all_algos.push_back(&int8_nchw4_gemm_dotprod);
    all_algos.push_back(&int8_nchw4_implicit_gemm_dotprod);

    for (auto&& algo : all_algos) {
        m_all_algos_map.emplace(algo->info().desc, algo);
    }
}

MEGDNN_DEF_GET_ALGO_FROM_DESC(BatchConvBiasForwardImpl)

BatchConvBiasForwardImpl::AlgoPack BatchConvBiasForwardImpl::sm_algo_pack;

BatchConvBiasForwardImpl::AlgoBase::SizeArgs::SizeArgs(
        BatchConvBiasForwardImpl* o, const TensorLayout& src,
        const TensorLayout& filter, const TensorLayout& bias,
        const TensorLayout& z, const TensorLayout& dst)
        : opr{o},
          src_layout{src},
          filter_layout{filter},
          bias_layout{bias},
          z_layout{z},
          dst_layout{dst} {}

BatchConvBiasForwardImpl::AlgoBase::ExecArgs::ExecArgs(
        BatchConvBiasForwardImpl* opr, _megdnn_tensor_in src,
        _megdnn_tensor_in filter, _megdnn_tensor_in bias, _megdnn_tensor_in z,
        _megdnn_tensor_out dst, _megdnn_workspace workspace)
        : SizeArgs(opr, src.layout, filter.layout, bias.layout, z.layout,
                   dst.layout),
          src_tensor{&src},
          filter_tensor{&filter},
          bias_tensor{&bias},
          z_tensor{&z},
          dst_tensor{&dst},
          workspace{workspace} {}

std::string BatchConvBiasForwardImpl::AlgoBase::SizeArgs::to_string() const {
    auto&& param = opr->param();
    MEGDNN_MARK_USED_VAR(param);
    return megdnn_mangle(ssprintf(
            "src=%s, filter=%s, bias=%s, z=%s, dst=%s, "
            "pad=%ux%u, stride=%ux%u, dilate=%ux%u, xcorr=%d, "
            "dtype=(%s(src),%s(flt),%s(bias),%s(z))->(%s(dst))",
            src_layout.to_string().c_str(), filter_layout.to_string().c_str(),
            bias_layout.to_string().c_str(), z_layout.to_string().c_str(),
            dst_layout.to_string().c_str(), param.pad_h, param.pad_w,
            param.stride_h, param.stride_w, param.dilate_h, param.dilate_w,
            static_cast<int>(param.mode), src_layout.dtype.name(),
            filter_layout.dtype.name(), bias_layout.dtype.name(),
            z_layout.dtype.name(), dst_layout.dtype.name()));
}

// vim: syntax=cpp.doxygen
