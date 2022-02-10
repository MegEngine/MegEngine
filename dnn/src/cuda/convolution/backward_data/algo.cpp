/**
 * \file dnn/src/cuda/convolution/backward_data/algo.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "./algo.h"
#include "src/cuda/utils.h"

using namespace megdnn;
using namespace cuda;

ConvolutionBackwardDataImpl::AlgoPack::AlgoPack() {
    non_cudnn_algos.push_back(&chanwise);
    non_cudnn_algos.push_back(&chanwise_small);
    non_cudnn_algos.push_back(&matmul);

    all_algos.push_back(&chanwise);        // prefer chanwise
    all_algos.push_back(&chanwise_small);  // prefer small chanwise

    fill_cudnn_algos();
    for (auto&& i : cudnn) {
        all_algos.push_back(&i);
    }
    all_algos.push_back(&matmul);

    fill_int8_dp4a_algos();
    for (auto&& algo : int8_nchw4_dotprod) {
        all_algos.push_back(&algo);
        int8_algos.push_back(&algo);
    }

    fill_int8_imma_algos();
    for (auto&& algo : int8_nhwc_imma) {
        all_algos.push_back(&algo);
        int8_algos.push_back(&algo);
    }
    fill_dwconv_algos();

    int8_algos.push_back(&int8_nchw_dotprod);
    all_algos.push_back(&int8_nchw_dotprod);

    all_algos.push_back(&bfloat16);
    bfloat16_algos.push_back(&bfloat16);
    all_algos.push_back(&group);

    for (auto&& algo : all_algos) {
        m_all_algos_map.emplace(algo->info().desc, algo);
    }
}

void ConvolutionBackwardDataImpl::AlgoPack::fill_dwconv_algos() {
    {
        using AlgoParam = AlgoFloat32NCHWFMAImplicitBatchedGemm::AlgoParam;
        /// preferred algo
        implbmm_nchw_fma.emplace_back(AlgoParam{64, 128, 8, 32, 64, 8, 2});
        implbmm_nchw_fma.emplace_back(AlgoParam{128, 128, 8, 32, 64, 8, 2});
        implbmm_nchw_fma.emplace_back(AlgoParam{128, 64, 8, 64, 32, 8, 2});
        implbmm_nchw_fma.emplace_back(AlgoParam{128, 32, 8, 64, 32, 8, 2});
        implbmm_nchw_fma.emplace_back(AlgoParam{32, 128, 8, 32, 64, 8, 2});
        implbmm_nchw_fma.emplace_back(AlgoParam{64, 64, 8, 32, 64, 8, 2});
        implbmm_nchw_fma.emplace_back(AlgoParam{32, 64, 8, 32, 64, 8, 2});
        implbmm_nchw_fma.emplace_back(AlgoParam{32, 32, 8, 32, 32, 8, 2});
        implbmm_nchw_fma.emplace_back(AlgoParam{64, 32, 8, 64, 32, 8, 2});
        for (auto&& algo : implbmm_nchw_fma) {
            all_algos.push_back(&algo);
        }
    }
#if CUDA_VERSION >= 10020
    {
        using AlgoParam = AlgoFloat16NCHWHMMAImplicitBatchedGemm::AlgoParam;
        /// preferred algo
        implbmm_nchw_hmma.emplace_back(AlgoParam{64, 128, 32, 32, 32, 32, 8, 8, 4, 2});
        implbmm_nchw_hmma.emplace_back(AlgoParam{128, 128, 32, 32, 32, 32, 8, 8, 4, 2});
        implbmm_nchw_hmma.emplace_back(AlgoParam{128, 256, 32, 64, 64, 32, 8, 8, 4, 2});
        implbmm_nchw_hmma.emplace_back(AlgoParam{128, 64, 32, 32, 32, 32, 8, 8, 4, 2});
        implbmm_nchw_hmma.emplace_back(AlgoParam{64, 64, 32, 32, 32, 32, 8, 8, 4, 2});
        for (auto&& algo : implbmm_nchw_hmma) {
            all_algos.push_back(&algo);
        }
    }
#endif
}

MEGDNN_DEF_GET_ALGO_FROM_DESC(ConvolutionBackwardDataImpl)

ConvolutionBackwardDataImpl::AlgoCUDNN* ConvolutionBackwardDataImpl::AlgoPack::
        cudnn_from_enum(cudnnConvolutionBwdDataAlgo_t algo) {
    for (auto&& i : cudnn) {
        if (i.cudnn_enum() == algo)
            return &i;
    }
    megdnn_throw(ssprintf(
            "can not find cudnn bwd_data algorithm %d", static_cast<int>(algo)));
}

ConvolutionBackwardDataImpl::AlgoPack ConvolutionBackwardDataImpl::sm_algo_pack;

ConvolutionBackwardDataImpl::AlgoBase::SizeArgs::SizeArgs(
        const ConvolutionBackwardDataImpl* o, const TensorLayout& filter,
        const TensorLayout& diff, const TensorLayout& grad)
        : SizeArgs(
                  o, filter, o->make_canonized_filter_meta(grad.ndim, filter), diff,
                  grad) {}

ConvolutionBackwardDataImpl::AlgoBase::SizeArgs::SizeArgs(
        const ConvolutionBackwardDataImpl* o, const TensorLayout& filter,
        const CanonizedFilterMeta& filter_meta, const TensorLayout& diff,
        const TensorLayout& grad)
        : handle{concrete_handle(o->handle())},
          filter_meta{filter_meta},
          diff_layout{&diff},
          grad_layout{&grad},
          filter_layout{&filter},
          opr{o} {}

ConvolutionBackwardDataImpl::AlgoBase::ExecArgs::ExecArgs(
        const ConvolutionBackwardDataImpl* opr, _megdnn_tensor_in filter,
        _megdnn_tensor_in diff, _megdnn_tensor_out grad, _megdnn_workspace workspace)
        : SizeArgs(opr, filter.layout, diff.layout, grad.layout),
          filter_tensor{&filter},
          diff_tensor{&diff},
          grad_tensor{&grad},
          workspace{workspace} {}

std::string ConvolutionBackwardDataImpl::AlgoBase::SizeArgs::to_string() const {
    auto&& fm = filter_meta;
    MEGDNN_MARK_USED_VAR(fm);
    return ssprintf(
            "filter=%u{%u,%u,%u,%u}, diff=%s, grad=%s, "
            "pad=%ux%u, stride=%ux%u, dilate=%ux%u, xcorr=%d, dtype=%s,%s",
            fm.group, fm.ocpg, fm.icpg, fm.spatial[0], fm.spatial[1],
            diff_layout->to_string().c_str(), grad_layout->to_string().c_str(),
            fm.padding[0], fm.padding[1], fm.stride[0], fm.stride[1], fm.dilation[0],
            fm.dilation[1], !fm.should_flip, diff_layout->dtype.name(),
            grad_layout->dtype.name());
}

// vim: syntax=cpp.doxygen
