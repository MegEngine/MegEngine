/**
 * \file dnn/src/x86/pooling/opr_impl.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#pragma once
#include "src/fallback/pooling/opr_impl.h"

namespace megdnn {
namespace x86 {

class PoolingImpl : public fallback::PoolingImpl {
private:
    class AlgoMeanW2S2AVX;
    class AlgoMeanW2S2SSE3;
    class AlgoMaxW2S2SSE;
    class AlgoMaxW3S3SSE;
#if MEGDNN_X86_WITH_MKL_DNN
    class AlgoMKLDNNNCHW;
    class AlgoMKLDNNNCHW88;
#endif
    class AlgoFallback;
    class AlgoPack;
    static AlgoPack sm_algo_pack;

public:
    using fallback::PoolingImpl::PoolingImpl;
    class AlgoBase;
    void exec(_megdnn_tensor_in src, _megdnn_tensor_out dst,
              _megdnn_workspace) override;
    size_t get_workspace_in_bytes(const TensorLayout&,
                                  const TensorLayout&) override;

    static size_t constexpr MAX_SPATIAL_DIM = 2;

    const char* get_algorithm_set_name() const override {
        return "X86_POOLING_FORWARD";
    }
    Algorithm* get_algorithm_from_desc(const AlgorithmDesc& desc) override;

    AlgorithmInfo get_algorithm_info_heuristic(
            const TensorLayout& src, const TensorLayout& dst,
            size_t workspace_limit_in_bytes, const AlgoAttribute& positive_attr,
            const AlgoAttribute& negative_attr) {
        return get_algorithm_heuristic(src, dst, workspace_limit_in_bytes,
                                       positive_attr, negative_attr)
                ->info();
    }

    static const AlgoPack& algo_pack() { return sm_algo_pack; }

    bool is_fallback_algo(Algorithm* algo) {
        return strcmp(algo->name(), "FALLBACK_POOLING") == 0;
    }

protected:
    std::vector<Algorithm*> get_all_algorithms(
            const TensorLayout& src, const TensorLayout& dst) override;
    Algorithm* get_algorithm_heuristic(
            const TensorLayout& src, const TensorLayout& dst,
            size_t workspace_limit_in_bytes, const AlgoAttribute& positive_attr,
            const AlgoAttribute& negative_attr) override;
};

WorkspaceBundle get_bundle(const TensorLayout& src, const TensorLayout& dst,
                           const param::Pooling& param);

}  // namespace x86
}  // namespace megdnn
// vim: syntax=cpp.doxygen
