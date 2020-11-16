/**
 * \file dnn/src/rocm/convolution/opr_impl.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include "megdnn/oprs/nn.h"
#include "src/common/utils.h"

namespace megdnn {
namespace rocm {

class ConvolutionForwardImpl : public ConvolutionForward {
public:
    using ConvolutionForward::ConvolutionForward;
    void exec(_megdnn_tensor_in src, _megdnn_tensor_in filter,
              _megdnn_tensor_out dst,
              const PreprocessedFilter* preprocessed_filter,
              _megdnn_workspace workspace) override;
    AlgorithmInfo get_algorithm_info_heuristic(
            const TensorLayout& src, const CanonizedFilterMeta& filter,
            const TensorLayout& dst, size_t workspace_limit_in_bytes,
            bool reproducible) {
        return get_algorithm_heuristic(src, filter, dst,
                                       workspace_limit_in_bytes, reproducible)
                ->info();
    }
    size_t get_workspace_in_bytes(const TensorLayout& src,
                                  const TensorLayout& filter,
                                  const TensorLayout& dst,
                                  const PreprocessedFilter*) override;

    size_t get_preprocess_workspace_in_bytes(const TensorLayout&,
                                             const TensorLayout&,
                                             const TensorLayout&) override {
        return 0;
    }

    void exec_preprocess(const TensorLayout&, _megdnn_tensor_in,
                         const TensorLayout&, PreprocessedFilter*,
                         _megdnn_workspace) override {
        megdnn_throw("convolution exec_preprocess has not implemented yet");
    }

    SmallVector<TensorLayout> deduce_preprocessed_filter_layout(
            const TensorLayout&, const TensorLayout&,
            const TensorLayout&) override {
        return {};
    }
    const char* get_algorithm_set_name() const override;

    class AlgoBase;
    class AlgoMIOpen;
    class AlgoMatmul;
    class AlgoInplaceMatmul;
    class Algo1x1;
    class Algo1x1LargeBatch;
    class AlgoChanwise;

    class AlgoPack;

    static const AlgoPack& algo_pack() { return sm_algo_pack; }
    static AlgoBase* get_algo_from_desc(const AlgorithmDesc& desc);

private:
    std::vector<Algorithm*> get_all_algorithms(
            const TensorLayout& src, const TensorLayout& filter,
            const TensorLayout& dst) override;
    Algorithm* get_algorithm_heuristic(const TensorLayout& src,
                                       const TensorLayout& filter,
                                       const TensorLayout& dst,
                                       size_t workspace_limit_in_bytes,
                                       bool reproducible) override;
    Algorithm* get_algorithm_heuristic(const TensorLayout& src,
                                       const CanonizedFilterMeta& filter,
                                       const TensorLayout& dst,
                                       size_t workspace_limit_in_bytes,
                                       bool reproducible);

    static AlgoPack sm_algo_pack;
};

class ConvolutionBackwardDataImpl : public ConvolutionBackwardData {
public:
    using ConvolutionBackwardData::ConvolutionBackwardData;
    void exec(_megdnn_tensor_in filter, _megdnn_tensor_in diff,
              _megdnn_tensor_out grad, _megdnn_workspace workspace) override;
    AlgorithmInfo get_algorithm_info_heuristic(
            const CanonizedFilterMeta& filter, const TensorLayout& diff,
            const TensorLayout& grad, size_t workspace_limit_in_bytes,
            bool reproducible) {
        return get_algorithm_heuristic(filter, diff, grad,
                                       workspace_limit_in_bytes, reproducible)
                ->info();
    }
    size_t get_workspace_in_bytes(const TensorLayout& filter,
                                  const TensorLayout& diff,
                                  const TensorLayout& grad) override;
    const char* get_algorithm_set_name() const override;

    class AlgoBase;
    class AlgoMIOpen;
    class AlgoMatmul;
    class AlgoChanwise;

    class AlgoPack;

    static const AlgoPack& algo_pack() { return sm_algo_pack; }
    static AlgoBase* get_algo_from_desc(const AlgorithmDesc& desc);

private:
    std::vector<Algorithm*> get_all_algorithms(
            const TensorLayout& filter, const TensorLayout& diff,
            const TensorLayout& grad) override;
    Algorithm* get_algorithm_heuristic(const TensorLayout& filter,
                                       const TensorLayout& diff,
                                       const TensorLayout& grad,
                                       size_t workspace_limit_in_bytes,
                                       bool reproducible) override;
    Algorithm* get_algorithm_heuristic(const CanonizedFilterMeta& filter,
                                       const TensorLayout& diff,
                                       const TensorLayout& grad,
                                       size_t workspace_limit_in_bytes,
                                       bool reproducible);

    static AlgoPack sm_algo_pack;
};

class ConvolutionBackwardFilterImpl : public ConvolutionBackwardFilter {
public:
    using ConvolutionBackwardFilter::ConvolutionBackwardFilter;
    void exec(_megdnn_tensor_in src, _megdnn_tensor_in diff,
              _megdnn_tensor_out grad, _megdnn_workspace workspace) override;
    AlgorithmInfo get_algorithm_info_heuristic(const TensorLayout& src,
                                               const TensorLayout& diff,
                                               const CanonizedFilterMeta& grad,
                                               size_t workspace_limit_in_bytes,
                                               bool reproducible) {
        return get_algorithm_heuristic(src, diff, grad,
                                       workspace_limit_in_bytes, reproducible)
                ->info();
    }
    size_t get_workspace_in_bytes(const TensorLayout& src,
                                  const TensorLayout& diff,
                                  const TensorLayout& grad) override;
    const char* get_algorithm_set_name() const override;

    class AlgoBase;
    class AlgoMIOpen;
    class AlgoMatmul;
    class AlgoChanwise;

    class AlgoPack;

    static AlgoBase* get_algo_from_desc(const AlgorithmDesc& desc);
    static const AlgoPack& algo_pack() { return sm_algo_pack; }

private:
    std::vector<Algorithm*> get_all_algorithms(
            const TensorLayout& src, const TensorLayout& diff,
            const TensorLayout& grad) override;
    Algorithm* get_algorithm_heuristic(const TensorLayout& src,
                                       const TensorLayout& diff,
                                       const TensorLayout& grad,
                                       size_t workspace_limit_in_bytes,
                                       bool reproducible) override;
    Algorithm* get_algorithm_heuristic(const TensorLayout& src,
                                       const TensorLayout& diff,
                                       const CanonizedFilterMeta& grad,
                                       size_t workspace_limit_in_bytes,
                                       bool reproducible);

    static AlgoPack sm_algo_pack;
};

}  // namespace rocm
}  // namespace megdnn

// vim: syntax=cpp.doxygen
