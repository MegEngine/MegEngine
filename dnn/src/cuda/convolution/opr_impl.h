/**
 * \file dnn/src/cuda/convolution/opr_impl.h
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

#include "megdnn/oprs/nn.h"
#include "src/common/utils.h"

namespace megdnn {
namespace cuda {

class ConvolutionForwardImpl : public ConvolutionForward {
public:
    using ConvolutionForward::ConvolutionForward;
    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_in filter, _megdnn_tensor_out dst,
            const PreprocessedFilter* preprocessed_filter,
            _megdnn_workspace workspace) override;

    size_t get_workspace_in_bytes(
            const TensorLayout& src, const TensorLayout& filter,
            const TensorLayout& dst,
            const PreprocessedFilter* preprocessed_filter) override;
    const char* get_algorithm_set_name() const override;

    SmallVector<TensorLayout> deduce_preprocessed_filter_layout(
            const TensorLayout&, const TensorLayout&, const TensorLayout&) override {
        return {};
    }
    size_t get_preprocess_workspace_in_bytes(
            const TensorLayout&, const TensorLayout&, const TensorLayout&) override {
        return 0;
    }
    void exec_preprocess(
            const TensorLayout&, _megdnn_tensor_in, const TensorLayout&,
            PreprocessedFilter*, _megdnn_workspace) override {
        megdnn_throw("cuda exec_preprocess has not implemeted yet");
    }

    Algorithm* get_algorithm_from_desc(const AlgorithmDesc& desc) override;

    class AlgoBase;
    class AlgoDefault;
    class AlgoPack;

    static const AlgoPack& algo_pack() { return sm_algo_pack; }

protected:
    std::vector<Algorithm*> get_all_algorithms(
            const TensorLayout& src, const TensorLayout& filter,
            const TensorLayout& dst) override;

    std::vector<Algorithm*> get_all_algorithms_safe(
            const TensorLayout& src, const TensorLayout& filter,
            const TensorLayout& dst) override;
    Algorithm* get_algorithm_heuristic(
            const TensorLayout& src, const TensorLayout& filter,
            const TensorLayout& dst, size_t workspace_limit_in_bytes,
            const AlgoAttribute& positive_attr,
            const AlgoAttribute& negative_attr) override;

private:
    static AlgoPack sm_algo_pack;
};

class ConvolutionBackwardDataImpl : public ConvolutionBackwardData {
public:
    using ConvolutionBackwardData::ConvolutionBackwardData;
    void exec(
            _megdnn_tensor_in filter, _megdnn_tensor_in diff, _megdnn_tensor_out grad,
            _megdnn_workspace workspace) override;
    AlgorithmInfo get_algorithm_info_heuristic(
            const TensorLayout& filter, const TensorLayout& diff,
            const TensorLayout& grad, size_t workspace_limit_in_bytes,
            const AlgoAttribute& positive_attr, const AlgoAttribute& negative_attr) {
        return get_algorithm_heuristic(
                       filter, diff, grad, workspace_limit_in_bytes, positive_attr,
                       negative_attr)
                ->info();
    }

    size_t get_workspace_in_bytes(
            const TensorLayout& filter, const TensorLayout& diff,
            const TensorLayout& grad) override;
    const char* get_algorithm_set_name() const override;

    class AlgoBase;
    class AlgoCUDNN;
    class AlgoMatmul;
    class AlgoChanwise;
    class AlgoChanwiseSmall;
    class AlgoGroupConvGeneral;
    class AlgoBFloat16;
    class AlgoInt8NCHW4DotProdImplicitGemm;
    class AlgoInt8NCHWDotProdImplicitGemm;
    class AlgoInt8NHWCIMMAImplicitGemm;

    class AlgoPack;

    static const AlgoPack& algo_pack() { return sm_algo_pack; }

    Algorithm* get_algorithm_from_desc(const AlgorithmDesc& desc) override;

protected:
    std::vector<Algorithm*> get_all_algorithms(
            const TensorLayout& filter, const TensorLayout& diff,
            const TensorLayout& grad) override;

    std::vector<Algorithm*> get_all_algorithms_safe(
            const TensorLayout& filter, const TensorLayout& diff,
            const TensorLayout& grad) override;
    Algorithm* get_algorithm_heuristic(
            const TensorLayout& filter, const TensorLayout& diff,
            const TensorLayout& grad, size_t workspace_limit_in_bytes,
            const AlgoAttribute& positive_attr,
            const AlgoAttribute& negative_attr) override;

private:
    static AlgoPack sm_algo_pack;
};

class ConvolutionBackwardFilterImpl : public ConvolutionBackwardFilter {
public:
    using ConvolutionBackwardFilter::ConvolutionBackwardFilter;
    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_in diff, _megdnn_tensor_out grad,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout& src, const TensorLayout& diff,
            const TensorLayout& grad) override;
    AlgorithmInfo get_algorithm_info_heuristic(
            const TensorLayout& filter, const TensorLayout& diff,
            const TensorLayout& grad, size_t workspace_limit_in_bytes,
            const AlgoAttribute& positive_attr, const AlgoAttribute& negative_attr) {
        return get_algorithm_heuristic(
                       filter, diff, grad, workspace_limit_in_bytes, positive_attr,
                       negative_attr)
                ->info();
    }

    const char* get_algorithm_set_name() const override;

    class AlgoBase;
    class AlgoCUDNN;
    class AlgoMatmul;
    class AlgoChanwise;
    class AlgoGroupConvGeneral;
    class AlgoBFloat16;

    class AlgoPack;

    static const AlgoPack& algo_pack() { return sm_algo_pack; }

    Algorithm* get_algorithm_from_desc(const AlgorithmDesc& desc) override;

protected:
    std::vector<Algorithm*> get_all_algorithms(
            const TensorLayout& src, const TensorLayout& diff,
            const TensorLayout& grad) override;

    std::vector<Algorithm*> get_all_algorithms_safe(
            const TensorLayout& src, const TensorLayout& diff,
            const TensorLayout& grad) override;
    Algorithm* get_algorithm_heuristic(
            const TensorLayout& src, const TensorLayout& diff, const TensorLayout& grad,
            size_t workspace_limit_in_bytes, const AlgoAttribute& positive_attr,
            const AlgoAttribute& negative_attr) override;

private:
    static AlgoPack sm_algo_pack;
};

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
