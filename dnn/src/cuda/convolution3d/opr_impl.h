/**
 * \file dnn/src/cuda/convolution3d/opr_impl.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#pragma once

#include "megdnn/oprs/nn.h"

namespace megdnn {
namespace cuda {

class Convolution3DForwardImpl : public Convolution3DForward {
public:
    using Convolution3DForward::Convolution3DForward;
    void exec(_megdnn_tensor_in src, _megdnn_tensor_in filter,
              _megdnn_tensor_out dst, _megdnn_workspace workspace) override;
    AlgorithmInfo get_algorithm_info_heuristic(const TensorLayout& src,
                                       const CanonizedFilterMeta& filter,
                                       const TensorLayout& dst,
                                       size_t workspace_limit_in_bytes,
                                       bool reproducible) {
        return get_algorithm_heuristic(src, filter, dst,
                                       workspace_limit_in_bytes, reproducible)
                ->info();
    }
    size_t get_workspace_in_bytes(const TensorLayout& src,
                                  const TensorLayout& filter,
                                  const TensorLayout& dst) override;
    const char* get_algorithm_set_name() const override;
    class AlgoBase;
    class AlgoCUDNN;
    class Algo1x1x1;
    class AlgoInplaceMatmul;
    class AlgoChanwise;
    class AlgoGroupConvGeneral;
    class AlgoPack;
    static const AlgoPack& algo_pack() { return sm_algo_pack; }
    static AlgoBase* get_algo_from_desc(const AlgorithmDesc& desc);

protected:
    std::vector<Algorithm*> get_all_algorithms(
            const TensorLayout& src, const TensorLayout& filter,
            const TensorLayout& dst) override;
    Algorithm* get_algorithm_heuristic(const TensorLayout& src,
                                       const TensorLayout& filter,
                                       const TensorLayout& dst,
                                       size_t workspace_limit_in_bytes,
                                       bool reproducible) override;

private:
    Algorithm* get_algorithm_heuristic(const TensorLayout& src,
                                       const CanonizedFilterMeta& filter,
                                       const TensorLayout& dst,
                                       size_t workspace_limit_in_bytes,
                                       bool reproducible);


    static AlgoPack sm_algo_pack;
};

class Convolution3DBackwardDataImpl : public Convolution3DBackwardData {
public:
    using Convolution3DBackwardData::Convolution3DBackwardData;
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
    class AlgoCUDNN;
    class AlgoInplaceMatmul;
    class AlgoChanwise;
    class AlgoGroupConvGeneral;

    class AlgoPack;

    static const AlgoPack& algo_pack() { return sm_algo_pack; }
    static AlgoBase* get_algo_from_desc(const AlgorithmDesc& desc);

protected:
    std::vector<Algorithm*> get_all_algorithms(
            const TensorLayout& filter, const TensorLayout& diff,
            const TensorLayout& grad) override;
    Algorithm* get_algorithm_heuristic(const TensorLayout& filter,
                                       const TensorLayout& diff,
                                       const TensorLayout& grad,
                                       size_t workspace_limit_in_bytes,
                                       bool reproducible) override;

private:
    Algorithm* get_algorithm_heuristic(const CanonizedFilterMeta& filter,
                                       const TensorLayout& diff,
                                       const TensorLayout& grad,
                                       size_t workspace_limit_in_bytes,
                                       bool reproducible);

    static AlgoPack sm_algo_pack;
};

class Convolution3DBackwardFilterImpl : public Convolution3DBackwardFilter {
public:
    using Convolution3DBackwardFilter::Convolution3DBackwardFilter;
    void exec(_megdnn_tensor_in src, _megdnn_tensor_in diff,
              _megdnn_tensor_out grad, _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(const TensorLayout& src,
                                  const TensorLayout& diff,
                                  const TensorLayout& grad) override;
    AlgorithmInfo get_algorithm_info_heuristic(const TensorLayout& src,
                                               const TensorLayout& diff,
                                               const CanonizedFilterMeta& grad,
                                               size_t workspace_limit_in_bytes,
                                               bool reproducible) {
        return get_algorithm_heuristic(src, diff, grad,
                                       workspace_limit_in_bytes, reproducible)
                ->info();
    }

    const char* get_algorithm_set_name() const override;

    class AlgoBase;
    class AlgoCUDNN;
    class AlgoInplaceMatmul;
    class AlgoChanwise;
    class AlgoGroupConvGeneral;

    class AlgoPack;

    static const AlgoPack& algo_pack() { return sm_algo_pack; }
    static AlgoBase* get_algo_from_desc(const AlgorithmDesc& desc);

protected:
    std::vector<Algorithm*> get_all_algorithms(
            const TensorLayout& src, const TensorLayout& diff,
            const TensorLayout& grad) override;
    Algorithm* get_algorithm_heuristic(const TensorLayout& src,
                                       const TensorLayout& diff,
                                       const TensorLayout& grad,
                                       size_t workspace_limit_in_bytes,
                                       bool reproducible) override;

private:
    Algorithm* get_algorithm_heuristic(const TensorLayout& src,
                                       const TensorLayout& diff,
                                       const CanonizedFilterMeta& grad,
                                       size_t workspace_limit_in_bytes,
                                       bool reproducible);

    static AlgoPack sm_algo_pack;
};
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
