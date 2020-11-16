/**
 * \file dnn/src/cuda/local_share/opr_impl.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "megdnn/oprs.h"
#include "src/cuda/utils.h"

namespace megdnn {
namespace cuda {

class LocalShareForwardImpl : public LocalShareForward {
public:
    using LocalShareForward::LocalShareForward;
    void exec(_megdnn_tensor_in src, _megdnn_tensor_in filter,
              _megdnn_tensor_out dst, _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(const TensorLayout& src,
                                  const TensorLayout& filter,
                                  const TensorLayout& dst) override;
    const char* get_algorithm_set_name() const override;

    class AlgoBase;
    class AlgoCHWNBatchSizeAware;
    class AlgoCHWNBatchSizeAwareSmallImage;
    class AlgoBatchedMatMul;

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
    static AlgoPack sm_algo_pack;
};

class LocalShareBackwardDataImpl : public LocalShareBackwardData {
public:
    using LocalShareBackwardData::LocalShareBackwardData;
    void exec(_megdnn_tensor_in filter, _megdnn_tensor_in diff,
              _megdnn_tensor_out grad, _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(const TensorLayout& filter,
                                  const TensorLayout& diff,
                                  const TensorLayout& grad) override;
    const char* get_algorithm_set_name() const override;

    class AlgoBase;
    class AlgoImplicitGemm;
    class AlgoBatchedMatMul;

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
    static AlgoPack sm_algo_pack;
};

class LocalShareBackwardFilterImpl : public LocalShareBackwardFilter {
public:
    using LocalShareBackwardFilter::LocalShareBackwardFilter;
    void exec(_megdnn_tensor_in src, _megdnn_tensor_in diff,
              _megdnn_tensor_out grad, _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(const TensorLayout& src,
                                  const TensorLayout& diff,
                                  const TensorLayout& grad) override;
    const char* get_algorithm_set_name() const override;

    class AlgoBase;
    class AlgoImplicitGemm;
    class AlgoBatchedMatMul;

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
    static AlgoPack sm_algo_pack;
};

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
