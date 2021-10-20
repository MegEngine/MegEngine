/**
 * \file dnn/src/naive/pooling/opr_impl.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "megdnn/oprs.h"
#include "src/common/utils.h"

namespace megdnn {
namespace naive {

class PoolingForwardImpl : public PoolingForward {
public:
    using PoolingForward::PoolingForward;
    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(const TensorLayout&, const TensorLayout&) override;

private:
    WorkspaceBundle get_workspace_bundle(
            void* ptr, const TensorLayout&, const TensorLayout&) const;

    const char* get_algorithm_set_name() const override { return "DEFALUT"; }

    Algorithm* get_algorithm_from_desc(const AlgorithmDesc&) override;

    std::vector<Algorithm*> get_all_algorithms(
            const TensorLayout& src, const TensorLayout& dst) override;
    std::vector<Algorithm*> get_all_algorithms_safe(
            const TensorLayout& src, const TensorLayout& dst) override;

    Algorithm* get_algorithm_heuristic(
            const TensorLayout& src, const TensorLayout& dst,
            size_t workspace_limit_in_bytes, const AlgoAttribute& positive_attr,
            const AlgoAttribute& negative_attr) override;
};

class PoolingBackwardImpl : public PoolingBackward {
public:
    using PoolingBackward::PoolingBackward;
    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_in dst, _megdnn_tensor_in diff,
            _megdnn_tensor_out grad, _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout&, const TensorLayout&, const TensorLayout&,
            const TensorLayout&) override;

    const char* get_algorithm_set_name() const override { return "DEFALUT"; }

    Algorithm* get_algorithm_from_desc(const AlgorithmDesc&) override;

    std::vector<Algorithm*> get_all_algorithms(
            const TensorLayout& src, const TensorLayout& dst, const TensorLayout& diff,
            const TensorLayout& grad) override;
    std::vector<Algorithm*> get_all_algorithms_safe(
            const TensorLayout& src, const TensorLayout& dst, const TensorLayout& diff,
            const TensorLayout& grad) override;

    Algorithm* get_algorithm_heuristic(
            const TensorLayout& src, const TensorLayout& dst, const TensorLayout& diff,
            const TensorLayout& grad, size_t workspace_limit_in_bytes,
            const AlgoAttribute& positive_attr,
            const AlgoAttribute& negative_attr) override;

private:
    WorkspaceBundle get_workspace_bundle(
            void* ptr, const TensorLayout&, const TensorLayout&, const TensorLayout&,
            const TensorLayout&) const;
};

}  // namespace naive
}  // namespace megdnn
// vim: syntax=cpp.doxygen
