#pragma once
#include "megdnn/oprs.h"

namespace megdnn {
namespace naive {

class LocalShareForwardImpl : public LocalShareForward {
public:
    using LocalShareForward::LocalShareForward;
    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_in filter, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) override;

    size_t get_workspace_in_bytes(
            const TensorLayout&, const TensorLayout&, const TensorLayout&) override {
        return 0;
    }

    std::vector<Algorithm*> get_all_algorithms(
            const TensorLayout& /*src*/, const TensorLayout& /*filter*/,
            const TensorLayout& /*dst*/) override;

    std::vector<Algorithm*> get_all_algorithms_safe(
            const TensorLayout& /*src*/, const TensorLayout& /*filter*/,
            const TensorLayout& /*dst*/) override;

    Algorithm* get_algorithm_heuristic(
            const TensorLayout& /*src*/, const TensorLayout& /*filter*/,
            const TensorLayout& /*dst*/, size_t /*workspace_limit_in_bytes*/,
            const AlgoAttribute& /*positive_attr*/,
            const AlgoAttribute& /*negative_attr*/) override;

    Algorithm* get_algorithm_from_desc(const AlgorithmDesc&) override;
    const char* get_algorithm_set_name() const override { return "DEFAULT"; }
};

class LocalShareBackwardDataImpl : public LocalShareBackwardData {
public:
    using LocalShareBackwardData::LocalShareBackwardData;
    void exec(
            _megdnn_tensor_in filter, _megdnn_tensor_in diff, _megdnn_tensor_out grad,
            _megdnn_workspace workspace) override;

    size_t get_workspace_in_bytes(
            const TensorLayout&, const TensorLayout&, const TensorLayout&) override {
        return 0;
    }

    std::vector<Algorithm*> get_all_algorithms(
            const TensorLayout& /*filter*/, const TensorLayout& /*diff*/,
            const TensorLayout& /*grad*/) override;

    std::vector<Algorithm*> get_all_algorithms_safe(
            const TensorLayout& /*filter*/, const TensorLayout& /*diff*/,
            const TensorLayout& /*grad*/) override;

    Algorithm* get_algorithm_heuristic(
            const TensorLayout& /*filter*/, const TensorLayout& /*diff*/,
            const TensorLayout& /*grad*/, size_t /*workspace_limit_in_bytes*/,
            const AlgoAttribute& /*positive_attr*/,
            const AlgoAttribute& /*negative_attr*/) override;

    Algorithm* get_algorithm_from_desc(const AlgorithmDesc&) override;
    const char* get_algorithm_set_name() const override { return "DEFAULT"; }
};

class LocalShareBackwardFilterImpl : public LocalShareBackwardFilter {
public:
    using LocalShareBackwardFilter::LocalShareBackwardFilter;
    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_in diff, _megdnn_tensor_out grad,
            _megdnn_workspace workspace) override;

    size_t get_workspace_in_bytes(
            const TensorLayout&, const TensorLayout&, const TensorLayout&) override {
        return 0;
    }
    std::vector<Algorithm*> get_all_algorithms(
            const TensorLayout& /*src*/, const TensorLayout& /*diff*/,
            const TensorLayout& /*grad*/) override;

    std::vector<Algorithm*> get_all_algorithms_safe(
            const TensorLayout& /*src*/, const TensorLayout& /*diff*/,
            const TensorLayout& /*grad*/) override;

    Algorithm* get_algorithm_heuristic(
            const TensorLayout& /*src*/, const TensorLayout& /*diff*/,
            const TensorLayout& /*grad*/, size_t /*workspace_limit_in_bytes*/,
            const AlgoAttribute& /*positive_attr*/,
            const AlgoAttribute& /*negative_attr*/) override;

    Algorithm* get_algorithm_from_desc(const AlgorithmDesc&) override;
    const char* get_algorithm_set_name() const override { return "DEFAULT"; }
};

}  // namespace naive
}  // namespace megdnn

// vim: syntax=cpp.doxygen
