#pragma once

#include "megdnn/oprs.h"

namespace megdnn {
namespace atlas {

class PoolingForwardImpl final : public PoolingForward {
public:
    using PoolingForward::PoolingForward;
    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout& src, const TensorLayout& dst) override;

    const char* get_algorithm_set_name() const override;
    Algorithm* get_algorithm_from_desc(const AlgorithmDesc& desc) override;

    AlgorithmInfo get_algorithm_info_heuristic(
            const TensorLayout& src, const TensorLayout& dst,
            size_t workspace_limit_in_bytes, const AlgoAttribute& positive_attr,
            const AlgoAttribute& negative_attr) {
        return get_algorithm_heuristic(
                       src, dst, workspace_limit_in_bytes, positive_attr, negative_attr)
                ->info();
    }

    class AlgoBase;
    class AlgoACL;
    class AlgoPack;

    static const AlgoPack& algo_pack() { return sm_algo_pack; }

protected:
    std::vector<Algorithm*> get_all_algorithms(
            const TensorLayout& src, const TensorLayout& dst) override;
    std::vector<Algorithm*> get_all_algorithms_safe(
            const TensorLayout& src, const TensorLayout& dst) override;
    Algorithm* get_algorithm_heuristic(
            const TensorLayout& src, const TensorLayout& dst,
            size_t workspace_limit_in_bytes, const AlgoAttribute& positive_attr,
            const AlgoAttribute& negative_attr) override;

private:
    static AlgoPack sm_algo_pack;
};

class PoolingBackwardImpl final : public PoolingBackward {
public:
    using PoolingBackward::PoolingBackward;
    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_in dst, _megdnn_tensor_in diff,
            _megdnn_tensor_out grad, _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout& src, const TensorLayout& dst, const TensorLayout& diff,
            const TensorLayout& grad) override;

    const char* get_algorithm_set_name() const override;
    Algorithm* get_algorithm_from_desc(const AlgorithmDesc& desc) override;

    AlgorithmInfo get_algorithm_info_heuristic(
            const TensorLayout& src, const TensorLayout& dst, const TensorLayout& diff,
            const TensorLayout& grad, size_t workspace_limit_in_bytes,
            const AlgoAttribute& positive_attr, const AlgoAttribute& negative_attr) {
        return get_algorithm_heuristic(
                       src, dst, diff, grad, workspace_limit_in_bytes, positive_attr,
                       negative_attr)
                ->info();
    }

    class AlgoBase;
    class AlgoACL;
    class AlgoPack;

    static const AlgoPack& algo_pack() { return sm_algo_pack; }

protected:
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
    static AlgoPack sm_algo_pack;
};

}  // namespace atlas
}  // namespace megdnn

// vim: syntax=cpp.doxygen
