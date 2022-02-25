#pragma once
#include "megdnn/oprs.h"
#include "src/common/utils.h"

namespace megdnn {
namespace naive {

class BatchConvBiasForwardImpl : public BatchConvBiasForward {
public:
    using BatchConvBiasForward::BatchConvBiasForward;
    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_in filter, _megdnn_tensor_in bias,
            _megdnn_tensor_in z, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) override;

    size_t get_workspace_in_bytes(
            const TensorLayout&, const TensorLayout&, const TensorLayout&,
            const TensorLayout&, const TensorLayout&) override;

    std::vector<Algorithm*> get_all_algorithms(
            const TensorLayout& src, const TensorLayout& filter,
            const TensorLayout& bias, const TensorLayout& z,
            const TensorLayout& dst) override;

    std::vector<Algorithm*> get_all_algorithms_safe(
            const TensorLayout& src, const TensorLayout& filter,
            const TensorLayout& bias, const TensorLayout& z,
            const TensorLayout& dst) override;

    Algorithm* get_algorithm_heuristic(
            const TensorLayout& src, const TensorLayout& filter,
            const TensorLayout& bias, const TensorLayout& z, const TensorLayout& dst,
            size_t workspace_limit_in_bytes, const AlgoAttribute& positive_attr,
            const AlgoAttribute& negative_attr) override;

    Algorithm* get_algorithm_from_desc(const AlgorithmDesc&) override;

    const char* get_algorithm_set_name() const override { return "DEFAULT"; }

private:
    WorkspaceBundle get_workspace_bundle(
            dt_byte* raw_ptr, const TensorLayout& src, const TensorLayout& filter,
            const TensorLayout& bias, const TensorLayout& z, const TensorLayout& dst);
};

}  // namespace naive
}  // namespace megdnn

// vim: syntax=cpp.doxygen
