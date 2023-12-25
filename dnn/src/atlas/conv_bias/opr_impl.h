#pragma once
#include "megdnn/oprs.h"

namespace megdnn {
namespace atlas {

class ConvBiasForwardImpl : public ConvBiasForward {
public:
    using ConvBiasForward::ConvBiasForward;
    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_in filter, _megdnn_tensor_in bias,
            _megdnn_tensor_in z, _megdnn_tensor_out dst,
            const PreprocessedFilter* preprocessed_filter,
            _megdnn_workspace workspace) override;

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

    size_t get_workspace_in_bytes(
            const TensorLayout& src, const TensorLayout& filter,
            const TensorLayout& bias, const TensorLayout& z, const TensorLayout& dst,
            const PreprocessedFilter* preprocessed_filter) override;

    size_t get_preprocess_workspace_in_bytes(
            const TensorLayout&, const TensorLayout&, const TensorLayout&,
            const TensorLayout&, const TensorLayout&) override {
        return 0;
    }
    SmallVector<TensorLayout> deduce_preprocessed_filter_layout(
            const TensorLayout&, const TensorLayout&, const TensorLayout&,
            const TensorLayout&, const TensorLayout&) override {
        return {};
    }

    void exec_preprocess(
            const TensorLayout&, _megdnn_tensor_in, _megdnn_tensor_in,
            const TensorLayout&, const TensorLayout&, PreprocessedFilter*,
            _megdnn_workspace) override {}

    const char* get_algorithm_set_name() const override;

    Algorithm* get_algorithm_from_desc(const AlgorithmDesc&) override;

    class AlgoBase;
    class AlgoDefault;
    class AlgoPack;

    static const AlgoPack& algo_pack() { return sm_algo_pack; }

private:
    static AlgoPack sm_algo_pack;
};

}  // namespace atlas
}  // namespace megdnn
// vim: syntax=cpp.doxygen
