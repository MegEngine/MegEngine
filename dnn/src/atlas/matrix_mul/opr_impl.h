#pragma once
#include "megdnn/oprs.h"

namespace megdnn {
namespace atlas {

class MatrixMulForwardImpl : public MatrixMulForward {
public:
    using MatrixMulForward::MatrixMulForward;
    void exec(
            _megdnn_tensor_in A, _megdnn_tensor_in B, _megdnn_tensor_out C,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout&, const TensorLayout&, const TensorLayout&) override;
    bool is_thread_safe() const override { return true; }
    const char* get_algorithm_set_name() const override;
    Algorithm* get_algorithm_from_desc(const AlgorithmDesc& desc) override;

    class AlgoBase;
    class AlgoACL;
    class AlgoPack;

    static const AlgoPack& algo_pack() { return sm_algo_pack; }

protected:
    std::vector<Algorithm*> get_all_algorithms(
            const TensorLayout& A, const TensorLayout& B,
            const TensorLayout& C) override;
    std::vector<Algorithm*> get_all_algorithms_safe(
            const TensorLayout& A, const TensorLayout& B,
            const TensorLayout& C) override;
    Algorithm* get_algorithm_heuristic(
            const TensorLayout& A, const TensorLayout& B, const TensorLayout& C,
            size_t workspace_limit_in_bytes, const AlgoAttribute& positive_attr,
            const AlgoAttribute& negative_attr) override;

private:
    static AlgoPack sm_algo_pack;
};

}  // namespace atlas
}  // namespace megdnn

// vim: syntax=cpp.doxygen
