#pragma once

#include "megdnn/oprs/nn.h"

namespace megdnn {
namespace cuda {

class DeformableConvForwardImpl : public DeformableConvForward {
public:
    using DeformableConvForward::DeformableConvForward;

    void exec(
            _megdnn_tensor_in im, _megdnn_tensor_in filter, _megdnn_tensor_in offset,
            _megdnn_tensor_in mask, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) override;

    size_t get_workspace_in_bytes(
            const TensorLayout& im, const TensorLayout& filter,
            const TensorLayout& offset, const TensorLayout& mask,
            const TensorLayout& dst) override;

    Algorithm* get_algorithm_heuristic(
            const TensorLayout& im, const CanonizedFilterMeta& filter,
            const TensorLayout& offset, const TensorLayout& mask,
            const TensorLayout& dst, size_t workspace_limit_in_bytes,
            const AlgoAttribute& positive_attr, const AlgoAttribute& negative_attr);

    const char* get_algorithm_set_name() const override;

    class AlgoBase;
    class AlgoMatmul;

    class AlgoPack;

    static const AlgoPack& algo_pack() { return sm_algo_pack; }
    Algorithm* get_algorithm_from_desc(const AlgorithmDesc& desc) override;

protected:
    std::vector<Algorithm*> get_all_algorithms(
            const TensorLayout& im, const TensorLayout& filter,
            const TensorLayout& offset, const TensorLayout& mask,
            const TensorLayout& dst) override;
    std::vector<Algorithm*> get_all_algorithms_safe(
            const TensorLayout& im, const TensorLayout& filter,
            const TensorLayout& offset, const TensorLayout& mask,
            const TensorLayout& dst) override;

    Algorithm* get_algorithm_heuristic(
            const TensorLayout& im, const TensorLayout& filter,
            const TensorLayout& offset, const TensorLayout& mask,
            const TensorLayout& dst, size_t workspace_limit_in_bytes,
            const AlgoAttribute& positive_attr,
            const AlgoAttribute& negative_attr) override;

private:
    static AlgoPack sm_algo_pack;
};

class DeformableConvBackwardFilterImpl : public DeformableConvBackwardFilter {
public:
    using DeformableConvBackwardFilter::DeformableConvBackwardFilter;

    void exec(
            _megdnn_tensor_in im, _megdnn_tensor_in offset, _megdnn_tensor_in mask,
            _megdnn_tensor_in out_grad, _megdnn_tensor_out filter_grad,
            _megdnn_workspace workspace) override;

    Algorithm* get_algorithm_heuristic(
            const TensorLayout& im, const TensorLayout& offset,
            const TensorLayout& mask, const TensorLayout& out_grad,
            const CanonizedFilterMeta& filter_grad, size_t workspace_limit_in_bytes,
            const AlgoAttribute& positive_attr, const AlgoAttribute& negative_attr);

    size_t get_workspace_in_bytes(
            const TensorLayout& im, const TensorLayout& offset,
            const TensorLayout& mask, const TensorLayout& out_grad,
            const TensorLayout& filter_grad) override;

    const char* get_algorithm_set_name() const override;

    class AlgoBase;
    class AlgoMatmul;

    class AlgoPack;

    static const AlgoPack& algo_pack() { return sm_algo_pack; }
    Algorithm* get_algorithm_from_desc(const AlgorithmDesc& desc) override;

protected:
    std::vector<Algorithm*> get_all_algorithms(
            const TensorLayout& im, const TensorLayout& offset,
            const TensorLayout& mask, const TensorLayout& out_grad,
            const TensorLayout& filter_grad) override;
    std::vector<Algorithm*> get_all_algorithms_safe(
            const TensorLayout& im, const TensorLayout& offset,
            const TensorLayout& mask, const TensorLayout& out_grad,
            const TensorLayout& filter_grad) override;

    Algorithm* get_algorithm_heuristic(
            const TensorLayout& im, const TensorLayout& offset,
            const TensorLayout& mask, const TensorLayout& out_grad,
            const TensorLayout& filter_grad, size_t workspace_limit_in_bytes,
            const AlgoAttribute& positive_attr,
            const AlgoAttribute& negative_attr) override;

private:
    static AlgoPack sm_algo_pack;
};

class DeformableConvBackwardDataImpl : public DeformableConvBackwardData {
public:
    using DeformableConvBackwardData::DeformableConvBackwardData;

    void exec(
            _megdnn_tensor_in im, _megdnn_tensor_in filter, _megdnn_tensor_in offset,
            _megdnn_tensor_in mask, _megdnn_tensor_in out_grad,
            _megdnn_tensor_out im_grad, _megdnn_tensor_out offset_grad,
            _megdnn_tensor_out mask_grad, _megdnn_workspace workspace) override;

    Algorithm* get_algorithm_heuristic(
            const TensorLayout& im, const CanonizedFilterMeta& filter,
            const TensorLayout& offset, const TensorLayout& mask,
            const TensorLayout& out_grad, const TensorLayout& im_grad,
            const TensorLayout& offset_grad, const TensorLayout& mask_grad,
            size_t workspace_limit_in_bytes, const AlgoAttribute& positive_attr,
            const AlgoAttribute& negative_attr);

    size_t get_workspace_in_bytes(
            const TensorLayout& im, const TensorLayout& filter,
            const TensorLayout& offset, const TensorLayout& mask,
            const TensorLayout& out_grad, const TensorLayout& im_grad,
            const TensorLayout& offset_grad, const TensorLayout& mask_grad) override;

    const char* get_algorithm_set_name() const override;

    class AlgoBase;
    class AlgoMatmul;

    class AlgoPack;

    static const AlgoPack& algo_pack() { return sm_algo_pack; }
    Algorithm* get_algorithm_from_desc(const AlgorithmDesc& desc) override;

protected:
    std::vector<Algorithm*> get_all_algorithms(
            const TensorLayout& im, const TensorLayout& filter,
            const TensorLayout& offset, const TensorLayout& mask,
            const TensorLayout& out_grad, const TensorLayout& im_grad,
            const TensorLayout& offset_grad, const TensorLayout& mask_grad) override;

    std::vector<Algorithm*> get_all_algorithms_safe(
            const TensorLayout& im, const TensorLayout& filter,
            const TensorLayout& offset, const TensorLayout& mask,
            const TensorLayout& out_grad, const TensorLayout& im_grad,
            const TensorLayout& offset_grad, const TensorLayout& mask_grad) override;

    Algorithm* get_algorithm_heuristic(
            const TensorLayout& im, const TensorLayout& filter,
            const TensorLayout& offset, const TensorLayout& mask,
            const TensorLayout& out_grad, const TensorLayout& im_grad,
            const TensorLayout& offset_grad, const TensorLayout& mask_grad,
            size_t workspace_limit_in_bytes, const AlgoAttribute& positive_attr,
            const AlgoAttribute& negative_attr) override;

private:
    static AlgoPack sm_algo_pack;
};

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
