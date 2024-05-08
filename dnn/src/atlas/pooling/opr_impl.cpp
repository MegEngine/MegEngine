#include "src/atlas/pooling/opr_impl.h"
#include "./algo.h"
#include "src/atlas/utils.h"
#include "src/common/algo_chooser.h"

namespace megdnn {
namespace atlas {

size_t PoolingForwardImpl::get_workspace_in_bytes(
        const TensorLayout& src, const TensorLayout& dst) {
    return get_dnn_workspace(this, src, dst);
}

const char* PoolingForwardImpl::get_algorithm_set_name() const {
    return "ACL_POOLING_FORWARD";
}

std::vector<PoolingForwardImpl::Algorithm*> PoolingForwardImpl::get_all_algorithms(
        const TensorLayout& src, const TensorLayout& dst) {
    return megdnn::get_all_algorithms<PoolingForwardImpl>({this, src, dst});
}
std::vector<PoolingForwardImpl::Algorithm*> PoolingForwardImpl::get_all_algorithms_safe(
        const TensorLayout& src, const TensorLayout& dst) {
    return megdnn::get_all_algorithms_safe<PoolingForwardImpl>({this, src, dst});
}

PoolingForwardImpl::Algorithm* PoolingForwardImpl::get_algorithm_heuristic(
        const TensorLayout& src, const TensorLayout& dst,
        size_t workspace_limit_in_bytes, const AlgoAttribute& positive_attr,
        const AlgoAttribute& negative_attr) {
    MEGDNN_MARK_USED_VAR(workspace_limit_in_bytes);

    AlgoBase::SizeArgs args(this, src, dst);
    for (auto&& iter : sm_algo_pack.all_algos) {
        if (iter->is_available_attribute(args, positive_attr, negative_attr)) {
            return iter;
        }
    }
    megdnn_throw(ssprintf(
            "require algorithm with attribute(%s) and without "
            "attribute(%s), but can't get suitable algo.\n",
            Algorithm::attribute_str(positive_attr).c_str(),
            Algorithm::attribute_str(negative_attr).c_str()));
    return nullptr;
}

void PoolingForwardImpl::exec(
        _megdnn_tensor_in src, _megdnn_tensor_out dst, _megdnn_workspace workspace) {
    check_exec(src.layout, dst.layout, workspace.size);
    {
        AlgoBase::ExecArgs args(this, src, dst, workspace);
        auto algo = get_algorithm(this, src.layout, dst.layout);
        algo->exec(args);
    }
}

// ============== Pooling Backward =================
size_t PoolingBackwardImpl::get_workspace_in_bytes(
        const TensorLayout& src, const TensorLayout& dst, const TensorLayout& diff,
        const TensorLayout& grad) {
    return get_dnn_workspace(this, src, dst, diff, grad);
}

const char* PoolingBackwardImpl::get_algorithm_set_name() const {
    return "ACL_POOLING_BACKWARD";
}

std::vector<Algorithm*> PoolingBackwardImpl::get_all_algorithms(
        const TensorLayout& src, const TensorLayout& dst, const TensorLayout& diff,
        const TensorLayout& grad) {
    return megdnn::get_all_algorithms<PoolingBackwardImpl>(
            {this, src, dst, diff, grad});
}
std::vector<Algorithm*> PoolingBackwardImpl::get_all_algorithms_safe(
        const TensorLayout& src, const TensorLayout& dst, const TensorLayout& diff,
        const TensorLayout& grad) {
    return megdnn::get_all_algorithms_safe<PoolingBackwardImpl>(
            {this, src, dst, diff, grad});
}

Algorithm* PoolingBackwardImpl::get_algorithm_heuristic(
        const TensorLayout& src, const TensorLayout& dst, const TensorLayout& diff,
        const TensorLayout& grad, size_t workspace_limit_in_bytes,
        const AlgoAttribute& positive_attr, const AlgoAttribute& negative_attr) {
    MEGDNN_MARK_USED_VAR(workspace_limit_in_bytes);

    AlgoBase::SizeArgs args(this, src, dst, diff, grad);
    for (auto iter : sm_algo_pack.all_algos) {
        if (iter->is_available_attribute(args, positive_attr, negative_attr)) {
            return iter;
        }
    }
    megdnn_throw(ssprintf(
            "require algorithm with attribute(%s) and without "
            "attribute(%s), but can't get suitable algo.\n",
            Algorithm::attribute_str(positive_attr).c_str(),
            Algorithm::attribute_str(negative_attr).c_str()));
    return nullptr;
}

void PoolingBackwardImpl::exec(
        _megdnn_tensor_in src, _megdnn_tensor_in dst, _megdnn_tensor_in diff,
        _megdnn_tensor_out grad, _megdnn_workspace workspace) {
    check_exec(src.layout, dst.layout, diff.layout, grad.layout, workspace.size);
    {
        AlgoBase::ExecArgs args(this, src, dst, diff, grad, workspace);
        auto algo =
                get_algorithm(this, src.layout, dst.layout, diff.layout, grad.layout);
        algo->exec(args);
    }
}

}  // namespace atlas
}  // namespace megdnn

// vim: syntax=cpp.doxygen