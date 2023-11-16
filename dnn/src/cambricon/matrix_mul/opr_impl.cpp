#include "src/cambricon/matrix_mul/opr_impl.h"
#include "src/cambricon/matrix_mul/algo.h"
#include "src/common/algo_chooser.h"

namespace megdnn {
namespace cambricon {

size_t MatrixMulForwardImpl::get_workspace_in_bytes(
        const TensorLayout& A, const TensorLayout& B, const TensorLayout& C) {
    return get_dnn_workspace(this, A, B, C);
}

void MatrixMulForwardImpl::exec(
        _megdnn_tensor_in A, _megdnn_tensor_in B, _megdnn_tensor_out C,
        _megdnn_workspace workspace) {
    check_exec(A.layout, B.layout, C.layout, workspace.size);
    AlgoBase::ExecArgs args(this, A, B, C, workspace);
    auto&& algo = get_algorithm(this, A.layout, B.layout, C.layout);
    algo->exec(args);
}

std::vector<MatrixMulForward::Algorithm*> MatrixMulForwardImpl::get_all_algorithms(
        const TensorLayout& A, const TensorLayout& B, const TensorLayout& C) {
    AlgoBase::SizeArgs args{this, A, B, C};
    return megdnn::get_all_algorithms<MatrixMulForwardImpl>(args);
}

std::vector<MatrixMulForward::Algorithm*> MatrixMulForwardImpl::get_all_algorithms_safe(
        const TensorLayout& A, const TensorLayout& B, const TensorLayout& C) {
    AlgoBase::SizeArgs args{this, A, B, C};
    return megdnn::get_all_algorithms_safe<MatrixMulForwardImpl>(args);
}

MatrixMulForward::Algorithm* MatrixMulForwardImpl::get_algorithm_heuristic(
        const TensorLayout& A, const TensorLayout& B, const TensorLayout& C,
        size_t workspace_limit_in_bytes, const AlgoAttribute& positive_attr,
        const AlgoAttribute& negative_attr) {
    AlgoBase::SizeArgs args{this, A, B, C};
    auto&& all_algos = algo_pack().all_algos;
    for (auto algo : all_algos) {
        if (algo->get_workspace_in_bytes(args) <= workspace_limit_in_bytes &&
            algo->is_available(args))
            return algo;
    }
    megdnn_throw(ssprintf("no usable matrix mul forward algorithm"));
    return nullptr;
}

MEGDNN_DEF_GET_ALGO_FROM_DESC(MatrixMulForwardImpl)

}  // namespace cambricon
}  // namespace megdnn

// vim: syntax=cpp.doxygen
