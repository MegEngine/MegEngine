#include "src/atlas/matrix_mul/opr_impl.h"
#include "src/atlas/matrix_mul/algo.h"
#include "src/atlas/utils.h"
#include "src/common/algo_chooser.h"

namespace megdnn {
namespace atlas {

MatrixMulForwardImpl::AlgoPack MatrixMulForwardImpl::sm_algo_pack;

const char* MatrixMulForwardImpl::get_algorithm_set_name() const {
    return "ACL_MATRIX_MUL_FORWARD";
}

std::vector<MatrixMulForwardImpl::Algorithm*> MatrixMulForwardImpl::get_all_algorithms(
        const TensorLayout& A, const TensorLayout& B, const TensorLayout& C) {
    AlgoBase::SizeArgs args{this, A, B, C};
    return megdnn::get_all_algorithms<MatrixMulForwardImpl>(args);
}

std::vector<MatrixMulForwardImpl::Algorithm*> MatrixMulForwardImpl::
        get_all_algorithms_safe(
                const TensorLayout& A, const TensorLayout& B, const TensorLayout& C) {
    AlgoBase::SizeArgs args{this, A, B, C};
    return megdnn::get_all_algorithms_safe<MatrixMulForwardImpl>(args);
}

MatrixMulForwardImpl::Algorithm* MatrixMulForwardImpl::get_algorithm_heuristic(
        const TensorLayout& A, const TensorLayout& B, const TensorLayout& C,
        size_t workspace_limit_in_bytes, const AlgoAttribute& positive_attr,
        const AlgoAttribute& negative_attr) {
    MEGDNN_MARK_USED_VAR(workspace_limit_in_bytes);

    AlgoBase::SizeArgs args(this, A, B, C);
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

size_t MatrixMulForwardImpl::get_workspace_in_bytes(
        const TensorLayout& A, const TensorLayout& B, const TensorLayout& C) {
    return get_dnn_workspace(this, A, B, C);
}

void MatrixMulForwardImpl::exec(
        _megdnn_tensor_in A, _megdnn_tensor_in B, _megdnn_tensor_out C,
        _megdnn_workspace workspace) {
    check_exec(A.layout, B.layout, C.layout, workspace.size);
    {
        AlgoBase::ExecArgs args(this, A, B, C, workspace);
        auto algo = get_algorithm(this, A.layout, B.layout, C.layout);
        algo->exec(args);
    }
}

}  // namespace atlas
}  // namespace megdnn

// vim: syntax=cpp.doxygen