#include "src/rocm/matrix_mul/opr_impl.h"
#include "hcc_detail/hcc_defs_prologue.h"

#include "./algos.h"
#include "src/common/algo_chooser.h"
#include "src/rocm/handle.h"
#include "src/rocm/utils.h"

using namespace megdnn;
using namespace rocm;

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
    AlgoBase::SizeArgs args{this, A, B, C};
    if (sm_algo_pack.blas.is_available_attribute(
                args, positive_attr, negative_attr, workspace_limit_in_bytes)) {
        return &sm_algo_pack.blas;
    }
    return megdnn::get_algo_match_attribute<MatrixMulForwardImpl>(
            sm_algo_pack.all_algos, args, workspace_limit_in_bytes,
            "matrix mul forward", positive_attr, negative_attr);
}

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

// vim: syntax=cpp.doxygen
