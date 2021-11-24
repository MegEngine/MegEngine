/**
 * \file dnn/src/naive/batched_matrix_mul/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/naive/batched_matrix_mul/opr_impl.h"
#include "src/common/utils.h"
#include "src/naive/handle.h"
#include "src/naive/matrix_mul/opr_impl.h"

namespace megdnn {
namespace naive {
BatchedMatrixMulForwardImpl::BatchedMatrixMulForwardImpl(Handle* handle)
        : BatchedMatrixMulForward(handle),
          m_opr(this->handle()->create_operator<MatrixMulForward>()) {}

size_t BatchedMatrixMulForwardImpl::get_workspace_in_bytes(
        const TensorLayout& A, const TensorLayout& B, const TensorLayout& C) {
    MEGDNN_MARK_USED_VAR(A);
    MEGDNN_MARK_USED_VAR(B);
    MEGDNN_MARK_USED_VAR(C);
    return 0;
}

void BatchedMatrixMulForwardImpl::exec(
        _megdnn_tensor_in A, _megdnn_tensor_in B, _megdnn_tensor_out C,
        _megdnn_workspace workspace) {
    check_exec(A.layout, B.layout, C.layout, workspace.size);

    m_opr->param() = this->param();
    auto N = A.layout.shape[0];

    auto Astrd = A.layout.dtype.size() * A.layout.stride[0],
         Bstrd = B.layout.dtype.size() * B.layout.stride[0],
         Cstrd = C.layout.dtype.size() * C.layout.stride[0];

    auto Aref = A.get_ref_ptr();
    auto Bref = B.get_ref_ptr();
    auto Cref = C.get_ref_ptr();

    rep(n, N) {
        //! all tensors should share the same RefPtr
        auto A_ref = Aref;
        A_ref += n * Astrd;
        auto B_ref = Bref;
        B_ref += n * Bstrd;
        auto C_ref = Cref;
        C_ref += n * Cstrd;
        TensorND A_{A.layout.remove_axis(0), A_ref};
        TensorND B_{B.layout.remove_axis(0), B_ref};
        TensorND C_{C.layout.remove_axis(0), C_ref};
        m_opr->exec(A_, B_, C_, workspace);
    }
}

std::vector<BatchedMatrixMulForward::Algorithm*> BatchedMatrixMulForwardImpl::
        get_all_algorithms(
                const TensorLayout& /*A*/, const TensorLayout& /*B*/,
                const TensorLayout& /*C*/) {
    return {static_cast<HandleImpl*>(handle())->default_batched_matmul_fwd_algo()};
}

std::vector<BatchedMatrixMulForward::Algorithm*> BatchedMatrixMulForwardImpl::
        get_all_algorithms_safe(
                const TensorLayout& /*A*/, const TensorLayout& /*B*/,
                const TensorLayout& /*C*/) {
    return {static_cast<HandleImpl*>(handle())->default_batched_matmul_fwd_algo()};
}

BatchedMatrixMulForward::Algorithm* BatchedMatrixMulForwardImpl::
        get_algorithm_heuristic(
                const TensorLayout& /*A*/, const TensorLayout& /*B*/,
                const TensorLayout& /*C*/, size_t /*workspace_limit_in_bytes*/,
                const AlgoAttribute& /*positive_attr*/,
                const AlgoAttribute& /*negative_attr*/) {
    return static_cast<HandleImpl*>(handle())->default_batched_matmul_fwd_algo();
}

BatchedMatrixMulForward::Algorithm* BatchedMatrixMulForwardImpl::
        get_algorithm_from_desc(const AlgorithmDesc& desc) {
    Algorithm* ret =
            static_cast<HandleImpl*>(handle())->default_batched_matmul_fwd_algo();
    megdnn_assert(desc == ret->info().desc);
    return ret;
}

}  // namespace naive
}  // namespace megdnn

// vim: syntax=cpp.doxygen
