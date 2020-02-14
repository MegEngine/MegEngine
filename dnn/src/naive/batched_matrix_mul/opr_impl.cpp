/**
 * \file dnn/src/naive/batched_matrix_mul/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/naive/batched_matrix_mul/opr_impl.h"
#include "src/naive/matrix_mul/opr_impl.h"
#include "src/naive/handle.h"
#include "src/common/utils.h"

namespace megdnn {
namespace naive {
BatchedMatrixMulForwardImpl::BatchedMatrixMulForwardImpl(Handle *handle):
    BatchedMatrixMulForward(handle),
    m_opr(this->handle()->create_operator<MatrixMulForward>())
{
}

size_t BatchedMatrixMulForwardImpl::get_workspace_in_bytes(
        const TensorLayout &A, const TensorLayout &B,
        const TensorLayout &C) {
    MEGDNN_MARK_USED_VAR(A);
    MEGDNN_MARK_USED_VAR(B);
    MEGDNN_MARK_USED_VAR(C);
    return 0;
}

void BatchedMatrixMulForwardImpl::exec(_megdnn_tensor_in A,
        _megdnn_tensor_in B,
        _megdnn_tensor_out C,
        _megdnn_workspace workspace) {
    check_exec(A.layout, B.layout, C.layout, workspace.size);

    m_opr->param() = this->param();
    auto N = A.layout.shape[0];
    TensorND A_, B_, C_;
    A_.raw_ptr = A.raw_ptr;
    A_.layout = A.layout.remove_axis(0);
    B_.raw_ptr = B.raw_ptr;
    B_.layout = B.layout.remove_axis(0);
    C_.raw_ptr = C.raw_ptr;
    C_.layout = C.layout.remove_axis(0);

    auto Astrd = A.layout.dtype.size() * A.layout.stride[0],
         Bstrd = B.layout.dtype.size() * B.layout.stride[0],
         Cstrd = C.layout.dtype.size() * C.layout.stride[0];

    auto advance_ptr = [](TensorND &dest, ptrdiff_t d) {
        dest.raw_ptr = static_cast<void*>(
                static_cast<dt_byte*>(dest.raw_ptr) + d);
    };

    rep(n, N) {
        m_opr->exec(A_, B_, C_, workspace);
        advance_ptr(A_, Astrd);
        advance_ptr(B_, Bstrd);
        advance_ptr(C_, Cstrd);
    }

}

} // namespace naive
} // namespace megdnn

// vim: syntax=cpp.doxygen


