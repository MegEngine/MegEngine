/**
 * \file dnn/src/naive/matrix_mul/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/naive/matrix_mul/opr_impl.h"

#include "src/common/utils.h"
#include "src/naive/handle.h"
#include "./matrix_mul_helper.h"

#include "midout.h"
MIDOUT_DECL(megdnn_naive_matmul)

namespace megdnn {
namespace naive {

size_t MatrixMulForwardImpl::get_workspace_in_bytes(const TensorLayout& A,
                                                    const TensorLayout& B,
                                                    const TensorLayout&) {
    MIDOUT_BEGIN(
            megdnn_naive_matmul,
            midout_iv("MatrixMulForwardImpl::get_workspace_in_bytes"_hash)) {
        if (A.dtype.enumv() == DTypeEnum::Quantized4Asymm ||
            A.dtype.enumv() == DTypeEnum::QuantizedS4) {
            return (A.span().dist_elem() + B.span().dist_elem()) *
                   sizeof(uint8_t);
        }
        return 0;
    }
    MIDOUT_END();
}

template <bool TA, bool TB>
void dispatch_ta_tb(_megdnn_tensor_in A, _megdnn_tensor_in B,
                    _megdnn_tensor_out C, _megdnn_workspace workspace,
                    const MatrixMul::Param& param) {
    auto M = C.layout.shape[0], N = C.layout.shape[1];
    auto K = A.layout.shape[param.transposeA ? 0 : 1];
    auto LDA = A.layout.stride[0], LDB = B.layout.stride[0],
         LDC = C.layout.stride[0];

    dispatch_ta_tb<TA, TB>(A.raw_ptr, B.raw_ptr, C.raw_ptr, workspace.raw_ptr,
                           M, N, K, LDA, LDB, LDC, A.layout.dtype,
                           B.layout.dtype, C.layout.dtype, param.format,
                           param.compute_mode);
}

void MatrixMulForwardImpl::exec_internal(_megdnn_tensor_in A,
                                         _megdnn_tensor_in B,
                                         _megdnn_tensor_out C,
                                         _megdnn_workspace workspace,
                                         const Param& param) {
#define DISPATCH(TA, TB)                                    \
    if (param.transposeA == TA && param.transposeB == TB) { \
        dispatch_ta_tb<TA, TB>(A, B, C, workspace, param);  \
        return;                                             \
    }
    DISPATCH(true, true);
    DISPATCH(true, false);
    DISPATCH(false, true);
    DISPATCH(false, false);
#undef DISPATCH
    megdnn_assert_internal(0);
}

void MatrixMulForwardImpl::exec(_megdnn_tensor_in A, _megdnn_tensor_in B,
                                _megdnn_tensor_out C,
                                _megdnn_workspace workspace) {
    MIDOUT_BEGIN(megdnn_naive_matmul,
                 midout_iv("MatrixMulForwardImpl::exec"_hash)) {
        check_exec(A.layout, B.layout, C.layout, workspace.size);
        auto p = param();
        MEGDNN_DISPATCH_CPU_KERN_OPR(exec_internal(A, B, C, workspace, p));
    }
    MIDOUT_END();
}

std::vector<MatrixMulForward::Algorithm*>
MatrixMulForwardImpl::get_all_algorithms(const TensorLayout& /*A*/,
                                         const TensorLayout& /*B*/,
                                         const TensorLayout& /*C*/)  {
    return {static_cast<HandleImpl*>(handle())->default_matmul_fwd_algo()};
}

MatrixMulForward::Algorithm* MatrixMulForwardImpl::get_algorithm_heuristic(
        const TensorLayout& /*A*/, const TensorLayout& /*B*/,
        const TensorLayout& /*C*/, size_t /*workspace_limit_in_bytes*/,
        const AlgoAttribute& /*positive_attr*/,
        const AlgoAttribute& /*negative_attr*/) {
    return static_cast<HandleImpl*>(handle())->default_matmul_fwd_algo();
}

MatrixMulForward::Algorithm* MatrixMulForwardImpl::get_algorithm_from_desc(
        const AlgorithmDesc& desc) {
    Algorithm* ret =
            static_cast<HandleImpl*>(handle())->default_matmul_fwd_algo();
    megdnn_assert(desc == ret->info().desc);
    return ret;
}

}  // namespace naive
}  // namespace megdnn

// vim: syntax=cpp.doxygen
