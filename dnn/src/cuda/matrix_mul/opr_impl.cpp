/**
 * \file dnn/src/cuda/matrix_mul/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/cuda/matrix_mul/opr_impl.h"
#include "./algos.h"
#include "src/common/algo_chooser.h"

#include <cuda.h>
#include "src/cuda/handle.h"
#include "src/cuda/matrix_mul/cublasLt_wrapper.h"
#include "src/cuda/utils.h"

namespace megdnn {
namespace cuda {

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
    if (sm_algo_pack.cublas.is_available_attribute(
                args, positive_attr, negative_attr, workspace_limit_in_bytes)) {
        return &sm_algo_pack.cublas;
    }
#if CUDA_VERSION >= 10010
    if (sm_algo_pack.cublas_lt.is_available_attribute(
                args, positive_attr, negative_attr, workspace_limit_in_bytes)) {
        return &sm_algo_pack.cublas_lt;
    }
#endif

#if CUDA_VERSION >= 10000
    if (sm_algo_pack.wmma_uint4x4x32.is_available_attribute(
                args, positive_attr, negative_attr, workspace_limit_in_bytes)) {
        return &sm_algo_pack.wmma_uint4x4x32;
    }
#endif

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

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
