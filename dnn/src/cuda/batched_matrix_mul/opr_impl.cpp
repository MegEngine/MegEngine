/**
 * \file dnn/src/cuda/batched_matrix_mul/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/cuda/batched_matrix_mul/opr_impl.h"
#include "src/cuda/batched_matrix_mul/algo.h"
#include "src/cuda/batched_matrix_mul/helper.cuh"

#include "src/common/algo_chooser.h"
#include "src/common/utils.cuh"
#include "src/cuda/handle.h"
#include "src/cuda/utils.h"

using namespace megdnn;
using namespace cuda;

using Algorithm = BatchedMatrixMulForwardImpl::Algorithm;

void BatchedMatrixMulForwardImpl::exec(
        _megdnn_tensor_in A, _megdnn_tensor_in B, _megdnn_tensor_out C,
        _megdnn_workspace workspace) {
    using namespace batched_matrix_mul;
    //!
    //! \Note (int8, int8) => int32 is supported
    //!    auto dtype=A.layout.dtype;
    //!    megdnn_assert(dtype.category() == DTypeCategory::FLOAT);
    AlgoBase::ExecArgs args(this, A, B, C, workspace);
    check_exec(A.layout, B.layout, C.layout, workspace.size);
    auto&& algo = megdnn::get_algorithm(this, A.layout, B.layout, C.layout);
    algo->exec(args);
}

size_t BatchedMatrixMulForwardImpl::get_workspace_in_bytes(
        const TensorLayout& A, const TensorLayout& B, const TensorLayout& C) {
    return get_dnn_workspace(this, A, B, C);
}

std::vector<Algorithm*> BatchedMatrixMulForwardImpl::get_all_algorithms(
        const TensorLayout& A, const TensorLayout& B, const TensorLayout& C) {
    std::vector<Algorithm*> ret;
    AlgoBase::SizeArgs args(this, A, B, C);
    for (auto&& algo : sm_algo_pack.all_algos) {
        if (algo->is_available(args))
            ret.push_back(algo);
    }
    return ret;
}
std::vector<Algorithm*> BatchedMatrixMulForwardImpl::get_all_algorithms_safe(
        const TensorLayout& A, const TensorLayout& B, const TensorLayout& C) {
    auto ret_safe = get_all_algorithms(A, B, C);
    megdnn_assert(!ret_safe.empty(), "no usable batchedmatrixmulForward fwd algorithm");
    return ret_safe;
}

Algorithm* BatchedMatrixMulForwardImpl::get_algorithm_heuristic(
        const TensorLayout& A, const TensorLayout& B, const TensorLayout& C,
        size_t workspace_limit_in_bytes, const AlgoAttribute& positive_attr,
        const AlgoAttribute& negative_attr) {
    MEGDNN_MARK_USED_VAR(workspace_limit_in_bytes);
    AlgoBase::SizeArgs args(this, A, B, C);
    if (sm_algo_pack.cublas.is_available_attribute(
                args, positive_attr, negative_attr)) {
        return &sm_algo_pack.cublas;
    }
#if CUDA_VERSION >= 10010
    else if (sm_algo_pack.cublasLt.is_available_attribute(
                     args, positive_attr, negative_attr)) {
        return &sm_algo_pack.cublasLt;
    }
#endif
    else if (sm_algo_pack.int8x8x32.is_available_attribute(
                     args, positive_attr, negative_attr)) {
        return &sm_algo_pack.int8x8x32;
    } else {
        if (sm_algo_pack.brute_force.is_available_attribute(
                    args, positive_attr, negative_attr)) {
            return &sm_algo_pack.brute_force;
        }
    }

    megdnn_throw(ssprintf(
            "no batched_matrix_mul algorithm without attribute(%s) with "
            "attribute(%s) args(%s) and "
            "workspace limit (%zu bytes)",
            Algorithm::attribute_str(negative_attr).c_str(),
            Algorithm::attribute_str(positive_attr).c_str(), args.to_string().c_str(),
            workspace_limit_in_bytes));
    return nullptr;
};

// vim: syntax=cpp.doxygen
