/**
 * \file dnn/src/fallback/matrix_mul/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/fallback/matrix_mul/opr_impl.h"

#include "src/common/metahelper.h"
#include "src/common/utils.h"
#include "src/fallback/matrix_mul/algos.h"
#include "src/fallback/matrix_mul/gemm_impl.h"
#include "src/fallback/matrix_mul/generic_strategy.h"
#include "src/naive/handle.h"
#include "src/naive/matrix_mul/opr_impl.h"
#include "src/common/algo_chooser.h"

using namespace megdnn;
using namespace fallback;

class MatrixMulImpl::AlgoPack : NonCopyableObj {
    AlgoF32K8x12x1 f32_k8x12x1;
public:
    AlgoGemv gemv;
    AlgoPack() {
        all_algos.emplace_back(&gemv);
        all_algos.emplace_back(&f32_k8x12x1);
    }
    SmallVector<AlgoBase*> all_algos;
};

SmallVector<MatrixMulImpl::AlgoBase*> MatrixMulImpl::algo_pack() {
    static AlgoPack s_algo_pack;
    return s_algo_pack.all_algos;
}

std::vector<MatrixMul::Algorithm*> MatrixMulImpl::get_all_algorithms(
        const TensorLayout& A, const TensorLayout& B, const TensorLayout& C) {
    std::vector<Algorithm*> gemm_algos, gemv_algos;
    auto kern_size_param = make_kern_size_param(A, B, C);
    for (auto&& algo : algo_pack()) {
        if (algo->usable(kern_size_param)) {
            if (algo->algoset() == AlgoBase::AlgoSet::ALGO_TYPE_GEMV) {
                // simple gemv
                gemv_algos.push_back(algo);
            } else {
                gemm_algos.push_back(algo);
            }
        }
    }
    gemv_algos.insert(gemv_algos.end(), gemm_algos.begin(), gemm_algos.end());
    return gemv_algos;
}

MatrixMul::Algorithm* MatrixMulImpl::get_algorithm_heuristic(
        const TensorLayout& A, const TensorLayout& B, const TensorLayout& C,
        size_t workspace_limit_in_bytes, bool reproducible) {
    auto kern_size_param = make_kern_size_param(A, B, C);
    if (auto algo = execution_policy().algorithm) {
        megdnn_assert(static_cast<AlgoBase*>(algo)->get_workspace(
                              kern_size_param) < workspace_limit_in_bytes);
        auto cur = megdnn::get_reproducible_algo<MatrixMulImpl>(
                static_cast<AlgoBase*>(algo), reproducible);
        if (cur)
            return cur;
        megdnn_throw(
                "require reproducible algorithm, but given algorithm is not "
                "reproducible");
    }

    auto algos = get_all_algorithms(A, B, C);
    for (auto&& algo : algos) {
        if (static_cast<AlgoBase*>(algo)->preferred_reproducible(
                    kern_size_param, reproducible) &&
            static_cast<AlgoBase*>(algo)->get_workspace(kern_size_param) <=
                    workspace_limit_in_bytes) {
            return algo;
        }
    }
    return nullptr;
}

MatrixMulImpl::KernSizeParam MatrixMulImpl::make_kern_size_param(
        const TensorLayout& A, const TensorLayout& B, const TensorLayout& C) {
    KernSizeParam kern_size_param;
    kern_size_param.A_type = A.dtype;
    kern_size_param.B_type = B.dtype;
    kern_size_param.C_type = C.dtype;
    kern_size_param.M = C.shape[0];
    kern_size_param.N = C.shape[1];
    kern_size_param.K = A[1 - param().transposeA];
    kern_size_param.LDA = A.stride[0];
    kern_size_param.LDB = B.stride[0];
    kern_size_param.LDC = C.stride[0];
    kern_size_param.trA = param().transposeA;
    kern_size_param.trB = param().transposeB;
    kern_size_param.compute_mode = param().compute_mode;
    kern_size_param.format = param().format;

    size_t pack_size = MatrixMulForward::pack_size(param().format);
    kern_size_param.K *= pack_size;
    kern_size_param.M *= pack_size;

    return kern_size_param;
}

MatrixMulImpl::KernParam MatrixMulImpl::make_kern_param(
        _megdnn_tensor_in A, _megdnn_tensor_in B, _megdnn_tensor_out C,
        _megdnn_workspace workspace) {
    KernParam kern_param;
    static_cast<KernSizeParam&>(kern_param) =
            make_kern_size_param(A.layout, B.layout, C.layout);
    kern_param.A_ptr = A.raw_ptr;
    kern_param.B_ptr = B.raw_ptr;
    kern_param.C_ptr = C.raw_ptr;
    kern_param.workspace_ptr = workspace.raw_ptr;
    kern_param.workspace_size = workspace.size;
    return kern_param;
}

size_t MatrixMulImpl::get_workspace_in_bytes(const TensorLayout& A,
                                             const TensorLayout& B,
                                             const TensorLayout& C) {
    if (auto algo = get_algorithm_heuristic(
                A, B, C, std::numeric_limits<size_t>::max(), false)) {
        auto kern_size_param = make_kern_size_param(A, B, C);
        return static_cast<AlgoBase*>(algo)->get_workspace(kern_size_param);
    }
    return 0;
}

void MatrixMulImpl::exec(_megdnn_tensor_in A, _megdnn_tensor_in B,
                         _megdnn_tensor_out C, _megdnn_workspace workspace) {
    check_exec(A.layout, B.layout, C.layout, workspace.size);

    if (auto algo = get_algorithm_heuristic(A.layout, B.layout, C.layout,
                                            std::numeric_limits<size_t>::max(),
                                            false)) {
        auto kern_param = make_kern_param(A, B, C, workspace);
        auto kern = static_cast<AlgoBase*>(algo)->get_kern(kern_param);
        auto run = [kern, kern_param]() { kern(kern_param); };
        static_cast<naive::HandleImpl*>(handle())->dispatch_kern(run);
        return;
    }

    naive::MatrixMulForwardImpl::exec(A, B, C, workspace);
}

// vim: syntax=cpp.doxygen
