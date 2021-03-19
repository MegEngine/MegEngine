/**
 * \file dnn/src/rocm/batched_matrix_mul/algos.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/rocm/batched_matrix_mul/algos.h"
#include "src/common/algo_base.h"

using namespace megdnn;
using namespace rocm;

BatchedMatrixMulForwardImpl::AlgoPack::AlgoPack() {
    all_algos.push_back(&blas);

    for (auto&& algo : all_algos) {
        m_all_algos_map.emplace(algo->info().desc, algo);
    }
}

BatchedMatrixMulForwardImpl::AlgoPack BatchedMatrixMulForwardImpl::sm_algo_pack;

MEGDNN_DEF_GET_ALGO_FROM_DESC(BatchedMatrixMulForwardImpl)

BatchedMatrixMulForwardImpl::AlgoBase::SizeArgs::SizeArgs(
        BatchedMatrixMulForwardImpl* o, const TensorLayout& A,
        const TensorLayout& B, const TensorLayout& C)
        : opr{o}, layout_a{A}, layout_b{B}, layout_c{C} {}

BatchedMatrixMulForwardImpl::AlgoBase::ExecArgs::ExecArgs(
        BatchedMatrixMulForwardImpl* opr, _megdnn_tensor_in A,
        _megdnn_tensor_in B, _megdnn_tensor_out C, _megdnn_workspace workspace)
        : SizeArgs(opr, A.layout, B.layout, C.layout),
          tensor_a{A},
          tensor_b{B},
          tensor_c{C},
          workspace{workspace} {}

std::string BatchedMatrixMulForwardImpl::AlgoBase::SizeArgs::to_string() const {
    auto&& param = opr->param();
    size_t m = layout_a.shape[0], n = layout_b.shape[1],
           k = layout_a.shape[param.transposeA ? 0 : 1];
    MEGDNN_MARK_USED_VAR(m);
    MEGDNN_MARK_USED_VAR(n);
    MEGDNN_MARK_USED_VAR(k);
    return ssprintf(
            "A={%zux%zu},B={%zux%zu},C={%zux%zu},Transpose A=%d,Transpose "
            "B=%d,ldA=%zu,ldB=%zu,ldC=%zu",
            m, k, k, n, m, n, param.transposeA, param.transposeB,
            layout_a.stride[0], layout_b.stride[0], layout_c.stride[0]);
}

// vim: syntax=cpp.doxygen
