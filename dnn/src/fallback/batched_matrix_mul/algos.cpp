/**
 * \file dnn/src/fallback/batched_matrix_mul/algos.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/fallback/batched_matrix_mul/algos.h"
#include "src/common/algo_base.h"
#include "src/naive/handle.h"

using namespace megdnn;
using namespace fallback;

BatchedMatrixMulForwardImpl::AlgoPack::AlgoPack() {
    all_algos.push_back(&algo_default);

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
            static_cast<size_t>(layout_a.stride[0]),
            static_cast<size_t>(layout_b.stride[0]),
            static_cast<size_t>(layout_c.stride[0]));
}

/* ===================== default algo ===================== */
size_t BatchedMatrixMulForwardImpl::AlgoDefault::get_workspace_in_bytes(
        const SizeArgs& args) const {
    auto opr = inplace_cpu_handle()->create_operator<MatrixMul>();
    auto A_ = args.layout_a.remove_axis(0), B_ = args.layout_b.remove_axis(0),
         C_ = args.layout_c.remove_axis(0);
    opr->param() = args.opr->param();
    return opr->get_workspace_in_bytes(A_, B_, C_);
}

void BatchedMatrixMulForwardImpl::AlgoDefault::exec(
        const ExecArgs& args) const {
    //! As megbrain may modify param when checking all transpose situations, so
    //! here we should copy the param when dispatching kern
    auto param = args.opr->param();
    auto kern = [args, param]() {
        auto N = args.layout_a.shape[0];
        TensorND A_, B_, C_;
        A_.raw_ptr = args.tensor_a.raw_ptr;
        A_.layout = args.layout_a.remove_axis(0);
        B_.raw_ptr = args.tensor_b.raw_ptr;
        B_.layout = args.layout_b.remove_axis(0);
        C_.raw_ptr = args.tensor_c.raw_ptr;
        C_.layout = args.layout_c.remove_axis(0);

        auto Astrd = args.layout_a.dtype.size() * args.layout_a.stride[0],
             Bstrd = args.layout_b.dtype.size() * args.layout_b.stride[0],
             Cstrd = args.layout_c.dtype.size() * args.layout_c.stride[0];

        auto advance_ptr = [](TensorND& dest, ptrdiff_t d) {
            dest.raw_ptr =
                    static_cast<void*>(static_cast<dt_byte*>(dest.raw_ptr) + d);
        };

        auto opr = inplace_cpu_handle()->create_operator<MatrixMul>();
        opr->param() = param;
        rep(n, N) {
            opr->exec(A_, B_, C_, args.workspace);
            advance_ptr(A_, Astrd);
            advance_ptr(B_, Bstrd);
            advance_ptr(C_, Cstrd);
        }
    };

    static_cast<naive::HandleImpl*>(args.opr->handle())->dispatch_kern(kern);
}

// vim: syntax=cpp.doxygen
