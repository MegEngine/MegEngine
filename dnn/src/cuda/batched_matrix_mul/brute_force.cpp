/**
 * \file dnn/src/cuda/batched_matrix_mul/brute_force.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "./algo.h"
#include "src/cuda/handle.h"
#include "src/cuda/utils.h"

using namespace megdnn;
using namespace cuda;

BatchedMatrixMulForwardImpl::AlgoBruteForce::AlgoBruteForce(
        MatrixMulForwardImpl::AlgoBase* algo)
        : m_algorithm(algo) {
    m_name = ssprintf("BRUTE_FORCE-%s", algo->name());
}
bool BatchedMatrixMulForwardImpl::AlgoBruteForce::is_available(
        const SizeArgs& args) const {
    MatrixMulForwardImpl mm{args.opr->handle()};
    mm.param() = {args.opr->param().transposeA, args.opr->param().transposeB};
    mm.execution_policy() = {m_algorithm->info()};

    auto mm_layout_a = args.layout_a.remove_axis(0);
    auto mm_layout_b = args.layout_b.remove_axis(0);
    auto mm_layout_c = args.layout_c.remove_axis(0);

    MatrixMulForwardImpl::AlgoBase::SizeArgs mm_args{&mm, mm_layout_a,
                                                     mm_layout_b, mm_layout_c};
    return m_algorithm->is_available(mm_args);
}
size_t BatchedMatrixMulForwardImpl::AlgoBruteForce::get_workspace_in_bytes(
        const SizeArgs& args) const {
    auto mm_opr = args.opr->handle()->create_operator<MatrixMulForward>();
    mm_opr->param() = {args.opr->param().transposeA,
                       args.opr->param().transposeB};
    mm_opr->execution_policy() = {m_algorithm->info()};

    return mm_opr->get_workspace_in_bytes(args.layout_a, args.layout_b,
                                          args.layout_c);
}
void BatchedMatrixMulForwardImpl::AlgoBruteForce::exec(
        const ExecArgs& args) const {
    auto N = args.layout_a.shape[0];
    auto&& mm_opr = args.opr->handle()->create_operator<MatrixMulForward>();
    mm_opr->param() = {args.opr->param().transposeA,
                       args.opr->param().transposeB};
    mm_opr->execution_policy() = {m_algorithm->info()};
    rep(n, N) {
        TensorND A_, B_, C_;
        auto tensor_n_from_batch = [n](const TensorND& in, TensorND& out) {
            out.raw_ptr = static_cast<void*>(static_cast<dt_byte*>(in.raw_ptr) +
                                             n * in.layout.stride[0] *
                                                     in.layout.dtype.size());
            out.layout = in.layout.remove_axis(0);
        };
        tensor_n_from_batch(args.tensor_a, A_);
        tensor_n_from_batch(args.tensor_b, B_);
        tensor_n_from_batch(args.tensor_c, C_);
        mm_opr->exec(A_, B_, C_, args.workspace);
    }
}
