/**
 * \file dnn/src/cuda/batched_matrix_mul/brute_force.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include <algorithm>
#include <memory>
#include "./algo.h"
#include "megdnn/opr_param_defs.h"
#include "src/common/algo_chooser.h"
#include "src/common/algo_base.h"
#include "src/cuda/handle.h"
#include "src/cuda/utils.h"

using namespace megdnn;
using namespace cuda;

namespace {
std::pair<TensorLayoutArray, MatrixMulForward::Param> sub_opr_config(
        const TensorLayout& layout_a, const TensorLayout& layout_b,
        const TensorLayout& layout_c, const BatchedMatrixMulForward* opr) {
    auto mm_layout_a = layout_a.remove_axis(0);
    auto mm_layout_b = layout_b.remove_axis(0);
    auto mm_layout_c = layout_c.remove_axis(0);

    return {{mm_layout_a, mm_layout_b, mm_layout_c}, opr->param()};
}

std::pair<TensorLayoutArray, std::unique_ptr<MatrixMulForward>> prepare_sub_opr(
        const BatchedMatrixMulForwardImpl::AlgoBase::SizeArgs& args) {
    auto matmul_opr = args.opr->handle()->create_operator<MatrixMulForward>();
    set_execution_policy<BatchedMatrixMulForward, MatrixMulForward*>(
            args.opr, matmul_opr.get());

    auto&& config = sub_opr_config(args.layout_a, args.layout_b, args.layout_c,
                                   args.opr);
    matmul_opr->param() = config.second;

    return {config.first, std::move(matmul_opr)};
}

}  // namespace

std::vector<Algorithm::SearchItem>
BatchedMatrixMulForwardImpl::AlgoBruteForce::get_subopr_list(
        const TensorLayoutArray& layouts, const OperatorBase* opr) const {
    const BatchedMatrixMulForwardImpl* bmm_opr =
            static_cast<const BatchedMatrixMulForwardImpl*>(opr);
    auto&& config = sub_opr_config(layouts[0], layouts[1], layouts[2], bmm_opr);

    std::string param_str;
    Algorithm::serialize_write_pod(config.second, param_str);
    return {{Algorithm::OprType::MATRIX_MUL_FORWARD, param_str, config.first}};
}

bool BatchedMatrixMulForwardImpl::AlgoBruteForce::is_available(
        const SizeArgs& args) const {
    auto config = prepare_sub_opr(args);

    return get_algorithm(
            static_cast<MatrixMulForwardImpl*>(config.second.get()),
            config.first[0], config.first[1], config.first[2]);
}
size_t BatchedMatrixMulForwardImpl::AlgoBruteForce::get_workspace_in_bytes(
        const SizeArgs& args) const {
    auto config = prepare_sub_opr(args);

    return config.second->get_workspace_in_bytes(
            config.first[0], config.first[1], config.first[2]);
}
void BatchedMatrixMulForwardImpl::AlgoBruteForce::exec(
        const ExecArgs& args) const {
    auto N = args.layout_a.shape[0];
    auto config = prepare_sub_opr(args);

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
        config.second->exec(A_, B_, C_, args.workspace);
    }
}
