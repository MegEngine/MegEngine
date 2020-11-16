/**
 * \file dnn/src/cuda/matrix_mul/bfloat16.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/cuda/handle.h"
#include "src/cuda/matrix_mul/algos.h"
#include "src/cuda/utils.h"

using namespace megdnn;
using namespace cuda;

MatrixMulForwardImpl::AlgoBFloat16::AlgoBFloat16(
        MatrixMulForwardImpl::AlgoBase* algorithm)
        : m_algorithm(algorithm) {
    megdnn_assert_internal(algorithm);
    m_name = ssprintf("MATMUL_BFLOAT16:%s", m_algorithm->name());
}

MatrixMulForwardImpl::AlgoBase::SizeArgs
MatrixMulForwardImpl::AlgoBFloat16::float_args(const SizeArgs& args) const {
    auto new_args = args;
    auto change_dtype = [](TensorLayout& layout) {
        if (layout.dtype == dtype::BFloat16()) {
            layout.dtype = dtype::Float32();
        }
    };
    change_dtype(new_args.layout_a);
    change_dtype(new_args.layout_b);
    change_dtype(new_args.layout_c);
    return new_args;
}

bool MatrixMulForwardImpl::AlgoBFloat16::is_available(
        const SizeArgs& args) const {
    auto fargs = float_args(args);
    return args.layout_a.dtype == dtype::BFloat16() &&
        m_algorithm->is_available(fargs);
}

WorkspaceBundle MatrixMulForwardImpl::AlgoBFloat16::get_workspace_bundle(
        void* ptr, const SizeArgs& args) const {
    auto fargs = float_args(args);
    SmallVector<size_t> sizes;
    auto get_workspace = [&sizes](const TensorLayout& src) {
        TensorLayout dst = src;
        if (dst.dtype == dtype::BFloat16()) {
            dst.dtype = dtype::Float32();
            sizes.push_back(dst.span().dist_byte());
        }
    };
    get_workspace(args.layout_a);
    get_workspace(args.layout_b);
    get_workspace(args.layout_c);
    sizes.push_back(m_algorithm->get_workspace_in_bytes(fargs));
    return {ptr, std::move(sizes)};
}

size_t MatrixMulForwardImpl::AlgoBFloat16::get_workspace_in_bytes(
        const SizeArgs& args) const {
    return get_workspace_bundle(nullptr, args).total_size_in_bytes();
}

void MatrixMulForwardImpl::AlgoBFloat16::exec(const ExecArgs& args) const {
    TensorND a = args.tensor_a;
    TensorND b = args.tensor_b;
    TensorND c = args.tensor_c;
    auto bundle = get_workspace_bundle(args.workspace.raw_ptr, args);
    auto ctypecvt = CompTypeCvter<dtype::BFloat16, dtype::Float32>(
            args.opr->handle(), &bundle);
    ctypecvt.src_to_comp_type(args.tensor_a, a)
            .src_to_comp_type(args.tensor_b, b)
            .src_to_comp_type(args.tensor_c, c);
    {
        auto matmul_opr =
                args.opr->handle()->create_operator<MatrixMulForward>();
        matmul_opr->param() = args.opr->param();
        matmul_opr->param().compute_mode = Param::ComputeMode::DEFAULT;
        matmul_opr->execution_policy() = {m_algorithm->info()};
        matmul_opr->exec(a, b, c, ctypecvt.workspace());
    }
    ctypecvt.comp_to_dst_type(c, args.tensor_c);
}

// vim: syntax=cpp.doxygen
