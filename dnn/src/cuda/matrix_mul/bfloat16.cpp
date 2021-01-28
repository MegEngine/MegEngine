/**
 * \file dnn/src/cuda/matrix_mul/bfloat16.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/cuda/handle.h"
#include "src/cuda/matrix_mul/algos.h"
#include "src/cuda/utils.h"
#include "src/common/algo_chooser.h"
#include "src/common/algo_base.h"

using namespace megdnn;
using namespace cuda;

namespace {
std::pair<TensorLayoutArray, MatrixMulForwardImpl::Param> sub_opr_config(
        const TensorLayoutArray& layouts, const MatrixMulForwardImpl* opr) {
    megdnn_assert(layouts.size() == 3);
    std::pair<TensorLayoutArray, MatrixMulForwardImpl::Param> ret;
    ret.first = layouts;
    auto change_dtype = [](TensorLayout& layout) {
        if (layout.dtype == dtype::BFloat16()) {
            layout.dtype = dtype::Float32();
        }
    };
    change_dtype(ret.first[0]);
    change_dtype(ret.first[1]);
    change_dtype(ret.first[2]);

    ret.second = opr->param();
    ret.second.compute_mode = MatrixMulForwardImpl::Param::ComputeMode::DEFAULT;
    return ret;
}

std::pair<TensorLayoutArray, std::unique_ptr<MatrixMulForward>> prepare_sub_opr(
        const MatrixMulForwardImpl::AlgoBase::SizeArgs& args) {
    auto&& config = sub_opr_config(
            {args.layout_a, args.layout_b, args.layout_c}, args.opr);
    auto matmul_opr = args.opr->handle()->create_operator<MatrixMulForward>();
    matmul_opr->param() = config.second;
    return {config.first, std::move(matmul_opr)};
}
}  // namespace

std::vector<Algorithm::SearchItem>
MatrixMulForwardImpl::AlgoBFloat16::get_subopr_list(
        const TensorLayoutArray& layouts, const OperatorBase* opr) const {
    auto&& config = sub_opr_config(
            layouts, static_cast<const MatrixMulForwardImpl*>(opr));

    std::string param_str;
    Algorithm::serialize_write_pod(config.second, param_str);
    return {{Algorithm::OprType::MATRIX_MUL_FORWARD, param_str, config.first}};
}

bool MatrixMulForwardImpl::AlgoBFloat16::is_available(
        const SizeArgs& args) const {
    auto config = prepare_sub_opr(args);
    return args.layout_a.dtype == dtype::BFloat16() &&
           get_algorithm(
                   static_cast<MatrixMulForwardImpl*>(config.second.get()),
                   config.first[0], config.first[1], config.first[2]);
}

WorkspaceBundle MatrixMulForwardImpl::AlgoBFloat16::get_workspace_bundle(
        void* ptr, const SizeArgs& args) const {
    auto config = prepare_sub_opr(args);

    SmallVector<size_t> sizes;
    auto get_workspace = [&sizes](const TensorLayout& src,
                                  const TensorLayout& dst) {
        if (src.dtype != dst.dtype) {
            sizes.push_back(dst.span().dist_byte());
        }
    };

    get_workspace(args.layout_a, config.first[0]);
    get_workspace(args.layout_b, config.first[1]);
    get_workspace(args.layout_c, config.first[2]);
    sizes.push_back(config.second->get_workspace_in_bytes(
            config.first[0], config.first[1], config.first[2]));
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
        auto config = prepare_sub_opr(args);
        config.second->exec(a, b, c, ctypecvt.workspace());
    }
    ctypecvt.comp_to_dst_type(c, args.tensor_c);
}

// vim: syntax=cpp.doxygen
