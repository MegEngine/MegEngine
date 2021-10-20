/**
 * \file dnn/src/cuda/local_share/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/cuda/local_share/opr_impl.h"
#include "./backward_data/algo.h"
#include "./backward_filter/algo.h"
#include "./forward/algo.h"
#include "src/common/algo_chooser.h"
#include "src/cuda/utils.h"

using namespace megdnn;
using namespace cuda;

/* ============== LocalShareForwardImpl ============== */
LocalShareForwardImpl::Algorithm* LocalShareForwardImpl::get_algorithm_heuristic(
        const TensorLayout& src, const TensorLayout& filter, const TensorLayout& dst,
        size_t workspace_limit_in_bytes, const AlgoAttribute& positive_attr,
        const AlgoAttribute& negative_attr) {
    AlgoBase::SizeArgs args(this, src, filter, dst);
    if (sm_algo_pack.batch_size_aware_chwn_small_image.is_available_attribute(
                args, positive_attr, negative_attr, workspace_limit_in_bytes)) {
        return &sm_algo_pack.batch_size_aware_chwn_small_image;
    }
    if (sm_algo_pack.batch_size_aware_chwn.is_available_attribute(
                args, positive_attr, negative_attr, workspace_limit_in_bytes)) {
        return &sm_algo_pack.batch_size_aware_chwn;
    }
    if (sm_algo_pack.batched_matmul.is_available_attribute(
                args, positive_attr, negative_attr, workspace_limit_in_bytes)) {
        return &sm_algo_pack.batched_matmul;
    }
    megdnn_throw(ssprintf(
            "no local share conv algorithm without attribute(%s) with "
            "attribute(%s), args(%s) and "
            "workspace limit (%zu bytes)",
            Algorithm::attribute_str(negative_attr).c_str(),
            Algorithm::attribute_str(positive_attr).c_str(), args.to_string().c_str(),
            workspace_limit_in_bytes));
}
std::vector<LocalShareForwardImpl::Algorithm*> LocalShareForwardImpl::
        get_all_algorithms(
                const TensorLayout& src, const TensorLayout& filter,
                const TensorLayout& dst) {
    AlgoBase::SizeArgs args{this, src, filter, dst};
    return megdnn::get_all_algorithms<LocalShareForwardImpl>(args);
}

std::vector<LocalShareForwardImpl::Algorithm*> LocalShareForwardImpl::
        get_all_algorithms_safe(
                const TensorLayout& src, const TensorLayout& filter,
                const TensorLayout& dst) {
    AlgoBase::SizeArgs args{this, src, filter, dst};
    return megdnn::get_all_algorithms_safe<LocalShareForwardImpl>(args);
}

size_t LocalShareForwardImpl::get_workspace_in_bytes(
        const TensorLayout& src, const TensorLayout& filter, const TensorLayout& dst) {
    return get_dnn_workspace(this, src, filter, dst);
}

void LocalShareForwardImpl::exec(
        _megdnn_tensor_in src, _megdnn_tensor_in filter, _megdnn_tensor_out dst,
        _megdnn_workspace workspace) {
    check_exec(src.layout, filter.layout, dst.layout, workspace.size);
    AlgoBase::ExecArgs args(this, src, filter, dst, workspace);
    auto algo = get_algorithm(this, src.layout, filter.layout, dst.layout);
    algo->exec(args);
}

const char* LocalShareForwardImpl::get_algorithm_set_name() const {
    return "CUDA_LOCAL_SHARE_CONV";
}

/* ============== LocalShareBackwardDataImpl ============== */
LocalShareBackwardDataImpl::Algorithm* LocalShareBackwardDataImpl::
        get_algorithm_heuristic(
                const TensorLayout& filter, const TensorLayout& diff,
                const TensorLayout& grad, size_t workspace_limit_in_bytes,
                const AlgoAttribute& positive_attr,
                const AlgoAttribute& negative_attr) {
    AlgoBase::SizeArgs args(this, filter, diff, grad);
    if (sm_algo_pack.implicit_gemm.is_available_attribute(
                args, positive_attr, negative_attr, workspace_limit_in_bytes)) {
        return &sm_algo_pack.implicit_gemm;
    }
    if (sm_algo_pack.batched_matmul.is_available_attribute(
                args, positive_attr, negative_attr, workspace_limit_in_bytes)) {
        return &sm_algo_pack.batched_matmul;
    }
    megdnn_throw(ssprintf(
            "no local share bwd data algorithm without attribute(%s) "
            "with attribute(%s) args(%s) and "
            "workspace limit (%zu bytes)",
            Algorithm::attribute_str(negative_attr).c_str(),
            Algorithm::attribute_str(positive_attr).c_str(), args.to_string().c_str(),
            workspace_limit_in_bytes));
}

std::vector<LocalShareBackwardDataImpl::Algorithm*> LocalShareBackwardDataImpl::
        get_all_algorithms(
                const TensorLayout& filter, const TensorLayout& diff,
                const TensorLayout& grad) {
    AlgoBase::SizeArgs args{this, filter, diff, grad};
    return megdnn::get_all_algorithms<LocalShareBackwardDataImpl>(args);
}

std::vector<LocalShareBackwardDataImpl::Algorithm*> LocalShareBackwardDataImpl::
        get_all_algorithms_safe(
                const TensorLayout& filter, const TensorLayout& diff,
                const TensorLayout& grad) {
    AlgoBase::SizeArgs args{this, filter, diff, grad};
    return megdnn::get_all_algorithms_safe<LocalShareBackwardDataImpl>(args);
}

size_t LocalShareBackwardDataImpl::get_workspace_in_bytes(
        const TensorLayout& filter, const TensorLayout& diff,
        const TensorLayout& grad) {
    return get_dnn_workspace(this, filter, diff, grad);
}

void LocalShareBackwardDataImpl::exec(
        _megdnn_tensor_in filter, _megdnn_tensor_in diff, _megdnn_tensor_out grad,
        _megdnn_workspace workspace) {
    AlgoBase::ExecArgs args(this, filter, diff, grad, workspace);
    auto algo = get_algorithm(this, filter.layout, diff.layout, grad.layout);
    algo->check_workspace(args, workspace).exec(args);
}

const char* LocalShareBackwardDataImpl::get_algorithm_set_name() const {
    return "CUDA_LOCAL_SHARE_CONV_BWD_DATA";
}

/* ============== LocalShareBackwardFilterImpl ============== */
LocalShareBackwardFilterImpl::Algorithm* LocalShareBackwardFilterImpl::
        get_algorithm_heuristic(
                const TensorLayout& src, const TensorLayout& diff,
                const TensorLayout& grad, size_t workspace_limit_in_bytes,
                const AlgoAttribute& positive_attr,
                const AlgoAttribute& negative_attr) {
    AlgoBase::SizeArgs args(this, src, diff, grad);
    if (sm_algo_pack.implicit_gemm.is_available_attribute(
                args, positive_attr, negative_attr, workspace_limit_in_bytes)) {
        return &sm_algo_pack.implicit_gemm;
    }
    if (sm_algo_pack.batched_matmul.is_available_attribute(
                args, positive_attr, negative_attr, workspace_limit_in_bytes)) {
        return &sm_algo_pack.batched_matmul;
    }
    megdnn_throw(ssprintf(
            "no local share bwd filter algorithm without "
            "attribute(%s) with attribute(%s), "
            "args(%s) and "
            "workspace limit (%zu bytes)",
            Algorithm::attribute_str(negative_attr).c_str(),
            Algorithm::attribute_str(positive_attr).c_str(), args.to_string().c_str(),
            workspace_limit_in_bytes));
}

std::vector<LocalShareBackwardFilterImpl::Algorithm*> LocalShareBackwardFilterImpl::
        get_all_algorithms(
                const TensorLayout& src, const TensorLayout& diff,
                const TensorLayout& grad) {
    AlgoBase::SizeArgs args{this, src, diff, grad};
    return megdnn::get_all_algorithms<LocalShareBackwardFilterImpl>(args);
}

std::vector<LocalShareBackwardFilterImpl::Algorithm*> LocalShareBackwardFilterImpl::
        get_all_algorithms_safe(
                const TensorLayout& src, const TensorLayout& diff,
                const TensorLayout& grad) {
    AlgoBase::SizeArgs args{this, src, diff, grad};
    return megdnn::get_all_algorithms_safe<LocalShareBackwardFilterImpl>(args);
}

size_t LocalShareBackwardFilterImpl::get_workspace_in_bytes(
        const TensorLayout& src, const TensorLayout& diff, const TensorLayout& grad) {
    return get_dnn_workspace(this, src, diff, grad);
}

void LocalShareBackwardFilterImpl::exec(
        _megdnn_tensor_in src, _megdnn_tensor_in diff, _megdnn_tensor_out grad,
        _megdnn_workspace workspace) {
    AlgoBase::ExecArgs args(this, src, diff, grad, workspace);
    auto algo = get_algorithm(this, src.layout, diff.layout, grad.layout);
    algo->check_workspace(args, workspace).exec(args);
}

const char* LocalShareBackwardFilterImpl::get_algorithm_set_name() const {
    return "CUDA_LOCAL_SHARE_CONV_BWD_FILTER";
}

// vim: syntax=cpp.doxygen
