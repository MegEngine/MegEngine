/**
 * \file dnn/src/rocm/convolution/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "hcc_detail/hcc_defs_prologue.h"

#include "./backward_data/algo.h"
#include "./backward_filter/algo.h"
#include "./forward/algo.h"
#include "./opr_impl.h"
#include "src/common/algo_chooser.h"

#include "src/rocm/utils.h"

using namespace megdnn;
using namespace rocm;

#define TO_STRING2(v) #v
#define TO_STRING(v)  TO_STRING2(v)
#define MIOPEN_VERSION_STR          \
    TO_STRING(MIOPEN_VERSION_MAJOR) \
    "." TO_STRING(MIOPEN_VERSION_MINOR) "." TO_STRING(MIOPEN_VERSION_PATCH)

/* ============== ConvolutionForwardImpl ============== */
ConvolutionForwardImpl::Algorithm* ConvolutionForwardImpl::get_algorithm_heuristic(
        const TensorLayout& src, const TensorLayout& filter, const TensorLayout& dst,
        size_t workspace_limit_in_bytes, const AlgoAttribute& positive_attr,
        const AlgoAttribute& negative_attr) {
    auto fm = check_layout_fwd(src, filter, dst);
    return get_algorithm_heuristic(
            src, fm, dst, workspace_limit_in_bytes, positive_attr, negative_attr);
}

ConvolutionForwardImpl::Algorithm* ConvolutionForwardImpl::get_algorithm_heuristic(
        const TensorLayout& src, const CanonizedFilterMeta& filter,
        const TensorLayout& dst, size_t workspace_limit_in_bytes,
        const AlgoAttribute& positive_attr, const AlgoAttribute& negative_attr) {
    AlgoBase::SizeArgs args(this, src, filter, dst);

    //! MIOpen auto-tuning need to run with actual tensors, so we cannot get
    //! best algorithm here.
    if (is_miopen_supported(args)) {
        auto algo = megdnn::get_algo_match_attribute<ConvolutionForwardImpl>(
                sm_algo_pack.miopen_algos[0], positive_attr, negative_attr);
        if (algo)
            return algo;
    }

    if (args.filter_meta.group > 1) {
        if (sm_algo_pack.chanwise.is_available_attribute(
                    args, positive_attr, negative_attr, workspace_limit_in_bytes)) {
            return &sm_algo_pack.chanwise;
        }
    }

    auto prefer_1x1 = [&args, positive_attr, negative_attr,
                       workspace_limit_in_bytes]() {
        const size_t MAX_BATCH_SIZE_FOR_1x1_MAT_ALGO = 4;
        size_t batch_size = args.src_layout->shape[0];

        if (batch_size > MAX_BATCH_SIZE_FOR_1x1_MAT_ALGO) {
            return false;
        }
        return sm_algo_pack.a1x1.is_available_attribute(
                args, positive_attr, negative_attr, workspace_limit_in_bytes);
    };

    if (prefer_1x1()) {
        return &sm_algo_pack.a1x1;
    }

    auto prefer_1x1_large_batch = [&args, positive_attr, negative_attr,
                                   workspace_limit_in_bytes]() {
        const size_t MIN_BATCH_SIZE_FOR_1x1_LARGE_BATCH_ALGO = 32;
        size_t batch_size = args.src_layout->shape[0];

        if (batch_size < MIN_BATCH_SIZE_FOR_1x1_LARGE_BATCH_ALGO) {
            return false;
        }
        return sm_algo_pack.batched_matrix_mul.is_available_attribute(
                args, positive_attr, negative_attr, workspace_limit_in_bytes);
    };

    if (prefer_1x1_large_batch()) {
        return &sm_algo_pack.batched_matrix_mul;
    }

    return megdnn::get_algo_match_attribute<ConvolutionForwardImpl>(
            sm_algo_pack.non_miopen_algos, args, workspace_limit_in_bytes,
            "rocm conv fwd", positive_attr, negative_attr);
}

std::vector<ConvolutionForwardImpl::Algorithm*> ConvolutionForwardImpl::
        get_all_algorithms(
                const TensorLayout& src, const TensorLayout& filter,
                const TensorLayout& dst) {
    return megdnn::get_all_algorithms<ConvolutionForwardImpl>({this, src, filter, dst});
}

std::vector<ConvolutionForwardImpl::Algorithm*> ConvolutionForwardImpl::
        get_all_algorithms_safe(
                const TensorLayout& src, const TensorLayout& filter,
                const TensorLayout& dst) {
    return megdnn::get_all_algorithms_safe<ConvolutionForwardImpl>(
            {this, src, filter, dst});
}

size_t ConvolutionForwardImpl::get_workspace_in_bytes(
        const TensorLayout& src, const TensorLayout& filter, const TensorLayout& dst,
        const PreprocessedFilter*) {
    TensorLayoutArray layouts{src, filter, dst};
    HeuristicCache::Key key{this->handle(), this->get_opr_type(),
                            layouts.data(), layouts.size(),
                            &this->param(), sizeof(this->param())};
    auto rst = HeuristicCache::instance().get(key);
    if (rst.policy.algo.valid()) {
        return rst.workspace;
    }

    AlgoBase::SizeArgs args(this, src, filter, dst);
    return get_algorithm(this, src, filter, dst)->get_workspace_in_bytes(args);
}

void ConvolutionForwardImpl::exec(
        _megdnn_tensor_in src, _megdnn_tensor_in filter, _megdnn_tensor_out dst,
        const PreprocessedFilter* preprocessed_filter, _megdnn_workspace workspace) {
    check_exec(
            src.layout, filter.layout, dst.layout, workspace.size, preprocessed_filter);
    AlgoBase::ExecArgs args(this, src, filter, dst, workspace);
    auto algo = get_algorithm(this, src.layout, filter.layout, dst.layout);
    algo->exec(args);
}

const char* ConvolutionForwardImpl::get_algorithm_set_name() const {
    return "ROCMCONV0+MIOPEN" MIOPEN_VERSION_STR;
}

/* ============== ConvolutionBackwardDataImpl ============== */

void ConvolutionBackwardDataImpl::exec(
        _megdnn_tensor_in filter, _megdnn_tensor_in diff, _megdnn_tensor_out grad,
        _megdnn_workspace workspace) {
    check_exec(filter.layout, diff.layout, grad.layout, workspace.size);
    AlgoBase::ExecArgs args(this, filter, diff, grad, workspace);
    auto algo = get_algorithm(this, filter.layout, diff.layout, grad.layout);
    algo->exec(args);
}

std::vector<ConvolutionBackwardDataImpl::Algorithm*> ConvolutionBackwardDataImpl::
        get_all_algorithms(
                const TensorLayout& filter, const TensorLayout& diff,
                const TensorLayout& grad) {
    return megdnn::get_all_algorithms<ConvolutionBackwardDataImpl>(
            {this, filter, diff, grad});
}

std::vector<ConvolutionBackwardDataImpl::Algorithm*> ConvolutionBackwardDataImpl::
        get_all_algorithms_safe(
                const TensorLayout& filter, const TensorLayout& diff,
                const TensorLayout& grad) {
    return megdnn::get_all_algorithms_safe<ConvolutionBackwardDataImpl>(
            {this, filter, diff, grad});
}

ConvolutionBackwardDataImpl::Algorithm* ConvolutionBackwardDataImpl::
        get_algorithm_heuristic(
                const TensorLayout& filter, const TensorLayout& diff,
                const TensorLayout& grad, size_t workspace_limit_in_bytes,
                const AlgoAttribute& positive_attr,
                const AlgoAttribute& negative_attr) {
    auto fm = check_layout_fwd(grad, filter, diff);
    return get_algorithm_heuristic(
            fm, diff, grad, workspace_limit_in_bytes, positive_attr, negative_attr);
}

ConvolutionBackwardDataImpl::Algorithm* ConvolutionBackwardDataImpl::
        get_algorithm_heuristic(
                const CanonizedFilterMeta& filter, const TensorLayout& diff,
                const TensorLayout& grad, size_t workspace_limit_in_bytes,
                const AlgoAttribute& positive_attr,
                const AlgoAttribute& negative_attr) {
    AlgoBase::SizeArgs args(this, filter, diff, grad);

    if (is_miopen_supported(args.as_fwd_args())) {
        auto algo = megdnn::get_algo_match_attribute<ConvolutionBackwardDataImpl>(
                sm_algo_pack.miopen_algos[0], positive_attr, negative_attr);
        if (algo)
            return algo;
    }

    if (args.filter_meta.group > 1 &&
        sm_algo_pack.chanwise.is_available_attribute(
                args, positive_attr, negative_attr, workspace_limit_in_bytes)) {
        return &sm_algo_pack.chanwise;
    }

    return megdnn::get_algo_match_attribute<ConvolutionBackwardDataImpl>(
            sm_algo_pack.non_miopen_algos, args, workspace_limit_in_bytes,
            "rocm conv bwd_data", positive_attr, negative_attr);
}

size_t ConvolutionBackwardDataImpl::get_workspace_in_bytes(
        const TensorLayout& filter, const TensorLayout& diff,
        const TensorLayout& grad) {
    TensorLayoutArray layouts{filter, diff, grad};
    HeuristicCache::Key key{this->handle(), this->get_opr_type(),
                            layouts.data(), layouts.size(),
                            &this->param(), sizeof(this->param())};
    auto rst = HeuristicCache::instance().get(key);
    if (rst.policy.algo.valid()) {
        return rst.workspace;
    }

    AlgoBase::SizeArgs args(this, filter, diff, grad);
    return get_algorithm(this, filter, diff, grad)->get_workspace_in_bytes(args);
}

const char* ConvolutionBackwardDataImpl::get_algorithm_set_name() const {
    return "ROCMCONV0+MIOPEN" MIOPEN_VERSION_STR;
}

/* ============== ConvolutionBackwardFilterImpl ============== */

void ConvolutionBackwardFilterImpl::exec(
        _megdnn_tensor_in src, _megdnn_tensor_in diff, _megdnn_tensor_out grad,
        _megdnn_workspace workspace) {
    check_exec(src.layout, diff.layout, grad.layout, workspace.size);
    AlgoBase::ExecArgs args(this, src, diff, grad, workspace);
    auto algo = get_algorithm(this, src.layout, diff.layout, grad.layout);
    algo->exec(args);
}

std::vector<ConvolutionBackwardFilterImpl::Algorithm*> ConvolutionBackwardFilterImpl::
        get_all_algorithms(
                const TensorLayout& src, const TensorLayout& diff,
                const TensorLayout& grad) {
    return megdnn::get_all_algorithms<ConvolutionBackwardFilterImpl>(
            {this, src, diff, grad});
}

std::vector<ConvolutionBackwardFilterImpl::Algorithm*> ConvolutionBackwardFilterImpl::
        get_all_algorithms_safe(
                const TensorLayout& src, const TensorLayout& diff,
                const TensorLayout& grad) {
    return megdnn::get_all_algorithms_safe<ConvolutionBackwardFilterImpl>(
            {this, src, diff, grad});
}

ConvolutionBackwardFilterImpl::Algorithm* ConvolutionBackwardFilterImpl::
        get_algorithm_heuristic(
                const TensorLayout& src, const TensorLayout& diff,
                const TensorLayout& grad, size_t workspace_limit_in_bytes,
                const AlgoAttribute& positive_attr,
                const AlgoAttribute& negative_attr) {
    auto fm = check_layout_fwd(src, grad, diff);
    return get_algorithm_heuristic(
            src, diff, fm, workspace_limit_in_bytes, positive_attr, negative_attr);
}

ConvolutionBackwardFilterImpl::Algorithm* ConvolutionBackwardFilterImpl::
        get_algorithm_heuristic(
                const TensorLayout& src, const TensorLayout& diff,
                const CanonizedFilterMeta& grad, size_t workspace_limit_in_bytes,
                const AlgoAttribute& positive_attr,
                const AlgoAttribute& negative_attr) {
    AlgoBase::SizeArgs args(this, src, diff, grad);

    if (is_miopen_supported(args.as_fwd_args())) {
        auto algo = megdnn::get_algo_match_attribute<ConvolutionBackwardFilterImpl>(
                sm_algo_pack.miopen_algos[0], positive_attr, negative_attr);
        if (algo)
            return algo;
    }

    if (args.grad_filter_meta.group > 1 &&
        sm_algo_pack.chanwise.is_available_attribute(
                args, positive_attr, negative_attr, workspace_limit_in_bytes)) {
        // prefer special chanwise impl
        return &sm_algo_pack.chanwise;
    }

    return megdnn::get_algo_match_attribute<ConvolutionBackwardFilterImpl>(
            sm_algo_pack.non_miopen_algos, args, workspace_limit_in_bytes,
            "rocm conv bwd_filter", positive_attr, negative_attr);
}

size_t ConvolutionBackwardFilterImpl::get_workspace_in_bytes(
        const TensorLayout& src, const TensorLayout& diff, const TensorLayout& grad) {
    TensorLayoutArray layouts{src, diff, grad};
    HeuristicCache::Key key{this->handle(), this->get_opr_type(),
                            layouts.data(), layouts.size(),
                            &this->param(), sizeof(this->param())};
    auto rst = HeuristicCache::instance().get(key);
    if (rst.policy.algo.valid()) {
        return rst.workspace;
    }

    AlgoBase::SizeArgs args(this, src, diff, grad);
    return get_algorithm(this, src, diff, grad)->get_workspace_in_bytes(args);
}

const char* ConvolutionBackwardFilterImpl::get_algorithm_set_name() const {
    return "ROCMCONV0+MIOPEN" MIOPEN_VERSION_STR;
}

// vim: syntax=cpp.doxygen
