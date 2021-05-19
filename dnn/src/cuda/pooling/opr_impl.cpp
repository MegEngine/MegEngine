/**
 * \file dnn/src/cuda/pooling/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#include "src/cuda/pooling/opr_impl.h"
#include "./algo.h"
#include "./pooling2d_qint.cuh"
#include "src/common/algo_chooser.h"
#include "src/cuda/relayout_format/opr_impl.h"
#include "src/cuda/utils.h"

namespace megdnn {
namespace cuda {

size_t PoolingForwardImpl::get_workspace_in_bytes(const TensorLayout& src,
                                                  const TensorLayout& dst) {
    AlgoBase::SizeArgs args(this, src, dst);
    return get_algorithm(this, src, dst)->get_workspace_in_bytes(args);
}

const char* PoolingForwardImpl::get_algorithm_set_name() const {
    return "CUDA_POOLING_FORWARD";
}

std::vector<PoolingForwardImpl::Algorithm*>
PoolingForwardImpl::get_all_algorithms(const TensorLayout& src,
                                       const TensorLayout& dst) {
    return megdnn::get_all_algorithms<PoolingForwardImpl>({this, src, dst});
}

PoolingForwardImpl::Algorithm* PoolingForwardImpl::get_algorithm_heuristic(
        const TensorLayout& src, const TensorLayout& dst,
        size_t workspace_limit_in_bytes, const AlgoAttribute& positive_attr,
        const AlgoAttribute& negative_attr) {
    MEGDNN_MARK_USED_VAR(workspace_limit_in_bytes);

    AlgoBase::SizeArgs args(this, src, dst);
    for (auto&& iter : sm_algo_pack.all_algos) {
        if (iter->is_available_attribute(args, positive_attr, negative_attr)) {
            return iter;
        }
    }
    megdnn_throw(
            ssprintf("require algorithm with attribute(%s) and without "
                     "attribute(%s), but can't get suitable algo.\n",
                     Algorithm::attribute_str(positive_attr).c_str(),
                     Algorithm::attribute_str(negative_attr).c_str()));
    return nullptr;
}

void PoolingForwardImpl::exec(_megdnn_tensor_in ssrc, _megdnn_tensor_out sdst,
                              _megdnn_workspace sworkspace) {
    check_exec(ssrc.layout, sdst.layout, sworkspace.size);
    {
        AlgoBase::ExecArgs args(this, ssrc, sdst, sworkspace);
        auto algo = get_algorithm(this, ssrc.layout, sdst.layout);
        algo->exec(args);
    }
}

const char* PoolingBackwardImpl::get_algorithm_set_name() const {
    return "CUDA_POOLING_BACKWARD";
}

std::vector<PoolingBackwardImpl::Algorithm*>
PoolingBackwardImpl::get_all_algorithms(const TensorLayout& src,
                                        const TensorLayout& dst,
                                        const TensorLayout& diff,
                                        const TensorLayout& grad) {
    return megdnn::get_all_algorithms<PoolingBackwardImpl>(
            {this, src, dst, diff, grad});
}

PoolingBackwardImpl::Algorithm* PoolingBackwardImpl::get_algorithm_heuristic(
        const TensorLayout& src, const TensorLayout& dst,
        const TensorLayout& diff, const TensorLayout& grad,
        size_t workspace_limit_in_bytes, const AlgoAttribute& positive_attr,
        const AlgoAttribute& negative_attr) {
    MEGDNN_MARK_USED_VAR(workspace_limit_in_bytes);

    AlgoBase::SizeArgs args(this, src, dst, diff, grad);
    for (auto iter : sm_algo_pack.all_algos) {
        if (iter->is_available_attribute(args, positive_attr, negative_attr)) {
            return iter;
        }
    }
    megdnn_throw(
            ssprintf("require algorithm with attribute(%s) and without "
                     "attribute(%s), but can't get suitable algo.\n",
                     Algorithm::attribute_str(positive_attr).c_str(),
                     Algorithm::attribute_str(negative_attr).c_str()));
    return nullptr;
}

void PoolingBackwardImpl::exec(_megdnn_tensor_in ssrc, _megdnn_tensor_in sdst,
                               _megdnn_tensor_in sdiff,
                               _megdnn_tensor_out sgrad,
                               _megdnn_workspace sworkspace) {
    check_exec(ssrc.layout, sdst.layout, sdiff.layout, sgrad.layout,
               sworkspace.size);
    {
        AlgoBase::ExecArgs args(this, ssrc, sdst, sdiff, sgrad, sworkspace);
        auto algo = get_algorithm(this, ssrc.layout, sdst.layout, sdiff.layout,
                                  sgrad.layout);
        algo->exec(args);
    }
}

size_t PoolingBackwardImpl::get_workspace_in_bytes(const TensorLayout& src,
                                                   const TensorLayout& dst,
                                                   const TensorLayout& diff,
                                                   const TensorLayout& grad) {
    AlgoBase::SizeArgs args(this, src, dst, diff, grad);
    return get_algorithm(this, src, dst, diff, grad)
            ->get_workspace_in_bytes(args);
}

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
