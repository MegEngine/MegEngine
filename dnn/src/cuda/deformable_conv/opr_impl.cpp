/**
 * \file dnn/src/cuda/deformable_conv/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/cuda/deformable_conv/fwd/algo.h"
#include "src/cuda/deformable_conv/bwd_flt/algo.h"
#include "src/cuda/deformable_conv/bwd_data/algo.h"

#include "src/common/algo_chooser.h"
#include "src/common/utils.h"
#include "src/cuda/deformable_conv/opr_impl.h"
#include "src/cuda/utils.h"

using namespace megdnn;
using namespace cuda;

using Fwd = DeformableConvForwardImpl;
using BwdFlt = DeformableConvBackwardFilterImpl;
using BwdData = DeformableConvBackwardDataImpl;

using AlgoFwd = Fwd::Algorithm;
using AlgoBwdFlt = BwdFlt::Algorithm;
using AlgoBwdData = BwdData::Algorithm;

/* ============== Fwd Implementation ============== */

size_t Fwd::get_workspace_in_bytes(const TensorLayout& im,
                                   const TensorLayout& filter,
                                   const TensorLayout& offset,
                                   const TensorLayout& mask,
                                   const TensorLayout& dst) {
    auto algo = get_algorithm(this, im, filter, offset, mask, dst);
    return algo->get_workspace_in_bytes({this, im, filter, offset, mask, dst});
}

std::vector<AlgoFwd*> Fwd::get_all_algorithms(const TensorLayout& /* im */,
                                              const TensorLayout& /* filter */,
                                              const TensorLayout& /* offset */,
                                              const TensorLayout& /* mask */,
                                              const TensorLayout& /* dst */) {
    std::vector<AlgoFwd*> algos;

    for (auto i : sm_algo_pack.all_algos)
        algos.push_back(static_cast<AlgoFwd*>(i));

    return algos;
}

AlgoFwd* Fwd::get_algorithm_heuristic(const TensorLayout& im,
                                      const TensorLayout& filter,
                                      const TensorLayout& offset,
                                      const TensorLayout& mask,
                                      const TensorLayout& dst,
                                      size_t workspace_limit_in_bytes,
                                      bool reproducible) {
    auto fm = make_canonized_filter_meta(im.ndim, filter, offset);
    return get_algorithm_heuristic(im, fm, offset, mask, dst,
                                   workspace_limit_in_bytes, reproducible);
}

AlgoFwd* Fwd::get_algorithm_heuristic(const TensorLayout& im,
                                      const CanonizedFilterMeta& filter,
                                      const TensorLayout& offset,
                                      const TensorLayout& mask,
                                      const TensorLayout& dst,
                                      size_t workspace_limit_in_bytes,
                                      bool reproducible) {
    AlgoBase::SizeArgs args(this, im, filter, offset, mask, dst);
    if (sm_algo_pack.algo_matmul.is_available_reproducible(
                args, reproducible, workspace_limit_in_bytes)) {
        return &sm_algo_pack.algo_matmul;
    }
    megdnn_throw(megdnn_mangle(
            ssprintf("no %s deformable conv fwd algorithm with args(%s) and "
                     "workspace limit (%zu bytes)",
                     reproducible ? "reproducible" : "usable",
                     args.to_string().c_str(), workspace_limit_in_bytes)));
}

const char* Fwd::get_algorithm_set_name() const {
    return "DEFORMABLE_CONV_FWD_CUDA";
};

void Fwd::exec(_megdnn_tensor_in im, _megdnn_tensor_in filter,
               _megdnn_tensor_in offset, _megdnn_tensor_in mask,
               _megdnn_tensor_out out, _megdnn_workspace workspace) {
    auto algo = get_algorithm(this, im.layout, filter.layout, offset.layout,
                              mask.layout, out.layout);

    AlgoBase::ExecArgs args(this, im, filter, offset, mask, out, workspace);

    algo->check_workspace(args, workspace).exec(args);
    return;
}

/* ============== BwdFlt Implementation ============== */

std::vector<AlgoBwdFlt*> BwdFlt::get_all_algorithms(const TensorLayout& /* im */, 
        const TensorLayout& /* offset */, const TensorLayout& /* mask */,
        const TensorLayout& /* out_grad */, const TensorLayout& /* filter_grad */) {
    std::vector<AlgoBwdFlt*> algos;
    for (auto i : sm_algo_pack.all_algos)
        algos.push_back(static_cast<AlgoBwdFlt*>(i));
    return algos;
}

AlgoBwdFlt* BwdFlt::get_algorithm_heuristic(
        const TensorLayout& im, const TensorLayout& offset,
        const TensorLayout& mask, const TensorLayout& out_grad,
        const TensorLayout& filter_grad,
        size_t workspace_limit_in_bytes, bool reproducible) {
    auto fm = make_canonized_filter_meta(im.ndim, filter_grad, offset);
    return get_algorithm_heuristic(im, offset, mask, out_grad, fm,
                                   workspace_limit_in_bytes, reproducible);
}

AlgoBwdFlt* BwdFlt::get_algorithm_heuristic(
        const TensorLayout& im, const TensorLayout& offset,
        const TensorLayout& mask, const TensorLayout& out_grad,
        const CanonizedFilterMeta& filter_grad,
        size_t workspace_limit_in_bytes, bool reproducible) {
    AlgoBase::SizeArgs args(this, im, offset, mask, out_grad, filter_grad);
    if (sm_algo_pack.algo_matmul.is_available_reproducible(
                args, reproducible, workspace_limit_in_bytes)) {
        return &sm_algo_pack.algo_matmul;
    }
    megdnn_throw(megdnn_mangle(ssprintf(
            "no %s deformable conv bwd filter algorithm with args(%s) and "
            "workspace limit (%zu bytes)",
            reproducible ? "reproducible" : "usable", args.to_string().c_str(),
            workspace_limit_in_bytes)));
}

size_t BwdFlt::get_workspace_in_bytes(
        const TensorLayout& im, const TensorLayout& offset, const TensorLayout& mask,
        const TensorLayout& out_grad, const TensorLayout& filter_grad) {
    auto algo = get_algorithm(this, im, offset, mask, out_grad, filter_grad);
    return algo->get_workspace_in_bytes({this, im, offset, mask, out_grad, filter_grad});
}

const char* BwdFlt::get_algorithm_set_name() const {
    return "DEFORMABLE_CONV_BWD_FILTER_CUDA";
};

void BwdFlt::exec(_megdnn_tensor_in im, _megdnn_tensor_in offset, _megdnn_tensor_in mask,
               _megdnn_tensor_in out_grad, _megdnn_tensor_out filter_grad,
               _megdnn_workspace workspace) {
    AlgoBase::ExecArgs args(this, im, offset, mask, out_grad, filter_grad, workspace);
    auto algo = get_algorithm(this, im.layout, offset.layout, mask.layout, out_grad.layout,
                              filter_grad.layout);
    algo->check_workspace(args, workspace).exec(args);
}

/* ============== BwdData Implementation ============== */

std::vector<AlgoBwdData*> BwdData::get_all_algorithms(
        const TensorLayout& /* im */, const TensorLayout& /* filter */,
        const TensorLayout& /* offset */, const TensorLayout& /* mask */, const TensorLayout& /* out_grad */,
        const TensorLayout& /* im_grad */, const TensorLayout& /* offset_grad */, const TensorLayout& /* mask_grad */) {
    std::vector<AlgoBwdData*> algos;
    for (auto i : sm_algo_pack.all_algos)
        algos.push_back(static_cast<AlgoBwdData*>(i));
    return algos;
}

AlgoBwdData* BwdData::get_algorithm_heuristic(
        const TensorLayout& im, const TensorLayout& filter,
        const TensorLayout& offset, const TensorLayout& mask,
        const TensorLayout& out_grad, const TensorLayout& im_grad,
        const TensorLayout& offset_grad, const TensorLayout& mask_grad,
        size_t workspace_limit_in_bytes, bool reproducible) {
    auto fm = make_canonized_filter_meta(im.ndim, filter, offset);
    return get_algorithm_heuristic(im, fm, offset, mask, out_grad, im_grad,
                                   offset_grad, mask_grad,
                                   workspace_limit_in_bytes, reproducible);
}

AlgoBwdData* BwdData::get_algorithm_heuristic(
        const TensorLayout& im, const CanonizedFilterMeta& filter,
        const TensorLayout& offset, const TensorLayout& mask,
        const TensorLayout& out_grad, const TensorLayout& im_grad,
        const TensorLayout& offset_grad, const TensorLayout& mask_grad,
        size_t workspace_limit_in_bytes, bool reproducible) {
    AlgoBase::SizeArgs args(this, im, filter, offset, mask, out_grad, im_grad,
                            offset_grad, mask_grad);
    if (sm_algo_pack.algo_matmul.is_available_reproducible(
                args, reproducible, workspace_limit_in_bytes)) {
        return &sm_algo_pack.algo_matmul;
    }
    megdnn_throw(megdnn_mangle(ssprintf(
            "no %s deformable conv bwd data algorithm with args(%s) and "
            "workspace limit (%zu bytes)",
            reproducible ? "reproducible" : "usable", args.to_string().c_str(),
            workspace_limit_in_bytes)));
}

size_t BwdData::get_workspace_in_bytes(
        const TensorLayout& im, const TensorLayout& filter,
        const TensorLayout& offset, const TensorLayout& mask,
        const TensorLayout& out_grad, const TensorLayout& im_grad,
        const TensorLayout& offset_grad, const TensorLayout& mask_grad) {
    auto algo = get_algorithm(this, im, filter, offset, mask, out_grad, 
                              im_grad, offset_grad, mask_grad);
    return algo->get_workspace_in_bytes({this, im, filter, offset, mask, out_grad, 
                                         im_grad, offset_grad, mask_grad});
}

const char* BwdData::get_algorithm_set_name() const {
    return "DEFORMABLE_CONV2_BWD_DATA_CUDA";
};

void BwdData::exec(_megdnn_tensor_in im, _megdnn_tensor_in filter,
               _megdnn_tensor_in offset, _megdnn_tensor_in mask,
               _megdnn_tensor_in out_grad, _megdnn_tensor_out im_grad,
               _megdnn_tensor_out offset_grad, _megdnn_tensor_out mask_grad,
               _megdnn_workspace workspace) {
    AlgoBase::ExecArgs args(this, im, filter, offset, mask, out_grad, im_grad,
                            offset_grad, mask_grad, workspace);
    auto algo = get_algorithm(this, im.layout, filter.layout, offset.layout,
                              mask.layout, out_grad.layout, im_grad.layout,
                              offset_grad.layout, mask_grad.layout);
    algo->check_workspace(args, workspace).exec(args);
}

// vim: syntax=cpp.doxygen
