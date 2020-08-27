/**
 * \file dnn/src/rocm/convolution/backward_filter/miopen.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "hcc_detail/hcc_defs_prologue.h"

#include "./algo.h"

#include "src/rocm/utils.h"
#include "src/rocm/miopen_wrapper.h"
#include "src/rocm/convolution/helper.h"

using namespace megdnn;
using namespace rocm;
using namespace convolution;

MIOpenCache<ConvolutionBackwardFilterImpl::AlgoBase::SizeArgs,
            miopenConvBwdWeightsAlgorithm_t>
        ConvolutionBackwardFilterImpl::AlgoMIOpen::sm_miopen_algo_cache;
MIOpenCache<ConvolutionBackwardFilterImpl::AlgoBase::SizeArgs, size_t>
        ConvolutionBackwardFilterImpl::AlgoMIOpen::sm_miopen_ws_cache;

bool ConvolutionBackwardFilterImpl::AlgoMIOpen::is_available(
        const SizeArgs& args) const {
    MIOpenBwdFilterDescs D;
    if (!is_miopen_supported(args.as_fwd_args()))
        return false;
    auto got = sm_miopen_ws_cache.get(args);
    if (got.first)
        return true;
    args.init_desc(D);
    size_t workspace_size;
    auto status = miopenConvolutionBackwardWeightsGetWorkSpaceSize(
            args.handle->miopen_handle(), D.diff_desc.desc, D.src_desc.desc,
            D.conv_desc.desc, D.grad_desc.desc, &workspace_size);
    if (status == miopenStatusSuccess) {
        sm_miopen_ws_cache.set(args, workspace_size);
        return true;
    }
    return false;
}

size_t ConvolutionBackwardFilterImpl::AlgoMIOpen::get_workspace_in_bytes(
        const SizeArgs& args) const {
    auto got = sm_miopen_ws_cache.get(args);
    if (got.first)
        return got.second;
    MIOpenBwdFilterDescs D;
    args.init_desc(D);
    size_t workspace_size;
    auto status = miopenConvolutionBackwardWeightsGetWorkSpaceSize(
            args.handle->miopen_handle(), D.diff_desc.desc, D.src_desc.desc,
            D.conv_desc.desc, D.grad_desc.desc, &workspace_size);
    megdnn_assert(status == miopenStatusSuccess,
                  "conv bwd_filter get workspace failed: %s; info: %s",
                  miopenGetErrorString(status), args.to_string().c_str());
    sm_miopen_ws_cache.set(args, workspace_size);
    return workspace_size;
}

miopenConvBwdWeightsAlgorithm_t
ConvolutionBackwardFilterImpl::AlgoMIOpen::find_best_algo(const ExecArgs& args) {
    auto find_algo = sm_miopen_algo_cache.get(args);
    if (find_algo.first)
        return find_algo.second;
    bool exhaustive_search = args.handle->enable_miopen_algo_search();
    MIOpenBwdFilterDescs D;
    args.init_desc(D);
    const int req_algo_count = 1;
    int ret_algo_count;
    miopenConvAlgoPerf_t algo_perf;
    miopen_check(miopenFindConvolutionBackwardWeightsAlgorithm(
            args.handle->miopen_handle(), D.diff_desc.desc,
            args.diff_tensor->raw_ptr, D.src_desc.desc,
            args.src_tensor->raw_ptr, D.conv_desc.desc, D.grad_desc.desc,
            args.grad_tensor->raw_ptr, req_algo_count, &ret_algo_count,
            &algo_perf, args.workspace.raw_ptr, args.workspace.size,
            exhaustive_search));
//    algo_perf.bwd_weights_algo = miopenConvolutionBwdWeightsAlgoGEMM;
    sm_miopen_algo_cache.set(args, algo_perf.bwd_weights_algo);
    return algo_perf.bwd_weights_algo;
}

void ConvolutionBackwardFilterImpl::AlgoMIOpen::exec(
        const ExecArgs& args) const {
    MIOpenBwdFilterDescs D;
    args.init_desc(D);
    auto algo = const_cast<ConvolutionBackwardFilterImpl::AlgoMIOpen*>(this)
                        ->find_best_algo(args);
    float alpha = 1.0f, beta = 0.0f;
    auto status = miopenConvolutionBackwardWeights(
            args.handle->miopen_handle(), &alpha, D.diff_desc.desc,
            args.diff_tensor->raw_ptr, D.src_desc.desc,
            args.src_tensor->raw_ptr, D.conv_desc.desc, algo, &beta,
            D.grad_desc.desc, args.grad_tensor->raw_ptr, args.workspace.raw_ptr,
            args.workspace.size);
    megdnn_assert(status == miopenStatusSuccess,
                  "conv bwd_filter failed: %s; info: %s",
                  miopenGetErrorString(status), args.to_string().c_str());
}

void ConvolutionBackwardFilterImpl::AlgoPack::fill_miopen_algos() {}

// vim: syntax=cpp.doxygen
