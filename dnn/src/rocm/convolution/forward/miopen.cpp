/**
 * \file dnn/src/rocm/convolution/forward/miopen.cpp
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

#include <mutex>
#include "src/rocm/convolution/helper.h"
#include "src/rocm/miopen_wrapper.h"
#include "src/rocm/utils.h"

using namespace megdnn;
using namespace rocm;
using namespace convolution;

MIOpenCache<ConvolutionForwardImpl::AlgoBase::SizeArgs,
            miopenConvFwdAlgorithm_t>
        ConvolutionForwardImpl::AlgoMIOpen::sm_miopen_algo_cache;
MIOpenCache<ConvolutionForwardImpl::AlgoBase::SizeArgs, size_t>
        ConvolutionForwardImpl::AlgoMIOpen::sm_miopen_ws_cache;

bool ConvolutionForwardImpl::AlgoMIOpen::is_available(
        const SizeArgs& args) const {
    if (!is_miopen_supported(args))
        return false;
    auto got = sm_miopen_ws_cache.get(args);
    if (got.first)
        return true;
    MIOpenForwardDescs D;
    args.init_desc(D);
    size_t workspace_size;
    auto status = miopenConvolutionForwardGetWorkSpaceSize(
            args.handle->miopen_handle(), D.filter_desc.desc, D.src_desc.desc,
            D.conv_desc.desc, D.dst_desc.desc, &workspace_size);
    if (status == miopenStatusSuccess) {
        sm_miopen_ws_cache.set(args, workspace_size);
        return true;
    }
    return false;
}

size_t ConvolutionForwardImpl::AlgoMIOpen::get_workspace_in_bytes(
        const SizeArgs& args) const {
    auto got = sm_miopen_ws_cache.get(args);
    if (got.first)
        return got.second;
    MIOpenForwardDescs D;
    args.init_desc(D);
    size_t workspace_size;
    auto status = miopenConvolutionForwardGetWorkSpaceSize(
            args.handle->miopen_handle(), D.filter_desc.desc, D.src_desc.desc,
            D.conv_desc.desc, D.dst_desc.desc, &workspace_size);
    megdnn_assert(status == miopenStatusSuccess,
                  "conv fwd get workspace failed: %s; info: %s",
                  miopenGetErrorString(status), args.to_string().c_str());
    sm_miopen_ws_cache.set(args, workspace_size);
    return workspace_size;
}

miopenConvFwdAlgorithm_t ConvolutionForwardImpl::AlgoMIOpen::find_best_algo(
        const ExecArgs& args) {
    auto find_algo = sm_miopen_algo_cache.get(args);
    if (find_algo.first)
        return find_algo.second;
    bool exhaustive_search = args.handle->enable_miopen_algo_search();
    MIOpenForwardDescs D;
    args.init_desc(D);
    const int req_algo_count = 1;
    int ret_algo_count;
    miopenConvAlgoPerf_t algo_perf;
    miopen_check(miopenFindConvolutionForwardAlgorithm(
            args.handle->miopen_handle(), D.src_desc.desc,
            args.src_tensor->raw_ptr, D.filter_desc.desc,
            args.filter_tensor->raw_ptr, D.conv_desc.desc, D.dst_desc.desc,
            args.dst_tensor->raw_ptr, req_algo_count, &ret_algo_count,
            &algo_perf, args.workspace.raw_ptr, args.workspace.size,
            exhaustive_search));
    sm_miopen_algo_cache.set(args, algo_perf.fwd_algo);
    return algo_perf.fwd_algo;
}

void ConvolutionForwardImpl::AlgoMIOpen::exec(const ExecArgs& args) const {
    MIOpenForwardDescs D;
    args.init_desc(D);
    auto algo = const_cast<ConvolutionForwardImpl::AlgoMIOpen*>(this)
                        ->find_best_algo(args);
    float alpha = 1.0f, beta = 0.0f;
    auto status = miopenConvolutionForward(
            args.handle->miopen_handle(), &alpha, D.src_desc.desc,
            args.src_tensor->raw_ptr, D.filter_desc.desc,
            args.filter_tensor->raw_ptr, D.conv_desc.desc, algo, &beta,
            D.dst_desc.desc, args.dst_tensor->raw_ptr, args.workspace.raw_ptr,
            args.workspace.size);
    megdnn_assert(status == miopenStatusSuccess,
                  "conv fwd failed: %s; info: %s", miopenGetErrorString(status),
                  args.to_string().c_str());
}

void ConvolutionForwardImpl::AlgoPack::fill_miopen_algos() {
    megdnn_throw("MIOpen has implemented auto-tuning in the framework, so we do not need to choose algorithms manually");
}

// vim: syntax=cpp.doxygen
