/**
 * \file dnn/src/cuda/convolution/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/cuda/convolution/opr_impl.h"
#include "megdnn/dtype.h"
#include "src/common/algo_chooser.h"
#include "src/cuda/convolution/helper.h"
#include "src/cuda/convolution/forward/algos.h"
#include "src/cuda/convolution/backward_data/algo.h"
#include "src/cuda/convolution/backward_filter/algo.h"
#include "src/cuda/conv_bias/opr_impl.h"

#include "src/cuda/utils.h"

using namespace megdnn;
using namespace cuda;
using namespace convolution;

#define TO_STRING2(v) #v
#define TO_STRING(v) TO_STRING2(v)
#define CUDNN_VERSION_STR  \
    TO_STRING(CUDNN_MAJOR) \
    "." TO_STRING(CUDNN_MINOR) "." TO_STRING(CUDNN_PATCHLEVEL)

/* ============== ConvolutionForwardImpl ============== */
ConvolutionForwardImpl::Algorithm*
ConvolutionForwardImpl::get_algorithm_heuristic(
        const TensorLayout& src, const TensorLayout& filter,
        const TensorLayout& dst, size_t workspace_limit_in_bytes,
        const AlgoAttribute& positive_attr,
        const AlgoAttribute& negative_attr) {
    AlgoBase::SizeArgs args{this, src, filter, dst};
    MEGDNN_MARK_USED_VAR(workspace_limit_in_bytes);
    MEGDNN_MARK_USED_VAR(positive_attr);
    MEGDNN_MARK_USED_VAR(negative_attr);
    return &sm_algo_pack.algo_default;
}

std::vector<ConvolutionForwardImpl::Algorithm*>
ConvolutionForwardImpl::get_all_algorithms(const TensorLayout& src,
                                           const TensorLayout& filter,
                                           const TensorLayout& dst) {
    AlgoBase::SizeArgs args{this, src, filter, dst};
    return megdnn::get_all_algorithms<ConvolutionForwardImpl>(args);
}

size_t ConvolutionForwardImpl::get_workspace_in_bytes(
        const TensorLayout& src, const TensorLayout& filter,
        const TensorLayout& dst,
        const PreprocessedFilter* preprocessed_filter) {
    MEGDNN_MARK_USED_VAR(preprocessed_filter);
    AlgoBase::SizeArgs args{this, src, filter, dst};
    return megdnn::get_algorithm(this, src, filter, dst)
            ->get_workspace_in_bytes(args);
}

void ConvolutionForwardImpl::exec(_megdnn_tensor_in src,
                                  _megdnn_tensor_in filter,
                                  _megdnn_tensor_out dst,
                                  const PreprocessedFilter* preprocessed_filter,
                                  _megdnn_workspace workspace) {
    check_exec(src.layout, filter.layout, dst.layout, workspace.size,
               preprocessed_filter);
    AlgoBase::ExecArgs args(this, src, filter, dst, workspace);
    auto&& algo = get_algorithm(this, src.layout, filter.layout, dst.layout);
    algo->check_workspace(args, workspace).exec(args);
}

const char* ConvolutionForwardImpl::get_algorithm_set_name() const {
    return "CUDA CONVOLUTION_FORWARD";
}

/* ============== ConvolutionBackwardDataImpl ============== */

void ConvolutionBackwardDataImpl::exec(_megdnn_tensor_in filter,
                                       _megdnn_tensor_in diff,
                                       _megdnn_tensor_out grad,
                                       _megdnn_workspace workspace) {
    AlgoBase::ExecArgs args(this, filter, diff, grad, workspace);
    auto algo = get_algorithm(this, filter.layout, diff.layout, grad.layout);
    algo->check_workspace(args, workspace).exec(args);
}

std::vector<ConvolutionBackwardDataImpl::Algorithm*>
ConvolutionBackwardDataImpl::get_all_algorithms(const TensorLayout& filter,
                                                const TensorLayout& diff,
                                                const TensorLayout& grad) {
    return megdnn::get_all_algorithms<ConvolutionBackwardDataImpl>(
            {this, filter, diff, grad});
}

ConvolutionBackwardDataImpl::Algorithm*
ConvolutionBackwardDataImpl::get_algorithm_heuristic(
        const TensorLayout& filter, const TensorLayout& diff,
        const TensorLayout& grad, size_t workspace_limit_in_bytes,
        const AlgoAttribute& positive_attr,
        const AlgoAttribute& negative_attr) {
    auto fm = check_layout_fwd(grad, filter, diff);
    return get_algorithm_heuristic(filter, fm, diff, grad,
                                   workspace_limit_in_bytes, positive_attr,
                                   negative_attr);
}

ConvolutionBackwardDataImpl::Algorithm*
ConvolutionBackwardDataImpl::get_algorithm_heuristic(const TensorLayout& filter,
        const CanonizedFilterMeta& filter_meta, const TensorLayout& diff,
        const TensorLayout& grad, size_t workspace_limit_in_bytes,
        const AlgoAttribute& positive_attr,
        const AlgoAttribute& negative_attr) {
    AlgoBase::SizeArgs args(this, filter, filter_meta, diff, grad);

    if (args.filter_meta.group > 1 &&
        sm_algo_pack.chanwise.is_available_attribute(
                args, positive_attr, negative_attr, workspace_limit_in_bytes)) {
        // prefer special chanwise impl
        return &sm_algo_pack.chanwise;
    }

    if (args.filter_layout->dtype.enumv() ==
        DTypeTrait<dtype::QuantizedS8>::enumv) {
        return megdnn::get_algo_match_attribute<ConvolutionBackwardDataImpl>(
                sm_algo_pack.int8_algos, args, workspace_limit_in_bytes,
                "cuda conv bwd_data", positive_attr, negative_attr);
    }

    auto get_cudnn_algo =
            [this, &args, workspace_limit_in_bytes, positive_attr,
             negative_attr]() -> ConvolutionBackwardDataImpl::AlgoBase* {
        auto cudnn_handle = cuda::cudnn_handle(this->handle());
        CUDNNBwdDataDescs desc;
        args.init_desc(desc);

#if CUDNN_MAJOR >= 7
        MEGDNN_MARK_USED_VAR(negative_attr);
        int max_count = 0;
        cudnn_check(cudnnGetConvolutionBackwardDataAlgorithmMaxCount(
                cudnn_handle, &max_count));
        SmallVector<cudnnConvolutionBwdDataAlgoPerf_t> algo_perf(max_count);
        int ret_count = 0;
        cudnn_check(cudnnGetConvolutionBackwardDataAlgorithm_v7(
                cudnn_handle, desc.filter_desc.desc, desc.diff_desc.desc,
                desc.conv_desc.desc, desc.grad_desc.desc, max_count, &ret_count,
                algo_perf.data()));
        for (int i = 0; i < ret_count; ++i) {
            if (algo_perf[i].memory > workspace_limit_in_bytes)
                continue;
            if ((positive_attr & AlgoAttribute::REPRODUCIBLE)) {
                if (algo_perf[i].determinism == CUDNN_DETERMINISTIC) {
                    return reinterpret_cast<AlgoBase*>(
                            sm_algo_pack.cudnn_from_enum(algo_perf[i].algo));
                }
            } else {
                return reinterpret_cast<AlgoBase*>(
                        sm_algo_pack.cudnn_from_enum(algo_perf[i].algo));
            }
        }
        return nullptr;
#else
        cudnnConvolutionBwdDataAlgo_t algo;
        cudnn_check(cudnnGetConvolutionBackwardDataAlgorithm(
                cudnn_handle, desc.filter_desc.desc, desc.diff_desc.desc,
                desc.conv_desc.desc, desc.grad_desc.desc,
                CUDNN_CONVOLUTION_BWD_DATA_SPECIFY_WORKSPACE_LIMIT,
                workspace_limit_in_bytes, &algo));
        auto&& cast_algo =
                reinterpret_cast<AlgoBase*>(sm_algo_pack.cudnn_from_enum(algo));
        return reinterpret_cast<AlgoBase*>(
                megdnn::get_algo_match_attribute<ConvolutionBackwardDataImpl>(
                        cast_algo, positive_attr, negative_attr));
#endif
    };

    if (is_cudnn_supported(args.as_fwd_args())) {
        if (auto algo = get_cudnn_algo())
            return algo;
    }

    if (args.filter_meta.group > 1) {
        auto orig_args = args;
        TensorLayout a, b;
        AlgoGroupConvGeneral::modify_size_args(args, a, b);
        if (is_cudnn_supported(args.as_fwd_args())) {
            if (auto algo = get_cudnn_algo())
                return sm_algo_pack.algo2gconv.at(algo);
        }
        args = orig_args;
    }

    if (args.filter_layout->dtype.enumv() !=
        DTypeTrait<dtype::BFloat16>::enumv) {
        return megdnn::get_algo_match_attribute<ConvolutionBackwardDataImpl>(
                sm_algo_pack.non_cudnn_algos, args, workspace_limit_in_bytes,
                "cuda conv bwd_data", positive_attr, negative_attr);
    } else {
        return megdnn::get_algo_match_attribute<ConvolutionBackwardDataImpl>(
                sm_algo_pack.bfloat16_algos, args, workspace_limit_in_bytes,
                "cuda conv bwd_data", positive_attr, negative_attr);
    }
}

size_t ConvolutionBackwardDataImpl::get_workspace_in_bytes(
        const TensorLayout& filter, const TensorLayout& diff,
        const TensorLayout& grad) {
    AlgoBase::SizeArgs args(this, filter, diff, grad);
    return get_algorithm(this, filter, args.filter_meta, diff, grad)
            ->get_workspace_in_bytes(args);
}

const char* ConvolutionBackwardDataImpl::get_algorithm_set_name() const {
    return "CUDACONV0+CUDNN" CUDNN_VERSION_STR;
}

/* ============== ConvolutionBackwardFilterImpl ============== */

void ConvolutionBackwardFilterImpl::exec(_megdnn_tensor_in src,
                                         _megdnn_tensor_in diff,
                                         _megdnn_tensor_out grad,
                                         _megdnn_workspace workspace) {
    AlgoBase::ExecArgs args(this, src, diff, grad, workspace);
    auto algo = get_algorithm(this, src.layout, diff.layout, grad.layout,
                              args.grad_filter_meta);
    algo->check_workspace(args, workspace).exec(args);
}

std::vector<ConvolutionBackwardFilterImpl::Algorithm*>
ConvolutionBackwardFilterImpl::get_all_algorithms(const TensorLayout& src,
                                                  const TensorLayout& diff,
                                                  const TensorLayout& grad) {
    return megdnn::get_all_algorithms<ConvolutionBackwardFilterImpl>(
            {this, src, diff, grad});
}

ConvolutionBackwardFilterImpl::Algorithm*
ConvolutionBackwardFilterImpl::get_algorithm_heuristic(
        const TensorLayout& src, const TensorLayout& diff,
        const TensorLayout& grad, size_t workspace_limit_in_bytes,
        const AlgoAttribute& positive_attr,
        const AlgoAttribute& negative_attr) {
    auto fm = check_layout_fwd(src, grad, diff);
    return get_algorithm_heuristic(src, diff, grad, fm,
                                   workspace_limit_in_bytes, positive_attr,
                                   negative_attr);
}

ConvolutionBackwardFilterImpl::Algorithm*
ConvolutionBackwardFilterImpl::get_algorithm_heuristic(
        const TensorLayout& src, const TensorLayout& diff,
        const TensorLayout& grad, const CanonizedFilterMeta& grad_meta,
        size_t workspace_limit_in_bytes,
        const AlgoAttribute& positive_attr,
        const AlgoAttribute& negative_attr) {
    AlgoBase::SizeArgs args(this, src, diff, grad, grad_meta);

    if (args.grad_filter_meta.group > 1 &&
        sm_algo_pack.chanwise.is_available_attribute(
                args, positive_attr, negative_attr, workspace_limit_in_bytes)) {
        // prefer special chanwise impl
        return &sm_algo_pack.chanwise;
    }

    auto get_cudnn_algo =
            [this, &args, workspace_limit_in_bytes, positive_attr,
             negative_attr]() -> ConvolutionBackwardFilterImpl::AlgoBase* {
        auto cudnn_handle = cuda::cudnn_handle(this->handle());
        CUDNNBwdFilterDescs desc;
        args.init_desc(desc);

        // disable, segfault in megbrain, need further investigate.
#if 0
        auto is_heuristic_success =
                convolution::PerformanceModelBackwardFilter::
                        get_algo_backward_filter_success(
                                args, desc, workspace_limit_in_bytes, &algo);
        if (is_heuristic_success) {
            return sm_algo_pack.cudnn_from_enum(algo);
        }
#endif
#if CUDNN_MAJOR >= 7
        MEGDNN_MARK_USED_VAR(negative_attr);
        int max_count = 0;
        cudnn_check(cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(
                cudnn_handle, &max_count));
        SmallVector<cudnnConvolutionBwdFilterAlgoPerf_t> algo_perf(max_count);
        int ret_count = 0;
        cudnn_check(cudnnGetConvolutionBackwardFilterAlgorithm_v7(
                cudnn_handle, desc.src_desc.desc, desc.diff_desc.desc,
                desc.conv_desc.desc, desc.grad_desc.desc, max_count, &ret_count,
                algo_perf.data()));
        for (int i = 0; i < ret_count; ++i) {
            if (algo_perf[i].memory > workspace_limit_in_bytes)
                continue;
            if ((positive_attr & AlgoAttribute::REPRODUCIBLE)) {
                if (algo_perf[i].determinism == CUDNN_DETERMINISTIC) {
                    return reinterpret_cast<AlgoBase*>(
                            sm_algo_pack.cudnn_from_enum(algo_perf[i].algo));
                }
            } else {
                return reinterpret_cast<AlgoBase*>(
                        sm_algo_pack.cudnn_from_enum(algo_perf[i].algo));
            }
        }
        return nullptr;
#else
        cudnnConvolutionBwdFilterAlgo_t algo;
        cudnn_check(cudnnGetConvolutionBackwardFilterAlgorithm(
                cudnn_handle, desc.src_desc.desc, desc.diff_desc.desc,
                desc.conv_desc.desc, desc.grad_desc.desc,
                CUDNN_CONVOLUTION_BWD_FILTER_SPECIFY_WORKSPACE_LIMIT,
                workspace_limit_in_bytes, &algo));
        auto&& cast_algo =
                reinterpret_cast<AlgoBase*>(sm_algo_pack.cudnn_from_enum(algo));
        return reinterpret_cast<AlgoBase*>(
                megdnn::get_algo_match_attribute<ConvolutionBackwardFilterImpl>(
                        cast_algo, positive_attr, negative_attr));
#endif
    };

    if (is_cudnn_supported(args.as_fwd_args())) {
        if (auto algo = get_cudnn_algo())
            return algo;
    }

    if (args.grad_filter_meta.group > 1) {
        auto orig_args = args;
        TensorLayout a, b;
        AlgoGroupConvGeneral::modify_size_args(args, a, b);
        if (is_cudnn_supported(args.as_fwd_args())) {
            if (auto algo = get_cudnn_algo())
                return sm_algo_pack.algo2gconv.at(algo);
        }
        args = orig_args;
    }

    if (args.src_layout->dtype.enumv() != DTypeTrait<dtype::BFloat16>::enumv) {
        return megdnn::get_algo_match_attribute<ConvolutionBackwardFilterImpl>(
                sm_algo_pack.non_cudnn_algos, args, workspace_limit_in_bytes,
                "cuda conv bwd_filter", positive_attr, negative_attr);
    } else {
        return megdnn::get_algo_match_attribute<ConvolutionBackwardFilterImpl>(
                sm_algo_pack.bfloat16_algos, args, workspace_limit_in_bytes,
                "cuda conv bwd_filter", positive_attr, negative_attr);
    }
}

size_t ConvolutionBackwardFilterImpl::get_workspace_in_bytes(
        const TensorLayout& src, const TensorLayout& diff,
        const TensorLayout& grad) {
    AlgoBase::SizeArgs args(this, src, diff, grad);
    return get_algorithm(this, src, diff, grad, args.grad_filter_meta)
            ->get_workspace_in_bytes(args);
}

const char* ConvolutionBackwardFilterImpl::get_algorithm_set_name() const {
    return "CUDACONV0+CUDNN" CUDNN_VERSION_STR;
}

// vim: syntax=cpp.doxygen
