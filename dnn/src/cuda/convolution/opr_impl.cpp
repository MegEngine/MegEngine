/**
 * \file dnn/src/cuda/convolution/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/cuda/convolution/opr_impl.h"
#include "megdnn/dtype.h"
#include "src/cuda/convolution/helper.h"
#include "src/cuda/convolution/backward_data/algo.h"
#include "src/cuda/convolution/backward_filter/algo.h"
#include "src/cuda/conv_bias/opr_impl.h"

#include "src/cuda/utils.h"

using namespace megdnn;
using namespace cuda;
using namespace convolution;

#define TO_STRING2(v) #v
#define TO_STRING(v) TO_STRING2(v)
#define CUDNN_VERSION_STR TO_STRING(CUDNN_MAJOR) "." \
    TO_STRING(CUDNN_MINOR) "." TO_STRING(CUDNN_PATCHLEVEL)

/* ============== ConvolutionForwardImpl ============== */
ConvolutionForwardImpl::ConvBiasExtraData
ConvolutionForwardImpl::conv_bias_extra_data(const TensorLayout& src,
                                             const TensorLayout& filter,
                                             const TensorLayout& dst) {
    auto conv_param = param();
    DType bias_type;
    if (src.dtype.enumv() == DTypeEnum::QuantizedS8) {
        bias_type = dtype::QuantizedS32(
                src.dtype.param<dtype::QuantizedS8>().scale *

                filter.dtype.param<dtype::QuantizedS8>().scale);
    } else if (src.dtype.enumv() == DTypeEnum::Quantized8Asymm) {
        bias_type = dtype::QuantizedS32(
                src.dtype.param<dtype::Quantized8Asymm>().scale *

                filter.dtype.param<dtype::Quantized8Asymm>().scale);
    } else if (src.dtype.enumv() == DTypeEnum::Uint8 ||
               src.dtype.enumv() == DTypeEnum::Int8) {
        bias_type = dtype::Int32{};
    } else if (src.dtype.enumv() == DTypeEnum::Quantized4Asymm) {
        bias_type = dtype::QuantizedS32(
                src.dtype.param<dtype::Quantized4Asymm>().scale *

                filter.dtype.param<dtype::Quantized4Asymm>().scale);
    } else {
        megdnn_assert(src.dtype.category() == DTypeCategory::FLOAT);
        bias_type = src.dtype;
    }
    ConvBiasExtraData ret = {this->handle()->create_operator<ConvBiasForward>(),
                             TensorLayout(bias_type), TensorLayout(dst.dtype)};
    ret.convbias_opr->param() = {param::ConvBias::NonlineMode::IDENTITY,
                                 conv_param.mode,
                                 conv_param.sparse,
                                 conv_param.format,
                                 conv_param.pad_h,
                                 conv_param.pad_w,
                                 conv_param.stride_h,
                                 conv_param.stride_w,
                                 conv_param.dilate_h,
                                 conv_param.dilate_w,
                                 conv_param.compute_mode};
    ret.convbias_opr->execution_policy() = {this->execution_policy().algo};
    return ret;
}

ConvolutionForwardImpl::Algorithm*
ConvolutionForwardImpl::get_algorithm_heuristic(const TensorLayout& src,
                                                const TensorLayout& filter,
                                                const TensorLayout& dst,
                                                size_t workspace_limit_in_bytes,
                                                bool reproducible) {
    auto extra_data = conv_bias_extra_data(src, filter, dst);
    return static_cast<ConvBiasForwardImpl*>(extra_data.convbias_opr.get())
            ->get_algorithm_heuristic(src, filter, extra_data.bias_layout,
                                      extra_data.z_layout, dst,
                                      workspace_limit_in_bytes, reproducible);
}

std::vector<ConvolutionForwardImpl::Algorithm*>
ConvolutionForwardImpl::get_all_algorithms(const TensorLayout& src,
                                           const TensorLayout& filter,
                                           const TensorLayout& dst) {
    auto extra_data = conv_bias_extra_data(src, filter, dst);
    return static_cast<ConvBiasForwardImpl*>(extra_data.convbias_opr.get())
            ->get_all_algorithms(src, filter, extra_data.bias_layout,
                                 extra_data.z_layout, dst);
}

size_t ConvolutionForwardImpl::get_workspace_in_bytes(
        const TensorLayout& src, const TensorLayout& filter,
        const TensorLayout& dst,
        const PreprocessedFilter* preprocessed_filter) {
    auto extra_data = conv_bias_extra_data(src, filter, dst);
    return static_cast<ConvBiasForwardImpl*>(extra_data.convbias_opr.get())
            ->get_workspace_in_bytes(
                    src, filter, extra_data.bias_layout, extra_data.z_layout,
                    dst,
                    reinterpret_cast<const ConvolutionBase<
                            param::ConvBias>::PreprocessedFilter*>(
                            preprocessed_filter));
}

void ConvolutionForwardImpl::exec(_megdnn_tensor_in src,
                                  _megdnn_tensor_in filter,
                                  _megdnn_tensor_out dst,
                                  const PreprocessedFilter* preprocessed_filter,
                                  _megdnn_workspace workspace) {
    auto extra_data =
            conv_bias_extra_data(src.layout, filter.layout, dst.layout);
    TensorND bias(nullptr, extra_data.bias_layout);
    TensorND z(nullptr, extra_data.z_layout);
    return static_cast<ConvBiasForwardImpl*>(extra_data.convbias_opr.get())
            ->exec(src, filter, bias, z, dst,
                   reinterpret_cast<const ConvolutionBase<
                           param::ConvBias>::PreprocessedFilter*>(
                           preprocessed_filter),
                   workspace);
}

const char* ConvolutionForwardImpl::get_algorithm_set_name() const {
    return "CUDACONV0+CUDNN" CUDNN_VERSION_STR;
}

/* ============== ConvolutionBackwardDataImpl ============== */

void ConvolutionBackwardDataImpl::exec(_megdnn_tensor_in filter,
        _megdnn_tensor_in diff,
        _megdnn_tensor_out grad,
        _megdnn_workspace workspace) {
    AlgoBase::ExecArgs args(this, filter, diff, grad, workspace);
    auto algo = get_algorithm(this, filter.layout, args.filter_meta,
                              diff.layout, grad.layout);
    algo->check_workspace(args, workspace).exec(args);
}

std::vector<ConvolutionBackwardDataImpl::Algorithm *>
ConvolutionBackwardDataImpl::get_all_algorithms(const TensorLayout &filter,
        const TensorLayout &diff,
        const TensorLayout &grad) {
    return megdnn::get_all_algorithms<ConvolutionBackwardDataImpl>(
            {this, filter, diff, grad});
}

ConvolutionBackwardDataImpl::Algorithm*
ConvolutionBackwardDataImpl::get_algorithm_heuristic(
        const TensorLayout& filter, const TensorLayout& diff,
        const TensorLayout& grad, size_t workspace_limit_in_bytes,
        bool reproducible) {
    auto fm = check_layout_fwd(grad, filter, diff);
    return get_algorithm_heuristic(filter, fm, diff, grad,
                                   workspace_limit_in_bytes, reproducible);
}

ConvolutionBackwardDataImpl::Algorithm*
ConvolutionBackwardDataImpl::get_algorithm_heuristic(const TensorLayout& filter,
        const CanonizedFilterMeta& filter_meta, const TensorLayout& diff,
        const TensorLayout& grad, size_t workspace_limit_in_bytes,
        bool reproducible) {
    AlgoBase::SizeArgs args(this, filter, filter_meta, diff, grad);

    if (args.filter_meta.group > 1 &&
        sm_algo_pack.chanwise.is_available_reproducible(
                args, reproducible, workspace_limit_in_bytes)) {
        // prefer special chanwise impl
        return &sm_algo_pack.chanwise;
    }

    auto get_cudnn_algo =
            [this, &args, workspace_limit_in_bytes,
             reproducible]() -> ConvolutionBackwardDataImpl::AlgoBase* {
        auto cudnn_handle = cuda::cudnn_handle(this->handle());
        CUDNNBwdDataDescs desc;
        args.init_desc(desc);

#if CUDNN_MAJOR >= 7
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
            if (reproducible) {
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
                megdnn::get_reproducible_algo<ConvolutionBackwardDataImpl>(
                        cast_algo, reproducible));
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
        if (reproducible) {
            return megdnn::get_reproducible_algo<ConvolutionBackwardDataImpl>(
                    sm_algo_pack.non_cudnn_algos, args,
                    workspace_limit_in_bytes, "cuda conv bwd_data");
        } else {
            return megdnn::get_usable_algo<ConvolutionBackwardDataImpl>(
                    sm_algo_pack.non_cudnn_algos, args,
                    workspace_limit_in_bytes, "cuda conv bwd_data");
        }
    } else {
        if (reproducible) {
            return megdnn::get_reproducible_algo<ConvolutionBackwardDataImpl>(
                    sm_algo_pack.bfloat16_algos, args, workspace_limit_in_bytes,
                    "cuda conv bwd_data");
        } else {
            return megdnn::get_usable_algo<ConvolutionBackwardDataImpl>(
                    sm_algo_pack.bfloat16_algos, args, workspace_limit_in_bytes,
                    "cuda conv bwd_data");
        }
    }
}

size_t ConvolutionBackwardDataImpl::get_workspace_in_bytes(
        const TensorLayout &filter,
        const TensorLayout &diff,
        const TensorLayout &grad) {
    AlgoBase::SizeArgs args(this, filter, diff, grad);
    return get_algorithm(this, filter, args.filter_meta, diff, grad)->
        get_workspace_in_bytes(args);
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
    auto algo = get_algorithm(this, src.layout, diff.layout,
            grad.layout, args.grad_filter_meta);
    algo->check_workspace(args, workspace).exec(args);
}

std::vector<ConvolutionBackwardFilterImpl::Algorithm *>
ConvolutionBackwardFilterImpl::get_all_algorithms(const TensorLayout &src,
        const TensorLayout &diff,
        const TensorLayout &grad) {
    return megdnn::get_all_algorithms<ConvolutionBackwardFilterImpl>(
            {this, src, diff, grad});
}

ConvolutionBackwardFilterImpl::Algorithm*
ConvolutionBackwardFilterImpl::get_algorithm_heuristic(
        const TensorLayout& src, const TensorLayout& diff,
        const TensorLayout& grad, size_t workspace_limit_in_bytes,
        bool reproducible) {
    auto fm = check_layout_fwd(src, grad, diff);
    return get_algorithm_heuristic(src, diff, grad, fm,
                                   workspace_limit_in_bytes, reproducible);
}

ConvolutionBackwardFilterImpl::Algorithm*
ConvolutionBackwardFilterImpl::get_algorithm_heuristic(
        const TensorLayout& src, const TensorLayout& diff,
        const TensorLayout& grad, const CanonizedFilterMeta& grad_meta,
        size_t workspace_limit_in_bytes, bool reproducible) {
    AlgoBase::SizeArgs args(this, src, diff, grad, grad_meta);

    if (args.grad_filter_meta.group > 1 &&
        sm_algo_pack.chanwise.is_available_reproducible(
                args, reproducible, workspace_limit_in_bytes)) {
        // prefer special chanwise impl
        return &sm_algo_pack.chanwise;
    }

    auto get_cudnn_algo =
            [this, &args, workspace_limit_in_bytes,
             reproducible]() -> ConvolutionBackwardFilterImpl::AlgoBase* {
        auto cudnn_handle = cuda::cudnn_handle(this->handle());
        CUDNNBwdFilterDescs desc;
        args.init_desc(desc);

        //disable, segfault in megbrain, need further investigate.
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
            if (reproducible) {
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
                megdnn::get_reproducible_algo<ConvolutionBackwardFilterImpl>(
                        cast_algo, reproducible));
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
        if (reproducible) {
            return megdnn::get_reproducible_algo<ConvolutionBackwardFilterImpl>(
                    sm_algo_pack.non_cudnn_algos, args,
                    workspace_limit_in_bytes, "cuda conv bwd_filter");
        } else {
            return megdnn::get_usable_algo<ConvolutionBackwardFilterImpl>(
                    sm_algo_pack.non_cudnn_algos, args,
                    workspace_limit_in_bytes, "cuda conv bwd_filter");
        }
    } else {
        if (reproducible) {
            return megdnn::get_reproducible_algo<ConvolutionBackwardFilterImpl>(
                    sm_algo_pack.bfloat16_algos, args, workspace_limit_in_bytes,
                    "cuda conv bwd_filter");
        } else {
            return megdnn::get_usable_algo<ConvolutionBackwardFilterImpl>(
                    sm_algo_pack.bfloat16_algos, args, workspace_limit_in_bytes,
                    "cuda conv bwd_filter");
        }
    }
}

size_t ConvolutionBackwardFilterImpl::get_workspace_in_bytes(
        const TensorLayout &src,
        const TensorLayout &diff,
        const TensorLayout &grad) {
    AlgoBase::SizeArgs args(this, src, diff, grad);
    return get_algorithm(this, src, diff, grad, args.grad_filter_meta)->
        get_workspace_in_bytes(args);
}

const char* ConvolutionBackwardFilterImpl::get_algorithm_set_name() const {
    return "CUDACONV0+CUDNN" CUDNN_VERSION_STR;
}

// vim: syntax=cpp.doxygen
