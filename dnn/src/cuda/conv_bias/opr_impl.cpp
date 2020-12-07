/**
 * \file dnn/src/cuda/conv_bias/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#include "megdnn/dtype.h"
#include "src/cuda/conv_bias/algo.h"
#include "src/cuda/conv_bias/helper.h"
#include "src/cuda/conv_bias/opr_impl.h"
#include "src/cuda/handle.h"
#include "src/cuda/utils.h"

#include "src/common/algo_chooser.h"

#include "src/cuda/cudnn_with_check.h"

namespace megdnn {
namespace cuda {

void ConvBiasForwardImpl::exec(_megdnn_tensor_in src, _megdnn_tensor_in filter,
                               _megdnn_tensor_in bias, _megdnn_tensor_in z,
                               _megdnn_tensor_out dst,
                               const PreprocessedFilter* preprocessed_filter,
                               _megdnn_workspace workspace) {
    check_exec(src.layout, filter.layout, bias.layout, z.layout, dst.layout,
               workspace.size, preprocessed_filter);
    AlgoBase::ExecArgs args(this, src, filter, bias, z, dst, workspace,
                            preprocessed_filter);
    auto algo = get_algorithm(this, src.layout, filter.layout, bias.layout,
                              z.layout, dst.layout);
    algo->check_workspace(args, workspace).exec(args);
};

std::vector<ConvBiasForward::Algorithm*>
ConvBiasForwardImpl::get_all_algorithms(const TensorLayout& src,
                                        const TensorLayout& filter,
                                        const TensorLayout& bias,
                                        const TensorLayout& z,
                                        const TensorLayout& dst) {
    return megdnn::get_all_algorithms<ConvBiasForwardImpl>(
            {this, src, filter, bias, z, dst});
}

ConvBiasForward::Algorithm* ConvBiasForwardImpl::get_algorithm_heuristic(
        const TensorLayout& src, const TensorLayout& filter,
        const TensorLayout& bias, const TensorLayout& z,
        const TensorLayout& dst, size_t workspace_limit_in_bytes,
        bool reproducible) {
    using namespace conv_bias;
    AlgoBase::SizeArgs args{this, src, filter, bias, z, dst};
    auto dst_layout = *args.dst_layout;
    if (dst_layout.dtype.enumv() != args.bias_layout->dtype.enumv()) {
        dst_layout.dtype = DType();
        args.opr->check_or_deduce_dtype_fwd(args.src_layout->dtype,
                                            args.filter_layout->dtype,
                                            dst_layout.dtype);
    }
    auto conv_args = args;

    auto cudnn_conv_bias_act_from_enum_wrapper =
            [](cudnnConvolutionFwdAlgo_t algo) -> AlgoBase* {
        return sm_algo_pack.cudnn_conv_bias_act_from_enum(algo);
    };

    auto cudnn_conv_from_enum_wrapper =
            [](cudnnConvolutionFwdAlgo_t algo) -> AlgoBase* {
        return sm_algo_pack.cudnn_conv_from_enum(algo);
    };

    auto get_cudnn_algo =
            [this, &conv_args, &args, workspace_limit_in_bytes, reproducible](
                    const thin_function<AlgoBase*(cudnnConvolutionFwdAlgo_t)>&
                            cb) -> AlgoBase* {
        auto cudnn_handle = cuda::cudnn_handle(this->handle());
        CUDNNForwardDescs desc;
        conv_args.init_conv_desc(desc);
#if CUDNN_MAJOR >= 7
        int max_count = 0;
        cudnn_check(cudnnGetConvolutionForwardAlgorithmMaxCount(cudnn_handle,
                                                                &max_count));
        SmallVector<cudnnConvolutionFwdAlgoPerf_t> algo_perf(max_count);
        int ret_count = 0;
        cudnn_check(cudnnGetConvolutionForwardAlgorithm_v7(
                cudnn_handle, desc.src_desc.desc, desc.filter_desc.desc,
                desc.conv_desc.conv_desc, desc.dst_desc.desc, max_count,
                &ret_count, algo_perf.data()));
        for (int i = 0; i < ret_count; ++i) {
            auto conv_bias_algo = cb(algo_perf[i].algo);
            if (conv_bias_algo->is_available_reproducible(
                        args, reproducible, workspace_limit_in_bytes))
                return conv_bias_algo;
        }
#else
        cudnnConvolutionFwdAlgo_t algo;
        cudnn_check(cudnnGetConvolutionForwardAlgorithm(
                cudnn_handle, desc.src_desc.desc, desc.filter_desc.desc,
                desc.conv_desc.conv_desc, desc.dst_desc.desc,
                CUDNN_CONVOLUTION_FWD_SPECIFY_WORKSPACE_LIMIT,
                workspace_limit_in_bytes, &algo));

        auto conv_bias_algo = cb(algo);
        if (conv_bias_algo->is_available_reproducible(args, reproducible,
                                                      workspace_limit_in_bytes))
            return conv_bias_algo;
#endif
        return nullptr;
    };

    auto get_1x1_algo = [workspace_limit_in_bytes,
                         reproducible](const AlgoBase::SizeArgs& size_arg)
            -> ConvBiasForwardImpl::AlgoBase* {
        if (sm_algo_pack.batched_matmul.is_available_reproducible(
                    size_arg, reproducible, workspace_limit_in_bytes)) {
            return &sm_algo_pack.batched_matmul;
        } else if (sm_algo_pack.a1x1.is_available_reproducible(
                           size_arg, reproducible, workspace_limit_in_bytes)) {
            return &sm_algo_pack.a1x1;
        }
        return nullptr;
    };

    const bool is_chanwise =
            (args.filter_meta.format == Param::Format::NCHW &&
             args.filter_meta.group == src[1]) ||
            (args.filter_meta.format == Param::Format::NCHW4 &&
             args.filter_meta.group == src[1] * 4) ||
            (args.filter_meta.format == Param::Format::NCHW32 &&
             args.filter_meta.group == src[1] * 32);
    // prefer special chanwise impl since as the group conv of cudnn
    // whose version is lower than v7.5.0 is still slower than our
    // implementation in many channel-wise cases
    const bool slow_cudnn_chanwise_impl =
            CUDNN_MAJOR < 7 || (CUDNN_MAJOR == 7 && CUDNN_MINOR < 5);
    //! choose CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM default for large image
    const int hw_size = src[2] * src[3];
    //! choose dnn when stride != 1, may need calibrate for different cudnn
    //! version
    const bool prefer_dnn_chanwise =
            slow_cudnn_chanwise_impl || args.filter_meta.stride[0] != 1 ||
            args.filter_meta.stride[1] != 1 || hw_size < 512;
    //! avoid bad case in cudnn, check dnn chanwise impl first
    if (is_chanwise) {
        if (prefer_dnn_chanwise) {
            if (sm_algo_pack.chanwise.is_available_reproducible(
                        args, reproducible, workspace_limit_in_bytes))
                return &sm_algo_pack.chanwise;
            if (sm_algo_pack.chanwise8x8x32.is_available_reproducible(
                        args, reproducible, workspace_limit_in_bytes))
                return &sm_algo_pack.chanwise8x8x32;
        } else {
            conv_args.dst_layout = &dst_layout;
            if (is_cudnn_supported(conv_args)) {
                if (auto algo = get_cudnn_algo(cudnn_conv_from_enum_wrapper)) {
                    return algo;
                }
            }
        }
    }

    //! Prefer CUDNN CONVBIAS.
    bool cudnn_conv_bias_act_supported = false;
    for (auto&& algo : sm_algo_pack.cudnn_conv_bias_activations) {
        if (algo.is_available_reproducible(args, reproducible,
                                           workspace_limit_in_bytes)) {
            cudnn_conv_bias_act_supported = true;
            break;
        }
    }

    if (cudnn_conv_bias_act_supported) {
        if (auto algo = get_cudnn_algo(cudnn_conv_bias_act_from_enum_wrapper))
            return algo;
    }

    int batch = src[0];
    if (batch == 1 && sm_algo_pack.a1x1.is_available_reproducible(
                              args, reproducible, workspace_limit_in_bytes)) {
        return &sm_algo_pack.a1x1;
    }

    // modify conv_args dst_layout
    conv_args.dst_layout = &dst_layout;
    if (is_cudnn_supported(conv_args)) {
        if (auto algo = get_cudnn_algo(cudnn_conv_from_enum_wrapper))
            return algo;
    }

    if (args.filter_meta.group > 1) {
        auto orig_args = conv_args;
        TensorLayout src, dst, bias;
        AlgoGroupConvGeneral::modify_size_args(conv_args, src, dst, bias);
        if (auto algo = get_1x1_algo(conv_args)) {
            return sm_algo_pack.algo2gconv.at(algo);
        }
        if (is_cudnn_supported(conv_args)) {
            if (auto algo = get_cudnn_algo(cudnn_conv_from_enum_wrapper)) {
                return sm_algo_pack.algo2gconv.at(algo);
            }
        }
        conv_args = orig_args;
    }

    if (auto algo = get_1x1_algo(args)) {
        return algo;
    }

    if (args.src_layout->dtype.enumv() != DTypeTrait<dtype::BFloat16>::enumv) {
        if (reproducible) {
            return megdnn::get_reproducible_algo<ConvBiasForwardImpl>(
                    sm_algo_pack.non_cudnn_algos, args,
                    workspace_limit_in_bytes, "cuda convbias fwd");
        } else {
            return megdnn::get_usable_algo<ConvBiasForwardImpl>(
                    sm_algo_pack.non_cudnn_algos, args,
                    workspace_limit_in_bytes, "cuda convbias fwd");
        }
    } else {
        if (reproducible) {
            return megdnn::get_reproducible_algo<ConvBiasForwardImpl>(
                    sm_algo_pack.bfloat16_algos, args, workspace_limit_in_bytes,
                    "cuda convbias fwd");
        } else {
            return megdnn::get_usable_algo<ConvBiasForwardImpl>(
                    sm_algo_pack.bfloat16_algos, args, workspace_limit_in_bytes,
                    "cuda convbias fwd");
        }
    }
}

const char* ConvBiasForwardImpl::get_algorithm_set_name() const {
    return "CONV_BIAS_CUDA";
}

size_t ConvBiasForwardImpl::get_workspace_in_bytes(
        const TensorLayout& src, const TensorLayout& filter,
        const TensorLayout& bias, const TensorLayout& z,
        const TensorLayout& dst,
        const PreprocessedFilter* preprocessed_filter) {
    AlgoBase::SizeArgs args{
            this, src, filter, bias, z, dst, preprocessed_filter};
    return get_algorithm(this, src, filter, bias, z, dst)
            ->get_workspace_in_bytes(args);
};

size_t ConvBiasForwardImpl::get_preprocess_workspace_in_bytes(
        const TensorLayout& src, const TensorLayout& filter,
        const TensorLayout& bias, const TensorLayout& z,
        const TensorLayout& dst) {
    AlgoBase::SizeArgs args{this, src, filter, bias, z, dst};
    return get_algorithm(this, src, filter, bias, z, dst)
            ->get_preprocess_workspace_in_bytes(args);
}

SmallVector<TensorLayout>
ConvBiasForwardImpl::deduce_preprocessed_filter_layout(
        const TensorLayout& src, const TensorLayout& filter,
        const TensorLayout& bias, const TensorLayout& z,
        const TensorLayout& dst) {
    AlgoBase::SizeArgs args{this, src, filter, bias, z, dst};
    return get_algorithm(this, src, filter, bias, z, dst)
            ->deduce_preprocessed_filter_layout(args);
}

void ConvBiasForwardImpl::exec_preprocess(
        const TensorLayout& src_layout, _megdnn_tensor_in filter,
        _megdnn_tensor_in bias, const TensorLayout& z_layout,
        const TensorLayout& dst_layout, PreprocessedFilter* preprocessed_filter,
        _megdnn_workspace workspace) {
    TensorND src{nullptr, src_layout}, dst{nullptr, dst_layout},
            z{nullptr, z_layout};
    AlgoBase::ExecArgs args(this, src, filter, bias, z, dst, workspace,
                            preprocessed_filter);
    auto algo = get_algorithm(this, src.layout, filter.layout, bias.layout,
                              z.layout, dst.layout);
    return algo->exec_preprocess(args);
}

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
