/**
 * \file dnn/src/cuda/convolution3d/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./opr_impl.h"
#include "./backward_data/algo.h"
#include "./backward_filter/algo.h"
#include "./forward/algo.h"
#include "./helper.h"

#include "src/cuda/utils.h"

using namespace megdnn;
using namespace cuda;
using namespace convolution3d;

#define TO_STRING2(v) #v
#define TO_STRING(v) TO_STRING2(v)
#define CUDNN_VERSION_STR  \
    TO_STRING(CUDNN_MAJOR) \
    "." TO_STRING(CUDNN_MINOR) "." TO_STRING(CUDNN_PATCHLEVEL)

/* ============== Convolution3DForwardImpl ============== */
Convolution3DForwardImpl::Algorithm*
Convolution3DForwardImpl::get_algorithm_heuristic(
        const TensorLayout& src, const TensorLayout& filter,
        const TensorLayout& dst, size_t workspace_limit_in_bytes,
        bool reproducible) {
    auto fm = check_layout_fwd(src, filter, dst);
    return get_algorithm_heuristic(src, fm, dst, workspace_limit_in_bytes,
                                   reproducible);
}
Convolution3DForwardImpl::Algorithm*
Convolution3DForwardImpl::get_algorithm_heuristic(
        const TensorLayout& src, const CanonizedFilterMeta& filter,
        const TensorLayout& dst, size_t workspace_limit_in_bytes,
        bool reproducible) {
    AlgoBase::SizeArgs args(this, src, filter, dst);

#if CUDNN_MAJOR < 7 || (CUDNN_MAJOR == 7 && CUDNN_MINOR < 5)
    if (args.filter_meta.group > 1) {
        // prefer special chanwise impl since as the group conv of cudnn whose
        // version is lower than v7.5.0 is still slower than our implementation
        // in many channel-wise cases
        if (sm_algo_pack.chanwise.is_available_reproducible(
                    args, reproducible, workspace_limit_in_bytes)) {
            return &sm_algo_pack.chanwise;
        }
    }
#endif

    auto prefer_1x1x1 = [&args, reproducible, workspace_limit_in_bytes]() {
        const size_t MAX_BATCH_SIZE_FOR_1x1x1_MAT_ALGO = 4;
        size_t batch_size = args.src_layout->shape[0];
        if (batch_size > MAX_BATCH_SIZE_FOR_1x1x1_MAT_ALGO) {
            return false;
        }
        return sm_algo_pack.a1x1x1.is_available_reproducible(
                args, reproducible, workspace_limit_in_bytes);
    };

    auto get_cudnn_algo =
            [this, &args, workspace_limit_in_bytes,
             reproducible]() -> Convolution3DForwardImpl::AlgoBase* {
        auto cudnn_handle = cuda::cudnn_handle(this->handle());
        cudnnConvolutionFwdAlgo_t algo;
        CUDNNForwardDescs desc;
        args.init_desc(desc);

        bool got = cudnn_get_convolution_fwd_algo_helper(
                cudnn_handle, desc.src_desc.desc, desc.filter_desc.desc,
                desc.conv_desc.desc, desc.dst_desc.desc,
                workspace_limit_in_bytes, &algo, reproducible);
        if (got) {
            return static_cast<AlgoBase*>(
                    megdnn::get_reproducible_algo<Convolution3DForwardImpl>(
                            sm_algo_pack.cudnn_from_enum(algo), reproducible));
        } else {
            return nullptr;
        }
    };
    if (prefer_1x1x1()) {
        return &sm_algo_pack.a1x1x1;
    }
    if (is_cudnn_supported(args)) {
        if (auto algo = get_cudnn_algo())
            return algo;
    }
    if (args.filter_meta.group > 1) {
        auto orig_args = args;
        TensorLayout a, b;
        AlgoGroupConvGeneral::modify_size_args(args, a, b);
        if (prefer_1x1x1()) {
            return sm_algo_pack.algo2gconv.at(&sm_algo_pack.a1x1x1);
        }
        if (is_cudnn_supported(args)) {
            if (auto algo = get_cudnn_algo())
                return sm_algo_pack.algo2gconv.at(algo);
        }
        args = orig_args;
    }

    if (reproducible) {
        return megdnn::get_reproducible_algo<Convolution3DForwardImpl>(
                sm_algo_pack.non_cudnn_algos, args, workspace_limit_in_bytes,
                "cuda conv3d fwd");
    } else {
        return megdnn::get_usable_algo<Convolution3DForwardImpl>(
                sm_algo_pack.non_cudnn_algos, args, workspace_limit_in_bytes,
                "cuda conv3d fwd");
    }
}

std::vector<Convolution3DForwardImpl::Algorithm*>
Convolution3DForwardImpl::get_all_algorithms(const TensorLayout& src,
                                             const TensorLayout& filter,
                                             const TensorLayout& dst) {
    return megdnn::get_all_algorithms<Convolution3DForwardImpl>(
            {this, src, filter, dst});
}

size_t Convolution3DForwardImpl::get_workspace_in_bytes(
        const TensorLayout& src, const TensorLayout& filter,
        const TensorLayout& dst) {
    AlgoBase::SizeArgs args(this, src, filter, dst);
    return get_algorithm(this, src, args.filter_meta, dst)
            ->get_workspace_in_bytes(args);
}

void Convolution3DForwardImpl::exec(_megdnn_tensor_in src,
                                    _megdnn_tensor_in filter,
                                    _megdnn_tensor_out dst,
                                    _megdnn_workspace workspace) {
    AlgoBase::ExecArgs args(this, src, filter, dst, workspace);
    auto algo = get_algorithm(this, src.layout, args.filter_meta, dst.layout);
    algo->check_workspace(args, workspace).exec(args);
}

const char* Convolution3DForwardImpl::get_algorithm_set_name() const {
    return "CUDACONV0+CUDNN" CUDNN_VERSION_STR;
}

void Convolution3DBackwardDataImpl::exec(_megdnn_tensor_in filter,
                                         _megdnn_tensor_in diff,
                                         _megdnn_tensor_out grad,
                                         _megdnn_workspace workspace) {
    AlgoBase::ExecArgs args(this, filter, diff, grad, workspace);
    auto algo = get_algorithm(this, args.filter_meta, diff.layout, grad.layout);
    algo->check_workspace(args, workspace).exec(args);
}

std::vector<Convolution3DBackwardDataImpl::Algorithm*>
Convolution3DBackwardDataImpl::get_all_algorithms(const TensorLayout& filter,
                                                  const TensorLayout& diff,
                                                  const TensorLayout& grad) {
    return megdnn::get_all_algorithms<Convolution3DBackwardDataImpl>(
            {this, filter, diff, grad});
}

Convolution3DBackwardDataImpl::Algorithm*
Convolution3DBackwardDataImpl::get_algorithm_heuristic(
        const TensorLayout& filter, const TensorLayout& diff,
        const TensorLayout& grad, size_t workspace_limit_in_bytes,
        bool reproducible) {
    auto fm = check_layout_fwd(grad, filter, diff);
    return get_algorithm_heuristic(fm, diff, grad, workspace_limit_in_bytes,
                                   reproducible);
}

Convolution3DBackwardDataImpl::Algorithm*
Convolution3DBackwardDataImpl::get_algorithm_heuristic(
        const CanonizedFilterMeta& filter, const TensorLayout& diff,
        const TensorLayout& grad, size_t workspace_limit_in_bytes,
        bool reproducible) {
    AlgoBase::SizeArgs args(this, filter, diff, grad);

    if (args.filter_meta.group > 1 &&
        sm_algo_pack.chanwise.is_available_reproducible(
                args, reproducible, workspace_limit_in_bytes)) {
        return &sm_algo_pack.chanwise;
    }

    auto get_cudnn_algo =
            [this, &args, workspace_limit_in_bytes,
             reproducible]() -> Convolution3DBackwardDataImpl::AlgoBase* {
        auto cudnn_handle = cuda::cudnn_handle(this->handle());
        cudnnConvolutionBwdDataAlgo_t algo;
        CUDNNBwdDataDescs desc;
        args.init_desc(desc);
        bool got = cudnn_get_convolution_bwd_data_algo_helper(
                cudnn_handle, desc.filter_desc.desc, desc.diff_desc.desc,
                desc.conv_desc.desc, desc.grad_desc.desc,
                workspace_limit_in_bytes, &algo, reproducible);
        if (got) {
            return static_cast<AlgoBase*>(megdnn::get_reproducible_algo<
                                          Convolution3DBackwardDataImpl>(
                    sm_algo_pack.cudnn_from_enum(algo), reproducible));
        } else {
            return nullptr;
        }
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

    if (reproducible) {
        return megdnn::get_reproducible_algo<Convolution3DBackwardDataImpl>(
                sm_algo_pack.non_cudnn_algos, args, workspace_limit_in_bytes,
                "cuda conv3d bwd data");
    } else {
        return megdnn::get_usable_algo<Convolution3DBackwardDataImpl>(
                sm_algo_pack.non_cudnn_algos, args, workspace_limit_in_bytes,
                "cuda conv3d bwd data");
    }
}

size_t Convolution3DBackwardDataImpl::get_workspace_in_bytes(
        const TensorLayout& filter, const TensorLayout& diff,
        const TensorLayout& grad) {
    AlgoBase::SizeArgs args(this, filter, diff, grad);
    return get_algorithm(this, args.filter_meta, diff, grad)
            ->get_workspace_in_bytes(args);
}

const char* Convolution3DBackwardDataImpl::get_algorithm_set_name() const {
    return "CUDACONV0+CUDNN" CUDNN_VERSION_STR;
}

void Convolution3DBackwardFilterImpl::exec(_megdnn_tensor_in src,
                                           _megdnn_tensor_in diff,
                                           _megdnn_tensor_out grad,
                                           _megdnn_workspace workspace) {
    AlgoBase::ExecArgs args(this, src, diff, grad, workspace);
    auto algo =
            get_algorithm(this, src.layout, diff.layout, args.grad_filter_meta);
    algo->check_workspace(args, workspace).exec(args);
}

std::vector<Convolution3DBackwardFilterImpl::Algorithm*>
Convolution3DBackwardFilterImpl::get_all_algorithms(const TensorLayout& src,
                                                    const TensorLayout& diff,
                                                    const TensorLayout& grad) {
    return megdnn::get_all_algorithms<Convolution3DBackwardFilterImpl>(
            {this, src, diff, grad});
}

Convolution3DBackwardFilterImpl::Algorithm*
Convolution3DBackwardFilterImpl::get_algorithm_heuristic(
        const TensorLayout& src, const TensorLayout& diff,
        const TensorLayout& grad, size_t workspace_limit_in_bytes,
        bool reproducible) {
    auto fm = check_layout_fwd(src, grad, diff);
    return get_algorithm_heuristic(src, diff, fm, workspace_limit_in_bytes,
                                   reproducible);
}

Convolution3DBackwardFilterImpl::Algorithm*
Convolution3DBackwardFilterImpl::get_algorithm_heuristic(
        const TensorLayout& src, const TensorLayout& diff,
        const CanonizedFilterMeta& grad, size_t workspace_limit_in_bytes,
        bool reproducible) {
    AlgoBase::SizeArgs args(this, src, diff, grad);

    if (args.grad_filter_meta.group > 1 &&
        sm_algo_pack.chanwise.is_available_reproducible(
                args, reproducible, workspace_limit_in_bytes)) {
        return &sm_algo_pack.chanwise;
    }

    auto get_cudnn_algo =
            [this, &args, workspace_limit_in_bytes,
             reproducible]() -> Convolution3DBackwardFilterImpl::AlgoBase* {
        auto cudnn_handle = cuda::cudnn_handle(this->handle());
        cudnnConvolutionBwdFilterAlgo_t algo;
        CUDNNBwdFilterDescs desc;
        args.init_desc(desc);
        bool got = cudnn_get_convolution_bwd_filter_algo_helper(
                cudnn_handle, desc.src_desc.desc, desc.diff_desc.desc,
                desc.conv_desc.desc, desc.grad_desc.desc,
                workspace_limit_in_bytes, &algo, reproducible);
        if (got) {
            return static_cast<AlgoBase*>(megdnn::get_reproducible_algo<
                                          Convolution3DBackwardFilterImpl>(
                    sm_algo_pack.cudnn_from_enum(algo), reproducible));
        } else {
            return nullptr;
        }
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

    if (reproducible) {
        return megdnn::get_reproducible_algo<Convolution3DBackwardFilterImpl>(
                sm_algo_pack.non_cudnn_algos, args, workspace_limit_in_bytes,
                "cuda conv3d bwd filter");
    } else {
        return megdnn::get_usable_algo<Convolution3DBackwardFilterImpl>(
                sm_algo_pack.non_cudnn_algos, args, workspace_limit_in_bytes,
                "cuda conv3d bwd filter");
    }
}

size_t Convolution3DBackwardFilterImpl::get_workspace_in_bytes(
        const TensorLayout& src, const TensorLayout& diff,
        const TensorLayout& grad) {
    AlgoBase::SizeArgs args(this, src, diff, grad);
    return get_algorithm(this, src, diff, args.grad_filter_meta)
            ->get_workspace_in_bytes(args);
}

const char* Convolution3DBackwardFilterImpl::get_algorithm_set_name() const {
    return "CUDACONV0+CUDNN" CUDNN_VERSION_STR;
}

// vim: syntax=cpp.doxygen
