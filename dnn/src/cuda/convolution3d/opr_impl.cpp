/**
 * \file dnn/src/cuda/convolution3d/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
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

#include "src/common/algo_chooser.h"
#include "src/cuda/utils.h"

using namespace megdnn;
using namespace cuda;
using namespace convolution3d;

#define TO_STRING2(v) #v
#define TO_STRING(v)  TO_STRING2(v)
#define CUDNN_VERSION_STR  \
    TO_STRING(CUDNN_MAJOR) \
    "." TO_STRING(CUDNN_MINOR) "." TO_STRING(CUDNN_PATCHLEVEL)

/* ============== Convolution3DForwardImpl ============== */
Convolution3DForwardImpl::Algorithm* Convolution3DForwardImpl::get_algorithm_heuristic(
        const TensorLayout& src, const TensorLayout& filter, const TensorLayout& dst,
        size_t workspace_limit_in_bytes, const AlgoAttribute& positive_attr,
        const AlgoAttribute& negative_attr) {
    AlgoBase::SizeArgs args(this, src, filter, dst);

#if CUDNN_MAJOR < 7 || (CUDNN_MAJOR == 7 && CUDNN_MINOR < 5)
    if (args.filter_meta.group > 1) {
        // prefer special chanwise impl since as the group conv of cudnn whose
        // version is lower than v7.5.0 is still slower than our implementation
        // in many channel-wise cases
        if (sm_algo_pack.chanwise.is_available_attribute(
                    args, positive_attr, negative_attr, workspace_limit_in_bytes)) {
            return &sm_algo_pack.chanwise;
        }
    }
#endif

    auto prefer_1x1x1 = [&args, positive_attr, negative_attr,
                         workspace_limit_in_bytes]() {
        const size_t MAX_BATCH_SIZE_FOR_1x1x1_MAT_ALGO = 4;
        size_t batch_size = args.src_layout->shape[0];
        if (batch_size > MAX_BATCH_SIZE_FOR_1x1x1_MAT_ALGO) {
            return false;
        }
        return sm_algo_pack.a1x1x1.is_available_attribute(
                args, positive_attr, negative_attr, workspace_limit_in_bytes);
    };

    auto get_cudnn_algo = [this, &args, workspace_limit_in_bytes, positive_attr,
                           negative_attr]() -> Convolution3DForwardImpl::AlgoBase* {
        auto cudnn_handle = cuda::cudnn_handle(this->handle());
        cudnnConvolutionFwdAlgo_t algo;
        CUDNNForwardDescs desc;
        args.init_desc(desc);

        bool got = cudnn_get_convolution_fwd_algo_helper(
                cudnn_handle, desc.src_desc.desc, desc.filter_desc.desc,
                desc.conv_desc.desc, desc.dst_desc.desc, workspace_limit_in_bytes,
                &algo, positive_attr, negative_attr);
        if (got) {
            return static_cast<AlgoBase*>(
                    megdnn::get_algo_match_attribute<Convolution3DForwardImpl>(
                            sm_algo_pack.cudnn_from_enum(algo), positive_attr,
                            negative_attr));
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

    if (args.filter_meta.group > 1 &&
        sm_algo_pack.group.is_available_attribute(
                args, positive_attr, negative_attr, workspace_limit_in_bytes)) {
        return &sm_algo_pack.group;
    }

    return megdnn::get_algo_match_attribute<Convolution3DForwardImpl>(
            sm_algo_pack.non_cudnn_algos, args, workspace_limit_in_bytes,
            "cuda conv3d fwd", positive_attr, negative_attr);
}

std::vector<Convolution3DForwardImpl::Algorithm*> Convolution3DForwardImpl::
        get_all_algorithms(
                const TensorLayout& src, const TensorLayout& filter,
                const TensorLayout& dst) {
    return megdnn::get_all_algorithms<Convolution3DForwardImpl>(
            {this, src, filter, dst});
}

std::vector<Convolution3DForwardImpl::Algorithm*> Convolution3DForwardImpl::
        get_all_algorithms_safe(
                const TensorLayout& src, const TensorLayout& filter,
                const TensorLayout& dst) {
    return megdnn::get_all_algorithms_safe<Convolution3DForwardImpl>(
            {this, src, filter, dst});
}

size_t Convolution3DForwardImpl::get_workspace_in_bytes(
        const TensorLayout& src, const TensorLayout& filter, const TensorLayout& dst) {
    return get_dnn_workspace(this, src, filter, dst);
}

void Convolution3DForwardImpl::exec(
        _megdnn_tensor_in src, _megdnn_tensor_in filter, _megdnn_tensor_out dst,
        _megdnn_workspace workspace) {
    check_exec(src.layout, filter.layout, dst.layout, workspace.size);
    AlgoBase::ExecArgs args(this, src, filter, dst, workspace);
    auto algo = get_algorithm(this, src.layout, filter.layout, dst.layout);
    algo->exec(args);
}

const char* Convolution3DForwardImpl::get_algorithm_set_name() const {
    return "CUDACONV0+CUDNN" CUDNN_VERSION_STR;
}

void Convolution3DBackwardDataImpl::exec(
        _megdnn_tensor_in filter, _megdnn_tensor_in diff, _megdnn_tensor_out grad,
        _megdnn_workspace workspace) {
    check_exec(filter.layout, diff.layout, grad.layout, workspace.size);
    AlgoBase::ExecArgs args(this, filter, diff, grad, workspace);
    auto algo = get_algorithm(this, filter.layout, diff.layout, grad.layout);
    algo->exec(args);
}

std::vector<Convolution3DBackwardDataImpl::Algorithm*> Convolution3DBackwardDataImpl::
        get_all_algorithms(
                const TensorLayout& filter, const TensorLayout& diff,
                const TensorLayout& grad) {
    return megdnn::get_all_algorithms<Convolution3DBackwardDataImpl>(
            {this, filter, diff, grad});
}

std::vector<Convolution3DBackwardDataImpl::Algorithm*> Convolution3DBackwardDataImpl::
        get_all_algorithms_safe(
                const TensorLayout& filter, const TensorLayout& diff,
                const TensorLayout& grad) {
    return megdnn::get_all_algorithms_safe<Convolution3DBackwardDataImpl>(
            {this, filter, diff, grad});
}

Convolution3DBackwardDataImpl::Algorithm* Convolution3DBackwardDataImpl::
        get_algorithm_heuristic(
                const TensorLayout& filter, const TensorLayout& diff,
                const TensorLayout& grad, size_t workspace_limit_in_bytes,
                const AlgoAttribute& positive_attr,
                const AlgoAttribute& negative_attr) {
    AlgoBase::SizeArgs args(this, filter, diff, grad);

    if (args.filter_meta.group > 1 &&
        sm_algo_pack.chanwise.is_available_attribute(
                args, positive_attr, negative_attr, workspace_limit_in_bytes)) {
        return &sm_algo_pack.chanwise;
    }

    auto get_cudnn_algo =
            [this, &args, workspace_limit_in_bytes, positive_attr,
             negative_attr]() -> Convolution3DBackwardDataImpl::AlgoBase* {
        auto cudnn_handle = cuda::cudnn_handle(this->handle());
        cudnnConvolutionBwdDataAlgo_t algo;
        CUDNNBwdDataDescs desc;
        args.init_desc(desc);
        bool got = cudnn_get_convolution_bwd_data_algo_helper(
                cudnn_handle, desc.filter_desc.desc, desc.diff_desc.desc,
                desc.conv_desc.desc, desc.grad_desc.desc, workspace_limit_in_bytes,
                &algo, positive_attr, negative_attr);
        if (got) {
            return static_cast<AlgoBase*>(
                    megdnn::get_algo_match_attribute<Convolution3DBackwardDataImpl>(
                            sm_algo_pack.cudnn_from_enum(algo), positive_attr,
                            negative_attr));
        } else {
            return nullptr;
        }
    };

    if (is_cudnn_supported(args.as_fwd_args())) {
        if (auto algo = get_cudnn_algo())
            return algo;
    }

    if (args.filter_meta.group > 1 &&
        sm_algo_pack.group.is_available_attribute(
                args, positive_attr, negative_attr, workspace_limit_in_bytes)) {
        return &sm_algo_pack.group;
    }

    return megdnn::get_algo_match_attribute<Convolution3DBackwardDataImpl>(
            sm_algo_pack.non_cudnn_algos, args, workspace_limit_in_bytes,
            "cuda conv3d bwd data", positive_attr, negative_attr);
}

size_t Convolution3DBackwardDataImpl::get_workspace_in_bytes(
        const TensorLayout& filter, const TensorLayout& diff,
        const TensorLayout& grad) {
    return get_dnn_workspace(this, filter, diff, grad);
}

const char* Convolution3DBackwardDataImpl::get_algorithm_set_name() const {
    return "CUDACONV0+CUDNN" CUDNN_VERSION_STR;
}

void Convolution3DBackwardFilterImpl::exec(
        _megdnn_tensor_in src, _megdnn_tensor_in diff, _megdnn_tensor_out grad,
        _megdnn_workspace workspace) {
    check_exec(src.layout, diff.layout, grad.layout, workspace.size);
    AlgoBase::ExecArgs args(this, src, diff, grad, workspace);
    auto algo = get_algorithm(this, src.layout, diff.layout, grad.layout);
    algo->exec(args);
}

std::vector<Convolution3DBackwardFilterImpl::Algorithm*>
Convolution3DBackwardFilterImpl::get_all_algorithms(
        const TensorLayout& src, const TensorLayout& diff, const TensorLayout& grad) {
    return megdnn::get_all_algorithms<Convolution3DBackwardFilterImpl>(
            {this, src, diff, grad});
}

std::vector<Convolution3DBackwardFilterImpl::Algorithm*>
Convolution3DBackwardFilterImpl::get_all_algorithms_safe(
        const TensorLayout& src, const TensorLayout& diff, const TensorLayout& grad) {
    return megdnn::get_all_algorithms_safe<Convolution3DBackwardFilterImpl>(
            {this, src, diff, grad});
}

Convolution3DBackwardFilterImpl::Algorithm* Convolution3DBackwardFilterImpl::
        get_algorithm_heuristic(
                const TensorLayout& src, const TensorLayout& diff,
                const TensorLayout& grad, size_t workspace_limit_in_bytes,
                const AlgoAttribute& positive_attr,
                const AlgoAttribute& negative_attr) {
    AlgoBase::SizeArgs args(this, src, diff, grad);

    if (args.grad_filter_meta.group > 1 &&
        sm_algo_pack.chanwise.is_available_attribute(
                args, positive_attr, negative_attr, workspace_limit_in_bytes)) {
        return &sm_algo_pack.chanwise;
    }

    auto get_cudnn_algo =
            [this, &args, workspace_limit_in_bytes, positive_attr,
             negative_attr]() -> Convolution3DBackwardFilterImpl::AlgoBase* {
        auto cudnn_handle = cuda::cudnn_handle(this->handle());
        cudnnConvolutionBwdFilterAlgo_t algo;
        CUDNNBwdFilterDescs desc;
        args.init_desc(desc);
        bool got = cudnn_get_convolution_bwd_filter_algo_helper(
                cudnn_handle, desc.src_desc.desc, desc.diff_desc.desc,
                desc.conv_desc.desc, desc.grad_desc.desc, workspace_limit_in_bytes,
                &algo, positive_attr, negative_attr);
        if (got) {
            return static_cast<AlgoBase*>(
                    megdnn::get_algo_match_attribute<Convolution3DBackwardFilterImpl>(
                            sm_algo_pack.cudnn_from_enum(algo), positive_attr,
                            negative_attr));
        } else {
            return nullptr;
        }
    };

    if (is_cudnn_supported(args.as_fwd_args())) {
        if (auto algo = get_cudnn_algo())
            return algo;
    }

    if (args.grad_filter_meta.group > 1 &&
        sm_algo_pack.group.is_available_attribute(
                args, positive_attr, negative_attr, workspace_limit_in_bytes)) {
        return &sm_algo_pack.group;
    }

    return megdnn::get_algo_match_attribute<Convolution3DBackwardFilterImpl>(
            sm_algo_pack.non_cudnn_algos, args, workspace_limit_in_bytes,
            "cuda conv3d bwd filter", positive_attr, negative_attr);
}

size_t Convolution3DBackwardFilterImpl::get_workspace_in_bytes(
        const TensorLayout& src, const TensorLayout& diff, const TensorLayout& grad) {
    return get_dnn_workspace(this, src, diff, grad);
}

const char* Convolution3DBackwardFilterImpl::get_algorithm_set_name() const {
    return "CUDACONV0+CUDNN" CUDNN_VERSION_STR;
}

// vim: syntax=cpp.doxygen
