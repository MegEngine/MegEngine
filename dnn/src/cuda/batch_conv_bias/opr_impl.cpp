/**
 * \file dnn/src/cuda/batch_conv_bias/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/cuda/batch_conv_bias/opr_impl.h"
#include "src/common/algo_chooser.h"
#include "src/cuda/batch_conv_bias/algo.h"
#include "src/cuda/utils.h"

using namespace megdnn;
using namespace cuda;

/* ============== BatchConvBiasForwardImpl ============== */
BatchConvBiasForwardImpl::Algorithm* BatchConvBiasForwardImpl::get_algorithm_heuristic(
        const TensorLayout& src, const TensorLayout& filter, const TensorLayout& bias,
        const TensorLayout& z, const TensorLayout& dst, size_t workspace_limit_in_bytes,
        const AlgoAttribute& positive_attr, const AlgoAttribute& negative_attr) {
    AlgoBase::SizeArgs args(this, src, filter, bias, z, dst);
    if (sm_algo_pack.int8_nchw4_gemm_dotprod.is_available_attribute(
                args, positive_attr, negative_attr, workspace_limit_in_bytes)) {
        return &sm_algo_pack.int8_nchw4_gemm_dotprod;
    }
    if (sm_algo_pack.int8_nchw4_implicit_gemm_dotprod.is_available_attribute(
                args, positive_attr, negative_attr, workspace_limit_in_bytes)) {
        return &sm_algo_pack.int8_nchw4_implicit_gemm_dotprod;
    }
    megdnn_throw(ssprintf(
            "no batch conv bias algorithm without attribute(%s) with "
            "attribute(%s) args(%s) and "
            "workspace limit (%zu bytes)",
            Algorithm::attribute_str(negative_attr).c_str(),
            Algorithm::attribute_str(positive_attr).c_str(), args.to_string().c_str(),
            workspace_limit_in_bytes));
}

std::vector<BatchConvBiasForwardImpl::Algorithm*> BatchConvBiasForwardImpl::
        get_all_algorithms(
                const TensorLayout& src, const TensorLayout& filter,
                const TensorLayout& bias, const TensorLayout& z,
                const TensorLayout& dst) {
    AlgoBase::SizeArgs args{this, src, filter, bias, z, dst};
    return megdnn::get_all_algorithms<BatchConvBiasForwardImpl>(args);
}
std::vector<BatchConvBiasForwardImpl::Algorithm*> BatchConvBiasForwardImpl::
        get_all_algorithms_safe(
                const TensorLayout& src, const TensorLayout& filter,
                const TensorLayout& bias, const TensorLayout& z,
                const TensorLayout& dst) {
    AlgoBase::SizeArgs args{this, src, filter, bias, z, dst};
    return megdnn::get_all_algorithms_safe<BatchConvBiasForwardImpl>(args);
}

size_t BatchConvBiasForwardImpl::get_workspace_in_bytes(
        const TensorLayout& src, const TensorLayout& filter, const TensorLayout& bias,
        const TensorLayout& z, const TensorLayout& dst) {
    return get_dnn_workspace(this, src, filter, bias, z, dst);
}

void BatchConvBiasForwardImpl::exec(
        _megdnn_tensor_in src, _megdnn_tensor_in filter, _megdnn_tensor_in bias,
        _megdnn_tensor_in z, _megdnn_tensor_out dst, _megdnn_workspace workspace) {
    check_exec(
            src.layout, filter.layout, bias.layout, z.layout, dst.layout,
            workspace.size);
    AlgoBase::ExecArgs args(this, src, filter, bias, z, dst, workspace);
    auto algo = get_algorithm(
            this, src.layout, filter.layout, bias.layout, z.layout, dst.layout);
    algo->exec(args);
}

const char* BatchConvBiasForwardImpl::get_algorithm_set_name() const {
    return "CUDA_BATCH_CONV_BIAS";
}

// vim: syntax=cpp.doxygen
