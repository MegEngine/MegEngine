/**
 * \file dnn/src/cuda/conv_bias/batched_matmul.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/common/algo_chooser.h"
#include "src/common/algo_base.h"
#include "src/common/conv_bias.h"
#include "src/cuda/batched_matrix_mul/algo.h"
#include "src/cuda/conv_bias/algo.h"
#include "src/cuda/handle.h"
#include "src/cuda/utils.cuh"

using namespace megdnn;
using namespace cuda;
using namespace conv_bias;

namespace {
std::pair<TensorLayoutArray, MatrixMulForward::Param> sub_opr_config(
        const ConvBiasForwardImpl::CanonizedFilterMeta& fm,
        const TensorLayout& src_layout, const TensorLayout&,
        const TensorLayout& dst_layout, const ConvBiasForwardImpl* opr) {
    // A {N, OC, IC}
    // B {N, IC, H * W}
    // C {N, OC, H * W}
    size_t batched = src_layout.shape[0];
    TensorLayout A, B, C;
    A = {{batched, fm.ocpg, fm.icpg}, fm.dtype};
    A.stride[0] = 0;
    B.ndim = 3;
    B.shape[1] = src_layout.shape[1];
    B.shape[2] = src_layout.shape[2] * src_layout.shape[3];
    B.shape[0] = batched;
    B.stride[2] = 1;
    B.stride[1] = src_layout.stride[1];
    B.stride[0] = src_layout.stride[0];
    B.dtype = src_layout.dtype;
    C = {{dst_layout.shape[0], dst_layout.shape[1], B.shape[2]},
         dst_layout.dtype};
    C.stride[2] = 1;
    C.stride[1] = dst_layout.stride[1];
    C.stride[0] = dst_layout.stride[0];

    MatrixMulForward::Param param;
    if (opr->param().compute_mode == param::Convolution::ComputeMode::FLOAT32) {
        param.compute_mode = param::MatrixMul::ComputeMode::FLOAT32;
    }

    return {{A, B, C}, param};
}

std::pair<TensorLayoutArray, std::unique_ptr<BatchedMatrixMulForward>>
prepare_sub_opr(const ConvBiasForwardImpl::AlgoBase::SizeArgs& args) {
    auto bmatmul_opr = args.handle->create_operator<BatchedMatrixMulForward>();
    set_execution_policy<ConvBiasForward, BatchedMatrixMulForward*>(
            args.opr, bmatmul_opr.get());
    auto&& config =
            sub_opr_config(args.filter_meta, *args.src_layout,
                           *args.filter_layout, *args.dst_layout, args.opr);
    bmatmul_opr->param() = config.second;

    return {config.first, std::move(bmatmul_opr)};
}
}  // namespace

std::vector<Algorithm::SearchItem>
ConvBiasForwardImpl::AlgoBatchedMatmul::get_subopr_list(
        const TensorLayoutArray& layouts, const OperatorBase* opr) const {
    const ConvBiasForwardImpl* conv_bias_opr =
            static_cast<const ConvBiasForwardImpl*>(opr);
    CanonizedFilterMeta fm =
            conv_bias_opr->check_layout_fwd(layouts[0], layouts[1], layouts[4]);
    auto&& config = sub_opr_config(fm, layouts[0], layouts[1], layouts[4],
                                   conv_bias_opr);

    std::string param_str;
    Algorithm::serialize_write_pod(config.second, param_str);
    return {{Algorithm::OprType::BATCHED_MATRIX_MUL_FORWARD, param_str,
             config.first}};
}

bool ConvBiasForwardImpl::AlgoBatchedMatmul::is_available(
        const SizeArgs& args) const {
    if (args.z_layout->ndim > 0)
        return false;

    auto config = prepare_sub_opr(args);
    //! The dst of batched matmul should be contiguous
    if (!config.first[2].is_contiguous()) return false;

    auto&& fm = args.filter_meta;
    return fm.format == Param::Format::NCHW &&
           (fm.dtype.enumv() == DTypeEnum::Float32 ||
            fm.dtype.enumv() == DTypeEnum::Float16) &&
           fm.spatial_ndim == 2 && fm.group == 1 && fm.dilation[0] == 1 &&
           fm.dilation[1] == 1 && fm.spatial[0] == 1 && fm.spatial[1] == 1 &&
           fm.padding[0] == 0 && fm.padding[1] == 0 && fm.stride[0] == 1 &&
           fm.stride[1] == 1 &&
           get_algorithm(static_cast<BatchedMatrixMulForwardImpl*>(
                                 config.second.get()),
                         config.first[0], config.first[1], config.first[2]);
}

WorkspaceBundle ConvBiasForwardImpl::AlgoBatchedMatmul::get_workspace_bundle(
        void* ptr, const SizeArgs& args) const {
    auto dst_layout = *args.dst_layout;
    SmallVector<size_t> sizes;
    if (dst_layout.dtype.enumv() != args.bias_layout->dtype.enumv()) {
        dst_layout.dtype = DType();
        args.opr->check_or_deduce_dtype_fwd(args.src_layout->dtype,
                                            args.filter_layout->dtype,
                                            dst_layout.dtype);
        sizes.push_back(dst_layout.span().dist_byte());
    }

    SizeArgs conv_args = args;
    conv_args.dst_layout = &dst_layout;

    auto config = prepare_sub_opr(args);

    sizes.insert(sizes.begin(),
                 config.second->get_workspace_in_bytes(
                         config.first[0], config.first[1], config.first[2]));
    return {ptr, std::move(sizes)};
}

size_t ConvBiasForwardImpl::AlgoBatchedMatmul::get_workspace_in_bytes(
        const SizeArgs& args) const {
    return get_workspace_bundle(nullptr, args).total_size_in_bytes();
}

void ConvBiasForwardImpl::AlgoBatchedMatmul::exec(const ExecArgs& args) const {
    auto bundle = get_workspace_bundle(args.workspace.raw_ptr, args);
    auto conv_dst_tensor = *args.dst_tensor;
    if (args.dst_layout->dtype.enumv() != args.bias_layout->dtype.enumv()) {
        conv_dst_tensor.raw_ptr = bundle.get(1);
        conv_dst_tensor.layout.dtype = DType();
        args.opr->check_or_deduce_dtype_fwd(args.src_layout->dtype,
                                            args.filter_layout->dtype,
                                            conv_dst_tensor.layout.dtype);
    }

    ExecArgs conv_args = args;
    conv_args.dst_tensor = &conv_dst_tensor;
    conv_args.dst_layout = &conv_dst_tensor.layout;
    {
        auto config = prepare_sub_opr(args);

        TensorND A{args.filter_tensor->raw_ptr, config.first[0]},
                B{args.src_tensor->raw_ptr, config.first[1]},
                C{args.dst_tensor->raw_ptr, config.first[2]};
        config.second->exec(A, B, C, bundle.get_workspace(0));
    }
    handle_bias_and_nonlinear(args.handle, args.nonlinear_mode,
                              &conv_dst_tensor, args.dst_tensor,
                              args.bias_tensor);
}

// vim: syntax=cpp.doxygen
