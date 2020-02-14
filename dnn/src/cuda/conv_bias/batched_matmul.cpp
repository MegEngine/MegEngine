/**
 * \file dnn/src/cuda/conv_bias/batched_matmul.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/common/conv_bias.h"
#include "src/cuda/conv_bias/algo.h"
#include "src/cuda/handle.h"
#include "src/cuda/utils.cuh"

using namespace megdnn;
using namespace cuda;
using namespace conv_bias;

bool ConvBiasForwardImpl::AlgoBatchedMatmul::is_available(
        const SizeArgs& args) const {
    if (args.z_layout->ndim > 0)
        return false;

    //! cudnn batched matmul with discontinuous stride has many bugs, so disable
    //! here.
    TensorLayout A, B, C;
    extract_matmul_layouts(args, A, B, C);
    if (!B.is_contiguous()) {
        return false;
    }
    auto&& fm = args.filter_meta;
    return fm.format == Param::Format::NCHW &&
           (fm.dtype.enumv() == DTypeEnum::Float32 ||
            fm.dtype.enumv() == DTypeEnum::Float16) &&
           fm.spatial_ndim == 2 && fm.group == 1 && fm.dilation[0] == 1 &&
           fm.dilation[1] == 1 && fm.spatial[0] == 1 && fm.spatial[1] == 1 &&
           fm.padding[0] == 0 && fm.padding[1] == 0 && fm.stride[0] == 1 &&
           fm.stride[1] == 1;
}

void ConvBiasForwardImpl::AlgoBatchedMatmul::extract_matmul_layouts(
        const SizeArgs& args, TensorLayout& A, TensorLayout& B,
        TensorLayout& C) {
    auto&& fm = args.filter_meta;
    // A {N, OC, IC}
    // B {N, IC, H * W}
    // C {N, OC, H * W}
    size_t batched = args.src_layout->shape[0];
    A = {{batched, fm.ocpg, fm.icpg}, fm.dtype};
    A.stride[0] = 0;
    B.ndim = 3;
    B.shape[1] = args.src_layout->shape[1];
    B.shape[2] = args.src_layout->shape[2] * args.src_layout->shape[3];
    B.shape[0] = batched;
    B.stride[2] = 1;
    B.stride[1] = args.src_layout->stride[1];
    B.stride[0] = args.src_layout->stride[0];
    B.dtype = args.src_layout->dtype;
    C = {{args.dst_layout->shape[0], args.dst_layout->shape[1], B.shape[2]},
         args.dst_layout->dtype};
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
    TensorLayout A, B, C;
    extract_matmul_layouts(conv_args, A, B, C);
    sizes.insert(
            sizes.begin(),
            args.handle->batched_matrix_mul()->get_workspace_in_bytes(A, B, C));
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
        TensorND A, B, C;
        extract_matmul_layouts(args, A.layout, B.layout, C.layout);
        A.raw_ptr = args.filter_tensor->raw_ptr;
        B.raw_ptr = args.src_tensor->raw_ptr;
        C.raw_ptr = args.dst_tensor->raw_ptr;
        auto mm = args.handle->batched_matrix_mul();
        mm->exec(A, B, C, bundle.get_workspace(0));
    }
    handle_bias_and_nonlinear(args.handle, args.nonlinear_mode,
                              &conv_dst_tensor, args.dst_tensor,
                              args.bias_tensor);
}

// vim: syntax=cpp.doxygen
