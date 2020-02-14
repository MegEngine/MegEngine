/**
 * \file dnn/src/cuda/conv_bias/1x1.cpp
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

bool ConvBiasForwardImpl::Algo1x1::is_available(const SizeArgs& args) const {
    if (args.z_layout->ndim > 0)
        return false;

    auto&& fm = args.filter_meta;
    return fm.format == Param::Format::NCHW &&
          (fm.dtype.enumv() == DTypeEnum::Float32 ||
           fm.dtype.enumv() == DTypeEnum::Float16) &&
          fm.spatial_ndim == 2 && fm.group == 1 && fm.dilation[0] == 1 &&
          fm.dilation[1] == 1 && fm.spatial[0] == 1 && fm.spatial[1] == 1 &&
          fm.padding[0] == 0 && fm.padding[1] == 0 && fm.stride[0] == 1 &&
          fm.stride[1] == 1;
}

void ConvBiasForwardImpl::Algo1x1::extract_matmul_layouts(const SizeArgs& args,
                                                          TensorLayout& A,
                                                          TensorLayout& B,
                                                          TensorLayout& C) {
    auto&& fm = args.filter_meta;
    A = {{fm.ocpg, fm.icpg}, fm.dtype};
    B.ndim = 2;
    B.shape[0] = args.src_layout->shape[1];
    B.shape[1] = args.src_layout->shape[2] * args.src_layout->shape[3];
    B.stride[0] = args.src_layout->stride[1];
    B.stride[1] = 1;
    B.dtype = args.src_layout->dtype;
    C = {{args.dst_layout->shape[1], B.shape[1]}, args.dst_layout->dtype};
}

WorkspaceBundle ConvBiasForwardImpl::Algo1x1::get_workspace_bundle(
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
    sizes.insert(sizes.begin(),
                 args.handle->matmul_opr()->get_workspace_in_bytes(A, B, C));
    return {ptr, std::move(sizes)};
}

size_t ConvBiasForwardImpl::Algo1x1::get_workspace_in_bytes(
        const SizeArgs& args) const {
    return get_workspace_bundle(nullptr, args).total_size_in_bytes();
}

void ConvBiasForwardImpl::Algo1x1::exec(const ExecArgs& args) const {
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
        extract_matmul_layouts(conv_args, A.layout, B.layout, C.layout);
        A.raw_ptr = conv_args.filter_tensor->raw_ptr;
        B.raw_ptr = conv_args.src_tensor->raw_ptr;
        C.raw_ptr = conv_args.dst_tensor->raw_ptr;
        size_t batch = conv_args.src_layout->shape[0];
        auto mm = conv_args.handle->matmul_opr();
        auto strd_B = conv_args.src_layout->stride[0] *
                      conv_args.src_layout->dtype.size(),
             strd_C = conv_args.dst_layout->stride[0] *
                      conv_args.dst_layout->dtype.size();
        for (size_t i = 0; i < batch; ++i) {
            mm->exec(A, B, C, bundle.get_workspace(0));
            incr_voidp(B.raw_ptr, strd_B);
            incr_voidp(C.raw_ptr, strd_C);
        }
    }
    handle_bias_and_nonlinear(args.handle, args.nonlinear_mode,
                              &conv_dst_tensor, args.dst_tensor,
                              args.bias_tensor);
}

// vim: syntax=cpp.doxygen
