/**
 * \file dnn/src/rocm/convolution/forward/1x1.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./algo.h"
#include "src/rocm/handle.h"
#include "src/rocm/utils.h.hip"

using namespace megdnn;
using namespace rocm;
using namespace convolution;

bool ConvolutionForwardImpl::Algo1x1::is_available(const SizeArgs& args) const {
    auto&& fm = args.filter_meta;
    const size_t MAX_WORKSPACE_SIZE = 2147483648;  // 2 * 1024^3

    if (!(fm.format == Param::Format::NCHW &&
          args.opr->param().compute_mode != Param::ComputeMode::FLOAT32 &&
          (fm.dtype.enumv() == DTypeEnum::Float32 ||
           fm.dtype.enumv() == DTypeEnum::Float16) &&
          fm.spatial_ndim == 2 && fm.group == 1 && fm.dilation[0] == 1 &&
          fm.dilation[1] == 1 && fm.spatial[0] == 1 && fm.spatial[1] == 1 &&
          fm.padding[0] == 0 && fm.padding[1] == 0 && fm.stride[0] == 1 &&
          fm.stride[1] == 1))
        return false;
    if (get_workspace_in_bytes(args) > MAX_WORKSPACE_SIZE) {
        return false;
    }
    return true;
}

void ConvolutionForwardImpl::Algo1x1::extract_matmul_layouts(
        const SizeArgs& args, TensorLayout& A, TensorLayout& B,
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
size_t ConvolutionForwardImpl::Algo1x1::get_workspace_in_bytes(
        const SizeArgs& args) const {
    TensorLayout A, B, C;
    extract_matmul_layouts(args, A, B, C);
    return args.handle->matmul_opr()->get_workspace_in_bytes(A, B, C);
}
void ConvolutionForwardImpl::Algo1x1::exec(const ExecArgs& args) const {
    TensorND A, B, C;
    extract_matmul_layouts(args, A.layout, B.layout, C.layout);
    A.raw_ptr = args.filter_tensor->raw_ptr;
    B.raw_ptr = args.src_tensor->raw_ptr;
    C.raw_ptr = args.dst_tensor->raw_ptr;
    size_t batch = args.src_layout->shape[0];
    auto mm = args.handle->matmul_opr();
    auto strd_B = args.src_layout->stride[0] * args.src_layout->dtype.size(),
         strd_C = args.dst_layout->stride[0] * args.dst_layout->dtype.size();
    for (size_t i = 0; i < batch; ++i) {
        mm->exec(A, B, C, args.workspace);
        incr_voidp(B.raw_ptr, strd_B);
        incr_voidp(C.raw_ptr, strd_C);
    }
}

/*
 *  Funcitons to handle large batch
 */
bool ConvolutionForwardImpl::Algo1x1LargeBatch::is_available(
        const SizeArgs& args) const {
    auto&& fm = args.filter_meta;
    return fm.format == Param::Format::NCHW &&
           args.opr->param().compute_mode != Param::ComputeMode::FLOAT32 &&
           (fm.dtype.enumv() == DTypeEnum::Float32 ||
            fm.dtype.enumv() == DTypeEnum::Float16) &&
           fm.spatial_ndim == 2 && fm.group == 1 && fm.dilation[0] == 1 &&
           fm.dilation[1] == 1 && fm.spatial[0] == 1 && fm.spatial[1] == 1 &&
           fm.padding[0] == 0 && fm.padding[1] == 0 && fm.stride[0] == 1 &&
           fm.stride[1] == 1;
}

void ConvolutionForwardImpl::Algo1x1LargeBatch::extract_matmul_layouts(
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

size_t ConvolutionForwardImpl::Algo1x1LargeBatch::get_workspace_in_bytes(
        const SizeArgs& args) const {
    TensorLayout A, B, C;
    extract_matmul_layouts(args, A, B, C);
    return args.handle->batched_matrix_mul()->get_workspace_in_bytes(A, B, C);
}

void ConvolutionForwardImpl::Algo1x1LargeBatch::exec(
        const ExecArgs& args) const {
    TensorND A, B, C;
    extract_matmul_layouts(args, A.layout, B.layout, C.layout);
    A.raw_ptr = args.filter_tensor->raw_ptr;
    B.raw_ptr = args.src_tensor->raw_ptr;
    C.raw_ptr = args.dst_tensor->raw_ptr;
    auto mm = args.handle->batched_matrix_mul();
    mm->exec(A, B, C, args.workspace);
}
// vim: syntax=cpp.doxygen
