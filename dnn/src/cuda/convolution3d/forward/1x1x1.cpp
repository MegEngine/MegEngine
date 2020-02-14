/**
 * \file dnn/src/cuda/convolution3d/forward/1x1x1.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./algo.h"
#include "src/cuda/handle.h"
#include "src/cuda/utils.cuh"
using namespace megdnn;
using namespace cuda;
using namespace convolution3d;

bool Convolution3DForwardImpl::Algo1x1x1::is_available(
        const SizeArgs &args) const {
    auto &&fm = args.filter_meta;
    const size_t MAX_WORKSPACE_SIZE = 2147483648; // 2 * 1024^3
    if (get_workspace_in_bytes(args) > MAX_WORKSPACE_SIZE) {
        return false;
    }
    return fm.format == Param::Format::NCDHW &&
        (fm.dtype_enum == DTypeEnum::Float32 ||
         fm.dtype_enum == DTypeEnum::Float16) &&
        fm.spatial_ndim == 3 && fm.group == 1 &&
        fm.dilation[0] == 1 && fm.dilation[1] == 1 &&
        fm.dilation[2] == 1 &&
        fm.spatial[0] == 1 && fm.spatial[1] == 1 &&
        fm.spatial[2] == 1 &&
        fm.padding[0] == 0 && fm.padding[1] == 0 &&
        fm.padding[2] == 0  &&
        fm.stride[0] == 1 && fm.stride[1] == 1 &&
        fm.stride[2] == 1;
}

void Convolution3DForwardImpl::Algo1x1x1::extract_matmul_layouts(
        const SizeArgs &args,
        TensorLayout &A, TensorLayout &B, TensorLayout &C) {
    auto &&fm = args.filter_meta;
    A = {{fm.ocpg, fm.icpg}, DType::from_enum(fm.dtype_enum)};
    B.ndim = 2;
    B.shape[0] = args.src_layout->shape[1];
    B.shape[1] = args.src_layout->shape[2] * args.src_layout->shape[3] * args.src_layout->shape[4];
    B.stride[0] = args.src_layout->stride[1];
    B.stride[1] = 1;
    B.dtype = args.src_layout->dtype;
    C = {{args.dst_layout->shape[1], B.shape[1]}, args.dst_layout->dtype};
}
size_t Convolution3DForwardImpl::Algo1x1x1::get_workspace_in_bytes(
        const SizeArgs &args) const {
    TensorLayout A, B, C;
    extract_matmul_layouts(args, A, B, C);
    return args.handle->matmul_opr()->get_workspace_in_bytes(A, B, C);
}
void Convolution3DForwardImpl::Algo1x1x1::exec(const ExecArgs &args) const {
    TensorND A, B, C;
    extract_matmul_layouts(args, A.layout, B.layout, C.layout);
    A.raw_ptr = args.filter_tensor->raw_ptr;
    B.raw_ptr = args.src_tensor->raw_ptr;
    C.raw_ptr = args.dst_tensor->raw_ptr;
    size_t batch = args.src_layout->shape[0];
    auto mm = args.handle->matmul_opr();
    auto strd_B = args.src_layout->stride[0] * args.src_layout->dtype.size(),
         strd_C = args.dst_layout->stride[0] * args.dst_layout->dtype.size();
    for (size_t i = 0; i < batch; ++ i) {
        mm->exec(A, B, C, args.workspace);
        incr_voidp(B.raw_ptr, strd_B);
        incr_voidp(C.raw_ptr, strd_C);
    }
}
// vim: syntax=cpp.doxygen

