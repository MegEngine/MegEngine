/**
 * \file dnn/src/cuda/convolution3d/backward_filter/inplace_matmul.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./algo.h"
#include "./inplace_matmul_impl.cuh"

using namespace megdnn;
using namespace cuda;

bool Convolution3DBackwardFilterImpl::AlgoInplaceMatmul::is_available(
        const SizeArgs &args) const {
    auto &&fm = args.grad_filter_meta;
    return args.grad_filter_meta.format == Param::Format::NCDHW &&
        args.src_layout->dtype == dtype::Float32() &&
        fm.group == 1 && fm.spatial_ndim == 3;
}

size_t Convolution3DBackwardFilterImpl::AlgoInplaceMatmul::get_workspace_in_bytes(
        const SizeArgs &) const {
    return 0;
}

void Convolution3DBackwardFilterImpl::AlgoInplaceMatmul::exec(
        const ExecArgs &args) const {
    auto &&fm = args.grad_filter_meta;
    size_t N = args.src_layout->shape[0],
           IC = fm.icpg,
           ID = args.src_layout->shape[2],
           IH = args.src_layout->shape[3],
           IW = args.src_layout->shape[4],
           OC = fm.ocpg,
           OD = args.diff_layout->shape[2],
           OH = args.diff_layout->shape[3],
           OW = args.diff_layout->shape[4],
           FD = fm.spatial[0],
           FH = fm.spatial[1],
           FW = fm.spatial[2],
           DD = fm.dilation[0], 
           DH = fm.dilation[1], 
           DW = fm.dilation[2]; 
    auto stream = args.handle->stream();

    convolution3d::exec_inplace_matmul_bwd_filter(
            args.diff_tensor->ptr<dt_float32>(),
            args.src_tensor->ptr<dt_float32>(),
            args.grad_tensor->ptr<dt_float32>(),
            N, 
            args.src_layout->stride[0], 
            args.diff_layout->stride[0], 
            IC, ID, IH, IW,
            OC, OD, OH, OW,
                FD, FH, FW,
            fm.padding[0], fm.padding[1], fm.padding[2], 
            fm.stride[0], fm.stride[1], fm.stride[2],
            DD, DH, DW,
            !fm.should_flip, stream);
}

// vim: syntax=cpp.doxygen

