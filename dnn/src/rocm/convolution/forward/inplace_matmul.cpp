/**
 * \file dnn/src/rocm/convolution/forward/inplace_matmul.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./algo.h"
#include "./inplace_matmul_impl.h.hip"

using namespace megdnn;
using namespace rocm;

bool ConvolutionForwardImpl::AlgoInplaceMatmul::is_available(
        const SizeArgs& args) const {
    auto&& fm = args.filter_meta;
    return args.filter_meta.format == Param::Format::NCHW &&
           args.src_layout->dtype == dtype::Float32() && fm.group == 1 &&
           fm.spatial_ndim == 2 && fm.dilation[0] == 1 && fm.dilation[1] == 1;
}

size_t ConvolutionForwardImpl::AlgoInplaceMatmul::get_workspace_in_bytes(
        const SizeArgs&) const {
    return 0;
}

void ConvolutionForwardImpl::AlgoInplaceMatmul::exec(
        const ExecArgs& args) const {
    auto&& fm = args.filter_meta;
    size_t N = args.src_layout->shape[0], IC = fm.icpg,
           IH = args.src_layout->shape[2], IW = args.src_layout->shape[3],
           OC = fm.ocpg, OH = args.dst_layout->shape[2],
           OW = args.dst_layout->shape[3], FH = fm.spatial[0],
           FW = fm.spatial[1];
    auto stream = args.handle->stream();
    convolution::exec_inplace_matmul_fwd(
            args.src_tensor->ptr<dt_float32>(),
            args.filter_tensor->ptr<dt_float32>(),
            args.dst_tensor->ptr<dt_float32>(), N, args.src_layout->stride[0],
            args.dst_layout->stride[0], IC, IH, IW, OC, OH, OW, FH, FW,
            fm.padding[0], fm.padding[1], fm.stride[0], fm.stride[1],
            !fm.should_flip, stream);
}

// vim: syntax=cpp.doxygen
