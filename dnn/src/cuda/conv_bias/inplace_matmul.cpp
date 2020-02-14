/**
 * \file dnn/src/cuda/conv_bias/inplace_matmul.cpp
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
#include "src/cuda/conv_bias/matmul/inplace_matmul_impl.cuh"

using namespace megdnn;
using namespace cuda;

bool ConvBiasForwardImpl::AlgoInplaceMatmul::is_available(
        const SizeArgs& args) const {
    if (args.z_layout->ndim > 0)
        return false;

    auto&& fm = args.filter_meta;
    return args.filter_meta.format == Param::Format::NCHW &&
           args.src_layout->dtype == dtype::Float32() && fm.group == 1 &&
           fm.spatial_ndim == 2 && fm.dilation[0] == 1 && fm.dilation[1] == 1;
}

size_t ConvBiasForwardImpl::AlgoInplaceMatmul::get_workspace_in_bytes(
        const SizeArgs& args) const {
    auto dst_layout = *args.dst_layout;
    if (dst_layout.dtype.enumv() != args.bias_layout->dtype.enumv()) {
        dst_layout.dtype = DType();
        args.opr->check_or_deduce_dtype_fwd(args.src_layout->dtype,
                                            args.filter_layout->dtype,
                                            dst_layout.dtype);
        return dst_layout.span().dist_byte();
    }
    return 0;
}

void ConvBiasForwardImpl::AlgoInplaceMatmul::exec(const ExecArgs& args) const {
    WorkspaceBundle bundle{args.workspace.raw_ptr,
                           {get_workspace_in_bytes(args)}};
    auto conv_dst_tensor = *args.dst_tensor;
    if (args.dst_layout->dtype.enumv() != args.bias_layout->dtype.enumv()) {
        conv_dst_tensor.raw_ptr = bundle.get(0);
        conv_dst_tensor.layout.dtype = DType();
        args.opr->check_or_deduce_dtype_fwd(args.src_layout->dtype,
                                            args.filter_layout->dtype,
                                            conv_dst_tensor.layout.dtype);
    }

    {
        auto&& fm = args.filter_meta;
        size_t N = args.src_layout->shape[0], IC = fm.icpg,
               IH = args.src_layout->shape[2], IW = args.src_layout->shape[3],
               OC = fm.ocpg, OH = conv_dst_tensor.layout.shape[2],
               OW = conv_dst_tensor.layout.shape[3], FH = fm.spatial[0],
               FW = fm.spatial[1];
        auto stream = args.handle->stream();
        conv_bias::exec_inplace_matmul_fwd(
                args.src_tensor->ptr<dt_float32>(),
                args.filter_tensor->ptr<dt_float32>(),
                conv_dst_tensor.ptr<dt_float32>(), N,
                args.src_layout->stride[0], conv_dst_tensor.layout.stride[0],
                IC, IH, IW, OC, OH, OW, FH, FW, fm.padding[0], fm.padding[1],
                fm.stride[0], fm.stride[1], !fm.should_flip, stream);
    }
    handle_bias_and_nonlinear(args.handle, args.nonlinear_mode,
                              &conv_dst_tensor, args.dst_tensor,
                              args.bias_tensor);
}

// vim: syntax=cpp.doxygen
