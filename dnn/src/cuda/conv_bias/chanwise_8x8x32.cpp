/**
 * \file dnn/src/cuda/conv_bias/chanwise_8x8x32.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "./algo.h"

#include "src/cuda/utils.h"
#include "src/cuda/conv_bias/chanwise/kern.cuh"
#include "src/common/conv_bias.h"
#include "src/common/elemwise/kern_defs.cuh"

using namespace megdnn;
using namespace cuda;
using namespace conv_bias;

bool ConvBiasForwardImpl::AlgoChanwise8x8x32::is_available(
        const SizeArgs& args) const {
    if (args.z_layout->ndim > 0)
        return false;
    using NonlineMode = param::ConvBias::NonlineMode;

    auto&& fm = args.filter_meta;
    return (args.nonlinear_mode == NonlineMode::IDENTITY ||
            args.nonlinear_mode == NonlineMode::RELU) &&
           args.filter_meta.format == Param::Format::NHWC &&
           args.src_layout->dtype == dtype::Int8() &&
           fm.dtype.enumv() == DTypeEnum::Int8 && fm.spatial_ndim == 2 &&
           fm.icpg == 1 && fm.ocpg == 1 && fm.group % 4 == 0;
}

size_t ConvBiasForwardImpl::AlgoChanwise8x8x32::get_workspace_in_bytes(
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

void ConvBiasForwardImpl::AlgoChanwise8x8x32::exec(const ExecArgs& args) const {
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
        auto kparam = chanwise::Param::from_fwd_args(args);
        auto stream = cuda_stream(args.handle);
        chanwise::run_fwd_8x8x32(conv_dst_tensor.ptr<dt_int32>(),
                                 args.src_tensor->ptr<dt_int8>(),
                                 args.filter_tensor->ptr<dt_int8>(), kparam,
                                 stream);
    }
    handle_bias_and_nonlinear(args.handle, args.nonlinear_mode,
                              &conv_dst_tensor, args.dst_tensor,
                              args.bias_tensor);
}

// vim: syntax=cpp.doxygen
