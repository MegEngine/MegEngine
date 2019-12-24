/**
 * \file dnn/src/cuda/conv_bias/chanwise.cpp
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
#include "src/cuda/conv_bias/chanwise/kern.cuh"
#include "src/cuda/utils.h"

using namespace megdnn;
using namespace cuda;
using namespace conv_bias;

bool ConvBiasForwardImpl::AlgoChanwise::is_available(
        const SizeArgs& args) const {
    if (args.src_layout->dtype == args.filter_layout->dtype &&
        args.src_layout->dtype == dtype::BFloat16()) {
        return false;
    }
    if (args.z_layout->ndim > 0)
        return false;

    auto&& fm = args.filter_meta;
    bool flag = args.filter_meta.format == Param::Format::NCHW &&
           args.src_layout->dtype.category() == DTypeCategory::FLOAT &&
           args.opr->param().compute_mode == Param::ComputeMode::DEFAULT &&
           fm.spatial_ndim == 2 && fm.icpg == 1 && fm.dilation[0] == 1 &&
           fm.dilation[1] == 1 && !fm.should_flip;
    return flag;
}

size_t ConvBiasForwardImpl::AlgoChanwise::get_workspace_in_bytes(
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

void ConvBiasForwardImpl::AlgoChanwise::exec(const ExecArgs& args) const {
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
        switch (args.src_layout->dtype.enumv()) {
            case DTypeEnum::Float32:
                chanwise::run_fwd(conv_dst_tensor.ptr<float>(),
                                  args.src_tensor->ptr<float>(),
                                  args.filter_tensor->ptr<float>(), kparam,
                                  stream);
                break;
            case DTypeEnum::Float16:
#if CUDA_VERSION >= 9000
                if (is_compute_capability_required(5, 3)) {
                    chanwise::run_fwd(
                            static_cast<half*>(conv_dst_tensor.raw_ptr),
                            static_cast<half*>(args.src_tensor->raw_ptr),
                            static_cast<half*>(args.filter_tensor->raw_ptr),
                            kparam, stream);
                } else {
                    chanwise::run_fwd(conv_dst_tensor.ptr<dt_float16>(),
                                      args.src_tensor->ptr<dt_float16>(),
                                      args.filter_tensor->ptr<dt_float16>(),
                                      kparam, stream);
                }
#else
                chanwise::run_fwd(conv_dst_tensor.ptr<dt_float16>(),
                                  args.src_tensor->ptr<dt_float16>(),
                                  args.filter_tensor->ptr<dt_float16>(), kparam,
                                  stream);
#endif
                break;
            default:
                megdnn_assert_internal(0);
        }
    }

    handle_bias_and_nonlinear(args.handle, args.nonlinear_mode,
                              &conv_dst_tensor, args.dst_tensor,
                              args.bias_tensor);
}

// vim: syntax=cpp.doxygen
