/**
 * \file dnn/src/cuda/convolution/backward_data/depthwise_large_filter.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/cuda/convolution/backward_data/algo.h"
#include "src/cuda/convolution/chanwise/kern.cuh"
#include "src/cuda/utils.h"

using namespace megdnn;
using namespace cuda;
using namespace convolution;

namespace {
inline bool is_available_depthwise_large_filter(const chanwise::Param& param) {
    auto&& device_prop = cuda::current_device_prop();
    int flt_smem_w = (param.flt_w + 3) / 4 * 4;
    int flt_smem_h = 3;
    int flt_reg_per_thread =
            flt_smem_w > 32 ? (flt_smem_w + 31) / 32 : 1 + flt_smem_w / 4;
    int ow = param.out_w > 64 ? 64 : param.out_w;
    int src_smem_w = ow + flt_smem_w - 1;
    int src_smem_h = flt_smem_h + param.flt_h - 1;
    int src_reg_per_thread = src_smem_w > 128 ? (flt_smem_w + 127) / 128
                                              : 1 + (ow + 3) / 4 + flt_smem_w / 4 - 1;
    int out_reg_per_thread = (ow + 3) / 4 * 4;
    if (device_prop.regsPerBlock < 4 * 32 *
                                           (flt_reg_per_thread + src_reg_per_thread +
                                            out_reg_per_thread) ||
        device_prop.sharedMemPerBlock <
                static_cast<size_t>(
                        flt_smem_w * flt_smem_h + src_smem_w * src_smem_h)) {
        return false;
    }
    return param.stride_h == 1 && param.stride_w == 1 && param.src_h == param.out_h &&
           param.src_w == param.out_w;
}
}  // anonymous namespace

bool ConvolutionBackwardDataImpl::AlgoDepthwiseLargeFilter::is_available(
        const SizeArgs& args) const {
    if (!args.grad_layout->is_contiguous() || !args.diff_layout->is_contiguous()) {
        return false;
    }
    if (args.diff_layout->dtype != args.filter_layout->dtype &&
        (args.diff_layout->dtype != dtype::Float32()
#if CUDA_VERSION >= 9000
         || args.diff_layout->dtype != dtype::Float16()
#endif
                 )) {
        return false;
    }

    auto param = chanwise::Param::from_fwd_args(args.as_fwd_args());
    auto&& fm = args.filter_meta;
    return fm.group > 1 && args.filter_meta.format == Param::Format::NCHW &&
           args.diff_layout->dtype.category() == DTypeCategory::FLOAT &&
           args.opr->param().compute_mode == Param::ComputeMode::DEFAULT &&
           fm.spatial_ndim == 2 && fm.icpg == 1 && fm.dilation[0] == 1 &&
           fm.dilation[1] == 1 && !fm.should_flip &&
           is_available_depthwise_large_filter(param);
}

size_t ConvolutionBackwardDataImpl::AlgoDepthwiseLargeFilter::get_workspace_in_bytes(
        const SizeArgs& args) const {
    return 0;
}

void ConvolutionBackwardDataImpl::AlgoDepthwiseLargeFilter::exec(
        const ExecArgs& args) const {
    auto kparam = chanwise::Param::from_fwd_args(args.as_fwd_args());
    auto stream = cuda_stream(args.handle);
    switch (args.diff_layout->dtype.enumv()) {
        case DTypeEnum::Float32:
            chanwise::run_bwd_depthwise_large_filter(
                    args.grad_tensor->ptr<float>(), args.diff_tensor->ptr<float>(),
                    args.filter_tensor->ptr<float>(), kparam, stream);
            break;
#if CUDA_VERSION >= 9000
        case DTypeEnum::Float16:
            chanwise::run_bwd_depthwise_large_filter(
                    static_cast<half*>(args.grad_tensor->raw_ptr()),
                    static_cast<half*>(args.diff_tensor->raw_ptr()),
                    static_cast<half*>(args.filter_tensor->raw_ptr()), kparam, stream);
            break;
#endif
        default:
            megdnn_assert_internal(0);
    }
}

// vim: syntax=cpp.doxygen
