/**
 * \file dnn/src/cuda/convolution/backward_data/chanwise_small.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/cuda/convolution/backward_data/algo.h"
#include "src/cuda/utils.h"
#include "src/cuda/convolution/chanwise/kern.cuh"

using namespace megdnn;
using namespace cuda;
using namespace convolution;

namespace {
inline bool is_available_small(const chanwise::Param& param) {
    return param.chl_mul == 1 && param.stride_h == 1 && param.stride_w == 1 &&
           param.src_h <= 32 && param.src_w <= 32 &&
           param.src_h == param.out_h && param.src_w == param.out_w &&
           param.pad_h < param.flt_h && param.pad_w < param.flt_w &&
           param.flt_h * param.flt_w <= (param.src_h + 1) / 2 * param.src_w;
}
}  // anonymous namespace

bool ConvolutionBackwardDataImpl::AlgoChanwiseSmall::is_available(
        const SizeArgs &args) const {
    if (args.diff_layout->dtype == args.filter_layout->dtype &&
        args.diff_layout->dtype == dtype::BFloat16()) {
        return false;
    }
#if CUDA_VERSION < 9000
    if (args.diff_layout->dtype.enumv() == DTypeEnum::Float16)
        return false;
#endif
    auto kparam = chanwise::Param::from_fwd_args(args.as_fwd_args());
    auto &&fm = args.filter_meta;
    return args.filter_meta.format == Param::Format::NCHW &&
        args.diff_layout->dtype.category() == DTypeCategory::FLOAT &&
           args.opr->param().compute_mode == Param::ComputeMode::DEFAULT &&
        fm.spatial_ndim == 2 && fm.icpg == 1 &&
        fm.dilation[0] == 1 && fm.dilation[1] == 1 &&
        !fm.should_flip && is_available_small(kparam);
}

size_t ConvolutionBackwardDataImpl::AlgoChanwiseSmall::get_workspace_in_bytes(
        const SizeArgs &) const {
    return 0;
}

void ConvolutionBackwardDataImpl::AlgoChanwiseSmall::exec(
        const ExecArgs &args) const {
    auto kparam = chanwise::Param::from_fwd_args(args.as_fwd_args());
    auto stream = cuda_stream(args.handle);
    switch (args.grad_layout->dtype.enumv()) {
        case DTypeEnum::Float32:
            return chanwise::run_bwd_data_small(args.grad_tensor->ptr<float>(),
                                     args.diff_tensor->ptr<float>(),
                                     args.filter_tensor->ptr<float>(), kparam,
                                     stream);
#if CUDA_VERSION >= 9000
        case DTypeEnum::Float16:
            return chanwise::run_bwd_data_small(
                    static_cast<half*>(args.grad_tensor->raw_ptr),
                    static_cast<half*>(args.diff_tensor->raw_ptr),
                    static_cast<half*>(args.filter_tensor->raw_ptr), kparam,
                    stream);
#endif
        default:
            break;
    }
    megdnn_assert_internal(0);
}

// vim: syntax=cpp.doxygen

