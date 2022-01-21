/**
 * \file dnn/src/cuda/conv_bias/chanwise/fwd_depthwise_large_filter.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "cuda.h"
#include "cuda_fp16.h"
// #include "src/cuda/conv_bias/chanwise/fwd_depthwise_large_filter.cuh"
#include "src/cuda/conv_bias/chanwise/kern.cuh"
#include "src/cuda/conv_bias/chanwise/kern_helper.cuh"
#include "src/cuda/conv_bias/chanwise/launch_config.cuh"
#include "src/cuda/fp16_help.cuh"

using namespace megdnn;
using namespace cuda;
using namespace conv_bias;
using namespace chanwise;

#include "src/cuda/conv_bias/chanwise/depthwise_large_filter_algo.inl"

namespace megdnn {
namespace cuda {
namespace conv_bias {
namespace chanwise {

// =====================================fwd=====================================

#define check

template <>
void run_fwd_depthwise_large_filter(
        float* dst, const float* src, const float* flt, const Param& param,
        cudaStream_t stream) {
    INSTANCE(DepthwiseConv2dDirection::DIRECTION_FORWARD)
}

}  // namespace chanwise
}  // namespace conv_bias
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cuda.doxygen
