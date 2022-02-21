/**
 * \file dnn/src/cuda/conv_bias/chanwise/bwd_large_filter.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./kern.cuh"
#include "./kern_helper.cuh"
#include "cuda.h"
#include "cuda_fp16.h"
#include "src/cuda/convolution/chanwise/launch_config.cuh"
#include "src/cuda/fp16_help.cuh"

using namespace megdnn;
using namespace cuda;
using namespace convolution;
using namespace chanwise;

#include "src/cuda/conv_bias/chanwise/depthwise_large_filter_algo.cuh"

namespace megdnn {
namespace cuda {
namespace convolution {
namespace chanwise {

// =====================================fwd=====================================

template <>
void run_bwd_depthwise_large_filter(
        float* dst, const float* src, const float* flt, const Param& param,
        cudaStream_t stream) {
    INSTANCE(float, float2, DepthwiseConv2dDirection::DIRECTION_BACKWARD)
}

#if CUDA_VERSION >= 9000
template <>
void run_bwd_depthwise_large_filter(
        __half* dst, const __half* src, const __half* flt, const Param& param,
        cudaStream_t stream) {
    INSTANCE(__half, __half2, DepthwiseConv2dDirection::DIRECTION_BACKWARD)
}
#endif

}  // namespace chanwise
}  // namespace convolution
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cuda.doxygen
