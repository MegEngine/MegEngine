/**
 * \file dnn/src/cuda/convpooling/conv_pooling.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include <cuda_runtime_api.h>

namespace megdnn {
namespace cuda {
namespace conv_pool {

#define NR_PXL_PER_THREAD   4
#define NR_THREAD_PER_BLOCK 192
#define MAX_SHARED_MEM_SIZE 32768 //32 * 1024
#define MAX_TEX_OBJ_SIZE    134217728 //2^27
#define HEIGHT_EQUALS_WITH_WEIGHT

enum PoolModeCu {
    AVERAGE = 0,
    MAX = 1
};

enum ConvModeCu {
    CROSS_CORRELATION = 0,
    CONVOLUTION = 1
};

enum NonlineModeCu{
    IDENTITY = 0,
    RELU = 1,
    SIGMOID = 2
};

void start_gpu_xcorr_pool_with_texture_obj(
        cudaStream_t stream,
        float *input,
        const float *kernel,
        float *output,
        size_t N,
        size_t IC, size_t IH, size_t IW,
        size_t OC, size_t OH, size_t OW,
        size_t FH, size_t FW,
        size_t /*PH*/, size_t /*PW*/,
        size_t /*SH*/, size_t /*SW*/,
        size_t pool_shape_h,
        size_t pool_shape_w,
        PoolModeCu poolMode,
        ConvModeCu convMode,
        NonlineModeCu nonlineMode,
        const float *bias);

} // namespace conv_pool
} // namespace cuda
} // namespace megdnn

// vim: syntax=cpp.doxygen
