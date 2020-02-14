/**
 * \file dnn/src/cuda/convolution/chanwise/kern_helper.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include "src/cuda/query_blocksize.cuh"
#include "src/cuda/utils.cuh"
#include "megdnn/dtype.h"

#include <cuda_runtime.h>
#include <stdint.h>
#include <algorithm>

namespace megdnn {
namespace cuda {
namespace convolution {
namespace chanwise {

    /*!
     * \brief return a / b and set mod to a % b
     */
    __device__ __forceinline__ uint32_t div_mod(
            uint32_t a, uint32_t b, uint32_t &mod) {
        uint32_t ret = a / b;
        mod = a - ret * b;
        return ret;
    }

    /*!
     * \brief copy a 2D matrix by all threads in a block
     * \param rs row stride
     */
    template<typename T>
    __device__ __forceinline__ void block_memcpy(
            T *dst, const T *src, uint32_t size) {
        for (uint32_t i = threadIdx.x; i < size; i += blockDim.x) {
            dst[i] = src[i];
        }
        __syncthreads();
    }

} // namespace chanwise
} // namespace convolution
} // namespace cuda
} // namespace megdnn

// vim: syntax=cuda.doxygen

