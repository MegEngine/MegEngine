/**
 * \file src/rocm/convolution/chanwise/kern_helper.h.hip
 *
 * This file is part of MegDNN, a deep neural network run-time library
 * developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2019 Megvii Inc. All rights reserved.
 */
#pragma once

#include "megdnn/dtype.h"
#include "src/rocm/utils.h.hip"

#include <stdint.h>
#include <algorithm>
#include "hip_header.h"

namespace megdnn {
namespace rocm {
namespace convolution {
namespace chanwise {

/*!
 * \brief return a / b and set mod to a % b
 */
__device__ __forceinline__ uint32_t div_mod(uint32_t a, uint32_t b,
                                            uint32_t& mod) {
    uint32_t ret = a / b;
    mod = a - ret * b;
    return ret;
}

/*!
 * \brief copy a 2D matrix by all threads in a block
 * \param rs row stride
 */
template <typename T>
__device__ __forceinline__ void block_memcpy(T* dst, const T* src,
                                             uint32_t size) {
    for (uint32_t i = threadIdx.x; i < size; i += blockDim.x) {
        dst[i] = src[i];
    }
    __syncthreads();
}

} // namespace chanwise
} // namespace convolution
} // namespace rocm
} // namespace megdnn

// vim: syntax=cuda.doxygen
