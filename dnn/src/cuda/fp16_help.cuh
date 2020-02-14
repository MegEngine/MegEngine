/**
 * \file dnn/src/cuda/fp16_help.cuh
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
#include "cuda.h"
#include "cuda_fp16.h"

namespace megdnn {
namespace cuda {

__device__ __forceinline__ float fma(const float a, const float b,
                                     const float c) {
    return a * b + c;
}

__device__ __forceinline__ float2 fma2(const float2 a, const float2 b,
                                       const float2 c) {
    return {a.x * b.x + c.x, a.y * b.y + c.y};
}

#if CUDA_VERSION >= 9000

__device__ __forceinline__ __half fma(const __half a, const __half b,
                                      const __half c) {
#if __CUDA_ARCH__ >= 530
    return __hfma(a, b, c);
#else
    return __float2half(__half2float(a) * __half2float(b) + __half2float(c));
#endif
}

__device__ __forceinline__ __half2 fma2(const __half2 a, const __half2 b,
                                        const __half2 c) {
#if __CUDA_ARCH__ >= 530
    return __hfma2(a, b, c);
#else
    return {__float2half(__half2float(a.x) * __half2float(b.x) +
                         __half2float(c.x)),
            __float2half(__half2float(a.y) * __half2float(b.y) +
                         __half2float(c.y))};
#endif
}

#endif  // CUDA_VERSION >= 9000

} // namespace cuda
} // namespace megdnn

// vim: syntax=cpp.doxygen
