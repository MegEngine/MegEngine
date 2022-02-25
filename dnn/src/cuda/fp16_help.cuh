#pragma once

#include <cuda_runtime_api.h>
#include "cuda.h"
#include "cuda_fp16.h"

namespace megdnn {
namespace cuda {

__device__ __forceinline__ float fma(const float a, const float b, const float c) {
    return a * b + c;
}

__device__ __forceinline__ float2 fma2(const float2 a, const float2 b, const float2 c) {
    return {a.x * b.x + c.x, a.y * b.y + c.y};
}

#if CUDA_VERSION >= 9000

__device__ __forceinline__ __half fma(const __half a, const __half b, const __half c) {
#if __CUDA_ARCH__ >= 530
    return __hfma(a, b, c);
#else
    return __float2half(__half2float(a) * __half2float(b) + __half2float(c));
#endif
}

__device__ __forceinline__ __half2
fma2(const __half2 a, const __half2 b, const __half2 c) {
#if __CUDA_ARCH__ >= 530
    return __hfma2(a, b, c);
#else
    return {__float2half(__half2float(a.x) * __half2float(b.x) + __half2float(c.x)),
            __float2half(__half2float(a.y) * __half2float(b.y) + __half2float(c.y))};
#endif
}

__device__ __forceinline__ __half2 hadd2(const __half2 a, const __half2 b) {
#if __CUDA_ARCH__ >= 530
    return __hadd2(a, b);
#else
    return {__float2half(__half2float(a.x) + __half2float(b.x)),
            __float2half(__half2float(a.y) + __half2float(b.y))};
#endif
}

__device__ __forceinline__ float2
fma2(const __half2 a, const __half2 b, const float2 c) {
    return {__half2float(a.x) * __half2float(b.x) + c.x,
            __half2float(a.y) * __half2float(b.y) + c.y};
}

#endif  // CUDA_VERSION >= 9000

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
