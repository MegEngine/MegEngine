/**
 * \file dnn/src/cuda/utils.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include "include/megdnn/dtype.h"
#include "src/common/utils.cuh"

#include <stdint.h>

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include <cusolverDn.h>
#include "cuda.h"
#include "src/cuda/cudnn_with_check.h"
#include "cutlass/cutlass.h"

#define cuda_check(_x)                                       \
    do {                                                     \
        cudaError_t _err = (_x);                             \
        if (_err != cudaSuccess) {                           \
            ::megdnn::cuda::__throw_cuda_error__(_err, #_x); \
        }                                                    \
    } while (0)

#define cublas_check(_x)                                       \
    do {                                                       \
        cublasStatus_t _err = (_x);                            \
        if (_err != CUBLAS_STATUS_SUCCESS) {                   \
            ::megdnn::cuda::__throw_cublas_error__(_err, #_x); \
        }                                                      \
    } while (0)

#define cudnn_check(_x)                                       \
    do {                                                      \
        cudnnStatus_t _err = (_x);                            \
        if (_err != CUDNN_STATUS_SUCCESS) {                   \
            ::megdnn::cuda::__throw_cudnn_error__(_err, #_x); \
        }                                                     \
    } while (0)

#define cusolver_check(_x)                                       \
    do {                                                         \
        cusolverStatus_t _err = (_x);                            \
        if (_err != CUSOLVER_STATUS_SUCCESS) {                   \
            ::megdnn::cuda::__throw_cusolver_error__(_err, #_x); \
        }                                                        \
    } while (0)

#define cucheck(_x)                                                 \
    do {                                                            \
        CUresult _err = (_x);                                       \
        if (_err != CUDA_SUCCESS) {                                 \
            ::megdnn::cuda::__throw_cuda_driver_error__(_err, #_x); \
        }                                                           \
    } while (0)

#define cutlass_check(_x)                                       \
    do {                                                        \
        cutlass::Status _err = (_x);                            \
        if (_err != cutlass::Status::kSuccess) {                \
            ::megdnn::cuda::__throw_cutlass_error__(_err, #_x); \
        }                                                       \
    } while (0)

#define after_kernel_launch()           \
    do {                                \
        cuda_check(cudaGetLastError()); \
    } while (0)

#if MEGDNN_THREADS_512
#define NR_THREADS 512
#define NR_THREADS_X 32
#define NR_THREADS_Y 16
#else
#define NR_THREADS 1024
#define NR_THREADS_X 32
#define NR_THREADS_Y 32
#endif

#define DIVUP(x, y) (((x) + (y)-1) / (y))

#define KERN_FOR(i, n)                                              \
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
         i += blockDim.x * gridDim.x)

namespace megdnn {
namespace cuda {

//! Error handling funcions
MEGDNN_NORETURN void __throw_cuda_error__(cudaError_t err, const char* msg);
MEGDNN_NORETURN void __throw_cudnn_error__(cudnnStatus_t err, const char* msg);
MEGDNN_NORETURN void __throw_cublas_error__(cublasStatus_t err,
                                            const char* msg);
MEGDNN_NORETURN void __throw_cusolver_error__(cusolverStatus_t err,
                                              const char* msg);
MEGDNN_NORETURN void __throw_cuda_driver_error__(CUresult err, const char* msg);
MEGDNN_NORETURN void __throw_cutlass_error__(cutlass::Status status,
                                             const char* msg);
MEGDNN_NORETURN void report_error(const char* msg);

template <typename T, size_t N>
struct array_wrapper {
    T data[N];
};

/*!
 * \brief convert size to uint32_t and check for not overflow
 *
 * throw exception with human readable message if size not in the interval (0,
 * Uint32Fastdiv::MAX_DIVIDEND)
 */
uint32_t safe_size_in_kern(size_t size);

#ifdef __CUDACC__
template <typename T>
inline __device__ void fill_shared_mem(T* shared, uint32_t n, const T& val) {
    uint32_t stride = blockDim.x * blockDim.y * blockDim.z;
    uint32_t i =
            (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
    for (; i < n; i += stride)
        shared[i] = val;
}
#endif

// ==========================DTypeParam wrapper=================================
// Division is inefficient in cuda, so we replace div scale with mul 1/scale,
// and we need a wrapper of DTypeParam to hold the reciprocal of scale.

template <typename Type>
struct CudaDTypeParamImpl;

template <typename DType>
using CudaDTypeParam = CudaDTypeParamImpl<typename DTypeTrait<DType>::ctype>;

template <>
struct CudaDTypeParamImpl<dt_quint8> : DTypeParamImpl<dt_quint8> {
    float inv_scale;
    CudaDTypeParamImpl() = default;
    CudaDTypeParamImpl(float scale, uint8_t zero_point)
            : DTypeParamImpl<dt_quint8>(scale, zero_point),
              inv_scale(1.0f / scale) {}
    CudaDTypeParamImpl(const DTypeParamImpl<dt_quint8>& param)
            : CudaDTypeParamImpl(param.scale, param.zero_point) {}

    __device__ dt_quint8 quantize(float in) const {
        float v = in * inv_scale;
        v = roundf(v);
        v = v + zero_point;
        v = fmin(fmax(0.f, v), 255.f);
        return static_cast<dt_quint8>(v);
    }
};

template <>
struct CudaDTypeParamImpl<dt_qint8> : DTypeParamImpl<dt_qint8> {
    float inv_scale;
    CudaDTypeParamImpl() = default;
    CudaDTypeParamImpl(float scale)
            : DTypeParamImpl<dt_qint8>(scale), inv_scale(1.0f / scale) {}
    CudaDTypeParamImpl(const DTypeParamImpl<dt_qint8>& param)
            : CudaDTypeParamImpl(param.scale) {}

    __device__ dt_qint8 quantize(float in) const {
        float v = in * inv_scale;
        v = roundf(v);
        v = fmin(fmax(-128.f, v), 127.f);
        return static_cast<dt_qint8>(v);
    }
};

template <>
struct CudaDTypeParamImpl<dt_qint32> : DTypeParamImpl<dt_qint32> {
    float inv_scale;
    CudaDTypeParamImpl() = default;
    CudaDTypeParamImpl(float scale)
            : DTypeParamImpl<dt_qint32>(scale), inv_scale(1.0f / scale) {}
    CudaDTypeParamImpl(const DTypeParamImpl<dt_qint32>& param)
            : CudaDTypeParamImpl(param.scale) {}

    __device__ dt_qint32 quantize(float in) const {
        float v = in * inv_scale;
        v = roundf(v);
        /*! \note: the maximal signed integer that can be correctly represented
         * as a single precision floating point number is 2147483520
         */
        v = fmin(fmax(-2147483648.f, v), 2147483520.f);
        return static_cast<dt_qint32>(v);
    }
};

template <>
struct CudaDTypeParamImpl<dt_quint4> : DTypeParamImpl<dt_quint4> {
    float inv_scale;
    CudaDTypeParamImpl() = default;
    CudaDTypeParamImpl(float scale, uint8_t zero_point)
            : DTypeParamImpl<dt_quint4>(scale, zero_point),
              inv_scale(1.0f / scale) {}
    CudaDTypeParamImpl(const DTypeParamImpl<dt_quint4>& param)
            : CudaDTypeParamImpl(param.scale, param.zero_point) {}

    __device__ uint8_t quantize(float in) const {
        float v = in * inv_scale;
        v = roundf(v);
        v = v + zero_point;
        v = fmin(fmax(0.f, v), 15.f);
        return static_cast<uint8_t>(v);
    }
};

#if MEGDNN_CC_CUDA
template <typename T>
static inline MEGDNN_DEVICE void atomic_add(T* address, T val);

template <>
MEGDNN_DEVICE void atomic_add(dt_float32* address, dt_float32 val) {
    ::atomicAdd(reinterpret_cast<float*>(address), static_cast<float>(val));
}

// overload atomicAdd for half precision
// Taken from:
// https://github.com/torch/cutorch/blob/master/lib/THC/THCAtomic.cuh
template <>
MEGDNN_DEVICE void atomic_add(dt_float16* address, dt_float16 val) {
#if (__CUDA_ARCH__ < 700 || __CUDACC_VER_MAJOR__ <= 9)
    unsigned int* address_as_ui = reinterpret_cast<unsigned int*>(
            reinterpret_cast<char*>(address) -
            (reinterpret_cast<size_t>(address) & 2));
    unsigned int old = *address_as_ui;
    unsigned int assumed;

    do {
        assumed = old;
        unsigned short data = reinterpret_cast<size_t>(address) & 2
                                      ? (old >> 16)
                                      : (old & 0xffff);
        dt_float16 hsum = *reinterpret_cast<dt_float16*>(&data);
        hsum += val;
        data = *reinterpret_cast<unsigned short*>(&hsum);
        old = reinterpret_cast<size_t>(address) & 2
                      ? (old & 0xffff) | (data << 16)
                      : (old & 0xffff0000) | data;
        old = ::atomicCAS(address_as_ui, assumed, old);
    } while (assumed != old);
#else
    ::atomicAdd(reinterpret_cast<__half*>(address), static_cast<__half>(val));
#endif
}

template <>
MEGDNN_DEVICE void atomic_add(dt_bfloat16* address, dt_bfloat16 val) {
    unsigned int* address_as_ui = reinterpret_cast<unsigned int*>(
            reinterpret_cast<char*>(address) -
            (reinterpret_cast<size_t>(address) & 2));
    unsigned int old = *address_as_ui;
    unsigned int assumed;

    do {
        assumed = old;
        unsigned short data = reinterpret_cast<size_t>(address) & 2
                                      ? (old >> 16)
                                      : (old & 0xffff);
        dt_bfloat16 hsum = *reinterpret_cast<dt_bfloat16*>(&data);
        hsum += val;
        data = *reinterpret_cast<unsigned short*>(&hsum);
        old = reinterpret_cast<size_t>(address) & 2
                      ? (old & 0xffff) | (data << 16)
                      : (old & 0xffff0000) | data;
        old = ::atomicCAS(address_as_ui, assumed, old);
    } while (assumed != old);
}

static inline MEGDNN_DEVICE void dot_prod(int a, int b, int c, int& d) {
#if __CUDA_ARCH__ >= 610
    // clang-format off
    asm volatile("dp4a.s32.s32 %0, %1, %2, %3;"
            : "=r"(d)
            : "r"(a), "r"(b), "r"(c));
    // clang-format on
#else
    d = 0;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        int8_t val_a = (a & 0xff), val_b = (b & 0xff);
        d += static_cast<int>(val_a) * static_cast<int>(val_b);
        a = (a >> 8), b = (b >> 8);
    }
    d += c;
#endif
}

// the following code is taken from cutlass:
// https://github.com/NVIDIA/cutlass/blob/master/cutlass/gemm/igemm_epilogue.h
// Note: using .rni integer rounding modifier, i.e. rounding to nearest integer,
// choosing even integer if source is equidistant between two integers. The
// reason not use roundf is that roundf() maps to an 8-instruction sequence on
// the device, which causes significant performance drop in some cases. For
// details, refer to
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html
MEGDNN_DEVICE __forceinline__ static int transform_float4_to_int8x4(
        float4 val) {
    int ix, iy, iz, iw;
    asm volatile("cvt.rni.s8.f32 %0, %1;" : "=r"(ix) : "f"(val.x));
    asm volatile("cvt.rni.s8.f32 %0, %1;" : "=r"(iy) : "f"(val.y));
    asm volatile("cvt.rni.s8.f32 %0, %1;" : "=r"(iz) : "f"(val.z));
    asm volatile("cvt.rni.s8.f32 %0, %1;" : "=r"(iw) : "f"(val.w));

    asm volatile("prmt.b32 %0, %0, %1, 0x1140;" : "+r"(ix) : "r"(iy));
    asm volatile("prmt.b32 %0, %0, %1, 0x1140;" : "+r"(iz) : "r"(iw));
    asm volatile("prmt.b32 %0, %0, %1, 0x5410;" : "+r"(ix) : "r"(iz));
    return ix;
}

MEGDNN_DEVICE __forceinline__ static float4 transform_int8x4_to_float4(
        int val) {
    int ix, iy, iz, iw = val;

    // Extract the 4 bytes
    asm volatile("prmt.b32 %0, %1, 0x0, 0x4440;" : "=r"(ix) : "r"(iw));
    asm volatile("prmt.b32 %0, %1, 0x0, 0x4441;" : "=r"(iy) : "r"(iw));
    asm volatile("prmt.b32 %0, %1, 0x0, 0x4442;" : "=r"(iz) : "r"(iw));
    asm volatile("prmt.b32 %0, %1, 0x0, 0x4443;" : "=r"(iw) : "r"(iw));
    // the floats
    float fx, fy, fz, fw;

    // convert to floats (make sure we generate I2F.F32.S8)
    asm volatile("cvt.rn.f32.s8 %0, %1;" : "=f"(fx) : "r"(ix));
    asm volatile("cvt.rn.f32.s8 %0, %1;" : "=f"(fy) : "r"(iy));
    asm volatile("cvt.rn.f32.s8 %0, %1;" : "=f"(fz) : "r"(iz));
    asm volatile("cvt.rn.f32.s8 %0, %1;" : "=f"(fw) : "r"(iw));

    return ::make_float4(fx, fy, fz, fw);
}

MEGDNN_DEVICE __forceinline__ static float4 operator*(float scalar,
                                                      float4 val) {
    return make_float4(scalar * val.x, scalar * val.y, scalar * val.z,
                       scalar * val.w);
}

MEGDNN_DEVICE __forceinline__ static float4 operator+(float4 lval,
                                                      float4 rval) {
    return make_float4(lval.x + rval.x, lval.y + rval.y, lval.z + rval.z,
                       lval.w + rval.w);
}
#endif
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
