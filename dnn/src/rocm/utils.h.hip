/**
 * \file src/rocm/utils.h.hip
 *
 * This file is part of MegDNN, a deep neural network run-time library
 * developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2019 Megvii Inc. All rights reserved.
 */
#pragma once

#include "src/common/utils.cuh"

#include <stdint.h>

#include "src/rocm/miopen_with_check.h"
#include "src/rocm/rocblas_header.h"
#define hip_check(_x)                                       \
    do {                                                    \
        hipError_t _err = (_x);                             \
        if (_err != hipSuccess) {                           \
            ::megdnn::rocm::__throw_hip_error__(_err, #_x); \
        }                                                   \
    } while (0)

#define rocblas_check(_x)                                       \
    do {                                                        \
        rocblas_status _err = (_x);                             \
        if (_err != rocblas_status_success) {                   \
            ::megdnn::rocm::__throw_rocblas_error__(_err, #_x); \
        }                                                       \
    } while (0)

#define miopen_check(_x)                                       \
    do {                                                       \
        miopenStatus_t _err = (_x);                            \
        if (_err != miopenStatusSuccess) {                     \
            ::megdnn::rocm::__throw_miopen_error__(_err, #_x); \
        }                                                      \
    } while (0)

#define after_kernel_launch()         \
    do {                              \
        hip_check(hipGetLastError()); \
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

namespace megdnn {
namespace rocm {

//! Error handling funcions
MEGDNN_NORETURN void __throw_hip_error__(hipError_t err, const char* msg);
MEGDNN_NORETURN void __throw_miopen_error__(miopenStatus_t err,
                                            const char* msg);
MEGDNN_NORETURN void __throw_rocblas_error__(rocblas_status err,
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

#ifdef __HIPCC__
template <typename T>
inline __device__ void fill_shared_mem(T* shared, uint32_t n, const T& val) {
    uint32_t stride = hipBlockDim_x * hipBlockDim_y * hipBlockDim_z;
    uint32_t i =
            (hipThreadIdx_z * hipBlockDim_y + hipThreadIdx_y) * hipBlockDim_x +
            hipThreadIdx_x;
    for (; i < n; i += stride)
        shared[i] = val;
}
#endif

}  // namespace rocm
}  // namespace megdnn

// vim: syntax=cpp.doxygen
