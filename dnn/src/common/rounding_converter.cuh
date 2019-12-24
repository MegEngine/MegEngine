/**
 * \file dnn/src/common/rounding_converter.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once
#include "megdnn/dtype.h"

#if MEGDNN_CC_HOST && !defined(__host__)
#define MEGDNN_HOST_DEVICE_SELF_DEFINE
#define __host__
#define __device__
#if __GNUC__ || __has_attribute(always_inline)
#define __forceinline__ inline __attribute__((always_inline))
#else
#define __forceinline__ inline
#endif
#endif

namespace megdnn {
namespace rounding {

template <typename T>
struct RoundingConverter;

template <>
struct RoundingConverter<float> {
    __host__ __device__ __forceinline__ float operator()(float x) const {
        return x;
    }
};

#ifndef MEGDNN_DISABLE_FLOAT16

template <>
struct RoundingConverter<half_float::half> {
    __host__ __device__ __forceinline__ half_float::half operator()(
            float x) const {
        return static_cast<half_float::half>(x);
    }
};

template <>
struct RoundingConverter<half_bfloat16::bfloat16> {
    __host__ __device__ __forceinline__ half_bfloat16::bfloat16 operator()(
            float x) const {
        return static_cast<half_bfloat16::bfloat16>(x);
    }
};

#endif  // #ifdef MEGDNN_DISABLE_FLOAT16

template <>
struct RoundingConverter<int8_t> {
    __host__ __device__ __forceinline__ int8_t operator()(float x) const {
#if MEGDNN_CC_HOST
        using std::round;
#endif
        return static_cast<int8_t>(round(x));
    }
};

template <>
struct RoundingConverter<uint8_t> {
    __host__ __device__ __forceinline__ uint8_t operator()(float x) const {
#if MEGDNN_CC_HOST
        using std::round;
#endif
        return static_cast<uint8_t>(round(x));
    }
};

}  // namespace rounding
}  // namespace megdnn

/* vim: set ft=cpp: */
