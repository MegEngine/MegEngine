/**
 * \file dnn/src/cuda/cv/kernel_common.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include "src/common/cv/enums.h"

#include "megdnn/basic_types.h"

#include <cassert>
#include <cfloat>
#include <climits>
#include <cstdio>
#include <limits>

typedef unsigned char uchar;
typedef unsigned char byte;

namespace megdnn {
namespace megcv {

// FIXME the implement is not the same as in the cv/help.h
template <typename T>
__host__ __device__ T saturate(const T x, const T lower, const T upper) {
    if (x < lower)
        return lower;
    if (x > upper)
        return upper;
    return x;
}

__device__ inline int saturate_cast(double val) {
    return round(val);
}

__device__ inline short saturate_cast_short(double x) {
    return x < -32768 ? -32768 : (x > 32767 ? 32767 : round(x));
}

__device__ inline void interpolate_linear_coefs(float x, float* coeffs) {
    coeffs[0] = 1 - x;
    coeffs[1] = x;
}

__host__ __device__ inline void interpolate_cubic_coefs(float x,
                                                        float* coeffs) {
    const float A = -0.75f;
    coeffs[0] = ((A * (x + 1) - 5 * A) * (x + 1) + 8 * A) * (x + 1) - 4 * A;
    coeffs[1] = ((A + 2) * x - (A + 3)) * x * x + 1;
    coeffs[2] = ((A + 2) * (1 - x) - (A + 3)) * (1 - x) * (1 - x) + 1;
    coeffs[3] = 1.f - coeffs[0] - coeffs[1] - coeffs[2];
}

__device__ inline void interpolate_lanczos4_coefs(float x, float* coeffs) {
    const float s45 = 0.70710678118654752440084436210485;
    const float cs[][2] = {{1, 0},  {-s45, -s45}, {0, 1},  {s45, -s45},
                           {-1, 0}, {s45, s45},   {0, -1}, {-s45, s45}};
    const float MEGCV_PI = 3.1415926536;

    if (x < FLT_EPSILON) {
        for (int i = 0; i < 8; i++)
            coeffs[i] = 0;
        coeffs[3] = 1;
        return;
    }

    float sum = 0;
    float y0 = -(x + 3) * MEGCV_PI * 0.25, s0 = sin(y0), c0 = cos(y0);
    for (int i = 0; i < 8; i++) {
        float y = -(x + 3 - i) * MEGCV_PI * 0.25;
        coeffs[i] = (float)((cs[i][0] * s0 + cs[i][1] * c0) / (y * y));
        sum += coeffs[i];
    }

    sum = 1.f / sum;
    for (int i = 0; i < 8; i++)
        coeffs[i] *= sum;
}

template <BorderMode bmode>
class BModeTrait {
public:
    static const BorderMode bmode1 = bmode;
};
template <>
class BModeTrait<BORDER_TRANSPARENT> {
public:
    static const BorderMode bmode1 = BORDER_REFLECT_101;
};

template <typename T>
class TypeTrait {
public:
    typedef T WorkType;
    MEGDNN_DEVICE static T min() { return std::numeric_limits<T>::min(); }
    MEGDNN_DEVICE static T max() { return std::numeric_limits<T>::max(); }
    static const bool need_saturate;
};
template <>
class TypeTrait<uchar> {
public:
    typedef int WorkType;
    MEGDNN_DEVICE static uchar min() { return 0; }
    MEGDNN_DEVICE static uchar max() { return 255; }
    static const bool need_saturate = true;
};
template <>
class TypeTrait<float> {
public:
    typedef float WorkType;
    MEGDNN_DEVICE static float min() { return 0; }
    MEGDNN_DEVICE static float max() { return 1; }
    static const bool need_saturate = false;
};

template <BorderMode bmode>
__device__ inline int border_interpolate(int p, int len);

template <>
__device__ inline int border_interpolate<BORDER_REPLICATE>(int p, int len) {
    if ((unsigned)p >= (unsigned)len) {
        p = p < 0 ? 0 : len - 1;
    }
    return p;
}

template <>
__device__ inline int border_interpolate<BORDER_REFLECT>(int p, int len) {
    if (len == 1)
        return 0;

    do {
        if (p < 0)
            p = -p - 1;
        else
            p = len - 1 - (p - len);
    } while ((unsigned)p >= (unsigned)len);
    return p;
}

template <>
__device__ inline int border_interpolate<BORDER_REFLECT_101>(int p, int len) {
    if (len == 1)
        return 0;

    do {
        if (p < 0)
            p = -p;
        else
            p = len - 1 - (p - len) - 1;
    } while ((unsigned)p >= (unsigned)len);
    return p;
}

template <>
__device__ inline int border_interpolate<BORDER_WRAP>(int p, int len) {
    if ((unsigned)p >= (unsigned)len) {
        if (p < 0)
            p -= ((p - len + 1) / len) * len;

        p %= len;
    }
    return p;
}

template <>
__device__ inline int border_interpolate<BORDER_TRANSPARENT>(int p, int len) {
    if ((unsigned)p >= (unsigned)len) {
        p = -1;
    }
    return p;
}

template <>
__device__ inline int border_interpolate<BORDER_CONSTANT>(int p, int len) {
    // if ((unsigned)p >= (unsigned)len) {
    //    p = -1;
    //}
    return (unsigned)p >= (unsigned)len ? -1 : p;
}

template <InterpolationMode imode>
__device__ void interpolate_coefs(float x, float* coeffs);
template <>
__device__ inline void interpolate_coefs<INTER_NEAREST>(float x,
                                                        float* coeffs) {}
template <>
__device__ inline void interpolate_coefs<INTER_LINEAR>(float x, float* coeffs) {
    interpolate_linear_coefs(x, coeffs);
}
template <>
__device__ inline void interpolate_coefs<INTER_CUBIC>(float x, float* coeffs) {
    interpolate_cubic_coefs(x, coeffs);
}
template <>
__device__ inline void interpolate_coefs<INTER_LANCZOS4>(float x,
                                                         float* coeffs) {
    interpolate_lanczos4_coefs(x, coeffs);
}

template <InterpolationMode imode>
class IModeTrait {
public:
    static const int ksize;
};
template <>
class IModeTrait<INTER_NEAREST> {
public:
    static const int ksize = 1;
};
template <>
class IModeTrait<INTER_LINEAR> {
public:
    static const int ksize = 2;
};

template <>
class IModeTrait<INTER_CUBIC> {
public:
    static const int ksize = 4;
};
template <>
class IModeTrait<INTER_LANCZOS4> {
public:
    static const int ksize = 8;
};

}  // namespace megcv
}  // namespace megdnn

// vim: syntax=cpp.doxygen
