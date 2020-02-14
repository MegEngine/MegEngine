/**
 * By downloading, copying, installing or using the software you agree to this license.
 * If you do not agree to this license, do not download, install,
 * copy or use the software.
 *
 *
 *                           License Agreement
 *                For Open Source Computer Vision Library
 *                        (3-clause BSD License)
 *
 * Copyright (C) 2000-2020, Intel Corporation, all rights reserved.
 * Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
 * Copyright (C) 2009-2016, NVIDIA Corporation, all rights reserved.
 * Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
 * Copyright (C) 2015-2016, OpenCV Foundation, all rights reserved.
 * Copyright (C) 2015-2016, Itseez Inc., all rights reserved.
 * Copyright (C) 2019-2020, Xperience AI, all rights reserved.
 * Third party copyrights are property of their respective owners.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 *   * Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *   * Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *
 *   * Neither the names of the copyright holders nor the names of the contributors
 *     may be used to endorse or promote products derived from this software
 *     without specific prior written permission.
 *
 * This software is provided by the copyright holders and contributors "as is" and
 * any express or implied warranties, including, but not limited to, the implied
 * warranties of merchantability and fitness for a particular purpose are disclaimed.
 * In no event shall copyright holders or contributors be liable for any direct,
 * indirect, incidental, special, exemplary, or consequential damages
 * (including, but not limited to, procurement of substitute goods or services;
 * loss of use, data, or profits; or business interruption) however caused
 * and on any theory of liability, whether in contract, strict liability,
 * or tort (including negligence or otherwise) arising in any way out of
 * the use of this software, even if advised of the possibility of such damage.
 *
 * ---------------------------------------------------------------------------
 * \file dnn/src/common/cv/helper.h
 *
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *
 * This file has been modified by Megvii ("Megvii Modifications").
 * All Megvii Modifications are Copyright (C) 2014-2019 Megvii Inc. All rights reserved.
 *
 * ---------------------------------------------------------------------------
 */

#pragma once

#include <cassert>
#include <climits>
#include <cmath>
#include <cstddef>

#include "./aligned_allocator.h"
#include "./common.h"
#include "src/common/utils.h"

#include "megdnn/basic_types.h"
#include "megdnn/opr_param_defs.h"

#if defined(__SSE2__)
#include <xmmintrin.h>
#endif

#define MegCVException(expr)                \
    do {                                    \
        megdnn_throw(megdnn_mangle(#expr)); \
    } while (0)

namespace megdnn {

namespace megcv {

template <typename T>
using AlignedVector = std::vector<T, ah::aligned_allocator<T, 16>>;

static inline size_t align_size(size_t sz, int n) {
    megdnn_assert((n & (n - 1)) == 0);
    return (sz + n - 1) & -n;
}

static inline int clip(int x, int a, int b) {
    return x >= a ? (x < b ? x : b - 1) : a;
}

template <typename _Tp>
static inline _Tp* align_ptr(_Tp* ptr, int n = (int)sizeof(_Tp)) {
    return (_Tp*)(((size_t)ptr + n - 1) & -n);
}

template <typename T>
inline T saturate(T x, T lower, T upper) {
    return (x < lower ? lower : (x >= upper ? upper - 1 : x));
}

// common functions
template <typename T>
T modf(T x, T* iptr) {
    T ival;
    T rval(std::modf(x, &ival));
    *iptr = ival;
    return rval;
}

template <typename T>
int round(T value) {
    T intpart, fractpart;
    fractpart = modf(value, &intpart);
    if ((fabs(fractpart) != 0.5) || ((((int)intpart) % 2) != 0))
        return (int)(value + (value >= 0 ? 0.5 : -0.5));
    else
        return (int)intpart;
}
template <typename DT, typename ST>
static inline DT saturate_cast(ST x) {
    return x;
}

template <>
inline unsigned char saturate_cast<unsigned char, int>(int x) {
    return (unsigned char)((unsigned)x <= UCHAR_MAX ? x
                                                    : x > 0 ? UCHAR_MAX : 0);
}

template <>
inline short saturate_cast<short, int>(int x) {
    return (short)((unsigned)(x - SHRT_MIN) <= (unsigned)USHRT_MAX
                           ? x
                           : x > 0 ? SHRT_MAX : SHRT_MIN);
}

template <typename ST>
static inline int cv_round(ST value);

template <>
inline int cv_round<float>(float value) {
#if defined(__SSE2__)
    __m128 t = _mm_set_ss(value);
    return _mm_cvtss_si32(t);
#elif defined(__GNUC__)
    return (int)lrintf(value);
#else
    /* it's ok if round does not comply with IEEE754 standard;
     the tests should allow +/-1 difference when the tested functions use round
   */
    return (int)(value + (value >= 0 ? 0.5f : -0.5f));
#endif
}

template <>
inline int cv_round<double>(double value) {
#if defined(__SSE2__)
    __m128d t = _mm_set_sd(value);
    return _mm_cvtsd_si32(t);
#elif defined(__GNUC__)
    return (int)lrint(value);
#else
    /* it's ok if round does not comply with IEEE754 standard;
     the tests should allow +/-1 difference when the tested functions use round
   */
    return (int)(value + (value >= 0 ? 0.5f : -0.5f));
#endif
}

template <>
inline int saturate_cast<int, float>(float x) {
    return cv_round(x);
}

template <>
inline short saturate_cast<short, float>(float x) {
    return saturate_cast<short, int>(saturate_cast<int, float>(x));
}

template <>
inline int saturate_cast<int, double>(double x) {
    return cv_round(x);
}

template <typename ST, typename DT, int bits>
struct FixedPtCast {
    typedef ST type1;
    typedef DT rtype;
    enum { SHIFT = bits, DELTA = 1 << (bits - 1) };

    DT operator()(ST val) const {
        return saturate_cast<DT>((val + DELTA) >> SHIFT);
    }
};

template <typename ST, typename DT>
struct FixedPtCastEx {
    typedef ST type1;
    typedef DT rtype;

    FixedPtCastEx() : SHIFT(0), DELTA(0) {}
    FixedPtCastEx(int bits) : SHIFT(bits), DELTA(bits ? 1 << (bits - 1) : 0) {}
    DT operator()(ST val) const { return saturate_cast<DT>(val + DELTA); }
    int SHIFT, DELTA;
};

template <>
struct FixedPtCastEx<int, uchar> {
    typedef int type1;
    typedef uchar rtype;

    FixedPtCastEx() : SHIFT(0), DELTA(0) {}
    FixedPtCastEx(int bits) : SHIFT(bits), DELTA(bits ? 1 << (bits - 1) : 0) {}
    uchar operator()(int val) const {
        return saturate_cast<uchar>((val + DELTA) >> SHIFT);
    }
    int SHIFT, DELTA;
};

template <typename ST, typename DT>
struct Cast {
    typedef ST type1;
    typedef DT rtype;

    DT operator()(ST val) const { return saturate_cast<DT>(val); }
};

template <param::WarpPerspective::BorderMode bmode>
static inline int border_interpolate(int p, int len) {
    using BorderMode = param::WarpPerspective::BorderMode;
    if ((unsigned)p < (unsigned)len)
        ;
    else if (bmode == BorderMode::BORDER_REPLICATE)
        p = p < 0 ? 0 : len - 1;
    else if (bmode == BorderMode::BORDER_REFLECT ||
             bmode == BorderMode::BORDER_REFLECT_101) {
        int delta = (bmode == BorderMode::BORDER_REFLECT_101);
        if (len == 1)
            return 0;
        do {
            if (p < 0)
                p = -p - 1 + delta;
            else
                p = len - 1 - (p - len) - delta;
        } while ((unsigned)p >= (unsigned)len);
    } else if (bmode == BorderMode::BORDER_WRAP) {
        if (p < 0)
            p -= ((p - len + 1) / len) * len;
        while (p >= len) {
            p -= len;
        }
    } else if (bmode == BorderMode::BORDER_CONSTANT ||
               bmode == BorderMode::BORDER_TRANSPARENT)
        p = -1;
    else
        megdnn_throw("Unknown/unsupported border type");
    return p;
}

namespace gaussian_blur {

using BorderMode = param::GaussianBlur::BorderMode;

#include "./bordermode-inl.h"

}  // namespace gaussian_blur

}  // namespace megcv
}  // namespace megdnn

// vim: syntax=cpp.doxygen
