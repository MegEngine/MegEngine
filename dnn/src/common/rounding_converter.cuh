/**
 * \file dnn/src/common/rounding_converter.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once
#include "megdnn/dtype.h"

namespace megdnn {
namespace rounding {

template <typename T>
struct RoundingConverter;

template <>
struct RoundingConverter<float> {
    MEGDNN_HOST MEGDNN_DEVICE MEGDNN_FORCE_INLINE float operator()(
            float x) const {
        return x;
    }
};

#ifndef MEGDNN_DISABLE_FLOAT16

template <>
struct RoundingConverter<half_float::half> {
    MEGDNN_HOST MEGDNN_DEVICE MEGDNN_FORCE_INLINE half_float::half operator()(
            float x) const {
        return static_cast<half_float::half>(x);
    }
};

template <>
struct RoundingConverter<half_bfloat16::bfloat16> {
    MEGDNN_HOST MEGDNN_DEVICE MEGDNN_FORCE_INLINE half_bfloat16::bfloat16
    operator()(float x) const {
        return static_cast<half_bfloat16::bfloat16>(x);
    }
};

#endif  // #ifdef MEGDNN_DISABLE_FLOAT16

template <>
struct RoundingConverter<int8_t> {
    MEGDNN_HOST MEGDNN_DEVICE MEGDNN_FORCE_INLINE int8_t
    operator()(float x) const {
#if MEGDNN_CC_HOST
        using std::round;
#endif
        return static_cast<int8_t>(round(x));
    }
};

template <>
struct RoundingConverter<uint8_t> {
    MEGDNN_HOST MEGDNN_DEVICE MEGDNN_FORCE_INLINE uint8_t
    operator()(float x) const {
#if MEGDNN_CC_HOST
        using std::max;
        using std::min;
        using std::round;
#endif
        x = min(255.0f, max(0.0f, x));  //! FIXME!!! check other places
        return static_cast<uint8_t>(round(x));
    }
};

template <>
struct RoundingConverter<dt_qint4> {
    MEGDNN_HOST MEGDNN_DEVICE MEGDNN_FORCE_INLINE dt_qint4
    operator()(float x) const {
#if MEGDNN_CC_HOST
        using std::round;
#endif
        return static_cast<dt_qint4>(round(x));
    }
};

template <>
struct RoundingConverter<dt_quint4> {
    MEGDNN_HOST MEGDNN_DEVICE MEGDNN_FORCE_INLINE dt_quint4
    operator()(float x) const {
#if MEGDNN_CC_HOST
        using std::round;
#endif
        return static_cast<dt_quint4>(round(x));
    }
};

}  // namespace rounding
}  // namespace megdnn

/* vim: set ft=cpp: */
