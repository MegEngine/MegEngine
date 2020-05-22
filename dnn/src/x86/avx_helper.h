/**
 * \file dnn/src/x86/avx_helper.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include "megdnn/arch.h"

#include <immintrin.h>
#include <avxintrin.h>
#include <avx2intrin.h>
#include <fmaintrin.h>

#if !defined (__clang__)
#pragma GCC target ("avx")
#endif

namespace megdnn {
namespace x86 {

MEGDNN_ATTRIBUTE_TARGET("avx")
static inline __m256 _mm256_loadu2_m128_emulate(
        const float *hiaddr, const float *loaddr) {
    return _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_loadu_ps(loaddr)),
            _mm_loadu_ps(hiaddr), 1);
}

template <typename ctype, size_t len>
struct Vector;

template <>
struct Vector<float, 8> {
    __m256 value;
    Vector() {}
    Vector(const float v) MEGDNN_ATTRIBUTE_TARGET("avx") {
        value = _mm256_set1_ps(v);
    }
    Vector(const Vector& lr) MEGDNN_ATTRIBUTE_TARGET("avx") {
        value = lr.value;
    }
    Vector(const Vector&& lr) MEGDNN_ATTRIBUTE_TARGET("avx") {
        value = std::move(lr.value);
    }
    Vector(const __m256& v) MEGDNN_ATTRIBUTE_TARGET("avx") { value = v; }
    static Vector load(const float* addr) MEGDNN_ATTRIBUTE_TARGET("avx") {
        Vector v;
        v.value = _mm256_loadu_ps(addr);
        return v;
    }
    static void save(float* addr, const Vector& v)
            MEGDNN_ATTRIBUTE_TARGET("avx") {
        _mm256_storeu_ps(addr, v.value);
    }
    void save(float* addr) MEGDNN_ATTRIBUTE_TARGET("avx") {
        save(addr, *this);
    }
    Vector operator+(const Vector& lr) MEGDNN_ATTRIBUTE_TARGET("avx") {
        Vector dst;
        dst.value = _mm256_add_ps(value, lr.value);
        return dst;
    }
    Vector& operator+=(const Vector& lr) MEGDNN_ATTRIBUTE_TARGET("avx") {
        value = _mm256_add_ps(value, lr.value);
        return *this;
    }
    Vector operator-(const Vector& lr) MEGDNN_ATTRIBUTE_TARGET("avx") {
        Vector dst;
        dst.value = _mm256_sub_ps(value, lr.value);
        return dst;
    }
    Vector& operator-=(const Vector& lr) MEGDNN_ATTRIBUTE_TARGET("avx") {
        value = _mm256_sub_ps(value, lr.value);
        return *this;
    }
    Vector operator*(float lr)MEGDNN_ATTRIBUTE_TARGET("avx") {
        Vector dst;
        dst.value = _mm256_mul_ps(value, _mm256_set1_ps(lr));
        return dst;
    }
    Vector operator*(const Vector& lr)MEGDNN_ATTRIBUTE_TARGET("avx") {
        Vector dst;
        dst.value = _mm256_mul_ps(value, lr.value);
        return dst;
    }
    Vector& operator*=(const Vector& lr) MEGDNN_ATTRIBUTE_TARGET("avx") {
        value = _mm256_mul_ps(value, lr.value);
        return *this;
    }
    Vector& operator=(const Vector& lr) MEGDNN_ATTRIBUTE_TARGET("avx") {
        value = lr.value;
        return *this;
    }
    Vector& operator=(const Vector&& lr) MEGDNN_ATTRIBUTE_TARGET("avx") {
        value = std::move(lr.value);
        return *this;
    }
    Vector operator-() MEGDNN_ATTRIBUTE_TARGET("avx") {
        Vector dst;
        dst.value = -value;
        return dst;
    }
};

#if !defined (__clang__)
#pragma GCC reset_options
#endif

} // namespace x86
} // namespace megdnn

// vim: syntax=cpp.doxygen
