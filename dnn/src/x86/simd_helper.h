/**
 * \file dnn/src/x86/simd_helper.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "src/x86/utils.h"
#include "megdnn/arch.h"

#include <immintrin.h>
#include <xmmintrin.h>
#include <avxintrin.h>
#include <fmaintrin.h>
#include <cmath>
#include <algorithm>

namespace megdnn {
namespace x86 {

template <SIMDType feature>
struct simd_traits {};

template <>
struct simd_traits<SIMDType::SSE> {
    using type = __m128;
    static MEGDNN_CONSTEXPR size_t width = 4;
    static type setzero() MEGDNN_ATTRIBUTE_TARGET("sse")
    {
        return _mm_setzero_ps();
    }
    static type set1(float a) MEGDNN_ATTRIBUTE_TARGET("sse")
    {
        return _mm_set1_ps(a);
    }
    static type loadu(const float *mem_addr) MEGDNN_ATTRIBUTE_TARGET("sse")
    {
        return _mm_loadu_ps(mem_addr);
    }
    static void storeu(float *mem_addr, type a) MEGDNN_ATTRIBUTE_TARGET("sse")
    {
        _mm_storeu_ps(mem_addr, a);
    }
    static type add(type a, type b) MEGDNN_ATTRIBUTE_TARGET("sse")
    {
        return _mm_add_ps(a, b);
    }
    static type sub(type a, type b) MEGDNN_ATTRIBUTE_TARGET("sse")
    {
        return _mm_sub_ps(a, b);
    }
    static type mul(type a, type b) MEGDNN_ATTRIBUTE_TARGET("sse")
    {
        return _mm_mul_ps(a, b);
    }
    static type fmadd(type a, type b, type c) MEGDNN_ATTRIBUTE_TARGET("sse")
    {
        return add(mul(a, b), c);
    }
    static type exp(type a) MEGDNN_ATTRIBUTE_TARGET("sse")
    {
        float b[4];
        _mm_store_ps(b, a);
        for (size_t i = 0; i < 4; ++i) b[i] = std::exp(b[i]);
        return _mm_load_ps(b);
    }
    static type log(type a) MEGDNN_ATTRIBUTE_TARGET("sse")
    {
        float b[4];
        _mm_store_ps(b, a);
        for (size_t i = 0; i < 4; ++i) b[i] = std::log(b[i]);
        return _mm_load_ps(b);
    }
};

struct simd_traits_avx_base {
    using type = __m256;
    static MEGDNN_CONSTEXPR size_t width = 8;
    static type set1(float a) MEGDNN_ATTRIBUTE_TARGET("avx")
    {
        return _mm256_set1_ps(a);
    }
    static type setzero() MEGDNN_ATTRIBUTE_TARGET("avx")
    {
        return _mm256_setzero_ps();
    }
    static type loadu(const float *mem_addr) MEGDNN_ATTRIBUTE_TARGET("avx")
    {
        return _mm256_loadu_ps(mem_addr);
    }
    static void storeu(float *mem_addr, type a) MEGDNN_ATTRIBUTE_TARGET("avx")
    {
        _mm256_storeu_ps(mem_addr, a);
    }
    static type add(type a, type b) MEGDNN_ATTRIBUTE_TARGET("avx")
    {
        return _mm256_add_ps(a, b);
    }
    static type sub(type a, type b) MEGDNN_ATTRIBUTE_TARGET("avx")
    {
        return _mm256_sub_ps(a, b);
    }
    static type mul(type a, type b) MEGDNN_ATTRIBUTE_TARGET("avx")
    {
        return _mm256_mul_ps(a, b);
    }
    static type exp(type a) MEGDNN_ATTRIBUTE_TARGET("avx")
    {
        float b[8];
        _mm256_storeu_ps(b, a);
        for (size_t i = 0; i < 8; ++i) b[i] = std::exp(b[i]);
        return _mm256_loadu_ps(b);
    }
    static type log(type a) MEGDNN_ATTRIBUTE_TARGET("avx")
    {
        float b[8];
        _mm256_storeu_ps(b, a);
        for (size_t i = 0; i < 8; ++i) b[i] = std::log(b[i]);
        return _mm256_loadu_ps(b);
    }
};

template <>
struct simd_traits<SIMDType::AVX>: simd_traits_avx_base {
    static type fmadd(type a, type b, type c) MEGDNN_ATTRIBUTE_TARGET("avx")
    {
        return add(mul(a, b), c);
    }
};

template <>
struct simd_traits<SIMDType::FMA>: simd_traits_avx_base {
    static type fmadd(type a, type b, type c) MEGDNN_ATTRIBUTE_TARGET("fma")
    {
        return _mm256_fmadd_ps(a, b, c);
    }
};

} // namespace x86
} // namespace megdnn

// vim: syntax=cpp.doxygen
