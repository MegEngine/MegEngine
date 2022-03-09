/**
 * \file dnn/src/fallback/general_intrinsic/gi_common.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2022 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "math.h"
#include "stdint.h"
#include "string.h"

#if defined(_WIN32)
#include <intrin.h>
#include <windows.h>
#else
#if defined(__arm__) || defined(__aarch64__)
#include "src/arm_common/simd_macro/marm_neon.h"
#endif
#if defined(__x86_64__) || defined(__i386__)
#include <cpuid.h>
#include <immintrin.h>
#endif
#endif

#if defined(__x86_64__) || defined(__i386__) || defined(_M_IX86) || defined(_M_X64)
#define GI_TARGET_X86
#endif

#if defined(__arm__) || defined(__aarch64__)
#define GI_TARGET_ARM
#endif

#ifdef _WIN32
//! GI stand for general intrinsic
#define GI_DECLSPEC_ALIGN(variable, alignment) DECLSPEC_ALIGN(alignment) variable
#else
#define GI_DECLSPEC_ALIGN(variable, alignment) \
    variable __attribute__((aligned(alignment)))
#endif

#if defined(_MSC_VER)
#define GI_FORCEINLINE __forceinline
#else
#define GI_FORCEINLINE __attribute__((always_inline)) inline
#endif

#if defined(_MSC_VER)
#define GI_INTERNAL_DATA extern "C"
#else
#define GI_INTERNAL_DATA extern "C" __attribute((visibility("hidden")))
#endif

#if defined(GI_TARGET_ARM)
#define GI_NEON_INTRINSICS
#if defined(__aarch64__)
#define GI_NEON64_INTRINSICS
#else
#define GI_NEON32_INTRINSICS
#endif
#elif defined(GI_TARGET_X86)
//#if defined(__FMA__)
//#define GI_FMA_INTRINSICS
//#define GI_AVX2_INTRINSICS
//#define GI_AVX_INTRINSICS
//#elif defined(__AVX2__)
//#define GI_AVX2_INTRINSICS
//#define GI_AVX_INTRINSICS
//#elif defined(__AVX__)
//#define GI_AVX_INTRINSICS
#if defined(__SSE4_2__)
#define GI_SSE42_INTRINSICS
#define GI_SSE2_INTRINSICS
#elif defined(__SSE2__)
#define GI_SSE2_INTRINSICS
#endif
#endif

#if defined(GI_AVX_INTRINSICS) || defined(GI_AVX2_INTRINSICS) || \
        defined(GI_FMA_INTRINSICS)
typedef __m256 GI_FLOAT32_t;
typedef __m256i GI_UINT8_t;
typedef __m256i GI_INT8_t;
typedef __m256i GI_INT16_t;
typedef __m256i GI_INT32_t;
typedef __m256i GI_UINT32_t;
#elif defined(GI_NEON_INTRINSICS)
typedef float32x4_t GI_FLOAT32_t;
typedef uint8x16_t GI_UINT8_t;
typedef int8x16_t GI_INT8_t;
typedef int16x8_t GI_INT16_t;
typedef int32x4_t GI_INT32_t;
typedef uint32x4_t GI_UINT32_t;
#elif defined(GI_SSE2_INTRINSICS) || defined(GI_SSE42_INTRINSICS)
typedef __m128 GI_FLOAT32_t;
typedef __m128i GI_UINT8_t;
typedef __m128i GI_INT8_t;
typedef __m128i GI_INT16_t;
typedef __m128i GI_INT32_t;
typedef __m128i GI_UINT32_t;
#else
typedef float GI_FLOAT32_t __attribute__((vector_size(16)));
typedef uint8_t GI_UINT8_t __attribute__((vector_size(16)));
typedef int8_t GI_INT8_t __attribute__((vector_size(16)));
typedef int16_t GI_INT16_t __attribute__((vector_size(16)));
typedef int32_t GI_INT32_t __attribute__((vector_size(16)));
typedef uint32_t GI_UINT32_t __attribute__((vector_size(16)));
#endif

//! general intrinsic support dynamic length simd, if avx or avx2 the simd
//! length is 256
#if defined(GI_AVX_INTRINSICS) || defined(GI_AVX2_INTRINSICS) || \
        defined(GI_FMA_INTRINSICS)
//! if neon and sse the simd lenght is 128
#define GI_SIMD_LEN      256
#define GI_SIMD_LEN_BYTE 32
#elif defined(GI_NEON_INTRINSICS) || defined(GI_SSE2_INTRINSICS) || \
        defined(GI_SSE42_INTRINSICS)
#define GI_SIMD_LEN      128
#define GI_SIMD_LEN_BYTE 16
#else
//! if no simd hardware support, the simd is implemented by C, default set to
//! 128
#define GI_SIMD_LEN      128
#define GI_SIMD_LEN_BYTE 16
#endif

#define Max(a, b) (a) > (b) ? (a) : (b)
#define Min(a, b) (a) < (b) ? (a) : (b)

#if defined(GI_NEON_INTRINSICS)
#if defined(__ARM_FEATURE_FMA) && defined(GI_NEON64_INTRINSICS)
#define v_fma_ps_f32(c, b, a)         vfmaq_f32((c), (b), (a))
#define v_fma_n_f32(c, b, a)          vfmaq_n_f32((c), (b), (a))
#define v_fma_lane_f32(c, b, a, lane) vfmaq_lane_f32((c), (b), (a), (lane))
#else
#define v_fma_ps_f32(c, b, a)         vmlaq_f32((c), (b), (a))
#define v_fma_n_f32(c, b, a)          vmlaq_n_f32((c), (b), (a))
#define v_fma_lane_f32(c, b, a, lane) vmlaq_lane_f32((c), (b), (a), (lane))
#endif
#endif

typedef struct {
    GI_INT32_t val[2];
} GI_INT32_V2_t;

typedef struct {
    GI_INT32_t val[4];
} GI_INT32_V4_t;

typedef struct {
    GI_FLOAT32_t val[2];
} GI_FLOAT32_V2_t;

typedef struct {
    GI_FLOAT32_t val[4];
} GI_FLOAT32_V4_t;

typedef struct {
    GI_INT16_t val[2];
} GI_INT16_V2_t;

typedef struct {
    GI_INT8_t val[2];
} GI_INT8_V2_t;

GI_FORCEINLINE
GI_INT32_t GiAndInt32(GI_INT32_t Vector1, GI_INT32_t Vector2) {
#if defined(GI_NEON_INTRINSICS)
    return vandq_s32(Vector1, Vector2);
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_and_si128(Vector1, Vector2);
#else
    return Vector1 & Vector2;
#endif
}

GI_FORCEINLINE
GI_INT32_t GiOrInt32(GI_INT32_t Vector1, GI_INT32_t Vector2) {
#if defined(GI_NEON_INTRINSICS)
    return vorrq_s32(Vector1, Vector2);
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_or_si128(Vector1, Vector2);
#else
    return Vector1 | Vector2;
#endif
}

GI_FORCEINLINE
GI_INT32_t GiAndNotInt32(GI_INT32_t VectorNot, GI_INT32_t Vector) {
#if defined(GI_NEON_INTRINSICS)
    return vandq_s32(vmvnq_s32(VectorNot), Vector);
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_andnot_si128(VectorNot, Vector);
#else
    return (~VectorNot) & Vector;
#endif
}

GI_FORCEINLINE
GI_INT32_t GiXorInt32(GI_INT32_t Vector1, GI_INT32_t Vector2) {
#if defined(GI_NEON_INTRINSICS)
    return veorq_s32(Vector1, Vector2);
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_xor_si128(Vector1, Vector2);
#else
    return Vector1 ^ Vector2;
#endif
}

// vim: syntax=cpp.doxygen
