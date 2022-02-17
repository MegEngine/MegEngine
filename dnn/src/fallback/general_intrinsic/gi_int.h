/**
 * \file dnn/src/fallback/general_intrinsic/gi_float.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2022 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "gi_common.h"

GI_FORCEINLINE
GI_INT32
GiBroadcastInt32(int32_t Value) {
#if defined(GI_NEON_INTRINSICS)
    return vdupq_n_s32(Value);
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_set1_epi32(Value);
#else
    GI_INT32 ret;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(int32_t); i++) {
        ret[i] = Value;
    }
    return ret;
#endif
}

GI_FORCEINLINE
GI_INT8
GiBroadcastInt8(int8_t Value) {
#if defined(GI_NEON_INTRINSICS)
    return vdupq_n_s8(Value);
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_set1_epi8(Value);
#else
    GI_INT8 ret;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(int8_t); i++) {
        ret[i] = Value;
    }
    return ret;
#endif
}

GI_FORCEINLINE
GI_INT32
GiLoadInt32(const int32_t* Buffer) {
#if defined(GI_NEON_INTRINSICS)
    return vld1q_s32(Buffer);
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_loadu_si128((const __m128i*)Buffer);
#else
    GI_INT32 ret;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(int32_t); i++) {
        ret[i] = Buffer[i];
    }
    return ret;
#endif
}

GI_FORCEINLINE
GI_INT8
GiLoadInt8(const int8_t* Buffer) {
#if defined(GI_NEON_INTRINSICS)
    return vld1q_s8(Buffer);
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_loadu_si128((const __m128i*)Buffer);
#else
    GI_INT8 ret;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(int8_t); i++) {
        ret[i] = Buffer[i];
    }
    return ret;
#endif
}

GI_FORCEINLINE
void GiStoreInt32(int32_t* Buffer, GI_INT32 Vector) {
#if defined(GI_NEON_INTRINSICS)
    vst1q_s32(Buffer, Vector);
#elif defined(GI_SSE2_INTRINSICS)
    _mm_storeu_si128((__m128i*)Buffer, Vector);
#else
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(int32_t); i++) {
        Buffer[i] = Vector[i];
    }
#endif
}

GI_FORCEINLINE
void GiStoreInt8(int8_t* Buffer, GI_INT8 Vector) {
#if defined(GI_NEON_INTRINSICS)
    vst1q_s8(Buffer, Vector);
#elif defined(GI_SSE2_INTRINSICS)
    _mm_storeu_si128((__m128i*)Buffer, Vector);
#else
    for (int i = 0; i < 16; i++) {
        Buffer[i] = Vector[i];
    }
#endif
}

GI_FORCEINLINE
void GiStoreLowInt8(int8_t* Buffer, GI_INT8 Vector) {
#if defined(GI_NEON_INTRINSICS)
    vst1_s8(Buffer, vget_low_s8(Vector));
#elif defined(GI_SSE2_INTRINSICS)
    _mm_storel_epi64((__m128i*)Buffer, Vector);
#else
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / 2 / sizeof(int8_t); i++) {
        Buffer[i] = Vector[i];
    }
#endif
}

GI_FORCEINLINE
void GiStoreHihgInt8(int8_t* Buffer, GI_INT8 Vector) {
#if defined(GI_NEON_INTRINSICS)
    vst1_s8(Buffer, vget_high_s8(Vector));
#elif defined(GI_SSE2_INTRINSICS)
    _mm_storel_epi64((__m128i*)Buffer, _mm_unpackhi_epi64(Vector, Vector));
#else
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / 2 / sizeof(int8_t); i++) {
        Buffer[i] = Vector[GI_SIMD_LEN_BYTE / 2 + i];
    }
#endif
}

GI_FORCEINLINE
GI_INT32
GiAddInt32(GI_INT32 Vector1, GI_INT32 Vector2) {
#if defined(GI_NEON_INTRINSICS)
    return vaddq_s32(Vector1, Vector2);
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_add_epi32(Vector1, Vector2);
#else
    return Vector1 + Vector2;
#endif
}

GI_FORCEINLINE
GI_INT32
GiSubtractInt32(GI_INT32 Vector1, GI_INT32 Vector2) {
#if defined(GI_NEON_INTRINSICS)
    return vsubq_s32(Vector1, Vector2);
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_sub_epi32(Vector1, Vector2);
#else
    return Vector1 - Vector2;
#endif
}

GI_FORCEINLINE
GI_INT32
GiMultiplyInt32(GI_INT32 Vector1, GI_INT32 Vector2) {
#if defined(GI_NEON_INTRINSICS)
    return vmulq_s32(Vector1, Vector2);
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_mul_epi32(Vector1, Vector2);
#else
    return Vector1 * Vector2;
#endif
}

GI_FORCEINLINE
GI_INT8
GiAndInt8(GI_INT8 Vector1, GI_INT8 Vector2) {
#if defined(GI_NEON_INTRINSICS)
    return vandq_s8(Vector1, Vector2);
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_and_si128(Vector1, Vector2);
#else
    return Vector1 & Vector2;
#endif
}

GI_FORCEINLINE
GI_INT8
GiOrInt8(GI_INT8 Vector1, GI_INT8 Vector2) {
#if defined(GI_NEON_INTRINSICS)
    return vorrq_s8(Vector1, Vector2);
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_or_si128(Vector1, Vector2);
#else
    return Vector1 | Vector2;
#endif
}

GI_FORCEINLINE
GI_INT8
GiAndNotInt8(GI_INT8 VectorNot, GI_INT8 Vector) {
#if defined(GI_NEON_INTRINSICS)
    return vandq_s8(vmvnq_s8(VectorNot), Vector);
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_andnot_si128(VectorNot, Vector);
#else
    return (~VectorNot) & Vector;
#endif
}

GI_FORCEINLINE
GI_INT8
GiXorInt8(GI_INT8 Vector1, GI_INT8 Vector2) {
#if defined(GI_NEON_INTRINSICS)
    return veorq_s8(Vector1, Vector2);
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_xor_si128(Vector1, Vector2);
#else
    return Vector1 ^ Vector2;
#endif
}

#if defined(GI_NEON_INTRINSICS)
#define GISHIFTLEFTINT32(i)                                          \
    GI_FORCEINLINE GI_INT32 GiShiftLeft##i##Int32(GI_INT32 Vector) { \
        return vshlq_n_s32(Vector, i);                               \
    }

#elif defined(GI_SSE2_INTRINSICS)

#define GISHIFTLEFTINT32(i)                                          \
    GI_FORCEINLINE GI_INT32 GiShiftLeft##i##Int32(GI_INT32 Vector) { \
        return _mm_slli_epi32(Vector, i);                            \
    }
#else
#define GISHIFTLEFTINT32(i)                                          \
    GI_FORCEINLINE GI_INT32 GiShiftLeft##i##Int32(GI_INT32 Vector) { \
        return Vector << i;                                          \
    }
#endif

GISHIFTLEFTINT32(0)
GISHIFTLEFTINT32(1)
GISHIFTLEFTINT32(2)
GISHIFTLEFTINT32(3)

#undef GISHIFTLEFTINT32

GI_FORCEINLINE
GI_INT32
GiBlendInt32(GI_INT32 Vector1, GI_INT32 Vector2, GI_INT32 Selection) {
    return GiOrInt32(GiAndInt32(Vector2, Selection), GiAndNotInt32(Selection, Vector1));
}

GI_FORCEINLINE
GI_INT8
GiBlendInt8(GI_INT8 Vector1, GI_INT8 Vector2, GI_INT8 Selection) {
    return GiOrInt8(GiAndInt8(Vector2, Selection), GiAndNotInt8(Selection, Vector1));
}

GI_FORCEINLINE
GI_INT32
GiMaximumInt32(GI_INT32 Vector1, GI_INT32 Vector2) {
#if defined(GI_NEON_INTRINSICS)
    return vmaxq_s32(Vector1, Vector2);
#elif defined(GI_SSE42_INTRINSICS)
    return _mm_max_epi32(Vector1, Vector2);
#elif defined(GI_SSE2_INTRINSICS)
    return GiBlendInt32(Vector2, Vector1, _mm_cmpgt_epi32(Vector1, Vector2));
#else
    return GiBlendInt32(Vector2, Vector1, Vector1 > Vector2);
#endif
}

GI_FORCEINLINE
GI_INT32
GiMinimumInt32(GI_INT32 Vector1, GI_INT32 Vector2) {
#if defined(GI_NEON_INTRINSICS)
    return vminq_s32(Vector1, Vector2);
#elif defined(GI_SSE42_INTRINSICS)
    return _mm_min_epi32(Vector1, Vector2);
#elif defined(GI_SSE2_INTRINSICS)
    return GiBlendInt32(Vector2, Vector1, _mm_cmpgt_epi32(Vector2, Vector1));
#else
    return GiBlendInt32(Vector2, Vector1, Vector2 > Vector1);
#endif
}

GI_FORCEINLINE
GI_INT8
GiBlendInt8x16(GI_INT8 Vector1, GI_INT8 Vector2, GI_INT8 Selection) {
    return GiOrInt8(GiAndInt8(Vector2, Selection), GiAndNotInt8(Selection, Vector1));
}

GI_FORCEINLINE
GI_INT8
GiMaximumInt8(GI_INT8 Vector1, GI_INT8 Vector2) {
#if defined(GI_NEON_INTRINSICS)
    return vmaxq_s8(Vector1, Vector2);
#elif defined(GI_SSE42_INTRINSICS)
    return _mm_max_epi8(Vector1, Vector2);
#elif defined(GI_SSE2_INTRINSICS)
    return GiBlendInt8(Vector2, Vector1, _mm_cmpgt_epi8(Vector1, Vector2));
#else
    return GiBlendInt8(Vector2, Vector1, Vector1 > Vector2);
#endif
}

GI_FORCEINLINE
GI_INT8
GiMinimumInt8(GI_INT8 Vector1, GI_INT8 Vector2) {
#if defined(GI_NEON_INTRINSICS)
    return vminq_s8(Vector1, Vector2);
#elif defined(GI_SSE42_INTRINSICS)
    return _mm_min_epi8(Vector1, Vector2);
#elif defined(GI_SSE2_INTRINSICS)
    return GiBlendInt8(Vector2, Vector1, _mm_cmpgt_epi8(Vector2, Vector1));
#else
    return GiBlendInt8(Vector2, Vector1, Vector2 > Vector1);
#endif
}

GI_FORCEINLINE
GI_INT16
GiMoveHighLongInt8(GI_INT8 Vector) {
#if defined(GI_NEON_INTRINSICS)
    return vmovl_s8(vget_high_s8(Vector));
#elif defined(GI_SSE42_INTRINSICS)
    return _mm_cvtepi8_epi16(_mm_unpackhi_epi64(Vector, Vector));
#elif defined(GI_SSE2_INTRINSICS)
    int16_t data[8];
    int8_t o_data[16];
    _mm_storeu_si128((__m128i*)o_data, Vector);
    for (int i = 0; i < 8; i++) {
        data[i] = o_data[8 + i];
    }
    return _mm_loadu_si16(data);
#else
    GI_INT16 ret;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / 2 / sizeof(int8_t); i++) {
        ret[i] = Vector[GI_SIMD_LEN_BYTE / 2 + i];
    }
    return ret;
#endif
}

GI_FORCEINLINE
GI_INT16
GiMoveLowLongInt8(GI_INT8 Vector) {
#if defined(GI_NEON_INTRINSICS)
    return vmovl_s8(vget_low_s8(Vector));
#elif defined(GI_SSE42_INTRINSICS)
    return _mm_cvtepi8_epi16(Vector);
#elif defined(GI_SSE2_INTRINSICS)
    int16_t data[8];
    int8_t o_data[16];
    _mm_storeu_si128((__m128i*)o_data, Vector);
    for (int i = 0; i < 8; i++) {
        data[i] = o_data[i];
    }
    return _mm_loadu_si16(data);
#else
    GI_INT16 ret;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / 2 / sizeof(int8_t); i++) {
        ret[i] = Vector[i];
    }
    return ret;
#endif
}

GI_FORCEINLINE
GI_INT32
GiMoveHighLongInt16(GI_INT16 Vector) {
#if defined(GI_NEON_INTRINSICS)
    return vmovl_s16(vget_high_s16(Vector));
#elif defined(GI_SSE42_INTRINSICS)
    return _mm_cvtepi16_epi32(_mm_unpackhi_epi64(Vector, Vector));
#elif defined(GI_SSE2_INTRINSICS)
    int32_t data[4];
    int16_t o_data[8];
    _mm_storeu_si128((__m128i*)o_data, Vector);
    for (int i = 0; i < 4; i++) {
        data[i] = o_data[4 + i];
    }
    return _mm_loadu_si32(data);
#else
    GI_INT32 ret;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / 2 / sizeof(int16_t); i++) {
        ret[i] = Vector[GI_SIMD_LEN_BYTE / 2 + i];
    }
    return ret;
#endif
}

GI_FORCEINLINE
GI_INT32
GiMoveLowLongInt16(GI_INT16 Vector) {
#if defined(GI_NEON_INTRINSICS)
    return vmovl_s16(vget_low_s16(Vector));
#elif defined(GI_SSE42_INTRINSICS)
    return _mm_cvtepi16_epi32(Vector);
#elif defined(GI_SSE2_INTRINSICS)
    int32_t data[4];
    int16_t o_data[8];
    _mm_storeu_si128((__m128i*)o_data, Vector);
    for (int i = 0; i < 4; i++) {
        data[i] = o_data[i];
    }
    return _mm_loadu_si32(data);
#else
    GI_INT32 ret;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / 2 / sizeof(int16_t); i++) {
        ret[i] = Vector[i];
    }
    return ret;
#endif
}

GI_FORCEINLINE
int16_t GiReduceAddInt8(GI_INT8 Vector) {
#if defined(GI_NEON64_INTRINSICS)
    return vaddlvq_s8(Vector);
#elif defined(GI_NEON32_INTRINSICS)
    int32_t sum = vpaddlq_s16(vpaddlq_s8(Vector));
    return (vgetq_lane_s32(sum, 0) + vgetq_lane_s32(sum, 1) + vgetq_lane_s32(sum, 2) +
            vgetq_lane_s32(sum, 3));
#elif defined(GI_SSE42_INTRINSICS)
    __m128i v0 = _mm_cvtepi8_epi16(Vector);
    __m128i v1 = _mm_cvtepi8_epi16(_mm_unpackhi_epi64(Vector, Vector));
    __m128i sum_int16 = _mm_add_epi16(v0, v1);
    __m128i v0_int32 = _mm_cvtepi16_epi32(sum_int16);
    __m128i v1_int32 = _mm_cvtepi16_epi32(_mm_unpackhi_epi64(sum_int16, sum_int16));
    __m128i sum = _mm_add_epi32(v0_int32, v1_int32);
    float ret = _mm_extract_epi32(sum, 0);
    ret += _mm_extract_epi32(sum, 1);
    ret += _mm_extract_epi32(sum, 2);
    ret += _mm_extract_epi32(sum, 3);
    return (int16_t)(ret);

#elif defined(GI_SSE2_INTRINSICS)
    __m64 low = GiGetLowInt8x16(Vector);
    __m64 high = GiGetHighInt8x16(Vector);
    __m128 v0 = _mm_cvtpi8_ps(low);
    __m128 v1 = _mm_cvtpi8_ps(_mm_unpackhi_pi32(low, low));
    __m128 v2 = _mm_cvtpi8_ps(high);
    __m128 v3 = _mm_cvtpi8_ps(_mm_unpackhi_pi32(high, high));
    __m128 sum0 = _mm_add_ps(v0, v1);
    __m128 sum1 = _mm_add_ps(v2, v3);
    __m128 sum = _mm_add_ps(sum0, sum1);
    float ret0 = _mm_cvtss_f32(sum);
    float ret1 = _mm_cvtss_f32(_mm_shuffle_ps(sum, sum, _MM_SHUFFLE(1, 1, 1, 1)));
    float ret2 = _mm_cvtss_f32(_mm_shuffle_ps(sum, sum, _MM_SHUFFLE(2, 2, 2, 2)));
    float ret3 = _mm_cvtss_f32(_mm_shuffle_ps(sum, sum, _MM_SHUFFLE(3, 3, 3, 3)));
    return (int16_t)(ret0 + ret1 + ret2 + ret3);
#else
    int32_t sum = 0;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(int32_t); i++) {
        sum += Vector[i];
    }
    return sum;
#endif
}

#define Max(a, b) (a) > (b) ? (a) : (b)
#define Min(a, b) (a) < (b) ? (a) : (b)

GI_FORCEINLINE
int8_t GiReduceMaxInt8(GI_INT8 Vector) {
#if defined(GI_NEON64_INTRINSICS)
    return vmaxvq_s8(Vector);
#elif defined(GI_NEON32_INTRINSICS)
    int8x8_t VectorLow = vget_low_s8(Vector);
    int8x8_t VectorHigh = vget_high_s8(Vector);
    VectorLow = vpmin_s8(VectorLow, VectorHigh);
    VectorLow = vpmin_s8(VectorLow, VectorHigh);
    return vget_lane_s8(VectorLow, 0);
#elif defined(GI_SSE42_INTRINSICS)
    __m128i v0 = _mm_cvtepi8_epi16(Vector);
    __m128i v1 = _mm_cvtepi8_epi16(_mm_unpackhi_epi64(Vector, Vector));
    __m128i max_int16 = _mm_max_epi16(v0, v1);
    __m128i v0_int32 = _mm_cvtepi16_epi32(max_int16);
    __m128i v1_int32 = _mm_cvtepi16_epi32(_mm_unpackhi_epi64(max_int16, max_int16));
    __m128i sum = _mm_max_epi32(v0_int32, v1_int32);
    int ret = _mm_extract_epi32(sum, 0);
    ret = Max(_mm_extract_epi32(sum, 1), ret);
    ret = Max(_mm_extract_epi32(sum, 2), ret);
    ret = Max(_mm_extract_epi32(sum, 3), ret);
    return (int8_t)ret;
#elif defined(GI_SSE2_INTRINSICS)
    __m64 low = GiGetLowInt8x16(Vector);
    __m64 high = GiGetHighInt8x16(Vector);
    __m128 v0 = _mm_cvtpi8_ps(low);
    __m128 v1 = _mm_cvtpi8_ps(_mm_unpackhi_pi32(low, low));
    __m128 v2 = _mm_cvtpi8_ps(high);
    __m128 v3 = _mm_cvtpi8_ps(_mm_unpackhi_pi32(high, high));
    __m128 sum0 = _mm_add_ps(v0, v1);
    __m128 sum1 = _mm_add_ps(v2, v3);
    __m128 sum = _mm_add_ps(sum0, sum1);
    float ret0 = _mm_cvtss_f32(sum);
    float ret1 = _mm_cvtss_f32(_mm_shuffle_ps(sum, sum, _MM_SHUFFLE(1, 1, 1, 1)));
    float ret2 = _mm_cvtss_f32(_mm_shuffle_ps(sum, sum, _MM_SHUFFLE(2, 2, 2, 2)));
    float ret3 = _mm_cvtss_f32(_mm_shuffle_ps(sum, sum, _MM_SHUFFLE(3, 3, 3, 3)));
    return (int8_t)(Max(Max(ret0, ret1), Max(ret2, ret3)));
#else
    int8_t max = Vector[0];
    for (size_t i = 1; i < GI_SIMD_LEN_BYTE / sizeof(int32_t); i++) {
        max = Max(max, Vector[i]);
    }
    return max;
#endif
}

GI_FORCEINLINE
int8_t GiReduceMinInt8(GI_INT8 Vector) {
#if defined(GI_NEON64_INTRINSICS)
    return vminvq_s8(Vector);
#elif defined(GI_NEON32_INTRINSICS)
    int8x8_t VectorLow = vget_low_s8(Vector);
    int8x8_t VectorHigh = vget_high_s8(Vector);
    VectorLow = vpmin_s8(VectorLow, VectorHigh);
    VectorLow = vpmin_s8(VectorLow, VectorHigh);
    return vget_lane_s8(VectorLow, 0);
#elif defined(GI_SSE42_INTRINSICS)
    __m128i v0 = _mm_cvtepi8_epi16(Vector);
    __m128i v1 = _mm_cvtepi8_epi16(_mm_unpackhi_epi64(Vector, Vector));
    __m128i min_int16 = _mm_min_epi16(v0, v1);
    __m128i v0_int32 = _mm_cvtepi16_epi32(min_int16);
    __m128i v1_int32 = _mm_cvtepi16_epi32(_mm_unpackhi_epi64(min_int16, min_int16));
    __m128i sum = _mm_min_epi32(v0_int32, v1_int32);
    int ret = _mm_extract_epi32(sum, 0);
    ret = Min(_mm_extract_epi32(sum, 1), ret);
    ret = Min(_mm_extract_epi32(sum, 2), ret);
    ret = Min(_mm_extract_epi32(sum, 3), ret);
    return (int8_t)ret;
#elif defined(GI_SSE2_INTRINSICS)
    __m64 low = GiGetLowInt8x16(Vector);
    __m64 high = GiGetHighInt8x16(Vector);
    __m128 v0 = _mm_cvtpi8_ps(low);
    __m128 v1 = _mm_cvtpi8_ps(_mm_unpackhi_pi32(low, low));
    __m128 v2 = _mm_cvtpi8_ps(high);
    __m128 v3 = _mm_cvtpi8_ps(_mm_unpackhi_pi32(high, high));
    __m128 sum0 = _mm_add_ps(v0, v1);
    __m128 sum1 = _mm_add_ps(v2, v3);
    __m128 sum = _mm_add_ps(sum0, sum1);
    float ret0 = _mm_cvtss_f32(sum);
    float ret1 = _mm_cvtss_f32(_mm_shuffle_ps(sum, sum, _MM_SHUFFLE(1, 1, 1, 1)));
    float ret2 = _mm_cvtss_f32(_mm_shuffle_ps(sum, sum, _MM_SHUFFLE(2, 2, 2, 2)));
    float ret3 = _mm_cvtss_f32(_mm_shuffle_ps(sum, sum, _MM_SHUFFLE(3, 3, 3, 3)));
    return (int8_t)(Min(Min(ret0, ret1), Min(ret2, ret3)));
#else
    int8_t min = Vector[0];
    for (size_t i = 1; i < GI_SIMD_LEN_BYTE / sizeof(int32_t); i++) {
        min = Min(min, Vector[i]);
    }
    return min;
#endif
}

#define Saturate(x, lower, upper) \
    (x) > (upper) ? (upper) : ((x) >= (lower) ? (x) : (lower))

//! convert to the short type with the lower bit fill the real data, the high bite
//! will repeat the lower bit
GI_FORCEINLINE
GI_INT8
GiCvtFromFloat32ToInt8(GI_FLOAT32 src) {
#if defined(GI_NEON_INTRINSICS)
#if __ARM_ARCH >= 8
    int32x4_t vres0 = vcvtaq_s32_f32(src);
    int16x8_t mid_s16 = vcombine_s16(vqmovn_s32(vres0), vqmovn_s32(vres0));
    int8x8_t ret = vqmovn_s16(vcombine_s16(vqmovn_s32(mid_s16), vqmovn_s32(mid_s16)));
    return vcombine_s16(ret, ret);
#else
    float32x4_t vzero = vdupq_n_f32(0.f);
    float32x4_t vfhalf = vdupq_n_f32(0.5f);
    float32x4_t vfneg_half = vdupq_n_f32(-0.5f);
    float32x4_t vinc0 = vbslq_f32(vcgeq_f32(src, vzero), vfhalf, vfneg_half);
    int32x4_t vres0 = vcvtq_s32_f32(vaddq_f32(src, vinc0));
    int16x8_t mid_s16 = vcombine_s16(vqmovn_s32(vres0), vqmovn_s32(vres0));
    int8x8_t ret = vqmovn_s16(vcombine_s16(vqmovn_s32(mid_s16), vqmovn_s32(mid_s16)));
    return vcombine_s16(ret, ret);
#endif
#elif defined(GI_SSE42_INTRINSICS)
    __m128 vfzero = _mm_set1_ps(0.f);
    __m128 vfhalf = _mm_set1_ps(0.5f);
    __m128 vfneg_half = _mm_set1_ps(-0.5f);
    __m128 vfmin_int8 = _mm_set1_ps(-128.f);
    __m128 vfmax_int8 = _mm_set1_ps(127.f);

    __m128 vinc0 = _mm_blendv_ps(vfneg_half, vfhalf, _mm_cmpge_ps(src, vfzero));
    __m128 vres0 = _mm_add_ps(src, vinc0);
    vres0 = _mm_round_ps(vres0, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    vres0 = _mm_min_ps(_mm_max_ps(vres0, vfmin_int8), vfmax_int8);

    __m128i vepi32_0 = _mm_cvtps_epi32(vres0);
    __m128i vepi16 = _mm_packs_epi32(vepi32_0, vepi32_0);
    __m128i vepi8 = _mm_packs_epi16(vepi16, vepi16);
    return vepi8;
#else
    GI_INT8 ret;
    int length = GI_SIMD_LEN_BYTE / sizeof(float);
    for (int i = 0; i < length; i++) {
        int8_t data = Saturate(round(src[i]), -128, 127);
        ret[i] = data;
        ret[length + i] = data;
        ret[2 * length + i] = data;
        ret[3 * length + i] = data;
    }
    return ret;
#endif
}

GI_FORCEINLINE
GI_INT8
GiCvtFromFloat32V2ToInt8(GI_FLOAT32_V2 vsrc) {
#if defined(GI_NEON_INTRINSICS)
#if __ARM_ARCH >= 8
    int32x4_t vres0 = vcvtaq_s32_f32(vsrc.val[0]);
    int32x4_t vres1 = vcvtaq_s32_f32(vsrc.val[1]);
    int8x8_t mid1 = vqmovn_s16(vcombine_s16(vqmovn_s32(vres0), vqmovn_s32(vres1)));
    return vcombine_s8(mid1, mid1);
#else
    float32x4_t vzero = vdupq_n_f32(0.f);
    float32x4_t vfhalf = vdupq_n_f32(0.5f);
    float32x4_t vfneg_half = vdupq_n_f32(-0.5f);
    float32x4_t vinc0 = vbslq_f32(vcgeq_f32(vsrc.val[0], vzero), vfhalf, vfneg_half);
    float32x4_t vinc1 = vbslq_f32(vcgeq_f32(vsrc.val[1], vzero), vfhalf, vfneg_half);
    int32x4_t vres0 = vcvtq_s32_f32(vaddq_f32(vsrc.val[0], vinc0));
    int32x4_t vres1 = vcvtq_s32_f32(vaddq_f32(vsrc.val[1], vinc1));
    int8x8_t mid1 = vqmovn_s16(vcombine_s16(vqmovn_s32(vres0), vqmovn_s32(vres1)));
    return vcombine_s8(mid1, mid1);
#endif
#elif defined(GI_SSE42_INTRINSICS)
    __m128 vfzero = _mm_set1_ps(0.f);
    __m128 vfhalf = _mm_set1_ps(0.5f);
    __m128 vfneg_half = _mm_set1_ps(-0.5f);
    __m128 vfmin_int8 = _mm_set1_ps(-128.f);
    __m128 vfmax_int8 = _mm_set1_ps(127.f);

    __m128 vinc0 = _mm_blendv_ps(vfneg_half, vfhalf, _mm_cmpge_ps(vsrc.val[0], vfzero));
    __m128 vinc1 = _mm_blendv_ps(vfneg_half, vfhalf, _mm_cmpge_ps(vsrc.val[1], vfzero));

    __m128 vres0 = _mm_add_ps(vsrc.val[0], vinc0);
    __m128 vres1 = _mm_add_ps(vsrc.val[1], vinc1);

    vres0 = _mm_round_ps(vres0, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    vres1 = _mm_round_ps(vres1, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);

    vres0 = _mm_min_ps(_mm_max_ps(vres0, vfmin_int8), vfmax_int8);
    vres1 = _mm_min_ps(_mm_max_ps(vres1, vfmin_int8), vfmax_int8);

    __m128i vepi32_0 = _mm_cvtps_epi32(vres0);
    __m128i vepi32_1 = _mm_cvtps_epi32(vres1);
    __m128i vepi16_0 = _mm_packs_epi32(vepi32_0, vepi32_1);
    __m128i vepi8 = _mm_packs_epi16(vepi16_0, vepi16_0);
    return vepi8;
#else
    GI_INT8 ret;
    int length = GI_SIMD_LEN_BYTE / sizeof(float);
    for (int i = 0; i < 2 * length; i++) {
        ret[i] = Saturate(round(vsrc.val[i / length][i % length]), -128, 127);
    }
    return ret;
#endif
}

GI_FORCEINLINE
GI_INT8
GiCvtFromFloat32V4ToInt8(GI_FLOAT32_V4 vsrc) {
#if defined(GI_NEON_INTRINSICS)
#if __ARM_ARCH >= 8
    int32x4_t vres0 = vcvtaq_s32_f32(vsrc.val[0]);
    int32x4_t vres1 = vcvtaq_s32_f32(vsrc.val[1]);
    int32x4_t vres2 = vcvtaq_s32_f32(vsrc.val[1]);
    int32x4_t vres3 = vcvtaq_s32_f32(vsrc.val[1]);
    int8x8_t mid1 = vqmovn_s16(vcombine_s16(vqmovn_s32(vres0), vqmovn_s32(vres1)));
    int8x8_t mid2 = vqmovn_s16(vcombine_s16(vqmovn_s32(vres2), vqmovn_s32(vres3)));
    return vcombine_s8(mid1, mid2);
#else
    float32x4_t vzero = vdupq_n_f32(0.f);
    float32x4_t vfhalf = vdupq_n_f32(0.5f);
    float32x4_t vfneg_half = vdupq_n_f32(-0.5f);
    float32x4_t vinc0 = vbslq_f32(vcgeq_f32(vsrc.val[0], vzero), vfhalf, vfneg_half);
    float32x4_t vinc1 = vbslq_f32(vcgeq_f32(vsrc.val[1], vzero), vfhalf, vfneg_half);
    float32x4_t vinc2 = vbslq_f32(vcgeq_f32(vsrc.val[2], vzero), vfhalf, vfneg_half);
    float32x4_t vinc3 = vbslq_f32(vcgeq_f32(vsrc.val[3], vzero), vfhalf, vfneg_half);
    int32x4_t vres0 = vcvtq_s32_f32(vaddq_f32(vsrc.val[0], vinc0));
    int32x4_t vres1 = vcvtq_s32_f32(vaddq_f32(vsrc.val[1], vinc1));
    int32x4_t vres2 = vcvtq_s32_f32(vaddq_f32(vsrc.val[2], vinc2));
    int32x4_t vres3 = vcvtq_s32_f32(vaddq_f32(vsrc.val[3], vinc3));
    int8x8_t mid1 = vqmovn_s16(vcombine_s16(vqmovn_s32(vres0), vqmovn_s32(vres1)));
    int8x8_t mid2 = vqmovn_s16(vcombine_s16(vqmovn_s32(vres2), vqmovn_s32(vres3)));
    return vcombine_s8(mid1, mid2);
#endif
#elif defined(GI_SSE42_INTRINSICS)
    __m128 vfzero = _mm_set1_ps(0.f);
    __m128 vfhalf = _mm_set1_ps(0.5f);
    __m128 vfneg_half = _mm_set1_ps(-0.5f);
    __m128 vfmin_int8 = _mm_set1_ps(-128.f);
    __m128 vfmax_int8 = _mm_set1_ps(127.f);

    __m128 vinc0 = _mm_blendv_ps(vfneg_half, vfhalf, _mm_cmpge_ps(vsrc.val[0], vfzero));
    __m128 vinc1 = _mm_blendv_ps(vfneg_half, vfhalf, _mm_cmpge_ps(vsrc.val[1], vfzero));
    __m128 vinc2 = _mm_blendv_ps(vfneg_half, vfhalf, _mm_cmpge_ps(vsrc.val[2], vfzero));
    __m128 vinc3 = _mm_blendv_ps(vfneg_half, vfhalf, _mm_cmpge_ps(vsrc.val[3], vfzero));

    __m128 vres0 = _mm_add_ps(vsrc.val[0], vinc0);
    __m128 vres1 = _mm_add_ps(vsrc.val[1], vinc1);
    __m128 vres2 = _mm_add_ps(vsrc.val[2], vinc2);
    __m128 vres3 = _mm_add_ps(vsrc.val[3], vinc3);

    vres0 = _mm_round_ps(vres0, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    vres1 = _mm_round_ps(vres1, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    vres2 = _mm_round_ps(vres2, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    vres3 = _mm_round_ps(vres1, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);

    vres0 = _mm_min_ps(_mm_max_ps(vres0, vfmin_int8), vfmax_int8);
    vres1 = _mm_min_ps(_mm_max_ps(vres1, vfmin_int8), vfmax_int8);
    vres2 = _mm_min_ps(_mm_max_ps(vres2, vfmin_int8), vfmax_int8);
    vres3 = _mm_min_ps(_mm_max_ps(vres3, vfmin_int8), vfmax_int8);

    __m128i vepi32_0 = _mm_cvtps_epi32(vres0);
    __m128i vepi32_1 = _mm_cvtps_epi32(vres1);
    __m128i vepi32_2 = _mm_cvtps_epi32(vres2);
    __m128i vepi32_3 = _mm_cvtps_epi32(vres3);
    __m128i vepi16_0 = _mm_packs_epi32(vepi32_0, vepi32_1);
    __m128i vepi16_1 = _mm_packs_epi32(vepi32_2, vepi32_3);
    __m128i vepi8 = _mm_packs_epi16(vepi16_0, vepi16_1);
    return vepi8;
#else
    GI_INT8 ret;
    int length = GI_SIMD_LEN_BYTE / sizeof(float);
    for (int i = 0; i < 4 * length; i++) {
        ret[i] = Saturate(round(vsrc.val[i / length][i % length]), -128, 127);
    }
    return ret;
#endif
}

// vim: syntax=cpp.doxygen
