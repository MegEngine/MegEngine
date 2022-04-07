/**
 * \file dnn/src/fallback/general_intrinsic/gi_int.h
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
GI_UINT32_t GiBroadcastUint32(int32_t Value) {
#if defined(GI_NEON_INTRINSICS)
    return vdupq_n_u32(Value);
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_set1_epi32(Value);
#else
    GI_UINT32_t ret;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(int32_t); i++) {
        ret[i] = Value;
    }
    return ret;
#endif
}

GI_FORCEINLINE
GI_INT32_t GiLoadInt32(const void* Buffer) {
#if defined(GI_NEON_INTRINSICS)
    return vld1q_s32((int32_t*)Buffer);
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_loadu_si128((const __m128i*)Buffer);
#else
    GI_INT32_t ret;
    const int32_t* ptr = (int32_t*)Buffer;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(int32_t); i++) {
        ret[i] = ptr[i];
    }
    return ret;
#endif
}

GI_FORCEINLINE
GI_INT16_t GiLoadInt16(const void* Buffer) {
#if defined(GI_NEON_INTRINSICS)
    return vld1q_s16((int16_t*)Buffer);
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_loadu_si128((const __m128i*)Buffer);
#else
    GI_INT16_t ret;
    const int16_t* ptr = (int16_t*)Buffer;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(int16_t); i++) {
        ret[i] = ptr[i];
    }
    return ret;
#endif
}

GI_FORCEINLINE
GI_INT8_t GiLoadInt8(const void* Buffer) {
#if defined(GI_NEON_INTRINSICS)
    return vld1q_s8((int8_t*)Buffer);
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_loadu_si128((const __m128i*)Buffer);
#else
    GI_INT8_t ret;
    const int8_t* ptr = (int8_t*)Buffer;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(int8_t); i++) {
        ret[i] = ptr[i];
    }
    return ret;
#endif
}

GI_FORCEINLINE
void GiStoreInt32(void* Buffer, GI_INT32_t Vector) {
#if defined(GI_NEON_INTRINSICS)
    vst1q_s32((int32_t*)Buffer, Vector);
#elif defined(GI_SSE2_INTRINSICS)
    _mm_storeu_si128((__m128i*)Buffer, Vector);
#else
    int32_t* ptr = (int32_t*)Buffer;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(int32_t); i++) {
        ptr[i] = Vector[i];
    }
#endif
}

#if defined(GI_NEON_INTRINSICS)
#define GISTORELANEINT32(i)                                                      \
    GI_FORCEINLINE void GiStoreLane##i##Int32(void* Buffer, GI_INT32_t Vector) { \
        vst1q_lane_s32((int32_t*)Buffer, Vector, i);                             \
    }

#elif defined(GI_SSE2_INTRINSICS)

#define GISTORELANEINT32(i)                                                         \
    GI_FORCEINLINE void GiStoreLane##i##Int32(void* Buffer, GI_INT32_t Vector) {    \
        GI_FLOAT32_t tmp = _mm_castsi128_ps(Vector);                                \
        _mm_store_ss(                                                               \
                (float*)Buffer, _mm_shuffle_ps(tmp, tmp, _MM_SHUFFLE(i, i, i, i))); \
    }
#else
#define GISTORELANEINT32(i)                                                      \
    GI_FORCEINLINE void GiStoreLane##i##Int32(void* Buffer, GI_INT32_t Vector) { \
        *((int32_t*)Buffer) = Vector[i];                                         \
    }
#endif

GISTORELANEINT32(0)
GISTORELANEINT32(1)
GISTORELANEINT32(2)
GISTORELANEINT32(3)

#undef GISTORELANEFLOAT32

GI_FORCEINLINE
GI_INT8_t GiReinterInt32ToInt8(GI_INT32_t Vector) {
#if defined(GI_NEON_INTRINSICS)
    return vreinterpretq_s8_s32(Vector);
#elif defined(GI_SSE2_INTRINSICS)
    return Vector;
#else
    return *(GI_INT8_t*)&Vector;
#endif
}

GI_FORCEINLINE
void GiStoreInt16(void* Buffer, GI_INT16_t Vector) {
#if defined(GI_NEON_INTRINSICS)
    vst1q_s16((int16_t*)Buffer, Vector);
#elif defined(GI_SSE2_INTRINSICS)
    _mm_storeu_si128((__m128i*)Buffer, Vector);
#else
    int16_t* ptr = (int16_t*)Buffer;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(int16_t); i++) {
        ptr[i] = Vector[i];
    }
#endif
}

GI_FORCEINLINE
void GiStoreInt8(void* Buffer, GI_INT8_t Vector) {
#if defined(GI_NEON_INTRINSICS)
    vst1q_s8((int8_t*)Buffer, Vector);
#elif defined(GI_SSE2_INTRINSICS)
    _mm_storeu_si128((__m128i*)Buffer, Vector);
#else
    int8_t* ptr = (int8_t*)Buffer;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(int8_t); i++) {
        ptr[i] = Vector[i];
    }
#endif
}

GI_FORCEINLINE
void GiStoreLowInt8(void* Buffer, GI_INT8_t Vector) {
#if defined(GI_NEON_INTRINSICS)
    vst1_s8((int8_t*)Buffer, vget_low_s8(Vector));
#elif defined(GI_SSE2_INTRINSICS)
    _mm_storel_epi64((__m128i*)Buffer, Vector);
#else
    int8_t* ptr = (int8_t*)Buffer;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / 2 / sizeof(int8_t); i++) {
        ptr[i] = Vector[i];
    }
#endif
}

GI_FORCEINLINE
void GiStoreHihgInt8(void* Buffer, GI_INT8_t Vector) {
#if defined(GI_NEON_INTRINSICS)
    vst1_s8((int8_t*)Buffer, vget_high_s8(Vector));
#elif defined(GI_SSE2_INTRINSICS)
    _mm_storel_epi64((__m128i*)Buffer, _mm_unpackhi_epi64(Vector, Vector));
#else
    int8_t* ptr = (int8_t*)Buffer;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / 2 / sizeof(int8_t); i++) {
        ptr[i] = Vector[GI_SIMD_LEN_BYTE / 2 + i];
    }
#endif
}

GI_FORCEINLINE
GI_INT32_t GiNegInt32(GI_INT32_t Vector) {
#if defined(GI_NEON32_INTRINSICS)
    return vnegq_s32(Vector);
#elif defined(GI_SSE2_INTRINSICS)
    GI_INT32_t zero = _mm_set1_epi32(0);
    return _mm_sub_epi32(zero, Vector);
#else
    return -Vector;
#endif
}

GI_FORCEINLINE
GI_INT8_t GiNegInt8(GI_INT8_t Vector) {
#if defined(GI_NEON32_INTRINSICS)
    return vnegq_s8(Vector);
#elif defined(GI_SSE2_INTRINSICS)
    GI_INT32_t zero = _mm_set1_epi8(0);
    return _mm_sub_epi8(zero, Vector);
#else
    return -Vector;
#endif
}

GI_FORCEINLINE
GI_UINT32_t GiTestAndSetUint32(GI_UINT32_t Vector1, GI_UINT32_t Vector2) {
#if defined(GI_NEON_INTRINSICS)
    return vtstq_u32(Vector1, Vector2);
#elif defined(GI_SSE2_INTRINSICS)
    GI_UINT32_t tmp = _mm_and_si128(Vector1, Vector2);
    return _mm_cmpeq_epi32(tmp, _mm_setzero_si128());
#else
    GI_UINT32_t ret;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(int32_t); i++) {
        ret[i] = Vector1[i] & Vector2[i] ? 0xFFFFFFFF : 0;
    }
    return ret;
#endif
}

GI_FORCEINLINE
GI_INT32_t GiAddInt32(GI_INT32_t Vector1, GI_INT32_t Vector2) {
#if defined(GI_NEON_INTRINSICS)
    return vaddq_s32(Vector1, Vector2);
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_add_epi32(Vector1, Vector2);
#else
    return Vector1 + Vector2;
#endif
}

GI_FORCEINLINE
GI_UINT32_t GiAddUint32(GI_UINT32_t Vector1, GI_UINT32_t Vector2) {
#if defined(GI_NEON_INTRINSICS)
    return vaddq_u32(Vector1, Vector2);
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_add_epi32(Vector1, Vector2);
#else
    return Vector1 + Vector2;
#endif
}

GI_FORCEINLINE
GI_INT16_t GiAddInt16(GI_INT16_t Vector1, GI_INT16_t Vector2) {
#if defined(GI_NEON_INTRINSICS)
    return vaddq_s16(Vector1, Vector2);
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_add_epi16(Vector1, Vector2);
#else
    return Vector1 + Vector2;
#endif
}

GI_FORCEINLINE
GI_INT8_t GiAddInt8(GI_INT8_t Vector1, GI_INT8_t Vector2) {
#if defined(GI_NEON_INTRINSICS)
    return vaddq_s8(Vector1, Vector2);
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_add_epi8(Vector1, Vector2);
#else
    return Vector1 + Vector2;
#endif
}

GI_FORCEINLINE
GI_INT32_t GiSubtractInt32(GI_INT32_t Vector1, GI_INT32_t Vector2) {
#if defined(GI_NEON_INTRINSICS)
    return vsubq_s32(Vector1, Vector2);
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_sub_epi32(Vector1, Vector2);
#else
    return Vector1 - Vector2;
#endif
}

GI_FORCEINLINE
GI_UINT32_t GiSubtractUint32(GI_UINT32_t Vector1, GI_UINT32_t Vector2) {
#if defined(GI_NEON_INTRINSICS)
    return vsubq_u32(Vector1, Vector2);
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_sub_epi32(Vector1, Vector2);
#else
    return Vector1 - Vector2;
#endif
}

GI_FORCEINLINE
GI_INT8_t GiSubtractInt8(GI_INT8_t Vector1, GI_INT8_t Vector2) {
#if defined(GI_NEON_INTRINSICS)
    return vsubq_s8(Vector1, Vector2);
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_sub_epi8(Vector1, Vector2);
#else
    return Vector1 - Vector2;
#endif
}

GI_FORCEINLINE
GI_INT32_t GiMultiplyInt32(GI_INT32_t Vector1, GI_INT32_t Vector2) {
#if defined(GI_NEON_INTRINSICS)
    return vmulq_s32(Vector1, Vector2);
#elif defined(GI_SSE2_INTRINSICS)
    GI_FLOAT32_t v0 = _mm_cvtepi32_ps(Vector1);
    GI_FLOAT32_t v1 = _mm_cvtepi32_ps(Vector2);
    return _mm_cvttps_epi32(_mm_mul_ps(v0, v1));
#else
    return Vector1 * Vector2;
#endif
}
//! in x86, there is no int multiply, so implement it naive
GI_FORCEINLINE
GI_INT8_t GiMultiplyInt8(GI_INT8_t Vector1, GI_INT8_t Vector2) {
#if defined(GI_NEON_INTRINSICS)
    return vmulq_s8(Vector1, Vector2);
#elif defined(GI_SSE2_INTRINSICS)
    int8_t v1[16], v2[16], res[16];
    _mm_storeu_si128((__m128i*)v1, Vector1);
    _mm_storeu_si128((__m128i*)v2, Vector2);
    for (size_t id = 0; id < 16; id++) {
        res[id] = v1[id] * v2[id];
    }
    return _mm_loadu_si128((__m128i*)res);
#else
    return Vector1 * Vector2;
#endif
}

GI_FORCEINLINE
GI_INT32_t GiMultiplyAddInt32(
        GI_INT32_t Vector1, GI_INT32_t Vector2, GI_INT32_t Vector3) {
#if defined(GI_NEON_INTRINSICS)
    return vmlaq_s32(Vector1, Vector2, Vector3);
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_add_epi32(Vector1, GiMultiplyInt32(Vector2, Vector3));
#else
    return Vector1 + Vector2 * Vector3;
#endif
}

GI_FORCEINLINE
GI_INT8_t GiMultiplyAddInt8(GI_INT8_t Vector1, GI_INT8_t Vector2, GI_INT8_t Vector3) {
#if defined(GI_NEON_INTRINSICS)
    return vmlaq_s8(Vector1, Vector2, Vector3);
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_add_epi8(Vector1, GiMultiplyInt8(Vector2, Vector3));
#else
    return Vector1 + Vector2 * Vector3;
#endif
}

GI_FORCEINLINE
GI_INT8_t GiAndInt8(GI_INT8_t Vector1, GI_INT8_t Vector2) {
#if defined(GI_NEON_INTRINSICS)
    return vandq_s8(Vector1, Vector2);
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_and_si128(Vector1, Vector2);
#else
    return Vector1 & Vector2;
#endif
}

GI_FORCEINLINE
GI_UINT32_t GiEOrUint32(GI_UINT32_t Vector1, GI_UINT32_t Vector2) {
#if defined(GI_NEON_INTRINSICS)
    return veorq_u32(Vector1, Vector2);
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_xor_si128(Vector1, Vector2);
#else
    return Vector1 ^ Vector2;
#endif
}

GI_FORCEINLINE
GI_INT8_t GiOrInt8(GI_INT8_t Vector1, GI_INT8_t Vector2) {
#if defined(GI_NEON_INTRINSICS)
    return vorrq_s8(Vector1, Vector2);
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_or_si128(Vector1, Vector2);
#else
    return Vector1 | Vector2;
#endif
}

GI_FORCEINLINE
GI_INT8_t GiAndNotInt8(GI_INT8_t VectorNot, GI_INT8_t Vector) {
#if defined(GI_NEON_INTRINSICS)
    return vandq_s8(vmvnq_s8(VectorNot), Vector);
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_andnot_si128(VectorNot, Vector);
#else
    GI_INT8_t Not = ~VectorNot;
    return (Not & Vector);
#endif
}

GI_FORCEINLINE
GI_INT8_t GiXorInt8(GI_INT8_t Vector1, GI_INT8_t Vector2) {
#if defined(GI_NEON_INTRINSICS)
    return veorq_s8(Vector1, Vector2);
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_xor_si128(Vector1, Vector2);
#else
    return Vector1 ^ Vector2;
#endif
}

GI_FORCEINLINE
GI_INT32_t GiShiftLeft23Int32(GI_INT32_t Vector) {
#if defined(GI_NEON_INTRINSICS)
    return vshlq_n_s32(Vector, 23);
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_slli_epi32(Vector, 23);
#else
    return Vector << 23;
#endif
}

GI_FORCEINLINE
GI_INT32_t GiShiftRight23Int32(GI_INT32_t Vector) {
#if defined(GI_NEON_INTRINSICS)
    return vshrq_n_s32(Vector, 23);
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_srai_epi32(Vector, 23);
#else
    return Vector >> 23;
#endif
}

GI_FORCEINLINE
GI_INT32_t GiBlendInt32(GI_INT32_t Vector1, GI_INT32_t Vector2, GI_INT32_t Selection) {
    return GiOrInt32(GiAndInt32(Vector2, Selection), GiAndNotInt32(Selection, Vector1));
}

GI_FORCEINLINE
GI_INT8_t GiBlendInt8(GI_INT8_t Vector1, GI_INT8_t Vector2, GI_INT8_t Selection) {
    return GiOrInt8(GiAndInt8(Vector2, Selection), GiAndNotInt8(Selection, Vector1));
}

GI_FORCEINLINE
GI_INT32_t GiAbsInt32(GI_INT32_t Vector) {
#if defined(GI_NEON_INTRINSICS)
    return vabsq_s32(Vector);
#elif defined(GI_SSE42_INTRINSICS)
    return _mm_abs_epi32(Vector);
#else
    GI_INT32_t ret;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(int32_t); i++) {
        ret[i] = Vector[i] > 0 ? Vector[i] : -Vector[i];
    }
    return ret;
#endif
}

GI_FORCEINLINE
GI_INT16_t GiAbsInt16(GI_INT16_t Vector) {
#if defined(GI_NEON_INTRINSICS)
    return vabsq_s16(Vector);
#elif defined(GI_SSE42_INTRINSICS)
    return _mm_abs_epi16(Vector);
#else
    GI_INT16_t ret;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(int16_t); i++) {
        ret[i] = Vector[i] > 0 ? Vector[i] : -Vector[i];
    }
    return ret;
#endif
}

GI_FORCEINLINE
GI_INT8_t GiAbsInt8(GI_INT8_t Vector) {
#if defined(GI_NEON_INTRINSICS)
    return vabsq_s8(Vector);
#elif defined(GI_SSE42_INTRINSICS)
    return _mm_abs_epi8(Vector);
#else
    GI_INT8_t ret;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(int8_t); i++) {
        ret[i] = Vector[i] > 0 ? Vector[i] : -Vector[i];
    }
    return ret;
#endif
}

GI_FORCEINLINE
GI_INT32_t GiMaximumInt32(GI_INT32_t Vector1, GI_INT32_t Vector2) {
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
GI_INT32_t GiMinimumInt32(GI_INT32_t Vector1, GI_INT32_t Vector2) {
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
GI_INT8_t GiBlendInt8x16(GI_INT8_t Vector1, GI_INT8_t Vector2, GI_INT8_t Selection) {
    return GiOrInt8(GiAndInt8(Vector2, Selection), GiAndNotInt8(Selection, Vector1));
}

GI_FORCEINLINE
GI_INT8_t GiMaximumInt8(GI_INT8_t Vector1, GI_INT8_t Vector2) {
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
GI_INT8_t GiMinimumInt8(GI_INT8_t Vector1, GI_INT8_t Vector2) {
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
GI_INT16_t GiMoveHighLongInt8(GI_INT8_t Vector) {
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
    return _mm_loadu_si128((__m128i*)data);
#else
    GI_INT16_t ret;
    int8_t* data = (int8_t*)&Vector;
    size_t half_length = GI_SIMD_LEN_BYTE / 2 / sizeof(int8_t);
    for (size_t i = 0; i < half_length; i++) {
        ret[i] = data[i + half_length];
    }
    return ret;
#endif
}

GI_FORCEINLINE
GI_INT16_t GiMoveLowLongInt8(GI_INT8_t Vector) {
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
    return _mm_loadu_si128((__m128i*)data);
#else
    GI_INT16_t ret;
    size_t half_length = GI_SIMD_LEN_BYTE / 2 / sizeof(int8_t);
    for (size_t i = 0; i < half_length; i++) {
        ret[i] = Vector[i];
    }
    return ret;
#endif
}

GI_FORCEINLINE
GI_INT32_t GiMoveHighLongInt16(GI_INT16_t Vector) {
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
    return _mm_loadu_si128((__m128i*)data);
#else
    GI_INT32_t ret;
    size_t half_length = GI_SIMD_LEN_BYTE / 2 / sizeof(int16_t);
    for (size_t i = 0; i < half_length; i++) {
        ret[i] = Vector[half_length + i];
    }
    return ret;
#endif
}

GI_FORCEINLINE
GI_INT32_t GiMoveLowLongInt16(GI_INT16_t Vector) {
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
    return _mm_loadu_si128((__m128i*)data);
#else
    GI_INT32_t ret;
    size_t half_length = GI_SIMD_LEN_BYTE / 2 / sizeof(int16_t);
    for (size_t i = 0; i < half_length; i++) {
        ret[i] = Vector[i];
    }
    return ret;
#endif
}

GI_FORCEINLINE
int32_t GiReduceAddInt8(GI_INT8_t Vector) {
#if defined(GI_NEON64_INTRINSICS)
    return vaddlvq_s8(Vector);
#elif defined(GI_NEON32_INTRINSICS)
    int32x4_t sum = vpaddlq_s16(vpaddlq_s8(Vector));
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
    __m64 low = _mm_movepi64_pi64(Vector);
    __m64 high = _mm_movepi64_pi64(_mm_unpackhi_epi64(Vector, Vector));
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
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(int8_t); i++) {
        sum += Vector[i];
    }
    return sum;
#endif
}

GI_FORCEINLINE
int8_t GiReduceMaxInt8(GI_INT8_t Vector) {
#if defined(GI_NEON64_INTRINSICS)
    return vmaxvq_s8(Vector);
#elif defined(GI_NEON32_INTRINSICS)
    int8x8_t VectorLow = vget_low_s8(Vector);
    int8x8_t VectorHigh = vget_high_s8(Vector);
    VectorLow = vpmax_s8(VectorLow, VectorHigh);
    VectorLow = vpmax_s8(VectorLow, VectorLow);
    VectorLow = vpmax_s8(VectorLow, VectorLow);
    VectorLow = vpmax_s8(VectorLow, VectorLow);
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
    __m64 low = _mm_movepi64_pi64(Vector);
    __m64 high = _mm_movepi64_pi64(_mm_unpackhi_epi64(Vector, Vector));
    __m128 v0 = _mm_cvtpi8_ps(low);
    __m128 v1 = _mm_cvtpi8_ps(_mm_unpackhi_pi32(low, low));
    __m128 v2 = _mm_cvtpi8_ps(high);
    __m128 v3 = _mm_cvtpi8_ps(_mm_unpackhi_pi32(high, high));
    __m128 max0 = _mm_max_ps(v0, v1);
    __m128 max1 = _mm_max_ps(v2, v3);
    __m128 max = _mm_max_ps(max0, max1);
    float ret0 = _mm_cvtss_f32(max);
    float ret1 = _mm_cvtss_f32(_mm_shuffle_ps(max, max, _MM_SHUFFLE(1, 1, 1, 1)));
    float ret2 = _mm_cvtss_f32(_mm_shuffle_ps(max, max, _MM_SHUFFLE(2, 2, 2, 2)));
    float ret3 = _mm_cvtss_f32(_mm_shuffle_ps(max, max, _MM_SHUFFLE(3, 3, 3, 3)));
    return (int8_t)(Max(Max(ret0, ret1), Max(ret2, ret3)));
#else
    int8_t max = Vector[0];
    for (size_t i = 1; i < GI_SIMD_LEN_BYTE / sizeof(int8_t); i++) {
        max = Max(max, Vector[i]);
    }
    return max;
#endif
}

GI_FORCEINLINE
int8_t GiReduceMinInt8(GI_INT8_t Vector) {
#if defined(GI_NEON64_INTRINSICS)
    return vminvq_s8(Vector);
#elif defined(GI_NEON32_INTRINSICS)
    int8x8_t VectorLow = vget_low_s8(Vector);
    int8x8_t VectorHigh = vget_high_s8(Vector);
    VectorLow = vpmin_s8(VectorLow, VectorHigh);
    VectorLow = vpmin_s8(VectorLow, VectorLow);
    VectorLow = vpmin_s8(VectorLow, VectorLow);
    VectorLow = vpmin_s8(VectorLow, VectorLow);
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
    __m64 low = _mm_movepi64_pi64(Vector);
    __m64 high = _mm_movepi64_pi64(_mm_unpackhi_epi64(Vector, Vector));
    __m128 v0 = _mm_cvtpi8_ps(low);
    __m128 v1 = _mm_cvtpi8_ps(_mm_unpackhi_pi32(low, low));
    __m128 v2 = _mm_cvtpi8_ps(high);
    __m128 v3 = _mm_cvtpi8_ps(_mm_unpackhi_pi32(high, high));
    __m128 min0 = _mm_min_ps(v0, v1);
    __m128 min1 = _mm_min_ps(v2, v3);
    __m128 min = _mm_min_ps(min0, min1);
    float ret0 = _mm_cvtss_f32(min);
    float ret1 = _mm_cvtss_f32(_mm_shuffle_ps(min, min, _MM_SHUFFLE(1, 1, 1, 1)));
    float ret2 = _mm_cvtss_f32(_mm_shuffle_ps(min, min, _MM_SHUFFLE(2, 2, 2, 2)));
    float ret3 = _mm_cvtss_f32(_mm_shuffle_ps(min, min, _MM_SHUFFLE(3, 3, 3, 3)));
    return (int8_t)(Min(Min(ret0, ret1), Min(ret2, ret3)));
#else
    int8_t min = Vector[0];
    for (size_t i = 1; i < GI_SIMD_LEN_BYTE / sizeof(int8_t); i++) {
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
GI_INT8_t GiCvtFromFloat32ToInt8(GI_FLOAT32_t src) {
#if defined(GI_NEON_INTRINSICS)
#if __ARM_ARCH >= 8
    int32x4_t vres0 = vcvtaq_s32_f32(src);
    int16x8_t mid_s16 = vcombine_s16(vqmovn_s32(vres0), vqmovn_s32(vres0));
    return vcombine_s8(vqmovn_s16(mid_s16), vqmovn_s16(mid_s16));
#else
    float32x4_t vinc0 = vbslq_f32(vcgeq_f32(src, vfzero), vfhalf, vfneg_half);
    int32x4_t vres0 = vcvtq_s32_f32(vaddq_f32(src, vinc0));
    int16x8_t mid_s16 = vcombine_s16(vqmovn_s32(vres0), vqmovn_s32(vres0));
    return vcombine_s8(vqmovn_s16(mid_s16), vqmovn_s16(mid_s16));
#endif
#elif defined(GI_SSE42_INTRINSICS)
    __m128 vinc0 = _mm_blendv_ps(vfneg_half, vfhalf, _mm_cmpge_ps(src, vfzero));
    __m128 vres0 = _mm_add_ps(src, vinc0);
    vres0 = _mm_round_ps(vres0, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    vres0 = _mm_min_ps(_mm_max_ps(vres0, vfmin_int8), vfmax_int8);

    __m128i vepi32_0 = _mm_cvtps_epi32(vres0);
    __m128i vepi16 = _mm_packs_epi32(vepi32_0, vepi32_0);
    __m128i vepi8 = _mm_packs_epi16(vepi16, vepi16);
    return vepi8;
#else
    GI_INT8_t ret;
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
GI_INT8_t GiCvtFromFloat32V2ToInt8(GI_FLOAT32_V2_t vsrc) {
#if defined(GI_NEON_INTRINSICS)
#if __ARM_ARCH >= 8
    int32x4_t vres0 = vcvtaq_s32_f32(vsrc.val[0]);
    int32x4_t vres1 = vcvtaq_s32_f32(vsrc.val[1]);
    int8x8_t mid1 = vqmovn_s16(vcombine_s16(vqmovn_s32(vres0), vqmovn_s32(vres1)));
    return vcombine_s8(mid1, mid1);
#else
    float32x4_t vinc0 = vbslq_f32(vcgeq_f32(vsrc.val[0], vfzero), vfhalf, vfneg_half);
    float32x4_t vinc1 = vbslq_f32(vcgeq_f32(vsrc.val[1], vfzero), vfhalf, vfneg_half);
    int32x4_t vres0 = vcvtq_s32_f32(vaddq_f32(vsrc.val[0], vinc0));
    int32x4_t vres1 = vcvtq_s32_f32(vaddq_f32(vsrc.val[1], vinc1));
    int8x8_t mid1 = vqmovn_s16(vcombine_s16(vqmovn_s32(vres0), vqmovn_s32(vres1)));
    return vcombine_s8(mid1, mid1);
#endif
#elif defined(GI_SSE42_INTRINSICS)
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
    GI_INT8_t ret;
    int length = GI_SIMD_LEN_BYTE / sizeof(float);
    for (int i = 0; i < 2 * length; i++) {
        ret[i] = Saturate(round(vsrc.val[i / length][i % length]), -128, 127);
    }
    return ret;
#endif
}

GI_FORCEINLINE
GI_INT8_t GiCvtFromFloat32V4ToInt8(GI_FLOAT32_V4_t vsrc) {
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
    float32x4_t vfzero = vdupq_n_f32(0.f);
    float32x4_t vfhalf = vdupq_n_f32(0.5f);
    float32x4_t vfneg_half = vdupq_n_f32(-0.5f);
    float32x4_t vinc0 = vbslq_f32(vcgeq_f32(vsrc.val[0], vfzero), vfhalf, vfneg_half);
    float32x4_t vinc1 = vbslq_f32(vcgeq_f32(vsrc.val[1], vfzero), vfhalf, vfneg_half);
    float32x4_t vinc2 = vbslq_f32(vcgeq_f32(vsrc.val[2], vfzero), vfhalf, vfneg_half);
    float32x4_t vinc3 = vbslq_f32(vcgeq_f32(vsrc.val[3], vfzero), vfhalf, vfneg_half);
    int32x4_t vres0 = vcvtq_s32_f32(vaddq_f32(vsrc.val[0], vinc0));
    int32x4_t vres1 = vcvtq_s32_f32(vaddq_f32(vsrc.val[1], vinc1));
    int32x4_t vres2 = vcvtq_s32_f32(vaddq_f32(vsrc.val[2], vinc2));
    int32x4_t vres3 = vcvtq_s32_f32(vaddq_f32(vsrc.val[3], vinc3));
    int8x8_t mid1 = vqmovn_s16(vcombine_s16(vqmovn_s32(vres0), vqmovn_s32(vres1)));
    int8x8_t mid2 = vqmovn_s16(vcombine_s16(vqmovn_s32(vres2), vqmovn_s32(vres3)));
    return vcombine_s8(mid1, mid2);
#endif
#elif defined(GI_SSE42_INTRINSICS)
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
    GI_INT8_t ret;
    int length = GI_SIMD_LEN_BYTE / sizeof(float);
    for (int i = 0; i < 4 * length; i++) {
        ret[i] = Saturate(round(vsrc.val[i / length][i % length]), -128, 127);
    }
    return ret;
#endif
}

// vim: syntax=cpp.doxygen
