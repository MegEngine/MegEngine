#pragma once

#include "gi_common.h"

GI_FORCEINLINE
GI_INT32_t GiReinterpretInt8AsInt32(GI_INT8_t In) {
#if defined(GI_NEON_INTRINSICS)
    return vreinterpretq_s32_s8(In);
#elif defined(GI_SSE2_INTRINSICS)
    return (GI_INT32_t)In;
#elif defined(GI_RVV_INTRINSICS)
    return vreinterpret_v_i8m1_i32m1(In);
#else
    return (GI_INT32_t)In;
#endif
}

GI_FORCEINLINE
GI_UINT32_t GiBroadcastUint32(int32_t Value) {
#if defined(GI_NEON_INTRINSICS)
    return vdupq_n_u32(Value);
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_set1_epi32(Value);
#elif defined(GI_RVV_INTRINSICS)
    return vmv_v_x_u32m1(Value, GI_SIMD_LEN_BYTE / sizeof(int32_t));
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
#elif defined(GI_RVV_INTRINSICS)
    return vle32_v_i32m1((int32_t*)Buffer, GI_SIMD_LEN_BYTE / sizeof(int32_t));
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
#elif defined(GI_RVV_INTRINSICS)
    return vle16_v_i16m1((int16_t*)Buffer, GI_SIMD_LEN_BYTE / sizeof(int16_t));
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
#elif defined(GI_RVV_INTRINSICS)
    return vle8_v_i8m1((int8_t*)Buffer, GI_SIMD_LEN_BYTE / sizeof(int8_t));
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
GI_INT8_V2_t GiLoadUzipInt8V2(const void* Buffer) {
#if defined(GI_NEON_INTRINSICS)
    return vld2q_s8((int8_t*)Buffer);
#elif defined(GI_SSE42_INTRINSICS)
    GI_INT8_t v0, v1;
    v0 = _mm_loadu_si128((const __m128i*)Buffer);
    v1 = _mm_loadu_si128((const __m128i*)((int8_t*)Buffer + 16));
    GI_INT8_V2_t ret;
    v0 = _mm_shuffle_epi8(
            v0, _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15));
    v1 = _mm_shuffle_epi8(
            v1, _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15));
    ret.val[0] = _mm_unpacklo_epi64(v0, v1);
    ret.val[1] = _mm_unpackhi_epi64(v0, v1);
    return ret;
#elif defined(GI_RVV_INTRINSICS)
    return vlseg2e8_v_i8m1x2((int8_t*)Buffer, GI_SIMD_LEN_BYTE / sizeof(int8_t));
#else
    int8_t data[2 * GI_SIMD_LEN_BYTE];
    const int8_t* ptr = (int8_t*)Buffer;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(int8_t); i++) {
        data[i] = ptr[2 * i];
        data[GI_SIMD_LEN_BYTE + i] = ptr[2 * i + 1];
    }
    GI_INT8_V2_t ret;
    ret.val[0] = GiLoadInt8(data);
    ret.val[1] = GiLoadInt8(data + GI_SIMD_LEN_BYTE);
    return ret;
#endif
}

GI_FORCEINLINE
GI_INT8_V3_t GiLoadUzipInt8V3(const void* Buffer) {
#if defined(GI_NEON_INTRINSICS)
    return vld3q_s8((int8_t*)Buffer);
#elif defined(GI_SSE2_INTRINSICS)
    GI_INT8_V3_t ret;
    __m128i t00 = _mm_loadu_si128((const __m128i*)Buffer);
    __m128i t01 = _mm_loadu_si128((const __m128i*)((uint8_t*)Buffer + 16));
    __m128i t02 = _mm_loadu_si128((const __m128i*)((uint8_t*)Buffer + 32));

    __m128i t10 = _mm_unpacklo_epi8(t00, _mm_unpackhi_epi64(t01, t01));
    __m128i t11 = _mm_unpacklo_epi8(_mm_unpackhi_epi64(t00, t00), t02);
    __m128i t12 = _mm_unpacklo_epi8(t01, _mm_unpackhi_epi64(t02, t02));

    __m128i t20 = _mm_unpacklo_epi8(t10, _mm_unpackhi_epi64(t11, t11));
    __m128i t21 = _mm_unpacklo_epi8(_mm_unpackhi_epi64(t10, t10), t12);
    __m128i t22 = _mm_unpacklo_epi8(t11, _mm_unpackhi_epi64(t12, t12));

    __m128i t30 = _mm_unpacklo_epi8(t20, _mm_unpackhi_epi64(t21, t21));
    __m128i t31 = _mm_unpacklo_epi8(_mm_unpackhi_epi64(t20, t20), t22);
    __m128i t32 = _mm_unpacklo_epi8(t21, _mm_unpackhi_epi64(t22, t22));

    ret.val[0] = _mm_unpacklo_epi8(t30, _mm_unpackhi_epi64(t31, t31));
    ret.val[1] = _mm_unpacklo_epi8(_mm_unpackhi_epi64(t30, t30), t32);
    ret.val[2] = _mm_unpacklo_epi8(t31, _mm_unpackhi_epi64(t32, t32));
    return ret;
#elif defined(GI_RVV_INTRINSICS)
    return vlseg3e8_v_i8m1x3((int8_t*)Buffer, GI_SIMD_LEN_BYTE / sizeof(int8_t));
#else
    GI_INT8_V3_t ret;
    GI_INT8_t ret0, ret1, ret2;
    size_t i, i3;
    for (i = i3 = 0; i < GI_SIMD_LEN_BYTE / sizeof(int8_t); i++, i3 += 3) {
        ret0[i] = *((int8_t*)Buffer + i3);
        ret1[i] = *((int8_t*)Buffer + i3 + 1);
        ret2[i] = *((int8_t*)Buffer + i3 + 2);
    }
    ret.val[0] = ret0;
    ret.val[1] = ret1;
    ret.val[2] = ret2;

    return ret;
#endif
}

GI_FORCEINLINE
GI_INT8_V4_t GiLoadUzipInt8V4(const void* Buffer) {
#if defined(GI_NEON_INTRINSICS)
    return vld4q_s8((int8_t*)Buffer);
#elif defined(GI_SSE2_INTRINSICS)
    GI_INT8_V4_t v;
    __m128i tmp3, tmp2, tmp1, tmp0;

    v.val[0] = _mm_loadu_si128((const __m128i*)Buffer);
    v.val[1] = _mm_loadu_si128((const __m128i*)((int8_t*)Buffer + 16));
    v.val[2] = _mm_loadu_si128((const __m128i*)((int8_t*)Buffer + 32));
    v.val[3] = _mm_loadu_si128((const __m128i*)((int8_t*)Buffer + 48));

    tmp0 = _mm_unpacklo_epi8(v.val[0], v.val[1]);
    tmp1 = _mm_unpacklo_epi8(v.val[2], v.val[3]);
    tmp2 = _mm_unpackhi_epi8(v.val[0], v.val[1]);
    tmp3 = _mm_unpackhi_epi8(v.val[2], v.val[3]);

    v.val[0] = _mm_unpacklo_epi8(tmp0, tmp2);
    v.val[1] = _mm_unpackhi_epi8(tmp0, tmp2);
    v.val[2] = _mm_unpacklo_epi8(tmp1, tmp3);
    v.val[3] = _mm_unpackhi_epi8(tmp1, tmp3);

    tmp0 = _mm_unpacklo_epi32(v.val[0], v.val[2]);
    tmp1 = _mm_unpackhi_epi32(v.val[0], v.val[2]);
    tmp2 = _mm_unpacklo_epi32(v.val[1], v.val[3]);
    tmp3 = _mm_unpackhi_epi32(v.val[1], v.val[3]);

    v.val[0] = _mm_unpacklo_epi8(tmp0, tmp2);
    v.val[1] = _mm_unpackhi_epi8(tmp0, tmp2);
    v.val[2] = _mm_unpacklo_epi8(tmp1, tmp3);
    v.val[3] = _mm_unpackhi_epi8(tmp1, tmp3);
    return v;
#elif defined(GI_RVV_INTRINSICS)
    return vlseg4e8_v_i8m1x4((int8_t*)Buffer, GI_SIMD_LEN_BYTE / sizeof(int8_t));
#else
    GI_INT8_V4_t ret;
    const int8_t* ptr = (int8_t*)Buffer;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(int8_t); i++) {
        ret.val[0][i] = ptr[4 * i];
        ret.val[1][i] = ptr[4 * i + 1];
        ret.val[2][i] = ptr[4 * i + 2];
        ret.val[3][i] = ptr[4 * i + 3];
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
#elif defined(GI_RVV_INTRINSICS)
    vse32_v_i32m1((int32_t*)Buffer, Vector, GI_SIMD_LEN_BYTE / sizeof(int32_t));
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
#elif defined(GI_RVV_INTRINSICS)

#define GISTORELANEINT32(i)                                                      \
    GI_FORCEINLINE void GiStoreLane##i##Int32(void* Buffer, GI_INT32_t Vector) { \
        int32_t t[GI_SIMD_LEN_BYTE / sizeof(int32_t)];                           \
        vse32_v_i32m1(t, Vector, GI_SIMD_LEN_BYTE / sizeof(int32_t));            \
        *((int32_t*)Buffer) = t[i];                                              \
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
#elif defined(GI_RVV_INTRINSICS)
    return vreinterpret_v_i32m1_i8m1(Vector);
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
#elif defined(GI_RVV_INTRINSICS)
    vse16_v_i16m1((int16_t*)Buffer, Vector, GI_SIMD_LEN_BYTE / sizeof(int16_t));
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
#elif defined(GI_RVV_INTRINSICS)
    vse8_v_i8m1((int8_t*)Buffer, Vector, GI_SIMD_LEN_BYTE / sizeof(int8_t));
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
#elif defined(GI_RVV_INTRINSICS)
    vse8_v_i8m1((int8_t*)Buffer, Vector, GI_SIMD_LEN_BYTE / sizeof(int8_t) / 2);
#else
    int8_t* ptr = (int8_t*)Buffer;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / 2 / sizeof(int8_t); i++) {
        ptr[i] = Vector[i];
    }
#endif
}

GI_FORCEINLINE
void GiStoreHighInt8(void* Buffer, GI_INT8_t Vector) {
#if defined(GI_NEON_INTRINSICS)
    vst1_s8((int8_t*)Buffer, vget_high_s8(Vector));
#elif defined(GI_SSE2_INTRINSICS)
    _mm_storel_epi64((__m128i*)Buffer, _mm_unpackhi_epi64(Vector, Vector));
#elif defined(GI_RVV_INTRINSICS)
    vuint8m1_t index;
#if GI_SIMD_LEN_BYTE == 16
    uint8_t index_128[16] = {8, 9, 10, 11, 12, 13, 14, 15, 0, 1, 2, 3, 4, 5, 6, 7};
    index = vle8_v_u8m1(index_128, 16);
#else
    uint8_t* index_p = (uint8_t*)&index;
    int32_t offset = GI_SIMD_LEN_BYTE / sizeof(int8_t) / 2;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(int8_t) / 2; i++) {
        index_p[i] = offset + i;
        index_p[offset + i] = i;
    }
#endif
    vint8m1_t g_d = vrgather_vv_i8m1(Vector, index, GI_SIMD_LEN_BYTE / sizeof(int8_t));
    vse8_v_i8m1((int8_t*)Buffer, g_d, GI_SIMD_LEN_BYTE / sizeof(int8_t) / 2);
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
#elif defined(GI_RVV_INTRINSICS)
    return vneg_v_i32m1(Vector, GI_SIMD_LEN_BYTE / sizeof(int32_t));
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
#elif defined(GI_RVV_INTRINSICS)
    return vneg_v_i8m1(Vector, GI_SIMD_LEN_BYTE / sizeof(int8_t));
#else
    return -Vector;
#endif
}

GI_FORCEINLINE
GI_UINT32_t GiTestAndSetUint32(GI_UINT32_t Vector1, GI_UINT32_t Vector2) {
#if defined(GI_NEON_INTRINSICS)
    return vtstq_u32(Vector1, Vector2);
#elif defined(GI_SSE2_INTRINSICS)
    __m128i zero, one, res;
    zero = _mm_setzero_si128();
    one = _mm_cmpeq_epi8(zero, zero);
    res = _mm_and_si128(Vector1, Vector2);
    res = _mm_cmpeq_epi32(res, zero);
    return _mm_xor_si128(res, one);
#elif defined(GI_RVV_INTRINSICS)
    //! rvv uint32_t mask only use bit 0 and 1, imp with naive
    GI_UINT32_FIXLEN_t a = GiUint32Type2FixLenType(Vector1);
    GI_UINT32_FIXLEN_t b = GiUint32Type2FixLenType(Vector2);
    GI_UINT32_FIXLEN_t ret;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(int32_t); i++) {
        ret[i] = a[i] & b[i] ? 0xFFFFFFFF : 0;
    }
    return GiFixLenType2GiUint32Type(ret);
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
#elif defined(GI_RVV_INTRINSICS)
    return vadd_vv_i32m1(Vector1, Vector2, GI_SIMD_LEN_BYTE / sizeof(int32_t));
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
#elif defined(GI_RVV_INTRINSICS)
    return vadd_vv_u32m1(Vector1, Vector2, GI_SIMD_LEN_BYTE / sizeof(uint32_t));
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
#elif defined(GI_RVV_INTRINSICS)
    return vadd_vv_i16m1(Vector1, Vector2, GI_SIMD_LEN_BYTE / sizeof(int16_t));
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
#elif defined(GI_RVV_INTRINSICS)
    return vadd_vv_i8m1(Vector1, Vector2, GI_SIMD_LEN_BYTE / sizeof(int8_t));
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
#elif defined(GI_RVV_INTRINSICS)
    return vsub_vv_i32m1(Vector1, Vector2, GI_SIMD_LEN_BYTE / sizeof(int32_t));
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
#elif defined(GI_RVV_INTRINSICS)
    return vsub_vv_u32m1(Vector1, Vector2, GI_SIMD_LEN_BYTE / sizeof(uint32_t));
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
#elif defined(GI_RVV_INTRINSICS)
    return vsub_vv_i8m1(Vector1, Vector2, GI_SIMD_LEN_BYTE / sizeof(int8_t));
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
#elif defined(GI_RVV_INTRINSICS)
    return vmul_vv_i32m1(Vector1, Vector2, GI_SIMD_LEN_BYTE / sizeof(int32_t));
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
#elif defined(GI_RVV_INTRINSICS)
    return vmul_vv_i8m1(Vector1, Vector2, GI_SIMD_LEN_BYTE / sizeof(int8_t));
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
#elif defined(GI_RVV_INTRINSICS)
    return vmadd_vv_i32m1(
            Vector2, Vector3, Vector1, GI_SIMD_LEN_BYTE / sizeof(int32_t));
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
#elif defined(GI_RVV_INTRINSICS)
    return vmadd_vv_i8m1(Vector2, Vector3, Vector1, GI_SIMD_LEN_BYTE / sizeof(int8_t));
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
#elif defined(GI_RVV_INTRINSICS)
    return vand_vv_i8m1(Vector1, Vector2, GI_SIMD_LEN_BYTE / sizeof(int8_t));
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
#elif defined(GI_RVV_INTRINSICS)
    return vxor_vv_u32m1(Vector1, Vector2, GI_SIMD_LEN_BYTE / sizeof(uint32_t));
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
#elif defined(GI_RVV_INTRINSICS)
    return vor_vv_i8m1(Vector1, Vector2, GI_SIMD_LEN_BYTE / sizeof(int8_t));
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
#elif defined(GI_RVV_INTRINSICS)
    GI_INT8_t not_v = vnot_v_i8m1(VectorNot, GI_SIMD_LEN_BYTE / sizeof(int8_t));
    return vand_vv_i8m1(not_v, Vector, GI_SIMD_LEN_BYTE / sizeof(int8_t));
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
#elif defined(GI_RVV_INTRINSICS)
    return vxor_vv_i8m1(Vector1, Vector2, GI_SIMD_LEN_BYTE / sizeof(int8_t));
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
#elif defined(GI_RVV_INTRINSICS)
    return vsll_vx_i32m1(Vector, 23, GI_SIMD_LEN_BYTE / sizeof(int32_t));
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
#elif defined(GI_RVV_INTRINSICS)
    return vsra_vx_i32m1(Vector, 23, GI_SIMD_LEN_BYTE / sizeof(int32_t));
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
#elif defined(GI_RVV_INTRINSICS)
    //! rvv do not have int abs now
    GI_INT32_t shift = vsra_vx_i32m1(Vector, 31, GI_SIMD_LEN_BYTE / sizeof(int32_t));
    GI_INT32_t t_add = vadd_vv_i32m1(Vector, shift, GI_SIMD_LEN_BYTE / sizeof(int32_t));
    return vxor_vv_i32m1(t_add, shift, GI_SIMD_LEN_BYTE / sizeof(int32_t));
#else
    GI_INT32_t ret;
    GI_INT32_NAIVE_t tmp_ret;
    GI_INT32_NAIVE_t s0;

    memcpy(&s0, &Vector, sizeof(GI_INT32_t));
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(int32_t); i++) {
        tmp_ret[i] = s0[i] > 0 ? s0[i] : -s0[i];
    }

    memcpy(&ret, &tmp_ret, sizeof(GI_INT32_t));
    return ret;
#endif
}

GI_FORCEINLINE
GI_INT16_t GiAbsInt16(GI_INT16_t Vector) {
#if defined(GI_NEON_INTRINSICS)
    return vabsq_s16(Vector);
#elif defined(GI_SSE42_INTRINSICS)
    return _mm_abs_epi16(Vector);
#elif defined(GI_RVV_INTRINSICS)
    //! rvv do not have int abs now
    GI_INT16_t shift = vsra_vx_i16m1(Vector, 15, GI_SIMD_LEN_BYTE / sizeof(int16_t));
    GI_INT16_t t_add = vadd_vv_i16m1(Vector, shift, GI_SIMD_LEN_BYTE / sizeof(int16_t));
    return vxor_vv_i16m1(t_add, shift, GI_SIMD_LEN_BYTE / sizeof(int16_t));
#else
    GI_INT16_t ret;
    GI_INT16_NAIVE_t tmp_ret;
    GI_INT16_NAIVE_t s0;

    memcpy(&s0, &Vector, sizeof(GI_INT16_t));
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(int16_t); i++) {
        tmp_ret[i] = s0[i] > 0 ? s0[i] : -s0[i];
    }
    memcpy(&ret, &tmp_ret, sizeof(GI_INT16_t));
    return ret;
#endif
}

GI_FORCEINLINE
GI_INT8_t GiAbsInt8(GI_INT8_t Vector) {
#if defined(GI_NEON_INTRINSICS)
    return vabsq_s8(Vector);
#elif defined(GI_SSE42_INTRINSICS)
    return _mm_abs_epi8(Vector);
#elif defined(GI_RVV_INTRINSICS)
    //! rvv do not have int abs now
    GI_INT8_t shift = vsra_vx_i8m1(Vector, 7, GI_SIMD_LEN_BYTE / sizeof(int8_t));
    GI_INT8_t t_add = vadd_vv_i8m1(Vector, shift, GI_SIMD_LEN_BYTE / sizeof(int8_t));
    return vxor_vv_i8m1(t_add, shift, GI_SIMD_LEN_BYTE / sizeof(int8_t));
#else
    GI_INT8_t ret;
    GI_INT8_NAIVE_t tmp_ret;
    GI_INT8_NAIVE_t s0;

    memcpy(&s0, &Vector, sizeof(GI_INT8_t));
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(int8_t); i++) {
        tmp_ret[i] = s0[i] > 0 ? s0[i] : -s0[i];
    }
    memcpy(&ret, &tmp_ret, sizeof(GI_INT8_t));
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
#elif defined(GI_RVV_INTRINSICS)
    return vmax_vv_i32m1(Vector1, Vector2, GI_SIMD_LEN_BYTE / sizeof(int32_t));
#else
    GI_INT32_t tmp;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(int32_t); i++) {
        tmp[i] = Vector1[i] > Vector2[i] ? 0xFFFFFFFF : 0;
    }
    return GiBlendInt32(Vector2, Vector1, tmp);
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
#elif defined(GI_RVV_INTRINSICS)
    return vmin_vv_i32m1(Vector1, Vector2, GI_SIMD_LEN_BYTE / sizeof(int32_t));
#else
    GI_INT32_t tmp;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(int32_t); i++) {
        tmp[i] = Vector2[i] > Vector1[i] ? 0xFFFFFFFF : 0;
    }
    return GiBlendInt32(Vector2, Vector1, tmp);
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
#elif defined(GI_RVV_INTRINSICS)
    return vmax_vv_i8m1(Vector1, Vector2, GI_SIMD_LEN_BYTE / sizeof(int8_t));
#else
    GI_INT8_t tmp;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(int8_t); i++) {
        tmp[i] = Vector1[i] > Vector2[i] ? 0xFF : 0;
    }
    return GiBlendInt8(Vector2, Vector1, tmp);
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
#elif defined(GI_RVV_INTRINSICS)
    return vmin_vv_i8m1(Vector1, Vector2, GI_SIMD_LEN_BYTE / sizeof(int8_t));
#else
    GI_INT8_t tmp;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(int8_t); i++) {
        tmp[i] = Vector2[i] > Vector1[i] ? 0xFF : 0;
    }
    return GiBlendInt8(Vector2, Vector1, tmp);
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
    for (size_t i = 0; i < 8; i++) {
        data[i] = o_data[8 + i];
    }
    return _mm_loadu_si128((__m128i*)data);
#elif defined(GI_RVV_INTRINSICS)
    vint16m2_t two = vwcvt_x_x_v_i16m2(Vector, GI_SIMD_LEN_BYTE / sizeof(int8_t));
    return vget_v_i16m2_i16m1(two, 1);
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
    for (size_t i = 0; i < 8; i++) {
        data[i] = o_data[i];
    }
    return _mm_loadu_si128((__m128i*)data);
#elif defined(GI_RVV_INTRINSICS)
    vint16m2_t two = vwcvt_x_x_v_i16m2(Vector, GI_SIMD_LEN_BYTE / sizeof(int8_t));
    return vget_v_i16m2_i16m1(two, 0);
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
    for (size_t i = 0; i < 4; i++) {
        data[i] = o_data[4 + i];
    }
    return _mm_loadu_si128((__m128i*)data);
#elif defined(GI_RVV_INTRINSICS)
    vint32m2_t two = vwcvt_x_x_v_i32m2(Vector, GI_SIMD_LEN_BYTE / sizeof(int16_t));
    return vget_v_i32m2_i32m1(two, 1);
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
    for (size_t i = 0; i < 4; i++) {
        data[i] = o_data[i];
    }
    return _mm_loadu_si128((__m128i*)data);
#elif defined(GI_RVV_INTRINSICS)
    vint32m2_t two = vwcvt_x_x_v_i32m2(Vector, GI_SIMD_LEN_BYTE / sizeof(int16_t));
    return vget_v_i32m2_i32m1(two, 0);
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
#elif defined(GI_RVV_INTRINSICS)
    vint16m1_t redsum = vundefined_i16m1();
    vint16m1_t zero = vmv_v_x_i16m1(0, GI_SIMD_LEN_BYTE / sizeof(int16_t));
    redsum = vwredsum_vs_i8m1_i16m1(
            redsum, Vector, zero, GI_SIMD_LEN_BYTE / sizeof(int8_t));
    int16_t ret = 0;
    vse16_v_i16m1(&ret, redsum, 1);
    return ret;
#else
    int16_t sum = 0;
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
#elif defined(GI_RVV_INTRINSICS)
    vint8m1_t max = vundefined_i8m1();
    vint8m1_t zero = vmv_v_x_i8m1(0, GI_SIMD_LEN_BYTE / sizeof(int8_t));
    max = vredmax_vs_i8m1_i8m1(max, Vector, zero, GI_SIMD_LEN_BYTE / sizeof(int8_t));
    int8_t ret = 0;
    vse8_v_i8m1(&ret, max, 1);
    return ret;
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
#elif defined(GI_RVV_INTRINSICS)
    vint8m1_t min = vundefined_i8m1();
    vint8m1_t zero = vmv_v_x_i8m1(0, GI_SIMD_LEN_BYTE / sizeof(int8_t));
    min = vredmin_vs_i8m1_i8m1(min, Vector, zero, GI_SIMD_LEN_BYTE / sizeof(int8_t));
    int8_t ret = 0;
    vse8_v_i8m1(&ret, min, 1);
    return ret;
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
    float32x4_t vinc0 = vbslq_f32(
            vcgeq_f32(src, GiBroadcastFloat32(0.0f)), GiBroadcastFloat32(0.5f),
            GiBroadcastFloat32(-0.5f));
    int32x4_t vres0 = vcvtq_s32_f32(vaddq_f32(src, vinc0));
    int16x8_t mid_s16 = vcombine_s16(vqmovn_s32(vres0), vqmovn_s32(vres0));
    return vcombine_s8(vqmovn_s16(mid_s16), vqmovn_s16(mid_s16));
#endif
#elif defined(GI_SSE42_INTRINSICS)
    __m128 vinc0 = _mm_blendv_ps(
            GiBroadcastFloat32(-0.5f), GiBroadcastFloat32(0.5f),
            _mm_cmpge_ps(src, GiBroadcastFloat32(0.0f)));
    __m128 vres0 = _mm_add_ps(src, vinc0);
    vres0 = _mm_round_ps(vres0, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    vres0 = _mm_min_ps(
            _mm_max_ps(vres0, GiBroadcastFloat32(-128.0f)), GiBroadcastFloat32(127.0f));

    __m128i vepi32_0 = _mm_cvtps_epi32(vres0);
    __m128i vepi16 = _mm_packs_epi32(vepi32_0, vepi32_0);
    __m128i vepi8 = _mm_packs_epi16(vepi16, vepi16);
    return vepi8;
#elif defined(GI_RVV_INTRINSICS)
    //! TODO: vfcvt_rtz_x_f_v_i32m1 is RVV 1.0 api, now xuantie D1 only support 0p7
    //! as a workaround, we imp this API by naive
    GI_INT8_NAIVE_t tmp_ret;
    GI_FLOAT32_FIXLEN_t s0 = GiFloat32Type2FixLenType(src);
    size_t length = GI_SIMD_LEN_BYTE / sizeof(float);
    for (size_t i = 0; i < length; i++) {
        int8_t data = Saturate(round(s0[i]), -128, 127);
        tmp_ret[i] = data;
        tmp_ret[length + i] = data;
        tmp_ret[2 * length + i] = data;
        tmp_ret[3 * length + i] = data;
    }
    return vle8_v_i8m1((const signed char*)&tmp_ret, GI_SIMD_LEN_BYTE / sizeof(int8_t));
#else
    GI_INT8_t ret;
    GI_INT8_NAIVE_t tmp_ret;
    GI_FLOAT32_NAIVE_t s0;
    memcpy(&s0, &src, sizeof(GI_INT32_t));
    size_t length = GI_SIMD_LEN_BYTE / sizeof(float);
    for (size_t i = 0; i < length; i++) {
        int8_t data = Saturate(round(s0[i]), -128, 127);
        tmp_ret[i] = data;
        tmp_ret[length + i] = data;
        tmp_ret[2 * length + i] = data;
        tmp_ret[3 * length + i] = data;
    }
    memcpy(&ret, &tmp_ret, sizeof(GI_INT8_t));
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
    GI_FLOAT32_t vfhalf = GiBroadcastFloat32(0.5f);
    GI_FLOAT32_t vfneg_half = GiBroadcastFloat32(-0.5f);
    float32x4_t vinc0 = vbslq_f32(
            vcgeq_f32(vsrc.val[0], GiBroadcastFloat32(0.0f)), vfhalf, vfneg_half);
    float32x4_t vinc1 = vbslq_f32(
            vcgeq_f32(vsrc.val[1], GiBroadcastFloat32(0.0f)), vfhalf, vfneg_half);
    int32x4_t vres0 = vcvtq_s32_f32(vaddq_f32(vsrc.val[0], vinc0));
    int32x4_t vres1 = vcvtq_s32_f32(vaddq_f32(vsrc.val[1], vinc1));
    int8x8_t mid1 = vqmovn_s16(vcombine_s16(vqmovn_s32(vres0), vqmovn_s32(vres1)));
    return vcombine_s8(mid1, mid1);
#endif
#elif defined(GI_SSE42_INTRINSICS)
    GI_FLOAT32_t vfhalf = GiBroadcastFloat32(0.5f);
    GI_FLOAT32_t vfneg_half = GiBroadcastFloat32(-0.5f);
    GI_FLOAT32_t vfmax_int8 = GiBroadcastFloat32(127.0f);
    __m128 vinc0 = _mm_blendv_ps(
            vfneg_half, vfhalf, _mm_cmpge_ps(vsrc.val[0], GiBroadcastFloat32(0.0f)));
    __m128 vinc1 = _mm_blendv_ps(
            vfneg_half, vfhalf, _mm_cmpge_ps(vsrc.val[1], GiBroadcastFloat32(0.0f)));

    __m128 vres0 = _mm_add_ps(vsrc.val[0], vinc0);
    __m128 vres1 = _mm_add_ps(vsrc.val[1], vinc1);

    vres0 = _mm_round_ps(vres0, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);
    vres1 = _mm_round_ps(vres1, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);

    vres0 = _mm_min_ps(_mm_max_ps(vres0, GiBroadcastFloat32(-128.0f)), vfmax_int8);
    vres1 = _mm_min_ps(_mm_max_ps(vres1, GiBroadcastFloat32(-128.0f)), vfmax_int8);

    __m128i vepi32_0 = _mm_cvtps_epi32(vres0);
    __m128i vepi32_1 = _mm_cvtps_epi32(vres1);
    __m128i vepi16_0 = _mm_packs_epi32(vepi32_0, vepi32_1);
    __m128i vepi8 = _mm_packs_epi16(vepi16_0, vepi16_0);
    return vepi8;
#elif defined(GI_RVV_INTRINSICS)
    //! TODO: vfcvt_rtz_x_f_v_i32m1 is RVV 1.0 api, now xuantie D1 only support 0p7
    //! as a workaround, we imp this API by naive
    GI_INT8_NAIVE_t tmp_ret;
    GI_FLOAT32_FIXLEN_V2_t s0 = GiFloat32Type2FixLenV2Type(vsrc);
    size_t length = GI_SIMD_LEN_BYTE / sizeof(float);
    for (size_t i = 0; i < 2 * length; i++) {
        int8_t data = Saturate(round(s0.val[i / length][i % length]), -128, 127);
        tmp_ret[i] = data;
        tmp_ret[i + length * 2] = data;
    }
    return vle8_v_i8m1((const signed char*)&tmp_ret, GI_SIMD_LEN_BYTE / sizeof(int8_t));
#else
    GI_INT8_t ret;
    GI_INT8_NAIVE_t tmp_ret;
    GI_FLOAT32_V2_NAIVE_t s0;
    memcpy(&s0, &vsrc, sizeof(GI_FLOAT32_V2_NAIVE_t));
    size_t length = GI_SIMD_LEN_BYTE / sizeof(float);
    for (size_t i = 0; i < 2 * length; i++) {
        int8_t data = Saturate(round(s0.val[i / length][i % length]), -128, 127);
        tmp_ret[i] = data;
        tmp_ret[i + length * 2] = data;
    }
    memcpy(&ret, &tmp_ret, sizeof(GI_INT8_t));
    return ret;
#endif
}

GI_FORCEINLINE
GI_INT8_t GiCvtFromFloat32V4ToInt8(GI_FLOAT32_V4_t vsrc) {
#if defined(GI_NEON_INTRINSICS)
#if __ARM_ARCH >= 8
    int32x4_t vres0 = vcvtaq_s32_f32(vsrc.val[0]);
    int32x4_t vres1 = vcvtaq_s32_f32(vsrc.val[1]);
    int32x4_t vres2 = vcvtaq_s32_f32(vsrc.val[2]);
    int32x4_t vres3 = vcvtaq_s32_f32(vsrc.val[3]);
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
    GI_FLOAT32_t vfzero = GiBroadcastFloat32(0.0f);
    GI_FLOAT32_t vfhalf = GiBroadcastFloat32(0.5f);
    GI_FLOAT32_t vfneg_half = GiBroadcastFloat32(-0.5f);
    GI_FLOAT32_t vfmin_int8 = GiBroadcastFloat32(-128.0f);
    GI_FLOAT32_t vfmax_int8 = GiBroadcastFloat32(127.0f);
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
    vres3 = _mm_round_ps(vres3, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC);

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
#elif defined(GI_RVV_INTRINSICS)
    //! TODO: vfcvt_rtz_x_f_v_i32m1 is RVV 1.0 api, now xuantie D1 only support 0p7
    //! as a workaround, we imp this API by naive
    GI_INT8_NAIVE_t tmp_ret;
    GI_FLOAT32_V4_NAIVE_t s0;
    s0.val[0] = GiFloat32Type2FixLenType(GiGetSubVectorFloat32V4(vsrc, 0));
    s0.val[1] = GiFloat32Type2FixLenType(GiGetSubVectorFloat32V4(vsrc, 1));
    s0.val[2] = GiFloat32Type2FixLenType(GiGetSubVectorFloat32V4(vsrc, 2));
    s0.val[3] = GiFloat32Type2FixLenType(GiGetSubVectorFloat32V4(vsrc, 3));
    size_t length = GI_SIMD_LEN_BYTE / sizeof(float);
    for (size_t i = 0; i < 4 * length; i++) {
        tmp_ret[i] =
                Saturate(round(s0.val[i / length][i % length]), INT8_MIN, INT8_MAX);
    }
    return vle8_v_i8m1((const signed char*)&tmp_ret, GI_SIMD_LEN_BYTE / sizeof(int8_t));
#else
    GI_INT8_t ret;
    GI_INT8_NAIVE_t tmp_ret;
    GI_FLOAT32_V4_NAIVE_t s0;
    memcpy(&s0, &vsrc, sizeof(GI_FLOAT32_V4_NAIVE_t));
    size_t length = GI_SIMD_LEN_BYTE / sizeof(float);
    for (size_t i = 0; i < 4 * length; i++) {
        tmp_ret[i] =
                Saturate(round(s0.val[i / length][i % length]), INT8_MIN, INT8_MAX);
    }
    memcpy(&ret, &tmp_ret, sizeof(GI_INT8_t));
    return ret;
#endif
}

GI_FORCEINLINE
GI_UINT8_t GiLoadUint8(const void* Buffer) {
#if defined(GI_NEON_INTRINSICS)
    return vld1q_u8((uint8_t*)Buffer);
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_loadu_si128((const __m128i*)Buffer);
#elif defined(GI_RVV_INTRINSICS)
    return vle8_v_u8m1((uint8_t*)Buffer, GI_SIMD_LEN_BYTE / sizeof(uint8_t));
#else
    GI_UINT8_t ret;
    const uint8_t* ptr = (uint8_t*)Buffer;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(uint8_t); i++) {
        ret[i] = ptr[i];
    }
    return ret;
#endif
}

GI_FORCEINLINE GI_UINT8_t GiReverseUint8(GI_UINT8_t a) {
#if defined(GI_NEON_INTRINSICS)
    GI_UINT8_t vec = vrev64q_u8(a);
    return vextq_u8(vec, vec, 8);
#elif defined(GI_SSE2_INTRINSICS)
    char d[16];
    _mm_storeu_si128((__m128i*)d, a);
    return _mm_setr_epi8(
            d[15], d[14], d[13], d[12], d[11], d[10], d[9], d[8], d[7], d[6], d[5],
            d[4], d[3], d[2], d[1], d[0]);
#elif defined(GI_RVV_INTRINSICS)
    vuint8m1_t index = vundefined_u8m1();
#if GI_SIMD_LEN_BYTE == 16
    uint8_t idx_num0[16] = {0xf, 0xe, 0xd, 0xc, 0xb, 0xa, 0x9, 0x8,
                            0x7, 0x6, 0x5, 0x4, 0x3, 0x2, 0x1, 0x0};
    index = vle8_v_u8m1((uint8_t*)idx_num0, 16);
#else
    uint8_t* index_p = (uint8_t*)&index;
    int32_t offset = GI_SIMD_LEN_BYTE / sizeof(int8_t);
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(int8_t); i++) {
        index_p[i] = offset - i - 1;
    }
#endif

    return vrgather_vv_u8m1(a, index, GI_SIMD_LEN_BYTE / sizeof(uint8_t));
#else
    GI_UINT8_t ret;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(uint8_t); i++) {
        ret[i] = a[GI_SIMD_LEN_BYTE / sizeof(uint8_t) - i - 1];
    }
    return ret;
#endif
}

GI_FORCEINLINE
void GiStoreUint8(void* Buffer, GI_UINT8_t Vector) {
#if defined(GI_NEON_INTRINSICS)
    vst1q_u8((uint8_t*)Buffer, Vector);
#elif defined(GI_SSE2_INTRINSICS)
    _mm_storeu_si128((__m128i*)Buffer, Vector);
#elif defined(GI_RVV_INTRINSICS)
    vse8_v_u8m1((uint8_t*)Buffer, Vector, GI_SIMD_LEN_BYTE / sizeof(uint8_t));
#else
    uint8_t* ptr = (uint8_t*)Buffer;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(uint8_t); i++) {
        ptr[i] = Vector[i];
    }
#endif
}

GI_FORCEINLINE
GI_UINT8_t GiLoadUzip0V3Uint8(const void* Buffer) {
#if defined(GI_NEON_INTRINSICS)
    uint8x16x3_t vec = vld3q_u8((uint8_t*)Buffer);
    return vec.val[0];
#elif defined(GI_SSE2_INTRINSICS)
    __m128i t00 = _mm_loadu_si128((const __m128i*)Buffer);
    __m128i t01 = _mm_loadu_si128((const __m128i*)((uint8_t*)Buffer + 16));
    __m128i t02 = _mm_loadu_si128((const __m128i*)((uint8_t*)Buffer + 32));

    __m128i t10 = _mm_unpacklo_epi8(t00, _mm_unpackhi_epi64(t01, t01));
    __m128i t11 = _mm_unpacklo_epi8(_mm_unpackhi_epi64(t00, t00), t02);
    __m128i t12 = _mm_unpacklo_epi8(t01, _mm_unpackhi_epi64(t02, t02));

    __m128i t20 = _mm_unpacklo_epi8(t10, _mm_unpackhi_epi64(t11, t11));
    __m128i t21 = _mm_unpacklo_epi8(_mm_unpackhi_epi64(t10, t10), t12);
    __m128i t22 = _mm_unpacklo_epi8(t11, _mm_unpackhi_epi64(t12, t12));

    __m128i t30 = _mm_unpacklo_epi8(t20, _mm_unpackhi_epi64(t21, t21));
    __m128i t31 = _mm_unpacklo_epi8(_mm_unpackhi_epi64(t20, t20), t22);

    return _mm_unpacklo_epi8(t30, _mm_unpackhi_epi64(t31, t31));
#elif defined(GI_RVV_INTRINSICS)
    return vlse8_v_u8m1((uint8_t*)Buffer, 3, GI_SIMD_LEN_BYTE / sizeof(uint8_t));
#else
    GI_UINT8_t ret;
    size_t i, i3;
    for (i = i3 = 0; i < GI_SIMD_LEN_BYTE / sizeof(uint8_t); i++, i3 += 3) {
        ret[i] = *((uint8_t*)Buffer + i3);
    }
    return ret;
#endif
}

GI_FORCEINLINE
GI_UINT8_t GiLoadUzip1V3Uint8(const void* Buffer) {
#if defined(GI_NEON_INTRINSICS)
    uint8x16x3_t vec = vld3q_u8((uint8_t*)Buffer);
    return vec.val[1];
#elif defined(GI_SSE2_INTRINSICS)
    __m128i t00 = _mm_loadu_si128((const __m128i*)Buffer);
    __m128i t01 = _mm_loadu_si128((const __m128i*)((uint8_t*)Buffer + 16));
    __m128i t02 = _mm_loadu_si128((const __m128i*)((uint8_t*)Buffer + 32));

    __m128i t10 = _mm_unpacklo_epi8(t00, _mm_unpackhi_epi64(t01, t01));
    __m128i t11 = _mm_unpacklo_epi8(_mm_unpackhi_epi64(t00, t00), t02);
    __m128i t12 = _mm_unpacklo_epi8(t01, _mm_unpackhi_epi64(t02, t02));

    __m128i t20 = _mm_unpacklo_epi8(t10, _mm_unpackhi_epi64(t11, t11));
    __m128i t21 = _mm_unpacklo_epi8(_mm_unpackhi_epi64(t10, t10), t12);
    __m128i t22 = _mm_unpacklo_epi8(t11, _mm_unpackhi_epi64(t12, t12));

    __m128i t30 = _mm_unpacklo_epi8(t20, _mm_unpackhi_epi64(t21, t21));
    __m128i t32 = _mm_unpacklo_epi8(t21, _mm_unpackhi_epi64(t22, t22));

    return _mm_unpacklo_epi8(_mm_unpackhi_epi64(t30, t30), t32);
#elif defined(GI_RVV_INTRINSICS)
    return vlse8_v_u8m1((uint8_t*)Buffer + 1, 3, GI_SIMD_LEN_BYTE / sizeof(uint8_t));
#else
    GI_UINT8_t ret;
    size_t i, i3;
    for (i = i3 = 0; i < GI_SIMD_LEN_BYTE / sizeof(uint8_t); i++, i3 += 3) {
        ret[i] = *((uint8_t*)Buffer + i3 + 1);
    }
    return ret;
#endif
}

GI_FORCEINLINE
GI_UINT8_t GiLoadUzip2V3Uint8(const void* Buffer) {
#if defined(GI_NEON_INTRINSICS)
    uint8x16x3_t vec = vld3q_u8((uint8_t*)Buffer);
    return vec.val[2];
#elif defined(GI_SSE2_INTRINSICS)
    __m128i t00 = _mm_loadu_si128((const __m128i*)Buffer);
    __m128i t01 = _mm_loadu_si128((const __m128i*)((uint8_t*)Buffer + 16));
    __m128i t02 = _mm_loadu_si128((const __m128i*)((uint8_t*)Buffer + 32));

    __m128i t10 = _mm_unpacklo_epi8(t00, _mm_unpackhi_epi64(t01, t01));
    __m128i t11 = _mm_unpacklo_epi8(_mm_unpackhi_epi64(t00, t00), t02);
    __m128i t12 = _mm_unpacklo_epi8(t01, _mm_unpackhi_epi64(t02, t02));

    __m128i t20 = _mm_unpacklo_epi8(t10, _mm_unpackhi_epi64(t11, t11));
    __m128i t21 = _mm_unpacklo_epi8(_mm_unpackhi_epi64(t10, t10), t12);
    __m128i t22 = _mm_unpacklo_epi8(t11, _mm_unpackhi_epi64(t12, t12));

    __m128i t31 = _mm_unpacklo_epi8(_mm_unpackhi_epi64(t20, t20), t22);
    __m128i t32 = _mm_unpacklo_epi8(t21, _mm_unpackhi_epi64(t22, t22));

    return _mm_unpacklo_epi8(t31, _mm_unpackhi_epi64(t32, t32));
#elif defined(GI_RVV_INTRINSICS)
    return vlse8_v_u8m1((uint8_t*)Buffer + 2, 3, GI_SIMD_LEN_BYTE / sizeof(uint8_t));
#else
    GI_UINT8_t ret;
    size_t i, i3;
    for (i = i3 = 0; i < GI_SIMD_LEN_BYTE / sizeof(uint8_t); i++, i3 += 3) {
        ret[i] = *((uint8_t*)Buffer + i3 + 2);
    }
    return ret;
#endif
}

GI_FORCEINLINE
void GiStoreZipUint8V3(void* Buffer, GI_UINT8_t a, GI_UINT8_t b, GI_UINT8_t c) {
#if defined(GI_NEON_INTRINSICS)
    uint8x16x3_t vec;
    vec.val[0] = a;
    vec.val[1] = b;
    vec.val[2] = c;
    vst3q_u8((uint8_t*)Buffer, vec);
#elif defined(GI_SSE2_INTRINSICS)
    __m128i z = _mm_setzero_si128();
    __m128i ab0 = _mm_unpacklo_epi8(a, b);
    __m128i ab1 = _mm_unpackhi_epi8(a, b);
    __m128i c0 = _mm_unpacklo_epi8(c, z);
    __m128i c1 = _mm_unpackhi_epi8(c, z);

    __m128i p00 = _mm_unpacklo_epi16(ab0, c0);
    __m128i p01 = _mm_unpackhi_epi16(ab0, c0);
    __m128i p02 = _mm_unpacklo_epi16(ab1, c1);
    __m128i p03 = _mm_unpackhi_epi16(ab1, c1);

    __m128i p10 = _mm_unpacklo_epi32(p00, p01);
    __m128i p11 = _mm_unpackhi_epi32(p00, p01);
    __m128i p12 = _mm_unpacklo_epi32(p02, p03);
    __m128i p13 = _mm_unpackhi_epi32(p02, p03);

    __m128i p20 = _mm_unpacklo_epi64(p10, p11);
    __m128i p21 = _mm_unpackhi_epi64(p10, p11);
    __m128i p22 = _mm_unpacklo_epi64(p12, p13);
    __m128i p23 = _mm_unpackhi_epi64(p12, p13);

    p20 = _mm_slli_si128(p20, 1);
    p22 = _mm_slli_si128(p22, 1);

    __m128i p30 = _mm_slli_epi64(_mm_unpacklo_epi32(p20, p21), 8);
    __m128i p31 = _mm_srli_epi64(_mm_unpackhi_epi32(p20, p21), 8);
    __m128i p32 = _mm_slli_epi64(_mm_unpacklo_epi32(p22, p23), 8);
    __m128i p33 = _mm_srli_epi64(_mm_unpackhi_epi32(p22, p23), 8);

    __m128i p40 = _mm_unpacklo_epi64(p30, p31);
    __m128i p41 = _mm_unpackhi_epi64(p30, p31);
    __m128i p42 = _mm_unpacklo_epi64(p32, p33);
    __m128i p43 = _mm_unpackhi_epi64(p32, p33);

    __m128i v0 = _mm_or_si128(_mm_srli_si128(p40, 2), _mm_slli_si128(p41, 10));
    __m128i v1 = _mm_or_si128(_mm_srli_si128(p41, 6), _mm_slli_si128(p42, 6));
    __m128i v2 = _mm_or_si128(_mm_srli_si128(p42, 10), _mm_slli_si128(p43, 2));

    _mm_storeu_si128((__m128i*)(Buffer), v0);
    _mm_storeu_si128((__m128i*)((uint8_t*)Buffer + 16), v1);
    _mm_storeu_si128((__m128i*)((uint8_t*)Buffer + 32), v2);

#elif defined(GI_RVV_INTRINSICS)
    vsse8_v_u8m1((uint8_t*)Buffer, 3, a, GI_SIMD_LEN_BYTE / sizeof(uint8_t));
    vsse8_v_u8m1((uint8_t*)Buffer + 1, 3, b, GI_SIMD_LEN_BYTE / sizeof(uint8_t));
    vsse8_v_u8m1((uint8_t*)Buffer + 2, 3, c, GI_SIMD_LEN_BYTE / sizeof(uint8_t));
#else
    size_t i, i3;
    for (i = i3 = 0; i < GI_SIMD_LEN_BYTE / sizeof(uint8_t); i++, i3 += 3) {
        *((uint8_t*)Buffer + i3) = a[i];
        *((uint8_t*)Buffer + i3 + 1) = b[i];
        *((uint8_t*)Buffer + i3 + 2) = c[i];
    }
#endif
}

#if defined(GI_NEON_INTRINSICS)
#define GiShiftRightInt16ToUint8(Vector, shift)       \
    __extension__({                                   \
        uint8x8_t vec = vqshrun_n_s16(Vector, shift); \
        uint8x16_t _ret = vcombine_u8(vec, vec);      \
        _ret;                                         \
    })
#elif defined(GI_SSE2_INTRINSICS)
#define GiShiftRightInt16ToUint8(Vector, shift)      \
    __extension__({                                  \
        __m128i vec = _mm_srai_epi16(Vector, shift); \
        __m128i _ret = _mm_packus_epi16(vec, vec);   \
        _ret;                                        \
    })
#elif defined(GI_RVV_INTRINSICS)
#define GiShiftRightInt16ToUint8(Vector, shift)                                      \
    __extension__({                                                                  \
        vint16m1_t src1 =                                                            \
                vsra_vx_i16m1(Vector, shift, GI_SIMD_LEN_BYTE / sizeof(int16_t));    \
        vint16m2_t max, min, dest;                                                   \
        max = vmv_v_x_i16m2(UINT8_MAX, GI_SIMD_LEN_BYTE / sizeof(int8_t));           \
        min = vmv_v_x_i16m2(0, GI_SIMD_LEN_BYTE / sizeof(int8_t));                   \
        vbool8_t mask;                                                               \
        dest = vset_v_i16m1_i16m2(dest, 0, src1);                                    \
        dest = vset_v_i16m1_i16m2(dest, 1, src1);                                    \
        mask = vmsgt_vv_i16m2_b8(dest, min, GI_SIMD_LEN_BYTE / sizeof(int8_t));      \
        dest = vmerge_vvm_i16m2(mask, min, dest, GI_SIMD_LEN_BYTE / sizeof(int8_t)); \
        mask = vmslt_vv_i16m2_b8(dest, max, GI_SIMD_LEN_BYTE / sizeof(int8_t));      \
        dest = vmerge_vvm_i16m2(mask, max, dest, GI_SIMD_LEN_BYTE / sizeof(int8_t)); \
        vuint8m1_t _ret = vreinterpret_v_i8m1_u8m1(                                  \
                vncvt_x_x_w_i8m1(dest, GI_SIMD_LEN_BYTE / sizeof(int8_t)));          \
        _ret;                                                                        \
    })
#else
#define GiShiftRightInt16ToUint8(Vector, shift)                           \
    __extension__({                                                       \
        GI_UINT8_t _ret;                                                  \
        for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(int16_t); i++) { \
            uint8_t val = Saturate(Vector[i] >> shift, 0, UINT8_MAX);     \
            _ret[i] = val;                                                \
            _ret[i + GI_SIMD_LEN_BYTE / sizeof(int16_t)] = val;           \
        }                                                                 \
        _ret;                                                             \
    })
#endif

GI_FORCEINLINE
GI_INT16_t GiCombineInt16Low(GI_INT16_t Vector0, GI_INT16_t Vector1) {
#if defined(GI_NEON_INTRINSICS)
    return vcombine_s16(vget_low_s16(Vector0), vget_low_s16(Vector1));
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_unpacklo_epi64(Vector0, Vector1);
#elif defined(GI_RVV_INTRINSICS)
    return vslideup_vx_i16m1(
            Vector0, Vector1, GI_SIMD_LEN_BYTE / sizeof(int16_t) / 2,
            GI_SIMD_LEN_BYTE / sizeof(int16_t));
#else
    GI_INT16_t ret;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(int32_t); i++) {
        ret[i] = Vector0[i];
        ret[i + GI_SIMD_LEN_BYTE / sizeof(int32_t)] = Vector1[i];
    }
    return ret;
#endif
}

GI_FORCEINLINE
GI_UINT8_t GiCombineUint8Low(GI_UINT8_t Vector0, GI_UINT8_t Vector1) {
#if defined(GI_NEON_INTRINSICS)
    return vcombine_u8(vget_low_u8(Vector0), vget_low_u8(Vector1));
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_unpacklo_epi64(Vector0, Vector1);
#elif defined(GI_RVV_INTRINSICS)
    return vslideup_vx_u8m1(
            Vector0, Vector1, GI_SIMD_LEN_BYTE / sizeof(uint8_t) / 2,
            GI_SIMD_LEN_BYTE / sizeof(uint8_t));
#else
    GI_UINT8_t ret;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(int16_t); i++) {
        ret[i] = Vector0[i];
        ret[i + GI_SIMD_LEN_BYTE / sizeof(int16_t)] = Vector1[i];
    }
    return ret;
#endif
}

GI_FORCEINLINE
GI_INT8_t GiZipV0Int8(GI_INT8_t Vector0, GI_INT8_t Vector1) {
#if defined(GI_NEON_INTRINSICS)
    return vzipq_s8(Vector0, Vector1).val[0];
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_unpacklo_epi8(Vector0, Vector1);
#elif defined(GI_RVV_INTRINSICS)
    int8_t mask_idx[16] = {1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0};
    vint8m1_t mask_vec =
            vle8_v_i8m1((int8_t*)mask_idx, GI_SIMD_LEN_BYTE / sizeof(int8_t));
    vint8m1_t zero = vmv_v_x_i8m1(0, GI_SIMD_LEN_BYTE / sizeof(int8_t));
    vbool8_t mask = vmsgt_vv_i8m1_b8(mask_vec, zero, GI_SIMD_LEN_BYTE / sizeof(int8_t));
    vuint8m1_t index0 = vundefined_u8m1();
    vuint8m1_t index1 = vundefined_u8m1();
#if GI_SIMD_LEN_BYTE == 16
    uint8_t idx_num0[16] = {0x0, 0x8, 0x1, 0x9, 0x2, 0xa, 0x3, 0xb,
                            0x4, 0xc, 0x5, 0xd, 0x6, 0xe, 0x7, 0xf};
    uint8_t idx_num1[16] = {0x8, 0x0, 0x9, 0x1, 0xa, 0x2, 0xb, 0x3,
                            0xc, 0x4, 0xd, 0x5, 0xe, 0x6, 0xf, 0x7};
    index0 = vle8_v_u8m1((uint8_t*)idx_num0, GI_SIMD_LEN_BYTE / sizeof(uint8_t));
    index1 = vle8_v_u8m1((uint8_t*)idx_num1, GI_SIMD_LEN_BYTE / sizeof(uint8_t));
#else
    uint8_t* index_p0 = (uint8_t*)&index0;
    uint8_t* index_p1 = (uint8_t*)&index1;
    int32_t offset = GI_SIMD_LEN_BYTE / sizeof(uint8_t) / 2;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(uint8_t) / 2; i++) {
        index_p0[2 * i] = i;
        index_p0[2 * i + 1] = i + offset;
        index_p1[2 * i] = i + offset;
        index_p1[2 * i + 1] = i;
    }
#endif

    Vector0 = vrgather_vv_i8m1(Vector0, index0, GI_SIMD_LEN_BYTE / sizeof(uint8_t));
    Vector1 = vrgather_vv_i8m1(Vector1, index1, GI_SIMD_LEN_BYTE / sizeof(uint8_t));
    return vmerge_vvm_i8m1(mask, Vector1, Vector0, GI_SIMD_LEN_BYTE / sizeof(uint8_t));
#else
    GI_INT8_t ret;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(int16_t); ++i) {
        ret[2 * i] = Vector0[i];
        ret[2 * i + 1] = Vector1[i];
    }
    return ret;
#endif
}

GI_FORCEINLINE
GI_INT8_t GiZipV1Int8(GI_INT8_t Vector0, GI_INT8_t Vector1) {
#if defined(GI_NEON_INTRINSICS)
    return vzipq_s8(Vector0, Vector1).val[1];
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_unpackhi_epi8(Vector0, Vector1);
#elif defined(GI_RVV_INTRINSICS)
    int8_t mask_idx[16] = {0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1};
    vint8m1_t mask_vec =
            vle8_v_i8m1((int8_t*)mask_idx, GI_SIMD_LEN_BYTE / sizeof(int8_t));
    vint8m1_t zero = vmv_v_x_i8m1(0, GI_SIMD_LEN_BYTE / sizeof(int8_t));
    vbool8_t mask = vmsgt_vv_i8m1_b8(mask_vec, zero, GI_SIMD_LEN_BYTE / sizeof(int8_t));
    vuint8m1_t index0 = vundefined_u8m1();
    vuint8m1_t index1 = vundefined_u8m1();
#if GI_SIMD_LEN_BYTE == 16
    uint8_t idx_num0[16] = {0x0, 0x8, 0x1, 0x9, 0x2, 0xa, 0x3, 0xb,
                            0x4, 0xc, 0x5, 0xd, 0x6, 0xe, 0x7, 0xf};
    uint8_t idx_num1[16] = {0x8, 0x0, 0x9, 0x1, 0xa, 0x2, 0xb, 0x3,
                            0xc, 0x4, 0xd, 0x5, 0xe, 0x6, 0xf, 0x7};
    index0 = vle8_v_u8m1((uint8_t*)idx_num0, GI_SIMD_LEN_BYTE / sizeof(uint8_t));
    index1 = vle8_v_u8m1((uint8_t*)idx_num1, GI_SIMD_LEN_BYTE / sizeof(uint8_t));
#else
    uint8_t* index_p0 = (uint8_t*)&index0;
    uint8_t* index_p1 = (uint8_t*)&index1;
    int32_t offset = GI_SIMD_LEN_BYTE / sizeof(uint8_t) / 2;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(uint8_t) / 2; i++) {
        index_p0[2 * i] = i;
        index_p0[2 * i + 1] = i + offset;
        index_p1[2 * i] = i + offset;
        index_p1[2 * i + 1] = i;
    }
#endif
    Vector0 = vrgather_vv_i8m1(Vector0, index1, GI_SIMD_LEN_BYTE / sizeof(uint8_t));
    Vector1 = vrgather_vv_i8m1(Vector1, index0, GI_SIMD_LEN_BYTE / sizeof(uint8_t));
    return vmerge_vvm_i8m1(mask, Vector0, Vector1, GI_SIMD_LEN_BYTE / sizeof(uint8_t));
#else
    GI_INT8_t ret;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(int16_t); ++i) {
        ret[2 * i] = Vector0[i + GI_SIMD_LEN_BYTE / sizeof(int16_t)];
        ret[2 * i + 1] = Vector1[i + GI_SIMD_LEN_BYTE / sizeof(int16_t)];
    }
    return ret;
#endif
}

GI_FORCEINLINE
GI_INT16_t GiReinterpretInt8AsInt16(GI_INT8_t In) {
#if defined(GI_NEON_INTRINSICS)
    return vreinterpretq_s16_s8(In);
#elif defined(GI_SSE2_INTRINSICS)
    return (GI_INT16_t)In;
#elif defined(GI_RVV_INTRINSICS)
    return vreinterpret_v_i8m1_i16m1(In);
#else
    return (GI_INT16_t)In;
#endif
}

GI_FORCEINLINE
GI_INT16_t GiZipV0Int16(GI_INT16_t Vector0, GI_INT16_t Vector1) {
#if defined(GI_NEON_INTRINSICS)
    return vzipq_s16(Vector0, Vector1).val[0];
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_unpacklo_epi16(Vector0, Vector1);
#elif defined(GI_RVV_INTRINSICS)
    int16_t mask_idx[8] = {1, 0, 1, 0, 1, 0, 1, 0};
    vint16m1_t mask_vec =
            vle16_v_i16m1((int16_t*)mask_idx, GI_SIMD_LEN_BYTE / sizeof(int16_t));
    vint16m1_t zero = vmv_v_x_i16m1(0, GI_SIMD_LEN_BYTE / sizeof(int16_t));
    vbool16_t mask =
            vmsgt_vv_i16m1_b16(mask_vec, zero, GI_SIMD_LEN_BYTE / sizeof(int16_t));
    vuint16m1_t index0 = vundefined_u16m1();
    vuint16m1_t index1 = vundefined_u16m1();
#if GI_SIMD_LEN_BYTE == 16
    uint16_t idx_num0[8] = {0x0, 0x4, 0x1, 0x5, 0x2, 0x6, 0x3, 0x7};
    uint16_t idx_num1[8] = {0x4, 0x0, 0x5, 0x1, 0x6, 0x2, 0x7, 0x3};
    index0 = vle16_v_u16m1((uint16_t*)idx_num0, GI_SIMD_LEN_BYTE / sizeof(uint16_t));
    index1 = vle16_v_u16m1((uint16_t*)idx_num1, GI_SIMD_LEN_BYTE / sizeof(uint16_t));
#else
    uint16_t* index_p0 = (uint16_t*)&index0;
    uint16_t* index_p1 = (uint16_t*)&index1;
    int32_t offset = GI_SIMD_LEN_BYTE / sizeof(uint16_t) / 2;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(uint16_t) / 2; i++) {
        index_p0[2 * i] = i;
        index_p0[2 * i + 1] = i + offset;
        index_p1[2 * i] = i + offset;
        index_p1[2 * i + 1] = i;
    }
#endif

    Vector0 = vrgather_vv_i16m1(Vector0, index0, GI_SIMD_LEN_BYTE / sizeof(uint16_t));
    Vector1 = vrgather_vv_i16m1(Vector1, index1, GI_SIMD_LEN_BYTE / sizeof(uint16_t));
    return vmerge_vvm_i16m1(mask, Vector1, Vector0, GI_SIMD_LEN_BYTE / sizeof(int16_t));
#else
    GI_INT16_t ret;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(int32_t); i++) {
        ret[2 * i] = Vector0[i];
        ret[2 * i + 1] = Vector1[i];
    }
    return ret;
#endif
}

GI_FORCEINLINE
GI_INT16_t GiZipV1Int16(GI_INT16_t Vector0, GI_INT16_t Vector1) {
#if defined(GI_NEON_INTRINSICS)
    return vzipq_s16(Vector0, Vector1).val[1];
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_unpackhi_epi16(Vector0, Vector1);
#elif defined(GI_RVV_INTRINSICS)
    int16_t mask_idx[8] = {0, 1, 0, 1, 0, 1, 0, 1};
    vint16m1_t mask_vec =
            vle16_v_i16m1((int16_t*)mask_idx, GI_SIMD_LEN_BYTE / sizeof(int16_t));
    vint16m1_t zero = vmv_v_x_i16m1(0, GI_SIMD_LEN_BYTE / sizeof(int16_t));
    vbool16_t mask =
            vmsgt_vv_i16m1_b16(mask_vec, zero, GI_SIMD_LEN_BYTE / sizeof(int16_t));
    vuint16m1_t index0 = vundefined_u16m1();
    vuint16m1_t index1 = vundefined_u16m1();
#if GI_SIMD_LEN_BYTE == 16
    uint16_t idx_num0[8] = {0x0, 0x4, 0x1, 0x5, 0x2, 0x6, 0x3, 0x7};
    uint16_t idx_num1[8] = {0x4, 0x0, 0x5, 0x1, 0x6, 0x2, 0x7, 0x3};
    index0 = vle16_v_u16m1((uint16_t*)idx_num0, GI_SIMD_LEN_BYTE / sizeof(uint16_t));
    index1 = vle16_v_u16m1((uint16_t*)idx_num1, GI_SIMD_LEN_BYTE / sizeof(uint16_t));
#else
    uint16_t* index_p0 = (uint16_t*)&index0;
    uint16_t* index_p1 = (uint16_t*)&index1;
    int32_t offset = GI_SIMD_LEN_BYTE / sizeof(uint16_t) / 2;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(uint16_t) / 2; i++) {
        index_p0[2 * i] = i;
        index_p0[2 * i + 1] = i + offset;
        index_p1[2 * i] = i + offset;
        index_p1[2 * i + 1] = i;
    }
#endif
    Vector0 = vrgather_vv_i16m1(Vector0, index1, GI_SIMD_LEN_BYTE / sizeof(uint16_t));
    Vector1 = vrgather_vv_i16m1(Vector1, index0, GI_SIMD_LEN_BYTE / sizeof(uint16_t));
    return vmerge_vvm_i16m1(mask, Vector0, Vector1, GI_SIMD_LEN_BYTE / sizeof(int16_t));
#else
    GI_INT16_t ret;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(int32_t); ++i) {
        ret[2 * i] = Vector0[i + GI_SIMD_LEN_BYTE / sizeof(int32_t)];
        ret[2 * i + 1] = Vector1[i + GI_SIMD_LEN_BYTE / sizeof(int32_t)];
    }
    return ret;
#endif
}

GI_FORCEINLINE
GI_INT32_t GiReinterpretInt16AsInt32(GI_INT16_t In) {
#if defined(GI_NEON_INTRINSICS)
    return vreinterpretq_s32_s16(In);
#elif defined(GI_SSE2_INTRINSICS)
    return (GI_INT32_t)In;
#elif defined(GI_RVV_INTRINSICS)
    return vreinterpret_v_i16m1_i32m1(In);
#else
    return (GI_INT32_t)In;
#endif
}

GI_FORCEINLINE
GI_INT32_t GiZipV0Int32(GI_INT32_t Vector0, GI_INT32_t Vector1) {
#if defined(GI_NEON_INTRINSICS)
    return vzipq_s32(Vector0, Vector1).val[0];
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_unpacklo_epi32(Vector0, Vector1);
#elif defined(GI_RVV_INTRINSICS)
    int32_t mask_idx[4] = {1, 0, 1, 0};
    vint32m1_t mask_vec =
            vle32_v_i32m1((int32_t*)mask_idx, GI_SIMD_LEN_BYTE / sizeof(int32_t));
    vint32m1_t zero = vmv_v_x_i32m1(0, GI_SIMD_LEN_BYTE / sizeof(int32_t));
    vbool32_t mask =
            vmsgt_vv_i32m1_b32(mask_vec, zero, GI_SIMD_LEN_BYTE / sizeof(int32_t));
    vuint32m1_t index0 = vundefined_u32m1();
    vuint32m1_t index1 = vundefined_u32m1();
#if GI_SIMD_LEN_BYTE == 16
    uint32_t idx_num0[4] = {0x0, 0x2, 0x1, 0x3};
    uint32_t idx_num1[4] = {0x2, 0x0, 0x3, 0x1};
    index0 = vle32_v_u32m1((uint32_t*)idx_num0, GI_SIMD_LEN_BYTE / sizeof(uint32_t));
    index1 = vle32_v_u32m1((uint32_t*)idx_num1, GI_SIMD_LEN_BYTE / sizeof(uint32_t));
#else
    uint32_t* index_p0 = (uint32_t*)&index0;
    uint32_t* index_p1 = (uint32_t*)&index1;
    size_t offset = GI_SIMD_LEN_BYTE / sizeof(uint32_t) / 2;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(uint32_t) / 2; i++) {
        index_p0[2 * i] = i;
        index_p0[2 * i + 1] = i + offset;
        index_p1[2 * i] = i + offset;
        index_p1[2 * i + 1] = i;
    }
#endif
    Vector0 = vrgather_vv_i32m1(Vector0, index0, GI_SIMD_LEN_BYTE / sizeof(uint32_t));
    Vector1 = vrgather_vv_i32m1(Vector1, index1, GI_SIMD_LEN_BYTE / sizeof(uint32_t));
    return vmerge_vvm_i32m1(mask, Vector1, Vector0, GI_SIMD_LEN_BYTE / sizeof(int32_t));
#else
    GI_INT32_t ret;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(int64_t); i++) {
        ret[2 * i] = Vector0[i];
        ret[2 * i + 1] = Vector1[i];
    }
    return ret;
#endif
}

GI_FORCEINLINE
GI_INT32_t GiZipV1Int32(GI_INT32_t Vector0, GI_INT32_t Vector1) {
#if defined(GI_NEON_INTRINSICS)
    return vzipq_s32(Vector0, Vector1).val[1];
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_unpackhi_epi32(Vector0, Vector1);
#elif defined(GI_RVV_INTRINSICS)
    int32_t mask_idx[4] = {0, 1, 0, 1};
    vint32m1_t mask_vec =
            vle32_v_i32m1((int32_t*)mask_idx, GI_SIMD_LEN_BYTE / sizeof(int32_t));
    vint32m1_t zero = vmv_v_x_i32m1(0, GI_SIMD_LEN_BYTE / sizeof(int32_t));
    vbool32_t mask =
            vmsgt_vv_i32m1_b32(mask_vec, zero, GI_SIMD_LEN_BYTE / sizeof(int32_t));
    vuint32m1_t index0 = vundefined_u32m1();
    vuint32m1_t index1 = vundefined_u32m1();
#if GI_SIMD_LEN_BYTE == 16
    uint32_t idx_num0[4] = {0x0, 0x2, 0x1, 0x3};
    uint32_t idx_num1[4] = {0x2, 0x0, 0x3, 0x1};
    index0 = vle32_v_u32m1((uint32_t*)idx_num0, GI_SIMD_LEN_BYTE / sizeof(uint32_t));
    index1 = vle32_v_u32m1((uint32_t*)idx_num1, GI_SIMD_LEN_BYTE / sizeof(uint32_t));
#else
    uint32_t* index_p0 = (uint32_t*)&index0;
    uint32_t* index_p1 = (uint32_t*)&index1;
    size_t offset = GI_SIMD_LEN_BYTE / sizeof(uint32_t) / 2;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(uint32_t) / 2; i++) {
        index_p0[2 * i] = i;
        index_p0[2 * i + 1] = i + offset;
        index_p1[2 * i] = i + offset;
        index_p1[2 * i + 1] = i;
    }
#endif
    Vector0 = vrgather_vv_i32m1(Vector0, index1, GI_SIMD_LEN_BYTE / sizeof(uint32_t));
    Vector1 = vrgather_vv_i32m1(Vector1, index0, GI_SIMD_LEN_BYTE / sizeof(uint32_t));
    return vmerge_vvm_i32m1(mask, Vector0, Vector1, GI_SIMD_LEN_BYTE / sizeof(int32_t));
#else
    GI_INT32_t ret;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(int64_t); i++) {
        ret[2 * i] = Vector0[i + GI_SIMD_LEN_BYTE / sizeof(int64_t)];
        ret[2 * i + 1] = Vector1[i + GI_SIMD_LEN_BYTE / sizeof(int64_t)];
    }
    return ret;
#endif
}

GI_FORCEINLINE
GI_INT32_t GiCombineInt32Low(GI_INT32_t Vector0, GI_INT32_t Vector1) {
#if defined(GI_NEON_INTRINSICS)
    return vcombine_s32(vget_low_s32(Vector0), vget_low_s32(Vector1));
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_unpacklo_epi64(Vector0, Vector1);
#elif defined(GI_RVV_INTRINSICS)
    return vslideup_vx_i32m1(
            Vector0, Vector1, GI_SIMD_LEN_BYTE / sizeof(int32_t) / 2,
            GI_SIMD_LEN_BYTE / sizeof(int32_t));
#else
    GI_INT32_t ret;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(int64_t); i++) {
        ret[i] = Vector0[i];
        ret[i + GI_SIMD_LEN_BYTE / sizeof(int64_t)] = Vector1[i];
    }
    return ret;
#endif
}

GI_FORCEINLINE
GI_INT32_t GiCombineInt32High(GI_INT32_t Vector0, GI_INT32_t Vector1) {
#if defined(GI_NEON_INTRINSICS)
    return vcombine_s32(vget_high_s32(Vector0), vget_high_s32(Vector1));
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_unpackhi_epi64(Vector0, Vector1);
#elif defined(GI_RVV_INTRINSICS)
    Vector0 = vslidedown_vx_i32m1(
            Vector0, Vector0, GI_SIMD_LEN_BYTE / sizeof(int32_t) / 2,
            GI_SIMD_LEN_BYTE / sizeof(int32_t));
    Vector1 = vslidedown_vx_i32m1(
            Vector1, Vector1, GI_SIMD_LEN_BYTE / sizeof(int32_t) / 2,
            GI_SIMD_LEN_BYTE / sizeof(int32_t));
    return vslideup_vx_i32m1(
            Vector0, Vector1, GI_SIMD_LEN_BYTE / sizeof(int32_t) / 2,
            GI_SIMD_LEN_BYTE / sizeof(int32_t));
#else
    GI_INT32_t ret;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(int64_t); i++) {
        ret[i] = Vector0[i + GI_SIMD_LEN_BYTE / sizeof(int64_t)];
        ret[i + GI_SIMD_LEN_BYTE / sizeof(int64_t)] =
                Vector1[i + +GI_SIMD_LEN_BYTE / sizeof(int64_t)];
    }
    return ret;
#endif
}

GI_FORCEINLINE
void GiStoreZipInt8V3(void* Buffer, GI_INT8_t a, GI_INT8_t b, GI_INT8_t c) {
#if defined(GI_NEON_INTRINSICS)
    int8x16x3_t vec;
    vec.val[0] = a;
    vec.val[1] = b;
    vec.val[2] = c;
    vst3q_s8((int8_t*)Buffer, vec);
#elif defined(GI_SSE2_INTRINSICS)
    __m128i z = _mm_setzero_si128();
    __m128i ab0 = _mm_unpacklo_epi8(a, b);
    __m128i ab1 = _mm_unpackhi_epi8(a, b);
    __m128i c0 = _mm_unpacklo_epi8(c, z);
    __m128i c1 = _mm_unpackhi_epi8(c, z);

    __m128i p00 = _mm_unpacklo_epi16(ab0, c0);
    __m128i p01 = _mm_unpackhi_epi16(ab0, c0);
    __m128i p02 = _mm_unpacklo_epi16(ab1, c1);
    __m128i p03 = _mm_unpackhi_epi16(ab1, c1);

    __m128i p10 = _mm_unpacklo_epi32(p00, p01);
    __m128i p11 = _mm_unpackhi_epi32(p00, p01);
    __m128i p12 = _mm_unpacklo_epi32(p02, p03);
    __m128i p13 = _mm_unpackhi_epi32(p02, p03);

    __m128i p20 = _mm_unpacklo_epi64(p10, p11);
    __m128i p21 = _mm_unpackhi_epi64(p10, p11);
    __m128i p22 = _mm_unpacklo_epi64(p12, p13);
    __m128i p23 = _mm_unpackhi_epi64(p12, p13);

    p20 = _mm_slli_si128(p20, 1);
    p22 = _mm_slli_si128(p22, 1);

    __m128i p30 = _mm_slli_epi64(_mm_unpacklo_epi32(p20, p21), 8);
    __m128i p31 = _mm_srli_epi64(_mm_unpackhi_epi32(p20, p21), 8);
    __m128i p32 = _mm_slli_epi64(_mm_unpacklo_epi32(p22, p23), 8);
    __m128i p33 = _mm_srli_epi64(_mm_unpackhi_epi32(p22, p23), 8);

    __m128i p40 = _mm_unpacklo_epi64(p30, p31);
    __m128i p41 = _mm_unpackhi_epi64(p30, p31);
    __m128i p42 = _mm_unpacklo_epi64(p32, p33);
    __m128i p43 = _mm_unpackhi_epi64(p32, p33);

    __m128i v0 = _mm_or_si128(_mm_srli_si128(p40, 2), _mm_slli_si128(p41, 10));
    __m128i v1 = _mm_or_si128(_mm_srli_si128(p41, 6), _mm_slli_si128(p42, 6));
    __m128i v2 = _mm_or_si128(_mm_srli_si128(p42, 10), _mm_slli_si128(p43, 2));

    _mm_storeu_si128((__m128i*)(Buffer), v0);
    _mm_storeu_si128((__m128i*)((int8_t*)Buffer + 16), v1);
    _mm_storeu_si128((__m128i*)((int8_t*)Buffer + 32), v2);

#elif defined(GI_RVV_INTRINSICS)
    vsseg3e8_v_i8m1((int8_t*)Buffer, a, b, c, GI_SIMD_LEN_BYTE / sizeof(int8_t));
#else
    size_t i, i3;
    for (i = i3 = 0; i < GI_SIMD_LEN_BYTE / sizeof(int8_t); i++, i3 += 3) {
        *((int8_t*)Buffer + i3) = a[i];
        *((int8_t*)Buffer + i3 + 1) = b[i];
        *((int8_t*)Buffer + i3 + 2) = c[i];
    }
#endif
}

#if defined(GI_NEON_INTRINSICS)
#define GiShiftRightInt32(Vector, n) vshrq_n_s32(Vector, n)
#elif defined(GI_SSE2_INTRINSICS)
#define GiShiftRightInt32(Vector, n) _mm_srai_epi32(Vector, n)
#elif defined(GI_RVV_INTRINSICS)
#define GiShiftRightInt32(Vector, n) \
    vsra_vx_i32m1(Vector, n, GI_SIMD_LEN_BYTE / sizeof(int32_t))
#else
GI_FORCEINLINE
GI_INT32_t ShiftRightNaive(GI_INT32_t src, const size_t shift) {
    GI_INT32_t ret;
    for (size_t idx = 0; idx < GI_SIMD_LEN_BYTE / sizeof(int32_t); ++idx) {
        ret[idx] = src[idx] >> shift;
    }
    return ret;
}
#define GiShiftRightInt32(Vector, n) ShiftRightNaive(Vector, n)

#endif

#if defined(GI_NEON_INTRINSICS)
#define GiShiftLeftInt32(Vector, n) vshlq_n_s32(Vector, n)
#elif defined(GI_SSE2_INTRINSICS)
#define GiShiftLeftInt32(Vector, n) _mm_slli_epi32(Vector, n)
#elif defined(GI_RVV_INTRINSICS)
#define GiShiftLeftInt32(Vector, n) \
    vsll_vx_i32m1(Vector, n, GI_SIMD_LEN_BYTE / sizeof(int32_t))
#else
GI_FORCEINLINE
GI_INT32_t ShiftLeftNaive(GI_INT32_t src, const size_t shift) {
    GI_INT32_t ret;
    for (size_t idx = 0; idx < GI_SIMD_LEN_BYTE / sizeof(int32_t); ++idx) {
        ret[idx] = src[idx] << shift;
    }
    return ret;
}
#define GiShiftLeftInt32(Vector, n) ShiftLeftNaive(Vector, n)
#endif

GI_FORCEINLINE
GI_INT16_t GiBroadcastInt16(int16_t Value) {
#if defined(GI_NEON_INTRINSICS)
    return vdupq_n_s16(Value);
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_set1_epi16(Value);
#elif defined(GI_RVV_INTRINSICS)
    return vmv_v_x_i16m1(Value, GI_SIMD_LEN_BYTE / sizeof(int16_t));
#else
    GI_INT16_t ret;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(int16_t); i++) {
        ret[i] = Value;
    }
    return ret;
#endif
}

GI_FORCEINLINE
GI_INT16_t GiAndInt16(GI_INT16_t Vector1, GI_INT16_t Vector2) {
#if defined(GI_NEON_INTRINSICS)
    return vandq_s16(Vector1, Vector2);
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_and_si128(Vector1, Vector2);
#elif defined(GI_RVV_INTRINSICS)
    return vand_vv_i16m1(Vector1, Vector2, GI_SIMD_LEN_BYTE / sizeof(int16_t));
#else
    GI_INT16_t ret;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(int16_t); i++) {
        ret[i] = Vector1[i] & Vector2[i];
    }
    return ret;
#endif
}

GI_FORCEINLINE
GI_INT16_t GiSubtractInt16(GI_INT16_t Vector1, GI_INT16_t Vector2) {
#if defined(GI_NEON_INTRINSICS)
    return vsubq_s16(Vector1, Vector2);
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_sub_epi16(Vector1, Vector2);
#elif defined(GI_RVV_INTRINSICS)
    return vsub_vv_i16m1(Vector1, Vector2, GI_SIMD_LEN_BYTE / sizeof(int16_t));
#else
    GI_INT16_t ret;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(int16_t); i++) {
        ret[i] = Vector1[i] - Vector2[i];
    }
    return ret;
#endif
}
GI_FORCEINLINE
GI_INT16_t GiCvtInt32ToInt16(GI_INT32_t Vector) {
#if defined(GI_NEON_INTRINSICS)
    int16x4_t vec = vqmovn_s32(Vector);
    return vcombine_s16(vec, vec);
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_packs_epi32(Vector, Vector);
#elif defined(GI_RVV_INTRINSICS)
    vint32m2_t dest = vundefined_i32m2();
    vint32m2_t max = vmv_v_x_i32m2(INT16_MAX, GI_SIMD_LEN_BYTE / sizeof(int16_t));
    vint32m2_t min = vmv_v_x_i32m2(INT16_MIN, GI_SIMD_LEN_BYTE / sizeof(int16_t));
    dest = vset_v_i32m1_i32m2(dest, 0, Vector);
    dest = vset_v_i32m1_i32m2(dest, 1, Vector);
    vbool16_t mask = vmsgt_vv_i32m2_b16(dest, min, GI_SIMD_LEN_BYTE / sizeof(int16_t));
    dest = vmerge_vvm_i32m2(mask, min, dest, GI_SIMD_LEN_BYTE / sizeof(int16_t));
    mask = vmslt_vv_i32m2_b16(dest, max, GI_SIMD_LEN_BYTE / sizeof(int16_t));
    dest = vmerge_vvm_i32m2(mask, max, dest, GI_SIMD_LEN_BYTE / sizeof(int16_t));
    return vncvt_x_x_w_i16m1(dest, GI_SIMD_LEN_BYTE / sizeof(int16_t));
#else
    GI_INT16_t ret;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(int32_t); i++) {
        ret[i] = Saturate(Vector[i], INT16_MIN, INT16_MAX);
        ret[i + GI_SIMD_LEN_BYTE / sizeof(int32_t)] = ret[i];
    }
    return ret;
#endif
}

GI_FORCEINLINE
GI_INT8_t GiInterleave4Int8(GI_INT8_t Vector) {
#if defined(GI_NEON_INTRINSICS)
    uint8x16_t idx = {0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15};
    return vqtbl1q_s8(Vector, idx);
#elif defined(GI_SSE2_INTRINSICS)
    __m128i src0 = _mm_shufflelo_epi16(Vector, 0xd8);
    src0 = _mm_shufflehi_epi16(src0, 0xd8);

    __m128i src1 = _mm_shuffle_epi32(src0, 0xd8);
    __m128i src2 = _mm_bsrli_si128(src1, 2);

    __m128i src3 = _mm_unpacklo_epi8(src1, src2);
    __m128i src4 = _mm_unpackhi_epi8(src1, src2);

    __m128i src5 = _mm_shuffle_epi32(src3, 0xd8);
    __m128i src6 = _mm_shuffle_epi32(src4, 0xd8);

    __m128i src7 = _mm_unpacklo_epi64(src5, src6);
    __m128i ans = _mm_shufflelo_epi16(src7, 0xd8);
    return _mm_shufflehi_epi16(ans, 0xd8);
#elif defined(GI_RVV_INTRINSICS)
    vuint8m1_t index = vundefined_u8m1();
#if GI_SIMD_LEN_BYTE == 16
    uint8_t idx[16] = {0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15};
    index = vle8_v_u8m1((uint8_t*)idx, GI_SIMD_LEN_BYTE / sizeof(uint8_t));
#else
    uint8_t* index_p = (uint8_t*)&index;
    size_t offset = GI_SIMD_LEN_BYTE / sizeof(uint8_t) / 4;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(uint8_t) / 4; i++) {
        index_p[i] = 4 * i;
        index_p[i + 1 * offset] = 4 * i + 1;
        index_p[i + 2 * offset] = 4 * i + 2;
        index_p[i + 3 * offset] = 4 * i + 3;
    }
#endif

    return vrgather_vv_i8m1(Vector, index, GI_SIMD_LEN_BYTE / sizeof(uint8_t));
#else
    GI_INT8_t ret;
    size_t offset = GI_SIMD_LEN_BYTE / sizeof(int32_t);
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(int32_t); i++) {
        ret[i] = Vector[i * 4 + 0];
        ret[i + 1 * offset] = Vector[i * 4 + 1];
        ret[i + 2 * offset] = Vector[i * 4 + 2];
        ret[i + 3 * offset] = Vector[i * 4 + 3];
    }
    return ret;
#endif
}

GI_FORCEINLINE
GI_INT16_t GiCvtUint8toInt16Low(GI_UINT8_t Vector) {
#if defined(GI_NEON_INTRINSICS)
    return vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(Vector)));
#elif defined(GI_SSE2_INTRINSICS)
    __m128i sign_mask = _mm_setzero_si128();
    return _mm_unpacklo_epi8(Vector, sign_mask);
#elif defined(GI_RVV_INTRINSICS)
    vuint16m2_t vec = vwcvtu_x_x_v_u16m2(Vector, GI_SIMD_LEN_BYTE / sizeof(uint8_t));
    return vreinterpret_v_u16m1_i16m1(vget_v_u16m2_u16m1(vec, 0));
#else
    GI_INT16_t ret;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(int16_t); i++) {
        ret[i] = (int16_t)Vector[i];
    }
    return ret;
#endif
}

GI_FORCEINLINE
GI_INT16_t GiCvtUint8toInt16High(GI_UINT8_t Vector) {
#if defined(GI_NEON_INTRINSICS)
    return vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(Vector)));
#elif defined(GI_SSE2_INTRINSICS)
    __m128i sign_mask = _mm_setzero_si128();
    return _mm_unpackhi_epi8(Vector, sign_mask);
#elif defined(GI_RVV_INTRINSICS)
    vuint16m2_t vec = vwcvtu_x_x_v_u16m2(Vector, GI_SIMD_LEN_BYTE / sizeof(uint8_t));
    return vreinterpret_v_u16m1_i16m1(vget_v_u16m2_u16m1(vec, 1));
#else
    GI_INT16_t ret;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(int16_t); i++) {
        ret[i] = (int16_t)Vector[i + GI_SIMD_LEN_BYTE / sizeof(int16_t)];
    }
    return ret;
#endif
}

GI_FORCEINLINE
GI_INT32_t GiMultiplyAddInt16LongLow(
        GI_INT32_t Vector0, GI_INT16_t Vector1, GI_INT16_t Vector2) {
#if defined(GI_NEON_INTRINSICS)
    return vmlal_s16(Vector0, vget_low_s16(Vector1), vget_low_s16(Vector2));
#elif defined(GI_SSE2_INTRINSICS)
    __m128i lo = _mm_mullo_epi16(Vector1, Vector2);
    __m128i hi = _mm_mulhi_epi16(Vector1, Vector2);
    return _mm_add_epi32(Vector0, _mm_unpacklo_epi16(lo, hi));
#elif defined(GI_RVV_INTRINSICS)
    vint32m2_t vec1 = vwcvt_x_x_v_i32m2(Vector1, GI_SIMD_LEN_BYTE / sizeof(int16_t));
    vint32m2_t vec2 = vwcvt_x_x_v_i32m2(Vector2, GI_SIMD_LEN_BYTE / sizeof(int16_t));
    return vmadd_vv_i32m1(
            vget_v_i32m2_i32m1(vec1, 0), vget_v_i32m2_i32m1(vec2, 0), Vector0,
            GI_SIMD_LEN_BYTE / sizeof(int32_t));
#else
    GI_INT32_t ret;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(int32_t); i++) {
        ret[i] = (int32_t)Vector0[i] + (int32_t)Vector1[i] * (int32_t)(Vector2[i]);
    }
    return ret;
#endif
}

GI_FORCEINLINE
GI_INT32_t GiMultiplyAddInt16LongHigh(
        GI_INT32_t Vector0, GI_INT16_t Vector1, GI_INT16_t Vector2) {
#if defined(GI_NEON_INTRINSICS)
    return vmlal_s16(Vector0, vget_high_s16(Vector1), vget_high_s16(Vector2));
#elif defined(GI_SSE2_INTRINSICS)
    __m128i lo = _mm_mullo_epi16(Vector1, Vector2);
    __m128i hi = _mm_mulhi_epi16(Vector1, Vector2);
    return _mm_add_epi32(Vector0, _mm_unpackhi_epi16(lo, hi));
#elif defined(GI_RVV_INTRINSICS)
    vint32m2_t vec1 = vwcvt_x_x_v_i32m2(Vector1, GI_SIMD_LEN_BYTE / sizeof(int16_t));
    vint32m2_t vec2 = vwcvt_x_x_v_i32m2(Vector2, GI_SIMD_LEN_BYTE / sizeof(int16_t));
    return vmadd_vv_i32m1(
            vget_v_i32m2_i32m1(vec1, 1), vget_v_i32m2_i32m1(vec2, 1), Vector0,
            GI_SIMD_LEN_BYTE / sizeof(int32_t));

#else
    GI_INT32_t ret;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(int32_t); i++) {
        size_t idx = GI_SIMD_LEN_BYTE / sizeof(int32_t) + i;
        ret[i] = (int32_t)Vector0[i] + (int32_t)Vector1[idx] * (int32_t)(Vector2[idx]);
    }
    return ret;
#endif
}

GI_FORCEINLINE
GI_UINT8_t GiCvtFromInt32V4ToUint8(
        GI_INT32_t Vector0, GI_INT32_t Vector1, GI_INT32_t Vector2,
        GI_INT32_t Vector3) {
#if defined(GI_NEON_INTRINSICS)
    return vcombine_u8(
            vqmovun_s16(vcombine_s16(vqmovn_s32(Vector0), vqmovn_s32(Vector1))),
            vqmovun_s16(vcombine_s16(vqmovn_s32(Vector2), vqmovn_s32(Vector3))));
#elif defined(GI_SSE2_INTRINSICS)
    __m128i vepi16_0 = _mm_packs_epi32(Vector0, Vector1);
    __m128i vepi16_1 = _mm_packs_epi32(Vector2, Vector3);
    return _mm_packus_epi16(vepi16_0, vepi16_1);
#elif defined(GI_RVV_INTRINSICS)
    vint32m4_t dest = vundefined_i32m4();
    dest = vset_v_i32m1_i32m4(dest, 0, Vector0);
    dest = vset_v_i32m1_i32m4(dest, 1, Vector1);
    dest = vset_v_i32m1_i32m4(dest, 2, Vector2);
    dest = vset_v_i32m1_i32m4(dest, 3, Vector3);
    vint32m4_t max = vmv_v_x_i32m4(UINT8_MAX, GI_SIMD_LEN_BYTE / sizeof(uint8_t));
    vint32m4_t min = vmv_v_x_i32m4(0, GI_SIMD_LEN_BYTE / sizeof(uint8_t));
    vbool8_t mask = vmsgt_vv_i32m4_b8(dest, min, GI_SIMD_LEN_BYTE / sizeof(uint8_t));
    dest = vmerge_vvm_i32m4(mask, min, dest, GI_SIMD_LEN_BYTE / sizeof(uint8_t));
    mask = vmslt_vv_i32m4_b8(dest, max, GI_SIMD_LEN_BYTE / sizeof(uint8_t));
    dest = vmerge_vvm_i32m4(mask, max, dest, GI_SIMD_LEN_BYTE / sizeof(uint8_t));
    vuint16m2_t ans16 = vreinterpret_v_i16m2_u16m2(
            vncvt_x_x_w_i16m2(dest, GI_SIMD_LEN_BYTE / sizeof(uint8_t)));
    return vncvt_x_x_w_u8m1(ans16, GI_SIMD_LEN_BYTE / sizeof(uint8_t));
#else
    GI_UINT8_t ret;
    size_t length = GI_SIMD_LEN_BYTE / sizeof(int32_t);
    for (size_t i = 0; i < length; i++) {
        ret[i] = Saturate(Vector0[i], 0, UINT8_MAX);
        ret[i + length] = Saturate(Vector1[i], 0, UINT8_MAX);
        ret[i + length * 2] = Saturate(Vector2[i], 0, UINT8_MAX);
        ret[i + length * 3] = Saturate(Vector3[i], 0, UINT8_MAX);
    }
    return ret;
#endif
}

GI_FORCEINLINE
GI_UINT8_t GiInterleave2Uint8(GI_UINT8_t Vector) {
#if defined(GI_NEON_INTRINSICS)
    uint8x16_t idx = {0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15};
    return vreinterpretq_u8_s8(vqtbl1q_s8(vreinterpretq_s8_u8(Vector), idx));
#elif defined(GI_SSE2_INTRINSICS)
    __m128i src1 = Vector;
    __m128i src2 = _mm_bsrli_si128(src1, 2);

    __m128i src3 = _mm_unpacklo_epi8(src1, src2);
    __m128i src4 = _mm_unpackhi_epi8(src1, src2);

    __m128i src5 = _mm_shuffle_epi32(src3, 0xd8);
    __m128i src6 = _mm_shuffle_epi32(src4, 0xd8);

    __m128i src7 = _mm_shufflelo_epi16(src5, 0xd8);
    __m128i src8 = _mm_shufflelo_epi16(src6, 0xd8);
    return _mm_unpacklo_epi32(src7, src8);
#elif defined(GI_RVV_INTRINSICS)
    vuint8m1_t index = vundefined_u8m1();
#if GI_SIMD_LEN_BYTE == 16
    uint8_t idx[16] = {0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15};
    index = vle8_v_u8m1((uint8_t*)idx, GI_SIMD_LEN_BYTE / sizeof(uint8_t));
#else
    uint8_t* index_p = (uint8_t*)&index;
    size_t offset = GI_SIMD_LEN_BYTE / sizeof(uint8_t) / 2;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(uint8_t) / 2; i++) {
        index_p[i] = 2 * i;
        index_p[i + offset] = 2 * i + 1;
    }
#endif

    return vrgather_vv_u8m1(Vector, index, GI_SIMD_LEN_BYTE / sizeof(uint8_t));
#else
    GI_UINT8_t ret;
    size_t offset = GI_SIMD_LEN_BYTE / sizeof(int16_t);
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(int16_t); i++) {
        ret[i] = Vector[2 * i];
        ret[i + offset] = Vector[2 * i + 1];
    }
    return ret;
#endif
}

// vim: syntax=cpp.doxygen
