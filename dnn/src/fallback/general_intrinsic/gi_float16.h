#pragma once

#include "gi_common.h"

#if defined(GI_SUPPORT_F16)

//! c + b * a
#if defined(GI_NEON_INTRINSICS)
#if defined(__ARM_FEATURE_FMA)
#define v_fma_ps_f16(c, b, a) vfmaq_f16((c), (b), (a))
#define v_fma_n_f16(c, b, a)  vfmaq_n_f16((c), (b), (a))
#else
#define v_fma_ps_f16(c, b, a) vaddq_f16((c), vmulq_f16((b), (a)))
#define v_fma_n_f16(c, b, a)  vaddq_f16((c), vmulq_f16((b), vdupq_n_f16(a)))
#endif
#endif

GI_FORCEINLINE
GI_FLOAT16_t GiBroadcastFloat16(gi_float16_t Value) {
#if defined(GI_NEON_INTRINSICS)
    return vdupq_n_f16(Value);
#elif defined(GI_RVV_INTRINSICS)
    return vfmv_v_f_f16m1(Value, GI_SIMD_LEN_BYTE / sizeof(gi_float16_t));
#endif
}

GI_FORCEINLINE
GI_FLOAT16_t GiLoadBroadcastFloat16(const gi_float16_t* Value) {
#if defined(GI_NEON_INTRINSICS)
    return vld1q_dup_f16(Value);
#elif defined(GI_RVV_INTRINSICS)
    return GiBroadcastFloat16(*Value);
#endif
}

GI_FORCEINLINE
GI_FLOAT32_V2_t GiCastFloat16ToFloat32(const GI_FLOAT16_t& fp16) {
#if defined(GI_NEON_INTRINSICS)
    GI_FLOAT32_V2_t ret;
    GiSetSubVectorFloat32V2(ret, 0, vcvt_f32_f16(vget_low_f16(fp16)));
    GiSetSubVectorFloat32V2(ret, 1, vcvt_f32_f16(vget_high_f16(fp16)));
    return ret;
#elif defined(GI_RVV_INTRINSICS)
    GI_FLOAT32_V2_t ret;
    vfloat32m2_t tmp =
            vfwcvt_f_f_v_f32m2(fp16, GI_SIMD_LEN_BYTE / sizeof(gi_float16_t));
    GiSetSubVectorFloat32V2(ret, 0, vget_v_f32m2_f32m1(tmp, 0));
    GiSetSubVectorFloat32V2(ret, 1, vget_v_f32m2_f32m1(tmp, 1));
    return ret;
#endif
}

GI_FORCEINLINE
GI_FLOAT16_t GiCastFloat32ToFloat16(const GI_FLOAT32_t& low, const GI_FLOAT32_t& high) {
#if defined(GI_NEON_INTRINSICS)
    return vcombine_f16(vcvt_f16_f32(low), vcvt_f16_f32(high));
#elif defined(GI_RVV_INTRINSICS)
    vfloat32m2_t tmp = vfmv_v_f_f32m2(0.0, GI_SIMD_LEN_BYTE / sizeof(gi_float16_t));
    tmp = vset_v_f32m1_f32m2(tmp, 0, low);
    tmp = vset_v_f32m1_f32m2(tmp, 1, high);
    return vfncvt_f_f_w_f16m1(tmp, GI_SIMD_LEN_BYTE / sizeof(gi_float16_t));
#endif
}

GI_FORCEINLINE
GI_FLOAT16_t GiZeroFloat16(void) {
    return GiBroadcastFloat16(0.0);
}

GI_FORCEINLINE
GI_FLOAT16_t GiLoadFloat16(const gi_float16_t* Buffer) {
#if defined(GI_NEON_INTRINSICS)
    return vld1q_f16(Buffer);
#elif defined(GI_RVV_INTRINSICS)
    return vle16_v_f16m1(Buffer, GI_SIMD_LEN_BYTE / sizeof(gi_float16_t));
#endif
}

// !return a + b * c
GI_FORCEINLINE
GI_FLOAT16_t GiMlaqFloat16(GI_FLOAT16_t a, GI_FLOAT16_t b, GI_FLOAT16_t c) {
#if defined(GI_NEON_INTRINSICS)
#if defined(__ARM_FEATURE_FMA)
    return vfmaq_f16(a, b, c);
#else
    return vaddq_f16(a, vmulq_f16(b, c));
#endif
#elif defined(GI_RVV_INTRINSICS)
    return vfmadd_vv_f16m1(b, c, a, GI_SIMD_LEN_BYTE / sizeof(gi_float16_t));
#endif
}

GI_FORCEINLINE
void GiStoreFloat16(gi_float16_t* Buffer, GI_FLOAT16_t Vector) {
#if defined(GI_NEON_INTRINSICS)
    vst1q_f16(Buffer, Vector);
#elif defined(GI_RVV_INTRINSICS)
    vse16_v_f16m1(Buffer, Vector, GI_SIMD_LEN_BYTE / sizeof(gi_float16_t));
#endif
}

GI_FORCEINLINE
GI_FLOAT16_t GiAddFloat16(GI_FLOAT16_t Vector1, GI_FLOAT16_t Vector2) {
#if defined(GI_NEON_INTRINSICS)
    return vaddq_f16(Vector1, Vector2);
#elif defined(GI_RVV_INTRINSICS)
    return vfadd_vv_f16m1(Vector1, Vector2, GI_SIMD_LEN_BYTE / sizeof(gi_float16_t));
#endif
}

GI_FORCEINLINE
GI_FLOAT16_t GiSubtractFloat16(GI_FLOAT16_t Vector1, GI_FLOAT16_t Vector2) {
#if defined(GI_NEON_INTRINSICS)
    return vsubq_f16(Vector1, Vector2);
#elif defined(GI_RVV_INTRINSICS)
    return vfsub_vv_f16m1(Vector1, Vector2, GI_SIMD_LEN_BYTE / sizeof(gi_float16_t));
#endif
}

GI_FORCEINLINE
GI_FLOAT16_t GiMultiplyFloat16(GI_FLOAT16_t Vector1, GI_FLOAT16_t Vector2) {
#if defined(GI_NEON_INTRINSICS)
    return vmulq_f16(Vector1, Vector2);
#elif defined(GI_RVV_INTRINSICS)
    return vfmul_vv_f16m1(Vector1, Vector2, GI_SIMD_LEN_BYTE / sizeof(gi_float16_t));
#endif
}

GI_FORCEINLINE
GI_FLOAT16_t GiMultiplyScalerFloat16(GI_FLOAT16_t Vector1, gi_float16_t Scaler) {
#if defined(GI_NEON_INTRINSICS)
    return vmulq_n_f16(Vector1, Scaler);
#elif defined(GI_RVV_INTRINSICS)
    return vfmul_vf_f16m1(Vector1, Scaler, GI_SIMD_LEN_BYTE / sizeof(gi_float16_t));
#endif
}

GI_FORCEINLINE
GI_FLOAT16_t GiMultiplyAddScalarFloat16(
        GI_FLOAT16_t VectorSum, GI_FLOAT16_t Vector, gi_float16_t Scalar) {
#if defined(GI_NEON_INTRINSICS)
    return v_fma_n_f16(VectorSum, Vector, Scalar);
#elif defined(GI_RVV_INTRINSICS)
    return vfmadd_vf_f16m1(
            Vector, Scalar, VectorSum, GI_SIMD_LEN_BYTE / sizeof(gi_float16_t));
#endif
}

GI_FORCEINLINE
GI_FLOAT16_t GiMultiplySubScalarFloat16(
        GI_FLOAT16_t VectorSub, GI_FLOAT16_t Vector, gi_float16_t Scalar) {
#if defined(GI_NEON_INTRINSICS)
    return vsubq_f16(VectorSub, vmulq_n_f16(Vector, Scalar));
#elif defined(GI_RVV_INTRINSICS)
    return vfnmsub_vf_f16m1(
            Vector, Scalar, VectorSub, GI_SIMD_LEN_BYTE / sizeof(gi_float16_t));
#endif
}

GI_FORCEINLINE
GI_FLOAT16_t GiMaximumFloat16(GI_FLOAT16_t Vector1, GI_FLOAT16_t Vector2) {
#if defined(GI_NEON_INTRINSICS)
    return vmaxq_f16(Vector1, Vector2);
#elif defined(GI_RVV_INTRINSICS)
    return vfmax_vv_f16m1(Vector1, Vector2, GI_SIMD_LEN_BYTE / sizeof(gi_float16_t));
#endif
}

GI_FORCEINLINE
GI_FLOAT16_t GiMinimumFloat16(GI_FLOAT16_t Vector1, GI_FLOAT16_t Vector2) {
#if defined(GI_NEON_INTRINSICS)
    return vminq_f16(Vector1, Vector2);
#elif defined(GI_RVV_INTRINSICS)
    return vfmin_vv_f16m1(Vector1, Vector2, GI_SIMD_LEN_BYTE / sizeof(gi_float16_t));
#endif
}

//! a + b * c[d]
#if defined(GI_NEON_INTRINSICS)
#define GiSimdFmaLaneFloat16(a, b, c, d) vfmaq_laneq_f16(a, b, c, d)
#elif defined(GI_RVV_INTRINSICS)
#define __rvv_fmaq_laneq_f16(__a, __b, __c, __lane)                            \
    __extension__({                                                            \
        gi_float16_t t[GI_SIMD_LEN_BYTE / sizeof(gi_float16_t)];               \
        vse16_v_f16m1(t, __c, GI_SIMD_LEN_BYTE / sizeof(gi_float16_t));        \
        GI_FLOAT16_t __ret = vfmadd_vf_f16m1(                                  \
                __b, t[__lane], __a, GI_SIMD_LEN_BYTE / sizeof(gi_float16_t)); \
        __ret;                                                                 \
    })
#define GiSimdFmaLaneFloat16(a, b, c, d) __rvv_fmaq_laneq_f16(a, b, c, d)
#endif

//! a - b * v[lane]
#if defined(GI_NEON_INTRINSICS)
#define GiFmsqLaneQFloat16(a, b, v, lane) vfmsq_laneq_f16(a, b, v, lane)
#elif defined(GI_RVV_INTRINSICS)
#define __rvv_fmsq_lane_float16(__a, __b, __c, __lane)                         \
    __extension__({                                                            \
        gi_float16_t t[GI_SIMD_LEN_BYTE / sizeof(gi_float16_t)];               \
        vse16_v_f16m1(t, __c, GI_SIMD_LEN_BYTE / sizeof(gi_float16_t));        \
        GI_FLOAT16_t __ret = vfnmsub_vf_f16m1(                                 \
                __b, t[__lane], __a, GI_SIMD_LEN_BYTE / sizeof(gi_float16_t)); \
        __ret;                                                                 \
    })
#define GiFmsqLaneQFloat16(a, b, c, d) __rvv_fmsq_lane_float16(a, b, c, d)
#endif

#endif