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
GiReinterpretAsInt32(GI_FLOAT32 In) {
#if defined(GI_NEON_INTRINSICS)
    return vreinterpretq_s32_f32(In);
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_castps_si128(In);
#else
    return GI_INT32(In);
#endif
}

GI_FORCEINLINE
GI_INT32
GiRoundAsInt32(GI_FLOAT32 Vector) {
#if defined(GI_NEON_INTRINSICS)
#if __ARM_ARCH >= 8
    return vcvtaq_s32_f32(Vector);
#else
    float32x4_t vzero = vdupq_n_f32(0.f);
    float32x4_t vfhalf = vdupq_n_f32(0.5f);
    float32x4_t vfneg_half = vdupq_n_f32(-0.5f);
    float32x4_t vinc0 = vbslq_f32(vcgeq_f32(Vector, vzero), vfhalf, vfneg_half);
    return vcvtq_s32_f32(vaddq_f32(Vector, vinc0));
#endif
#elif defined(GI_SSE2_INTRINSICS)
    __m128 vfzero = _mm_set1_ps(0.f);
    __m128 vfhalf = _mm_set1_ps(0.5f);
    __m128 vfneg_half = _mm_set1_ps(-0.5f);
    __m128 vinc0 = _mm_blendv_ps(vfneg_half, vfhalf, _mm_cmpge_ps(Vector, vfzero));
    __m128 vres0 = _mm_add_ps(Vector, vinc0);
    return _mm_castps_si128(
            _mm_round_ps(vres0, _MM_FROUND_TO_ZERO | _MM_FROUND_NO_EXC));
#else
    GI_INT32 ret;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(float); i++) {
        ret[i] = (int32_t)round(Vector[i]);
    }
    return ret;
#endif
}

GI_FORCEINLINE
GI_FLOAT32
GiCastToFloat32(GI_INT32 Vector) {
#if defined(GI_NEON_INTRINSICS)
    return vcvtq_f32_s32(Vector);
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_cvtepi32_ps(Vector);
#else
    GI_FLOAT32 ret;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(int32_t); i++) {
        ret[i] = float(Vector[i]);
    }
    return ret;
#endif
}

GI_FORCEINLINE
GI_FLOAT32
GiReinterpretAsFloat32(GI_INT32 Vector) {
#if defined(GI_NEON_INTRINSICS)
    return vreinterpretq_f32_s32(Vector);
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_castsi128_ps(Vector);
#else
    return GI_FLOAT32(Vector);
#endif
}

GI_FORCEINLINE
GI_FLOAT32
GiBroadcastFloat32(float Value) {
#if defined(GI_NEON_INTRINSICS)
    return vdupq_n_f32(Value);
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_set1_ps(Value);
#else
    GI_FLOAT32 ret;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(float); i++) {
        ret[i] = Value;
    }
    return ret;
#endif
}

GI_FORCEINLINE
GI_FLOAT32
GiBroadcastFloat32(const float* Value) {
#if defined(GI_NEON_INTRINSICS)
    return vld1q_dup_f32(Value);
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_load_ps1(Value);
#else
    GI_FLOAT32 ret;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(float); i++) {
        ret[i] = *Value;
    }
    return ret;
#endif
}

GI_FORCEINLINE
GI_FLOAT32
GiZeroFloat32(void) {
#if defined(GI_NEON_INTRINSICS)
    return vdupq_n_f32(0.0f);
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_setzero_ps();
#else
    return GiBroadcastFloat32(0.0f);
#endif
}

GI_FORCEINLINE
GI_FLOAT32
GiLoadFloat32(const float* Buffer) {
#if defined(GI_NEON_INTRINSICS)
    return vld1q_f32(Buffer);
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_loadu_ps(Buffer);
#else
    GI_FLOAT32 ret;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(float); i++) {
        ret[i] = Buffer[i];
    }
    return ret;
#endif
}

GI_FORCEINLINE
void GiStoreFloat32(float* Buffer, GI_FLOAT32 Vector) {
#if defined(GI_NEON_INTRINSICS)
    vst1q_f32(Buffer, Vector);
#elif defined(GI_SSE2_INTRINSICS)
    _mm_storeu_ps(Buffer, Vector);
#else
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(float); i++) {
        Buffer[i] = Vector[i];
    }
#endif
}

GI_FORCEINLINE
void GiStoreAlignedFloat32(float* Buffer, GI_FLOAT32 Vector) {
#if defined(GI_NEON_INTRINSICS)
    vst1q_f32(Buffer, Vector);
#elif defined(GI_SSE2_INTRINSICS)
    _mm_store_ps(Buffer, Vector);
#else
    GiStoreFloat32(Buffer, Vector);
#endif
}

#if defined(GI_NEON_INTRINSICS)
#define GISTORELANEFLOAT32(i)                                                       \
    GI_FORCEINLINE void GiStoreLane##i##Float32(float* Buffer, GI_FLOAT32 Vector) { \
        vst1q_lane_f32(Buffer, Vector, i);                                          \
    }

#elif defined(GI_SSE2_INTRINSICS)

#define GISTORELANEFLOAT32(i)                                                          \
    GI_FORCEINLINE void GiStoreLane##i##Float32(float* Buffer, GI_FLOAT32 Vector) {    \
        _mm_store_ss(Buffer, _mm_shuffle_ps(Vector, Vector, _MM_SHUFFLE(i, i, i, i))); \
    }
#else
#define GISTORELANEFLOAT32(i)                                                       \
    GI_FORCEINLINE void GiStoreLane##i##Float32(float* Buffer, GI_FLOAT32 Vector) { \
        *Buffer = Vector[i];                                                        \
    }
#endif

GISTORELANEFLOAT32(0)
GISTORELANEFLOAT32(1)
GISTORELANEFLOAT32(2)
GISTORELANEFLOAT32(3)

#undef GISTORELANEFLOAT32

#if defined(GI_NEON_INTRINSICS)
#define GIEXTRACTLANEFLOAT32(i)                                         \
    GI_FORCEINLINE float GiExtractLane##i##Float32(GI_FLOAT32 Vector) { \
        return vgetq_lane_f32(Vector, i);                               \
    }
#elif defined(GI_SSE2_INTRINSICS)

#define GIEXTRACTLANEFLOAT32(i)                                                        \
    GI_FORCEINLINE float GiExtractLane##i##Float32(GI_FLOAT32 Vector) {                \
        return _mm_cvtss_f32(_mm_shuffle_ps(Vector, Vector, _MM_SHUFFLE(i, i, i, i))); \
    }
#else
#define GIEXTRACTLANEFLOAT32(i)                                         \
    GI_FORCEINLINE float GiExtractLane##i##Float32(GI_FLOAT32 Vector) { \
        return Vector[i];                                               \
    }
#endif

GIEXTRACTLANEFLOAT32(0)
GIEXTRACTLANEFLOAT32(1)
GIEXTRACTLANEFLOAT32(2)
GIEXTRACTLANEFLOAT32(3)
#undef GIEXTRACTLANEFLOAT32

GI_FORCEINLINE
GI_FLOAT32
GiInterleaveLowFloat32(GI_FLOAT32 Vector1, GI_FLOAT32 Vector2) {
#if defined(GI_NEON64_INTRINSICS)
    return vzip1q_f32(Vector1, Vector2);
#elif defined(GI_NEON32_INTRINSICS)
    float32x4x2_t zipped = vzipq_f32(Vector1, Vector2);
    return zipped.val[0];
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_unpacklo_ps(Vector1, Vector2);
#else
    GI_FLOAT32 ret;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / 2 / sizeof(float); i++) {
        ret[2 * i] = Vector1[i];
        ret[2 * i + 1] = Vector2[i];
    }
    return ret;
#endif
}

GI_FORCEINLINE
GI_FLOAT32
GiInterleaveHighFloat32(GI_FLOAT32 Vector1, GI_FLOAT32 Vector2) {
#if defined(GI_NEON64_INTRINSICS)
    return vzip2q_f32(Vector1, Vector2);
#elif defined(GI_NEON32_INTRINSICS)
    float32x4x2_t zipped = vzipq_f32(Vector1, Vector2);
    return zipped.val[1];
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_unpackhi_ps(Vector1, Vector2);
#else
    GI_FLOAT32 ret;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / 2 / sizeof(float); i++) {
        ret[2 * i] = Vector1[GI_SIMD_LEN_BYTE / 2 + i];
        ret[2 * i + 1] = Vector2[GI_SIMD_LEN_BYTE / 2 + i];
    }
    return ret;
#endif
}

GI_FORCEINLINE
GI_FLOAT32
GiAddFloat32(GI_FLOAT32 Vector1, GI_FLOAT32 Vector2) {
#if defined(GI_NEON_INTRINSICS)
    return vaddq_f32(Vector1, Vector2);
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_add_ps(Vector1, Vector2);
#else
    return Vector1 + Vector2;
#endif
}

GI_FORCEINLINE
GI_FLOAT32
GiSubtractFloat32(GI_FLOAT32 Vector1, GI_FLOAT32 Vector2) {
#if defined(GI_NEON_INTRINSICS)
    return vsubq_f32(Vector1, Vector2);
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_sub_ps(Vector1, Vector2);
#else
    return Vector1 - Vector2;
#endif
}

GI_FORCEINLINE
GI_FLOAT32
GiMultiplyFloat32(GI_FLOAT32 Vector1, GI_FLOAT32 Vector2) {
#if defined(GI_NEON_INTRINSICS)
    return vmulq_f32(Vector1, Vector2);
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_mul_ps(Vector1, Vector2);
#else
    return Vector1 * Vector2;
#endif
}

GI_FORCEINLINE
GI_FLOAT32
GiMultiplyScalerFloat32(GI_FLOAT32 Vector1, float Scaler) {
#if defined(GI_NEON_INTRINSICS)
    return vmulq_n_f32(Vector1, Scaler);
#elif defined(GI_SSE2_INTRINSICS)
    GI_FLOAT32 Vector2 = _mm_set1_ps(Scaler);
    return _mm_mul_ps(Vector1, Vector2);
#else
    return Vector1 * Scaler;
#endif
}

GI_FORCEINLINE
GI_FLOAT32
GiMultiplyAddVecFloat32(GI_FLOAT32 VectorSum, GI_FLOAT32 Vector1, GI_FLOAT32 Vector2) {
#if defined(GI_NEON_INTRINSICS)
    return vmlaq_f32(VectorSum, Vector1, Vector2);
#elif defined(GI_FMA3_INTRINSICS)
    return _mm_fmadd_ps(Vector1, Vector2, VectorSum);
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_add_ps(_mm_mul_ps(Vector1, Vector2), VectorSum);
#else
    return Vector1 * Vector2 + VectorSum;
#endif
}

GI_FORCEINLINE
GI_FLOAT32
GiMultiplyAddScalarFloat32(GI_FLOAT32 VectorSum, GI_FLOAT32 Vector, float Scalar) {
#if defined(GI_NEON_INTRINSICS)
    return vmlaq_n_f32(VectorSum, Vector, Scalar);
#elif defined(GI_SSE2_INTRINSICS)
    return GiMultiplyAddVecFloat32(VectorSum, GiBroadcastFloat32(Scalar), Vector);
#else
    return VectorSum + Vector * Scalar;
#endif
}

#if defined(GI_NEON_INTRINSICS)
#define GIMULTIPLYADDLANFLOAT32(i)                                           \
    GI_FORCEINLINE GI_FLOAT32 GiMultiplyAddLan##i##Float32(                  \
            GI_FLOAT32 VectorSum, GI_FLOAT32 Vector1, GI_FLOAT32 Vector2) {  \
        return vmlaq_lane_f32(VectorSum, Vector1, vget_low_f32(Vector2), i); \
    }
GIMULTIPLYADDLANFLOAT32(0)
GIMULTIPLYADDLANFLOAT32(1)
#undef GIMULTIPLYADDLANFLOAT32
#define GIMULTIPLYADDLANFLOAT32(i)                                                \
    GI_FORCEINLINE GI_FLOAT32 GiMultiplyAddLan##i##Float32(                       \
            GI_FLOAT32 VectorSum, GI_FLOAT32 Vector1, GI_FLOAT32 Vector2) {       \
        return vmlaq_lane_f32(VectorSum, Vector1, vget_high_f32(Vector2), i - 2); \
    }
GIMULTIPLYADDLANFLOAT32(2)
GIMULTIPLYADDLANFLOAT32(3)
#undef GIMULTIPLYADDLANFLOAT32
#elif defined(GI_SSE2_INTRINSICS)

#define GIMULTIPLYADDLANFLOAT32(i)                                          \
    GI_FORCEINLINE GI_FLOAT32 GiMultiplyAddLan##i##Float32(                 \
            GI_FLOAT32 VectorSum, GI_FLOAT32 Vector1, GI_FLOAT32 Vector2) { \
        return GiMultiplyAddScalarFloat32(                                  \
                VectorSum, Vector1, GiExtractLane##i##Float32(Vector2));    \
    }
GIMULTIPLYADDLANFLOAT32(0)
GIMULTIPLYADDLANFLOAT32(1)
GIMULTIPLYADDLANFLOAT32(2)
GIMULTIPLYADDLANFLOAT32(3)
#undef GIMULTIPLYADDLANFLOAT32
#else
#define GIMULTIPLYADDLANFLOAT32(i)                                          \
    GI_FORCEINLINE GI_FLOAT32 GiMultiplyAddLan##i##Float32(                 \
            GI_FLOAT32 VectorSum, GI_FLOAT32 Vector1, GI_FLOAT32 Vector2) { \
        return VectorSum + Vector1 * Vector2[i];                            \
    }
GIMULTIPLYADDLANFLOAT32(0)
GIMULTIPLYADDLANFLOAT32(1)
GIMULTIPLYADDLANFLOAT32(2)
GIMULTIPLYADDLANFLOAT32(3)
#undef GIMULTIPLYADDLANFLOAT32
#endif

GI_FORCEINLINE
GI_FLOAT32
GiDivideFloat32(GI_FLOAT32 Vector1, GI_FLOAT32 Vector2) {
#if defined(GI_NEON64_INTRINSICS)
    return vdivq_f32(Vector1, Vector2);
#elif defined(GI_NEON32_INTRINSICS)
    float32x4_t recp = vrecpeq_f32(Vector2);
    recp = vmulq_f32(vrecpsq_f32(Vector2, recp), recp);
    return vmulq_f32(Vector1, recp);
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_div_ps(Vector1, Vector2);
#else
    return Vector1 / Vector2;
#endif
}

GI_FORCEINLINE
GI_FLOAT32
GiGreaterThanFloat32(GI_FLOAT32 Vector1, GI_FLOAT32 Vector2) {
#if defined(GI_NEON_INTRINSICS)
    return vreinterpretq_f32_u32(vcgtq_f32(Vector1, Vector2));
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_cmpgt_ps(Vector1, Vector2);
#else
    return Vector1 > Vector2;
#endif
}

GI_FORCEINLINE
GI_FLOAT32
GiAndFloat32(GI_FLOAT32 Vector1, GI_FLOAT32 Vector2) {
#if defined(GI_SSE2_INTRINSICS)
    return _mm_and_ps(Vector1, Vector2);
#else
    return GiReinterpretAsFloat32(
            GiAndInt32(GiReinterpretAsInt32(Vector1), GiReinterpretAsInt32(Vector2)));
#endif
}

GI_FORCEINLINE
GI_FLOAT32
GiOrFloat32(GI_FLOAT32 Vector1, GI_FLOAT32 Vector2) {
#if defined(GI_SSE2_INTRINSICS)
    return _mm_or_ps(Vector1, Vector2);
#else
    return GiReinterpretAsFloat32(
            GiOrInt32(GiReinterpretAsInt32(Vector1), GiReinterpretAsInt32(Vector2)));
#endif
}

GI_FORCEINLINE
GI_FLOAT32
GiAndNotFloat32(GI_FLOAT32 VectorNot, GI_FLOAT32 Vector) {
#if defined(GI_SSE2_INTRINSICS)
    return _mm_andnot_ps(VectorNot, Vector);
#else
    return GiReinterpretAsFloat32(GiAndNotInt32(
            GiReinterpretAsInt32(VectorNot), GiReinterpretAsInt32(Vector)));
#endif
}

GI_FORCEINLINE
GI_FLOAT32
GiXorFloat32(GI_FLOAT32 Vector1, GI_FLOAT32 Vector2) {
#if defined(GI_SSE2_INTRINSICS)
    return _mm_xor_ps(Vector1, Vector2);
#else
    return GiReinterpretAsFloat32(
            GiXorInt32(GiReinterpretAsInt32(Vector1), GiReinterpretAsInt32(Vector2)));
#endif
}

GI_FORCEINLINE
GI_FLOAT32
GiBlendFloat32(GI_FLOAT32 Vector1, GI_FLOAT32 Vector2, GI_FLOAT32 Selection) {
    return GiOrFloat32(
            GiAndFloat32(Vector2, Selection), GiAndNotFloat32(Selection, Vector1));
}

#define MIN_NAN(a, b) (isnan(a) || (a) < (b)) ? (a) : (b);
#define MAX_NAN(a, b) (isnan(a) || (a) > (b)) ? (a) : (b);

GI_FORCEINLINE
GI_FLOAT32
GiMaximumFloat32(GI_FLOAT32 Vector1, GI_FLOAT32 Vector2) {
#if defined(GI_NEON_INTRINSICS)
    return vmaxq_f32(Vector1, Vector2);
#else
    //! _mm_max_ps does not fellow the IEEE standard when input is NAN, so
    //! implement by C code
    GI_FLOAT32 max;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(float); i++) {
        max[i] = MAX_NAN(Vector1[i], Vector2[i]);
    }
    return max;
#endif
}

GI_FORCEINLINE
GI_FLOAT32
GiMinimumFloat32(GI_FLOAT32 Vector1, GI_FLOAT32 Vector2) {
#if defined(GI_NEON_INTRINSICS)
    return vminq_f32(Vector1, Vector2);
#else
    //! _mm_min_ps does not fellow the IEEE standard when input is NAN, so
    //! implement by C code
    GI_FLOAT32 min;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(float); i++) {
        min[i] = MIN_NAN(Vector1[i], Vector2[i]);
    }
    return min;
#endif
}

GI_FORCEINLINE
GI_FLOAT32
GiClampFloat32(GI_FLOAT32 Value, float LowerRange, float UpperRange) {
    Value = GiMaximumFloat32(GiBroadcastFloat32(LowerRange), Value);
    Value = GiMinimumFloat32(GiBroadcastFloat32(UpperRange), Value);
    return Value;
}

GI_FORCEINLINE
float GiReduceAddFloat32(GI_FLOAT32 Vector) {
#if defined(GI_NEON64_INTRINSICS)
    Vector = vpaddq_f32(Vector, Vector);
    Vector = vpaddq_f32(Vector, Vector);
    return vgetq_lane_f32(Vector, 0);
#elif defined(GI_NEON32_INTRINSICS)
    float32x2_t VectorLow = vget_low_f32(Vector);
    float32x2_t VectorHigh = vget_high_f32(Vector);
    VectorLow = vpadd_f32(VectorLow, VectorHigh);
    VectorLow = vpadd_f32(VectorLow, VectorHigh);
    return vget_lane_f32(VectorLow, 0);
#elif defined(GI_SSE2_INTRINSICS)
    Vector = GiAddFloat32(
            Vector, _mm_shuffle_ps(Vector, Vector, _MM_SHUFFLE(2, 3, 2, 3)));
    Vector = GiAddFloat32(
            Vector, _mm_shuffle_ps(Vector, Vector, _MM_SHUFFLE(1, 1, 1, 1)));
    return GiExtractLane0Float32(Vector);
#else
    float ret = 0;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(float); i++) {
        ret += Vector[i];
    }
    return ret;
#endif
}

GI_FORCEINLINE
float GiReduceMultiplyFloat32(GI_FLOAT32 Vector) {
#if defined(GI_NEON64_INTRINSICS)
    float32x2_t low = vget_low_f32(Vector);
    float32x2_t high = vget_high_f32(Vector);
    float32x2_t res = vmul_f32(low, high);
    return vget_lane_f32(res, 0) * vget_lane_f32(res, 1);
#elif defined(GI_SSE2_INTRINSICS)
    Vector = GiMultiplyFloat32(
            Vector, _mm_shuffle_ps(Vector, Vector, _MM_SHUFFLE(2, 3, 2, 3)));
    Vector = GiMultiplyFloat32(
            Vector, _mm_shuffle_ps(Vector, Vector, _MM_SHUFFLE(1, 1, 1, 1)));
    return GiExtractLane0Float32(Vector);
#else
    float ret = 1;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(float); i++) {
        ret *= Vector[i];
    }
    return ret;
#endif
}

#define Max(a, b) (a) > (b) ? (a) : (b)
#define Min(a, b) (a) < (b) ? (a) : (b)

GI_FORCEINLINE
float GiReduceMaximumFloat32(GI_FLOAT32 Vector) {
#if defined(GI_NEON64_INTRINSICS)
    return vmaxvq_f32(Vector);
#elif defined(GI_NEON32_INTRINSICS)
    float32x2_t VectorLow = vget_low_f32(Vector);
    float32x2_t VectorHigh = vget_high_f32(Vector);
    VectorLow = vpmax_f32(VectorLow, VectorHigh);
    VectorLow = vpmax_f32(VectorLow, VectorHigh);
    return vget_lane_f32(VectorLow, 0);
#elif defined(GI_SSE2_INTRINSICS)
    Vector = GiMaximumFloat32(
            Vector, _mm_shuffle_ps(Vector, Vector, _MM_SHUFFLE(2, 3, 2, 3)));
    Vector = GiMaximumFloat32(
            Vector, _mm_shuffle_ps(Vector, Vector, _MM_SHUFFLE(1, 1, 1, 1)));
    return GiExtractLane0Float32(Vector);
#else
    float ret = Vector[0];
    for (size_t i = 1; i < GI_SIMD_LEN_BYTE / sizeof(float); i++) {
        ret = MAX_NAN(ret, Vector[i]);
    }
    return ret;
#endif
}

GI_FORCEINLINE
float GiReduceMinimumFloat32(GI_FLOAT32 Vector) {
#if defined(GI_NEON64_INTRINSICS)
    return vminvq_f32(Vector);
#elif defined(GI_NEON32_INTRINSICS)
    float32x2_t VectorLow = vget_low_f32(Vector);
    float32x2_t VectorHigh = vget_high_f32(Vector);
    VectorLow = vpmin_f32(VectorLow, VectorHigh);
    VectorLow = vpmin_f32(VectorLow, VectorHigh);
    return vget_lane_f32(VectorLow, 0);
#elif defined(GI_SSE2_INTRINSICS)
    Vector = GiMinimumFloat32(
            Vector, _mm_shuffle_ps(Vector, Vector, _MM_SHUFFLE(2, 3, 2, 3)));
    Vector = GiMinimumFloat32(
            Vector, _mm_shuffle_ps(Vector, Vector, _MM_SHUFFLE(1, 1, 1, 1)));
    return GiExtractLane0Float32(Vector);
#else
    float ret = Vector[0];
    for (size_t i = 1; i < GI_SIMD_LEN_BYTE / sizeof(float); i++) {
        ret = MIN_NAN(ret, Vector[i]);
    }
    return ret;
#endif
}

// vim: syntax=cpp.doxygen
