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
GI_INT32_t GiReinterpretAsInt32(GI_FLOAT32_t In) {
#if defined(GI_NEON_INTRINSICS)
    return vreinterpretq_s32_f32(In);
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_castps_si128(In);
#else
    return (GI_INT32_t)In;
#endif
}

GI_FORCEINLINE
GI_UINT32_t GiReinterpretAsUint32(GI_FLOAT32_t In) {
#if defined(GI_NEON_INTRINSICS)
    return vreinterpretq_u32_f32(In);
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_castps_si128(In);
#else
    return (GI_UINT32_t)In;
#endif
}

GI_FORCEINLINE
GI_FLOAT32_t GiReintInt32ToFloat32(GI_INT32_t Vector) {
#if defined(GI_NEON_INTRINSICS)
    return vreinterpretq_f32_s32(Vector);
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_castsi128_ps(Vector);
#else
    return (GI_FLOAT32_t)Vector;
#endif
}

GI_FORCEINLINE
GI_FLOAT32_t GiReintUint32ToFloat32(GI_UINT32_t Vector) {
#if defined(GI_NEON_INTRINSICS)
    return vreinterpretq_f32_u32(Vector);
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_castsi128_ps(Vector);
#else
    return (GI_FLOAT32_t)Vector;
#endif
}

GI_FORCEINLINE
GI_INT32_t GiRoundAsInt32(GI_FLOAT32_t Vector) {
#if defined(GI_NEON_INTRINSICS)
#if __ARM_ARCH >= 8
    return vcvtaq_s32_f32(Vector);
#else
    float32x4_t vinc0 = vbslq_f32(vcgeq_f32(Vector, vfzero), vfhalf, vfneg_half);
    return vcvtq_s32_f32(vaddq_f32(Vector, vinc0));
#endif
#elif defined(GI_SSE42_INTRINSICS)
    __m128 vinc0 = _mm_blendv_ps(vfneg_half, vfhalf, _mm_cmpge_ps(Vector, vfzero));
    return _mm_cvttps_epi32(_mm_add_ps(Vector, vinc0));
#else
    GI_INT32_t ret;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(float); i++) {
        ret[i] = (int32_t)round(Vector[i]);
    }
    return ret;
#endif
}

GI_FORCEINLINE
GI_INT32_t GiCastToInt32(GI_FLOAT32_t Vector) {
#if defined(GI_NEON_INTRINSICS)
    return vcvtq_s32_f32(Vector);
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_cvttps_epi32(Vector);
#else
    GI_INT32_t ret;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(float); i++) {
        ret[i] = (int32_t)(Vector[i]);
    }
    return ret;
#endif
}

GI_FORCEINLINE
GI_FLOAT32_t GiCastToFloat32(GI_INT32_t Vector) {
#if defined(GI_NEON_INTRINSICS)
    return vcvtq_f32_s32(Vector);
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_cvtepi32_ps(Vector);
#else
    GI_FLOAT32_t ret;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(int32_t); i++) {
        ret[i] = (float)Vector[i];
    }
    return ret;
#endif
}

GI_FORCEINLINE
GI_FLOAT32_t GiLoadBroadcastFloat32(const float* Value) {
#if defined(GI_NEON_INTRINSICS)
    return vld1q_dup_f32(Value);
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_load_ps1(Value);
#else
    GI_FLOAT32_t ret;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(float); i++) {
        ret[i] = *Value;
    }
    return ret;
#endif
}

GI_FORCEINLINE
GI_FLOAT32_t GiZeroFloat32(void) {
#if defined(GI_NEON_INTRINSICS)
    return vdupq_n_f32(0.0f);
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_setzero_ps();
#else
    return GiBroadcastFloat32(0.0f);
#endif
}

GI_FORCEINLINE
GI_FLOAT32_t GiLoadFloat32(const float* Buffer) {
#if defined(GI_NEON_INTRINSICS)
    return vld1q_f32(Buffer);
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_loadu_ps(Buffer);
#else
    GI_FLOAT32_t ret;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(float); i++) {
        ret[i] = Buffer[i];
    }
    return ret;
#endif
}

GI_FORCEINLINE
void GiStoreFloat32(float* Buffer, GI_FLOAT32_t Vector) {
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

#if defined(GI_NEON_INTRINSICS)
#define GISTORELANEFLOAT32(i)                                                         \
    GI_FORCEINLINE void GiStoreLane##i##Float32(float* Buffer, GI_FLOAT32_t Vector) { \
        vst1q_lane_f32(Buffer, Vector, i);                                            \
    }

#elif defined(GI_SSE2_INTRINSICS)

#define GISTORELANEFLOAT32(i)                                                          \
    GI_FORCEINLINE void GiStoreLane##i##Float32(float* Buffer, GI_FLOAT32_t Vector) {  \
        _mm_store_ss(Buffer, _mm_shuffle_ps(Vector, Vector, _MM_SHUFFLE(i, i, i, i))); \
    }
#else
#define GISTORELANEFLOAT32(i)                                                         \
    GI_FORCEINLINE void GiStoreLane##i##Float32(float* Buffer, GI_FLOAT32_t Vector) { \
        *Buffer = Vector[i];                                                          \
    }
#endif

GISTORELANEFLOAT32(0)
GISTORELANEFLOAT32(1)
GISTORELANEFLOAT32(2)
GISTORELANEFLOAT32(3)

#undef GISTORELANEFLOAT32

#if defined(GI_NEON_INTRINSICS)
#define GIEXTRACTLANEFLOAT32(i)                                           \
    GI_FORCEINLINE float GiExtractLane##i##Float32(GI_FLOAT32_t Vector) { \
        return vgetq_lane_f32(Vector, i);                                 \
    }
#elif defined(GI_SSE2_INTRINSICS)

#define GIEXTRACTLANEFLOAT32(i)                                                        \
    GI_FORCEINLINE float GiExtractLane##i##Float32(GI_FLOAT32_t Vector) {              \
        return _mm_cvtss_f32(_mm_shuffle_ps(Vector, Vector, _MM_SHUFFLE(i, i, i, i))); \
    }
#else
#define GIEXTRACTLANEFLOAT32(i)                                           \
    GI_FORCEINLINE float GiExtractLane##i##Float32(GI_FLOAT32_t Vector) { \
        return Vector[i];                                                 \
    }
#endif

GIEXTRACTLANEFLOAT32(0)
GIEXTRACTLANEFLOAT32(1)
GIEXTRACTLANEFLOAT32(2)
GIEXTRACTLANEFLOAT32(3)
#undef GIEXTRACTLANEFLOAT32

GI_FORCEINLINE
GI_FLOAT32_t GiInterleaveLowFloat32(GI_FLOAT32_t Vector1, GI_FLOAT32_t Vector2) {
#if defined(GI_NEON64_INTRINSICS)
    return vzip1q_f32(Vector1, Vector2);
#elif defined(GI_NEON32_INTRINSICS)
    float32x4x2_t zipped = vzipq_f32(Vector1, Vector2);
    return zipped.val[0];
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_unpacklo_ps(Vector1, Vector2);
#else
    GI_FLOAT32_t ret;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / 2 / sizeof(float); i++) {
        ret[2 * i] = Vector1[i];
        ret[2 * i + 1] = Vector2[i];
    }
    return ret;
#endif
}

GI_FORCEINLINE
GI_FLOAT32_t GiInterleaveHighFloat32(GI_FLOAT32_t Vector1, GI_FLOAT32_t Vector2) {
#if defined(GI_NEON64_INTRINSICS)
    return vzip2q_f32(Vector1, Vector2);
#elif defined(GI_NEON32_INTRINSICS)
    float32x4x2_t zipped = vzipq_f32(Vector1, Vector2);
    return zipped.val[1];
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_unpackhi_ps(Vector1, Vector2);
#else
    GI_FLOAT32_t ret;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / 2 / sizeof(float); i++) {
        ret[2 * i] = Vector1[GI_SIMD_LEN_BYTE / 2 + i];
        ret[2 * i + 1] = Vector2[GI_SIMD_LEN_BYTE / 2 + i];
    }
    return ret;
#endif
}

GI_FORCEINLINE
GI_FLOAT32_t GiAddFloat32(GI_FLOAT32_t Vector1, GI_FLOAT32_t Vector2) {
#if defined(GI_NEON_INTRINSICS)
    return vaddq_f32(Vector1, Vector2);
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_add_ps(Vector1, Vector2);
#else
    return Vector1 + Vector2;
#endif
}

GI_FORCEINLINE
GI_FLOAT32_t GiSubtractFloat32(GI_FLOAT32_t Vector1, GI_FLOAT32_t Vector2) {
#if defined(GI_NEON_INTRINSICS)
    return vsubq_f32(Vector1, Vector2);
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_sub_ps(Vector1, Vector2);
#else
    return Vector1 - Vector2;
#endif
}

GI_FORCEINLINE
GI_FLOAT32_t GiMultiplyFloat32(GI_FLOAT32_t Vector1, GI_FLOAT32_t Vector2) {
#if defined(GI_NEON_INTRINSICS)
    return vmulq_f32(Vector1, Vector2);
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_mul_ps(Vector1, Vector2);
#else
    return Vector1 * Vector2;
#endif
}

GI_FORCEINLINE
GI_FLOAT32_t GiMultiplyScalerFloat32(GI_FLOAT32_t Vector1, float Scaler) {
#if defined(GI_NEON_INTRINSICS)
    return vmulq_n_f32(Vector1, Scaler);
#elif defined(GI_SSE2_INTRINSICS)
    GI_FLOAT32_t Vector2 = _mm_set1_ps(Scaler);
    return _mm_mul_ps(Vector1, Vector2);
#else
    return Vector1 * Scaler;
#endif
}

GI_FORCEINLINE
GI_FLOAT32_t GiMultiplyAddFloat32(
        GI_FLOAT32_t VectorSum, GI_FLOAT32_t Vector1, GI_FLOAT32_t Vector2) {
#if defined(GI_NEON_INTRINSICS)
    return v_fma_ps_f32(VectorSum, Vector1, Vector2);
#elif defined(GI_FMA3_INTRINSICS)
    return _mm_fmadd_ps(Vector1, Vector2, VectorSum);
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_add_ps(_mm_mul_ps(Vector1, Vector2), VectorSum);
#else
    return Vector1 * Vector2 + VectorSum;
#endif
}

GI_FORCEINLINE
GI_FLOAT32_t GiMultiplySubFloat32(
        GI_FLOAT32_t VectorSum, GI_FLOAT32_t Vector1, GI_FLOAT32_t Vector2) {
#if defined(GI_NEON_INTRINSICS)
    return vmlsq_f32(VectorSum, Vector1, Vector2);
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_sub_ps(VectorSum, _mm_mul_ps(Vector1, Vector2));
#else
    return VectorSum - Vector1 * Vector2;
#endif
}

GI_FORCEINLINE
GI_FLOAT32_t GiMultiplyAddScalarFloat32(
        GI_FLOAT32_t VectorSum, GI_FLOAT32_t Vector, float Scalar) {
#if defined(GI_NEON_INTRINSICS)
    return v_fma_n_f32(VectorSum, Vector, Scalar);
#elif defined(GI_SSE2_INTRINSICS)
    return GiMultiplyAddFloat32(VectorSum, GiBroadcastFloat32(Scalar), Vector);
#else
    return VectorSum + Vector * Scalar;
#endif
}

#if defined(GI_NEON_INTRINSICS)
#define GIMULTIPLYADDLANFLOAT32(i)                                                \
    GI_FORCEINLINE GI_FLOAT32_t GiMultiplyAddLan##i##Float32(                     \
            GI_FLOAT32_t VectorSum, GI_FLOAT32_t Vector1, GI_FLOAT32_t Vector2) { \
        return v_fma_lane_f32(VectorSum, Vector1, vget_low_f32(Vector2), i);      \
    }
GIMULTIPLYADDLANFLOAT32(0)
GIMULTIPLYADDLANFLOAT32(1)
#undef GIMULTIPLYADDLANFLOAT32
#define GIMULTIPLYADDLANFLOAT32(i)                                                \
    GI_FORCEINLINE GI_FLOAT32_t GiMultiplyAddLan##i##Float32(                     \
            GI_FLOAT32_t VectorSum, GI_FLOAT32_t Vector1, GI_FLOAT32_t Vector2) { \
        return v_fma_lane_f32(VectorSum, Vector1, vget_high_f32(Vector2), i - 2); \
    }
GIMULTIPLYADDLANFLOAT32(2)
GIMULTIPLYADDLANFLOAT32(3)
#undef GIMULTIPLYADDLANFLOAT32
#elif defined(GI_SSE2_INTRINSICS)

#define GIMULTIPLYADDLANFLOAT32(i)                                                \
    GI_FORCEINLINE GI_FLOAT32_t GiMultiplyAddLan##i##Float32(                     \
            GI_FLOAT32_t VectorSum, GI_FLOAT32_t Vector1, GI_FLOAT32_t Vector2) { \
        return GiMultiplyAddScalarFloat32(                                        \
                VectorSum, Vector1, GiExtractLane##i##Float32(Vector2));          \
    }
GIMULTIPLYADDLANFLOAT32(0)
GIMULTIPLYADDLANFLOAT32(1)
GIMULTIPLYADDLANFLOAT32(2)
GIMULTIPLYADDLANFLOAT32(3)
#undef GIMULTIPLYADDLANFLOAT32
#else
#define GIMULTIPLYADDLANFLOAT32(i)                                                \
    GI_FORCEINLINE GI_FLOAT32_t GiMultiplyAddLan##i##Float32(                     \
            GI_FLOAT32_t VectorSum, GI_FLOAT32_t Vector1, GI_FLOAT32_t Vector2) { \
        return VectorSum + Vector1 * Vector2[i];                                  \
    }
GIMULTIPLYADDLANFLOAT32(0)
GIMULTIPLYADDLANFLOAT32(1)
GIMULTIPLYADDLANFLOAT32(2)
GIMULTIPLYADDLANFLOAT32(3)
#undef GIMULTIPLYADDLANFLOAT32
#endif

GI_FORCEINLINE
GI_FLOAT32_t GiDivideFloat32(GI_FLOAT32_t Vector1, GI_FLOAT32_t Vector2) {
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
GI_FLOAT32_t GiRecpeSFloat32(GI_FLOAT32_t Vector1, GI_FLOAT32_t Vector2) {
#if defined(GI_NEON64_INTRINSICS)
    return vrecpsq_f32(Vector1, Vector2);
#elif defined(GI_SSE2_INTRINSICS)
    GI_FLOAT32_t two = _mm_set1_ps(2.0f);
    return _mm_sub_ps(two, _mm_mul_ps(Vector1, Vector2));
#else
    return (2.0f - Vector1 * Vector2);
#endif
}

GI_FORCEINLINE
GI_FLOAT32_t GiRecpeFloat32(GI_FLOAT32_t Vector) {
#if defined(GI_NEON32_INTRINSICS)
    return vrecpeq_f32(Vector);
#elif defined(GI_SSE2_INTRINSICS)
    GI_FLOAT32_t ones = _mm_set1_ps(1.0f);
    return _mm_div_ps(ones, Vector);
#else
    return 1 / Vector;
#endif
}

GI_FORCEINLINE
GI_FLOAT32_t GiNegFloat32(GI_FLOAT32_t Vector) {
#if defined(GI_NEON32_INTRINSICS)
    return vnegq_f32(Vector);
#elif defined(GI_SSE2_INTRINSICS)
    GI_FLOAT32_t zero = _mm_set1_ps(0.0f);
    return _mm_sub_ps(zero, Vector);
#else
    return -Vector;
#endif
}

GI_FORCEINLINE
GI_UINT32_t GiGreaterThanFloat32(GI_FLOAT32_t Vector1, GI_FLOAT32_t Vector2) {
#if defined(GI_NEON_INTRINSICS)
    return vcgtq_f32(Vector1, Vector2);
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_castps_si128(_mm_cmpgt_ps(Vector1, Vector2));
#else
    GI_UINT32_t ret;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(float); i++) {
        ret[i] = Vector1[i] > Vector2[i] ? 0xFFFFFFFF : 0;
    }
    return ret;
#endif
}

GI_FORCEINLINE
GI_UINT32_t GiLessThanEqFloat32(GI_FLOAT32_t Vector1, GI_FLOAT32_t Vector2) {
#if defined(GI_NEON_INTRINSICS)
    return vcleq_f32(Vector1, Vector2);
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_castps_si128(_mm_cmple_ps(Vector1, Vector2));
#else
    GI_UINT32_t ret;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(float); i++) {
        ret[i] = Vector1[i] <= Vector2[i] ? 0xFFFFFFFF : 0;
    }
    return ret;
#endif
}

GI_FORCEINLINE
GI_UINT32_t GiLessThanFloat32(GI_FLOAT32_t Vector1, GI_FLOAT32_t Vector2) {
#if defined(GI_NEON_INTRINSICS)
    return vcltq_f32(Vector1, Vector2);
#elif defined(GI_SSE2_INTRINSICS)
    return _mm_castps_si128(_mm_cmplt_ps(Vector1, Vector2));
#else
    GI_UINT32_t ret;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(float); i++) {
        ret[i] = Vector1[i] < Vector2[i] ? 0xFFFFFFFF : 0;
    }
    return ret;
#endif
}

GI_FORCEINLINE
GI_FLOAT32_t GiAndFloat32(GI_FLOAT32_t Vector1, GI_FLOAT32_t Vector2) {
#if defined(GI_SSE2_INTRINSICS)
    return _mm_and_ps(Vector1, Vector2);
#else
    return GiReintInt32ToFloat32(
            GiAndInt32(GiReinterpretAsInt32(Vector1), GiReinterpretAsInt32(Vector2)));
#endif
}

GI_FORCEINLINE
GI_FLOAT32_t GiOrFloat32(GI_FLOAT32_t Vector1, GI_FLOAT32_t Vector2) {
#if defined(GI_SSE2_INTRINSICS)
    return _mm_or_ps(Vector1, Vector2);
#else
    return GiReintInt32ToFloat32(
            GiOrInt32(GiReinterpretAsInt32(Vector1), GiReinterpretAsInt32(Vector2)));
#endif
}

GI_FORCEINLINE
GI_FLOAT32_t GiAndNotFloat32(GI_FLOAT32_t VectorNot, GI_FLOAT32_t Vector) {
#if defined(GI_SSE2_INTRINSICS)
    return _mm_andnot_ps(VectorNot, Vector);
#else
    return GiReintInt32ToFloat32(GiAndNotInt32(
            GiReinterpretAsInt32(VectorNot), GiReinterpretAsInt32(Vector)));
#endif
}

GI_FORCEINLINE
GI_FLOAT32_t GiXorFloat32(GI_FLOAT32_t Vector1, GI_FLOAT32_t Vector2) {
#if defined(GI_SSE2_INTRINSICS)
    return _mm_xor_ps(Vector1, Vector2);
#else
    return GiReintInt32ToFloat32(
            GiXorInt32(GiReinterpretAsInt32(Vector1), GiReinterpretAsInt32(Vector2)));
#endif
}

GI_FORCEINLINE
GI_FLOAT32_t GiBlendFloat32(
        GI_FLOAT32_t Vector1, GI_FLOAT32_t Vector2, GI_FLOAT32_t Selection) {
    return GiOrFloat32(
            GiAndFloat32(Vector2, Selection), GiAndNotFloat32(Selection, Vector1));
}

#define MIN_NAN(a, b) (isnan(a) || (a) < (b)) ? (a) : (b);
#define MAX_NAN(a, b) (isnan(a) || (a) > (b)) ? (a) : (b);

GI_FORCEINLINE
GI_FLOAT32_t GiBSLFloat32(
        GI_UINT32_t Selection, GI_FLOAT32_t Vector1, GI_FLOAT32_t Vector2) {
#if defined(GI_NEON_INTRINSICS)
    return vbslq_f32(Selection, Vector1, Vector2);
#else
    return GiBlendFloat32(Vector1, Vector2, GiReintUint32ToFloat32(Selection));
#endif
}

GI_FORCEINLINE
GI_FLOAT32_t GiMaximumFloat32(GI_FLOAT32_t Vector1, GI_FLOAT32_t Vector2) {
#if defined(GI_NEON_INTRINSICS)
    return vmaxq_f32(Vector1, Vector2);
#elif defined(GI_NEON32_INTRINSICS)
    return _mm_max_ps(Vector1, Vector2);
#else
    GI_FLOAT32_t max;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(float); i++) {
        max[i] = Max(Vector1[i], Vector2[i]);
    }
    return max;
#endif
}

GI_FORCEINLINE
GI_FLOAT32_t GiMinimumFloat32(GI_FLOAT32_t Vector1, GI_FLOAT32_t Vector2) {
#if defined(GI_NEON_INTRINSICS)
    return vminq_f32(Vector1, Vector2);
#elif defined(GI_NEON32_INTRINSICS)
    return _mm_min_ps(Vector1, Vector2);
#else
    GI_FLOAT32_t min;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(float); i++) {
        min[i] = Min(Vector1[i], Vector2[i]);
    }
    return min;
#endif
}

GI_FORCEINLINE
GI_FLOAT32_t GiMaxNanFloat32(GI_FLOAT32_t Vector1, GI_FLOAT32_t Vector2) {
#if defined(GI_NEON_INTRINSICS)
    return vmaxq_f32(Vector1, Vector2);
#else
    //! _mm_max_ps does not fellow the IEEE standard when input is NAN, so
    //! implement by C code
#define MAX_NAN(a, b) (isnan(a) || (a) > (b)) ? (a) : (b);
    GI_FLOAT32_t max;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(float); i++) {
        max[i] = MAX_NAN(Vector1[i], Vector2[i]);
    }
    return max;
#endif
}

GI_FORCEINLINE
GI_FLOAT32_t GiMinNanFloat32(GI_FLOAT32_t Vector1, GI_FLOAT32_t Vector2) {
#if defined(GI_NEON_INTRINSICS)
    return vminq_f32(Vector1, Vector2);
#else
    //! _mm_min_ps does not fellow the IEEE standard when input is NAN, so
    //! implement by C code
#define MIN_NAN(a, b) (isnan(a) || (a) < (b)) ? (a) : (b);
    GI_FLOAT32_t min;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(float); i++) {
        min[i] = MIN_NAN(Vector1[i], Vector2[i]);
    }
    return min;
#endif
}

GI_FORCEINLINE
GI_FLOAT32_t GiClampFloat32(GI_FLOAT32_t Value, float LowerRange, float UpperRange) {
    Value = GiMaximumFloat32(GiBroadcastFloat32(LowerRange), Value);
    Value = GiMinimumFloat32(GiBroadcastFloat32(UpperRange), Value);
    return Value;
}

GI_FORCEINLINE
float GiReduceAddFloat32(GI_FLOAT32_t Vector) {
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
float GiReduceMultiplyFloat32(GI_FLOAT32_t Vector) {
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
float GiReduceMaxNanFloat32(GI_FLOAT32_t Vector) {
#if defined(GI_NEON64_INTRINSICS)
    return vmaxvq_f32(Vector);
#elif defined(GI_NEON32_INTRINSICS)
    float32x2_t VectorLow = vget_low_f32(Vector);
    float32x2_t VectorHigh = vget_high_f32(Vector);
    VectorLow = vpmax_f32(VectorLow, VectorHigh);
    VectorLow = vpmax_f32(VectorLow, VectorHigh);
    return vget_lane_f32(VectorLow, 0);
#elif defined(GI_SSE2_INTRINSICS)
    Vector = GiMaxNanFloat32(
            Vector, _mm_shuffle_ps(Vector, Vector, _MM_SHUFFLE(2, 3, 2, 3)));
    Vector = GiMaxNanFloat32(
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
float GiReduceMinNanFloat32(GI_FLOAT32_t Vector) {
#if defined(GI_NEON64_INTRINSICS)
    return vminvq_f32(Vector);
#elif defined(GI_NEON32_INTRINSICS)
    float32x2_t VectorLow = vget_low_f32(Vector);
    float32x2_t VectorHigh = vget_high_f32(Vector);
    VectorLow = vpmin_f32(VectorLow, VectorHigh);
    VectorLow = vpmin_f32(VectorLow, VectorHigh);
    return vget_lane_f32(VectorLow, 0);
#elif defined(GI_SSE2_INTRINSICS)
    Vector = GiMinNanFloat32(
            Vector, _mm_shuffle_ps(Vector, Vector, _MM_SHUFFLE(2, 3, 2, 3)));
    Vector = GiMinNanFloat32(
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

GI_FORCEINLINE
GI_FLOAT32_t GiAbsFloat32(GI_FLOAT32_t Vector1) {
#if defined(GI_NEON64_INTRINSICS)
    return vabsq_f32(Vector1);
#elif defined(GI_SSE2_INTRINSICS)
    union {
        unsigned int int_val;
        float float_val;
    } value;
    value.int_val = 0x7fffffff;
    return _mm_and_ps(Vector1, _mm_set_ps1(value.float_val));
#else
    GI_FLOAT32_t ret;
    for (size_t i = 0; i < GI_SIMD_LEN_BYTE / sizeof(float); i++) {
        ret[i] = Vector1[i] > 0 ? Vector1[i] : -Vector1[i];
    }
    return ret;
#endif
}

// vim: syntax=cpp.doxygen
