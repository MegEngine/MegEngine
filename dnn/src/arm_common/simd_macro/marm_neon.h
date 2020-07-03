/**
 * \file dnn/src/arm_common/simd_macro/marm_neon.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#pragma once

#include <arm_neon.h>
#include "megdnn/arch.h"
#include "src/common/unroll_macro.h"

// GCC does not support __nodebug__, it reports:
// '__nodebug__' attribute directive ignored
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpragmas"
#pragma GCC diagnostic ignored "-Wattributes"
#define __ai      \
    static inline \
            __attribute__((__gnu_inline__, __always_inline__, __nodebug__))

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC && !MEGDNN_DISABLE_FLOAT16
#define MEGDNN_INC_ARM_FP16(_x) _x
#else
#define MEGDNN_INC_ARM_FP16(_x)
#endif  // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

//! copy from arm_neon, as in clang7.0 these function not exists
#ifdef __LITTLE_ENDIAN__
__ai float16x8_t vmlaq_f16(float16x8_t __p0, float16x8_t __p1,
                           float16x8_t __p2) {
    float16x8_t __ret;
    __ret = __p0 + __p1 * __p2;
    return __ret;
}
#else
__ai float16x8_t vmlaq_f16(float16x8_t __p0, float16x8_t __p1,
                           float16x8_t __p2) {
    float16x8_t __rev0;
    __rev0 = __builtin_shufflevector(__p0, __p0, 7, 6, 5, 4, 3, 2, 1, 0);
    float16x8_t __rev1;
    __rev1 = __builtin_shufflevector(__p1, __p1, 7, 6, 5, 4, 3, 2, 1, 0);
    float16x8_t __rev2;
    __rev2 = __builtin_shufflevector(__p2, __p2, 7, 6, 5, 4, 3, 2, 1, 0);
    float16x8_t __ret;
    __ret = __rev0 + __rev1 * __rev2;
    __ret = __builtin_shufflevector(__ret, __ret, 7, 6, 5, 4, 3, 2, 1, 0);
    return __ret;
}
#endif

#ifdef __LITTLE_ENDIAN__
#define vmlaq_lane_f16(__p0, __p1, __p2, __p3)                                \
    __extension__({                                                           \
        float16x8_t __s0 = __p0;                                              \
        float16x8_t __s1 = __p1;                                              \
        float16x4_t __s2 = __p2;                                              \
        float16x8_t __ret;                                                    \
        __ret = __s0 + __s1 * __builtin_shufflevector(__s2, __s2, __p3, __p3, \
                                                      __p3, __p3, __p3, __p3, \
                                                      __p3, __p3);            \
        __ret;                                                                \
    })
#else
#define vmlaq_lane_f16(__p0, __p1, __p2, __p3)                                 \
    __extension__({                                                            \
        float16x8_t __s0 = __p0;                                               \
        float16x8_t __s1 = __p1;                                               \
        float16x4_t __s2 = __p2;                                               \
        float16x8_t __rev0;                                                    \
        __rev0 = __builtin_shufflevector(__s0, __s0, 7, 6, 5, 4, 3, 2, 1, 0);  \
        float16x8_t __rev1;                                                    \
        __rev1 = __builtin_shufflevector(__s1, __s1, 7, 6, 5, 4, 3, 2, 1, 0);  \
        float16x4_t __rev2;                                                    \
        __rev2 = __builtin_shufflevector(__s2, __s2, 3, 2, 1, 0);              \
        float16x8_t __ret;                                                     \
        __ret = __rev0 + __rev1 * __builtin_shufflevector(                     \
                                          __rev2, __rev2, __p3, __p3, __p3,    \
                                          __p3, __p3, __p3, __p3, __p3);       \
        __ret = __builtin_shufflevector(__ret, __ret, 7, 6, 5, 4, 3, 2, 1, 0); \
        __ret;                                                                 \
    })
#endif

#if 0
//! As in arm_neon.h, `vdupq_n_f16` is macro, may be different with
//! `vdupq_n_f32`, So here just undefine the macro, and declare a function to
//! implement just as `vdupq_n_f32`.
#undef vdupq_n_f16
#ifdef __LITTLE_ENDIAN__
__ai float16x8_t vdupq_n_f16(float16_t __p0) {
    float16x8_t __ret;
    __ret = (float16x8_t){__p0, __p0, __p0, __p0, __p0, __p0, __p0, __p0};
    return __ret;
}

#else
__ai float16x8_t vdupq_n_f16(float16_t __p0) {
    float16x8_t __ret;
    __ret = (float16x8_t){__p0, __p0, __p0, __p0, __p0, __p0, __p0, __p0};
    __ret = __builtin_shufflevector(__ret, __ret, 7, 6, 5, 4, 3, 2, 1, 0);
    return __ret;
}
#endif
#endif

#ifdef __LITTLE_ENDIAN__
#define vmlaq_laneq_f16(__p0, __p1, __p2, __p3)                               \
    __extension__({                                                           \
        float16x8_t __s0 = __p0;                                              \
        float16x8_t __s1 = __p1;                                              \
        float16x8_t __s2 = __p2;                                              \
        float16x8_t __ret;                                                    \
        __ret = __s0 + __s1 * __builtin_shufflevector(__s2, __s2, __p3, __p3, \
                                                      __p3, __p3, __p3, __p3, \
                                                      __p3, __p3);            \
        __ret;                                                                \
    })
#else
#define vmlaq_laneq_f16(__p0, __p1, __p2, __p3)                                \
    __extension__({                                                            \
        float16x8_t __s0 = __p0;                                               \
        float16x8_t __s1 = __p1;                                               \
        float16x8_t __s2 = __p2;                                               \
        float16x8_t __rev0;                                                    \
        __rev0 = __builtin_shufflevector(__s0, __s0, 7, 6, 5, 4, 3, 2, 1, 0);  \
        float16x8_t __rev1;                                                    \
        __rev1 = __builtin_shufflevector(__s1, __s1, 7, 6, 5, 4, 3, 2, 1, 0);  \
        float16x8_t __rev2;                                                    \
        __rev2 = __builtin_shufflevector(__s2, __s2, 7, 6, 5, 4, 3, 2, 1, 0);  \
        float16x8_t __ret;                                                     \
        __ret = __rev0 + __rev1 * __builtin_shufflevector(                     \
                                          __rev2, __rev2, __p3, __p3, __p3,    \
                                          __p3, __p3, __p3, __p3, __p3);       \
        __ret = __builtin_shufflevector(__ret, __ret, 7, 6, 5, 4, 3, 2, 1, 0); \
        __ret;                                                                 \
    })
#endif

#if MEGDNN_ARMV7
#define vmlaq_low_lane_f16(__a, __b, __v, __lane)         \
    __extension__({                                       \
        auto c = vget_low_f16(__v);                       \
        auto __ret = vmlaq_lane_f16(__a, __b, c, __lane); \
        __ret;                                            \
    })

#define vmlaq_high_lane_f16(__a, __b, __v, __lane)              \
    __extension__({                                             \
        auto c = vget_high_f16(__v);                            \
        auto __ret = vmlaq_lane_f16(__a, __b, c, (__lane - 4)); \
        __ret;                                                  \
    })

//! FIXME: remove these funtion once llvm fix such bugs
//! As origin implentation in \c arm_neon.h may cause
//! \attention {error in backend: Do not know how to split this operator's
//! operand!}
///////////////////////////////////////////////////////////////////////
__ai float16x8_t vmulq_fix_f16(float16x8_t a, float16x8_t b) {
    float16x8_t ret;
    asm volatile("vmul.f16 %0, %1, %2\n" : "+w"(ret) : "w"(a), "w"(b));
    return ret;
}

__ai float16x8_t vmulq_n_fix_f16(float16x8_t a, __fp16 b) {
    float16x8_t ret;
    asm volatile(
            "vdup.16 q0, %2 \n"
            "vmul.f16 %0, %1, q0\n"
            : "+w"(ret)
            : "w"(a), "r"(b)
            : "q0");
    return ret;
}

__ai float16x4_t vmul_n_fix_f16(float16x4_t a, __fp16 b) {
    float16x4_t ret;
    asm volatile(
            "vdup.16 d0,%2\n"
            "vmul.f16 %0, %1, d0[0]\n"
            : "+w"(ret)
            : "w"(a), "r"(b)
            : "d0");
    return ret;
}
__ai float16x8_t vmlaq_fix_f16(float16x8_t a, float16x8_t b, float16x8_t c) {
    asm volatile("vmla.f16 %0, %1, %2\n" : "+w"(a) : "w"(b), "w"(c));
    return a;
}

__ai float16x8_t vaddq_fix_f16(float16x8_t a, float16x8_t b) {
    float16x8_t ret;
    asm volatile("vadd.f16 %0, %1, %2\n" : "+w"(ret) : "w"(a), "w"(b));
    return ret;
}

#undef vdupq_n_f16
__ai float16x8_t vdupq_n_f16(__fp16 a) {
    float16x8_t ret;
    asm volatile("vdup.16 %0, %1\n" : "+w"(ret) : "r"(a) :);
    return ret;
}

///////////////////////////////////////////////////////////////////////

#elif MEGDNN_AARCH64
#define vmlaq_low_lane_f16(__a, __b, __v, __lane) \
    vmlaq_laneq_f16(__a, __b, __v, __lane)

#define vmlaq_high_lane_f16(__a, __b, __v, __lane) \
    vmlaq_laneq_f16(__a, __b, __v, __lane)

//! FIXME: remove these funtion once llvm fix such bugs
//! As origin implentation in \c arm_neon.h may cause
//! \attention {error in backend: Do not know how to split this operator's
//! operand!}
///////////////////////////////////////////////////////////////////////

__ai float16x8_t vmulq_fix_f16(float16x8_t a, float16x8_t b) {
    return vmulq_f16(a, b);
}

__ai float16x8_t vmlaq_fix_f16(float16x8_t a, float16x8_t b, float16x8_t c) {
    return vmlaq_f16(a, b, c);
}

__ai float16x8_t vaddq_fix_f16(float16x8_t a, float16x8_t b) {
    return vaddq_f16(a, b);
}

#undef vdupq_n_f16
__ai float16x8_t vdupq_n_f16(__fp16 a) {
    float16x8_t ret;
    asm volatile("dup %0.8h, %w1\n" : "+w"(ret) : "r"(a) :);
    return ret;
}

///////////////////////////////////////////////////////////////////////

#endif

#endif  // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#if __ARM_FEATURE_DOTPROD

__ai int32x4_t vdotq2_s32(int8x16_t a, int8x16_t b) {
    int32x4_t c = vdupq_n_s32(0);
    return vdotq_s32(c, a, b);
}

__ai uint32x4_t vdotq2_u32(uint8x16_t a, uint8x16_t b) {
    uint32x4_t c = vdupq_n_u32(0);
    return vdotq_u32(c, a, b);
}

#define vdotq2_lane_s32(a, b, lane)        \
    __extension__({                        \
        int32x4_t c = vdupq_n_s32(0);      \
        c = vdotq_lane_s32(c, a, b, lane); \
        c;                                 \
    })

#define vdotq2_lane_u32(a, b, lane)        \
    __extension__({                        \
        uint32x4_t c = vdupq_n_u32(0);     \
        c = vdotq_lane_u32(c, a, b, lane); \
        c;                                 \
    })

__ai int32x2_t vdot2_s32(int8x8_t a, int8x8_t b) {
    int32x2_t c = vdup_n_s32(0);
    return vdot_s32(c, a, b);
}

__ai uint32x2_t vdot2_u8(uint8x8_t a, uint8x8_t b) {
    uint32x2_t c = vdup_n_u32(0);
    return vdot_u32(c, a, b);
}

#define vdot2_lane_s32(a, b, lane)        \
    __extension__({                       \
        int32x2_t c = vdup_n_s32(0);      \
        c = vdot_lane_s32(c, a, b, lane); \
        c;                                \
    })

#define vdot2_lane_u8(a, b, lane)         \
    __extension__({                       \
        uint32x2_t c = vdup_n_u32(0);     \
        c = vdot_lane_u32(c, a, b, lane); \
        c;                                \
    })

#endif  // __ARM_FEATURE_DOTPROD

#if __GNUC__ < 8
#undef vld1q_f32_x2
__ai float32x4x2_t vld1q_f32_x2(const float* p) {
    return {{vld1q_f32(p), vld1q_f32(p + 4)}};
}
#endif

#if __GNUC__ < 9
#undef vst1q_f32_x2
__ai void vst1q_f32_x2(const float* p, float32x4x2_t v) {
    vst1q_f32(const_cast<float*>(p), v.val[0]);
    vst1q_f32(const_cast<float*>(p) + 4, v.val[1]);
}
#endif

__ai int8x16_t vtranslq_s8(int8x8_t a) {
    int8x16_t ret;
#if MEGDNN_AARCH64
    asm volatile("ins %0.d[0], %1.d[0]\n" : "+w"(ret) : "w"(a) :);
#else
    asm volatile("vmov %e0, %P1\n" : "+w"(ret) : "w"(a) :);
#endif
    return ret;
}

__ai uint8x16_t vtranslq_u8(uint8x8_t a) {
    uint8x16_t ret;
#if MEGDNN_AARCH64
    asm volatile("ins %0.d[0], %1.d[0]\n" : "+w"(ret) : "w"(a) :);
#else
    asm volatile("vmov %e0, %P1\n" : "+w"(ret) : "w"(a) :);
#endif
    return ret;
}

#ifdef MEGDNN_TEGRA_X1
#define vset_lane_s16_fix_tx1(__elem, __vec, __index) \
    {                                                 \
        asm volatile("ins %0.h[" #__index "], %w1\n"  \
                     : "+w"(__vec)                    \
                     : "r"(__elem)                    \
                     :);                              \
    }
#else
#define vset_lane_s16_fix_tx1(__elem, __vec, __index) \
    __vec = vset_lane_s16(__elem, __vec, __index)
#endif

#if MEGDNN_ARMV7
__ai int32_t vaddlvq_s16(int16x8_t __p0) {
    int32_t __ret = 0;
    auto sum = vpaddlq_s16(__p0);
    __ret += (vgetq_lane_s32(sum, 0) + vgetq_lane_s32(sum, 1) +
              vgetq_lane_s32(sum, 2) + vgetq_lane_s32(sum, 3));
    return __ret;
}

__ai int16x8_t vmlal_high_s8(int16x8_t __p0, int8x16_t __p1, int8x16_t __p2) {
    int16x8_t __ret;
    __ret = vmlal_s8(__p0, vget_high_s8(__p1), vget_high_s8(__p2));
    return __ret;
}

__ai int16x8_t vmull_high_s8(int8x16_t __p0, int8x16_t __p1) {
    int16x8_t __ret;
    __ret = vmull_s8(vget_high_s8(__p0), vget_high_s8(__p1));
    return __ret;
}

//! armv7 : vmovl_xx(vget_high_xx()), armv8 : vmovl_high_xx()
__ai int16x8_t vmovl_high_s8(int8x16_t __p0) {
    return vmovl_s8(vget_high_s8(__p0));
}

__ai uint16x8_t vmovl_high_u8(uint8x16_t __p0) {
    return vmovl_u8(vget_high_u8(__p0));
}

__ai int32x4_t vmovl_high_s16(int16x8_t __p0) {
    return vmovl_s16(vget_high_s16(__p0));
}

__ai uint32x4_t vmovl_high_u16(uint16x8_t __p0) {
    return vmovl_u16(vget_high_u16(__p0));
}

__ai int64x2_t vmovl_high_s32(int32x4_t __p0) {
    return vmovl_s32(vget_high_s32(__p0));
}

__ai uint64x2_t vmovl_high_u32(uint32x4_t __p0) {
    return vmovl_u32(vget_high_u32(__p0));
}

__ai int64x2_t vzip1q_s64(int64x2_t& a, int64x2_t& b) {
    return vcombine_s64(vget_low_s64(a), vget_low_s64(b));
}

__ai int64x2_t vzip2q_s64(int64x2_t& a, int64x2_t& b) {
    return vcombine_s64(vget_high_s64(a), vget_high_s64(b));
}

__ai int32_t vaddv_s32(int32x2_t a) {
    return vget_lane_s32(a, 0) + vget_lane_s32(a, 1);
}

__ai int32_t vaddvq_s32(int32x4_t a) {
    return vgetq_lane_s32(a, 0) + vgetq_lane_s32(a, 1) +
           vgetq_lane_s32(a, 2) + vgetq_lane_s32(a, 3);
}

__ai float32_t vaddvq_f32(float32x4_t a) {
    return vgetq_lane_f32(a, 0) + vgetq_lane_f32(a, 1) +
           vgetq_lane_f32(a, 2) + vgetq_lane_f32(a, 3);
}

#endif  // MEGDNN_ARMV7

//! pack vmovl_low_xx() on armv7 and armv8
__ai int16x8_t vmovl_low_s8(int8x16_t __p0) {
    return vmovl_s8(vget_low_s8(__p0));
}

__ai uint16x8_t vmovl_low_u8(uint8x16_t __p0) {
    return vmovl_u8(vget_low_u8(__p0));
}

__ai int32x4_t vmovl_low_s16(int16x8_t __p0) {
    return vmovl_s16(vget_low_s16(__p0));
}

__ai uint32x4_t vmovl_low_u16(uint16x8_t __p0) {
    return vmovl_u16(vget_low_u16(__p0));
}

__ai int64x2_t vmovl_low_s32(int32x4_t __p0) {
    return vmovl_s32(vget_low_s32(__p0));
}

__ai uint64x2_t vmovl_low_u32(uint32x4_t __p0) {
    return vmovl_u32(vget_low_u32(__p0));
}

#if MEGDNN_ARMV7
#define vmlaq_low_lane_f32(__a, __b, __v, __lane)         \
    __extension__({                                       \
        auto c = vget_low_f32(__v);                       \
        auto __ret = vmlaq_lane_f32(__a, __b, c, __lane); \
        __ret;                                            \
    })

#define vmlaq_high_lane_f32(__a, __b, __v, __lane)              \
    __extension__({                                             \
        auto c = vget_high_f32(__v);                            \
        auto __ret = vmlaq_lane_f32(__a, __b, c, (__lane - 2)); \
        __ret;                                                  \
    })

#elif MEGDNN_AARCH64
__ai float64x2_t vbitq_f64(float64x2_t dst, float64x2_t v1, uint64x2_t mask) {
    asm volatile("bit %0.16b, %1.16b, %2.16b\n"
                 : "+w"(dst)
                 : "w"(v1), "w"(mask)
                 :);
    return dst;
}

#define vmlaq_low_lane_f32(__a, __b, __v, __lane) \
    vmlaq_laneq_f32(__a, __b, __v, __lane)

#define vmlaq_high_lane_f32(__a, __b, __v, __lane) \
    vmlaq_laneq_f32(__a, __b, __v, __lane)

#endif

#if MEGDNN_ARMV7
__ai int8x16_t vqtbl1q_s8(int8x16_t& a, uint8x16_t& idx) {
    int8x8_t src_low = vget_low_s8(a);
    int8x8_t src_high = vget_high_s8(a);
    return vcombine_s8(vtbl2_s8({src_low, src_high},
                                vget_low_s8(vreinterpretq_s8_u8(idx))),
                       vtbl2_s8({src_low, src_high},
                                vget_high_s8(vreinterpretq_s8_u8(idx))));
}
namespace {
template <int lane>
struct Vdup_laneq_s16_armv7 {
    __ai int16x4_t impl(int16x8_t vec);
};
#define cb(step)                                            \
    template <>                                             \
    struct Vdup_laneq_s16_armv7<step + 4> {                 \
        __ai int16x4_t impl(int16x8_t vec) {                \
            return vdup_lane_s16(vget_high_s16(vec), step); \
        }                                                   \
    };                                                      \
    template <>                                             \
    struct Vdup_laneq_s16_armv7<step> {                     \
        __ai int16x4_t impl(int16x8_t vec) {                \
            return vdup_lane_s16(vget_low_s16(vec), step);  \
        }                                                   \
    };

UNROLL_CALL_RAW(4, cb);
#undef cb
}  // namespace
#define vdup_laneq_s16(vec, lane) Vdup_laneq_s16_armv7<lane>::impl(vec)
namespace {
template <int lane>
struct Vfmaq_laneq_f32_armv7 {
    __ai float32x4_t impl(float32x4_t a, float32x4_t b, float32x4_t v);
};

template <>
struct Vfmaq_laneq_f32_armv7<0> {
    __ai float32x4_t impl(float32x4_t a, float32x4_t b, float32x4_t v) {
        return vmlaq_lane_f32(a, b, vget_low_f32(v), 0);
    }
};
template <>
struct Vfmaq_laneq_f32_armv7<1> {
    __ai float32x4_t impl(float32x4_t a, float32x4_t b, float32x4_t v) {
        return vmlaq_lane_f32(a, b, vget_low_f32(v), 1);
    }
};
template <>
struct Vfmaq_laneq_f32_armv7<2> {
    __ai float32x4_t impl(float32x4_t a, float32x4_t b, float32x4_t v) {
        return vmlaq_lane_f32(a, b, vget_high_f32(v), 0);
    }
};
template <>
struct Vfmaq_laneq_f32_armv7<3> {
    __ai float32x4_t impl(float32x4_t a, float32x4_t b, float32x4_t v) {
        return vmlaq_lane_f32(a, b, vget_high_f32(v), 1);
    }
};

template <int lane>
struct Vfmsq_laneq_f32_armv7 {
    __ai float32x4_t impl(float32x4_t a, float32x4_t b, float32x4_t v);
};

template <>
struct Vfmsq_laneq_f32_armv7<0> {
    __ai float32x4_t impl(float32x4_t a, float32x4_t b, float32x4_t v) {
        return vmlsq_lane_f32(a, b, vget_low_f32(v), 0);
    }
};
template <>
struct Vfmsq_laneq_f32_armv7<1> {
    __ai float32x4_t impl(float32x4_t a, float32x4_t b, float32x4_t v) {
        return vmlsq_lane_f32(a, b, vget_low_f32(v), 1);
    }
};
template <>
struct Vfmsq_laneq_f32_armv7<2> {
    __ai float32x4_t impl(float32x4_t a, float32x4_t b, float32x4_t v) {
        return vmlsq_lane_f32(a, b, vget_high_f32(v), 0);
    }
};
template <>
struct Vfmsq_laneq_f32_armv7<3> {
    __ai float32x4_t impl(float32x4_t a, float32x4_t b, float32x4_t v) {
        return vmlsq_lane_f32(a, b, vget_high_f32(v), 1);
    }
};
}  // namespace
#define vfmaq_laneq_f32(a, b, v, lane) \
    Vfmaq_laneq_f32_armv7<lane>::impl(a, b, v)

#define vfmsq_laneq_f32(a, b, v, lane) \
    Vfmsq_laneq_f32_armv7<lane>::impl(a, b, v)

#if __ARM_FEATURE_DOTPROD
namespace {
template <int lane>
struct Vdotq_laneq_s32_armv7 {
    __ai int32x4_t impl(int32x4_t a, int8x16_t b, int8x16_t v);
};
template <>
struct Vdotq_laneq_s32_armv7<0> {
    __ai int32x4_t impl(int32x4_t a, int8x16_t b, int8x16_t v) {
        return vdotq_lane_s32(a, b, vget_low_s32(v), 0);
    }
};
template <>
struct Vdotq_laneq_s32_armv7<1> {
    __ai int32x4_t impl(int32x4_t a, int8x16_t b, int8x16_t v) {
        return vdotq_lane_s32(a, b, vget_low_s32(v), 1);
    }
};
template <>
struct Vdotq_laneq_s32_armv7<2> {
    __ai int32x4_t impl(int32x4_t a, int8x16_t b, int8x16_t v) {
        return vdotq_lane_s32(a, b, vget_high_s32(v), 0);
    }
};
template <>
struct Vdotq_laneq_s32_armv7<3> {
    __ai int32x4_t impl(int32x4_t a, int8x16_t b, int8x16_t v) {
        return vdotq_lane_s32(a, b, vget_high_f32(v), 1);
    }
};
#define vdotq_laneq_s32(a, b, v, lane) \
    Vdotq_laneq_s32_armv7<lane>::impl(a, b, v)

}  // namespace
#endif

#endif

//! GCC split fmla with lane to dup+fmla when version < 9
//! https://gcc.gnu.org/bugzilla/show_bug.cgi?id=89101
#if MEGDNN_AARCH64
namespace {

template <int lane>
struct Vfmaq_laneq_f32_armv8 {
    __ai float32x4_t impl(float32x4_t a, float32x4_t b, float32x4_t v);
};
template <>
struct Vfmaq_laneq_f32_armv8<0> {
    __ai float32x4_t impl(float32x4_t a, float32x4_t b, float32x4_t v) {
        asm volatile("fmla %0.4s, %1.4s, %2.s[0]\n"
                     : "+w"(a)
                     : "w"(b), "w"(v)
                     :);
        return a;
    }
};
template <>
struct Vfmaq_laneq_f32_armv8<1> {
    __ai float32x4_t impl(float32x4_t a, float32x4_t b, float32x4_t v) {
        asm volatile("fmla %0.4s, %1.4s, %2.s[1]\n"
                     : "+w"(a)
                     : "w"(b), "w"(v)
                     :);
        return a;
    }
};
template <>
struct Vfmaq_laneq_f32_armv8<2> {
    __ai float32x4_t impl(float32x4_t a, float32x4_t b, float32x4_t v) {
        asm volatile("fmla %0.4s, %1.4s, %2.s[2]\n"
                     : "+w"(a)
                     : "w"(b), "w"(v)
                     :);
        return a;
    }
};
template <>
struct Vfmaq_laneq_f32_armv8<3> {
    __ai float32x4_t impl(float32x4_t a, float32x4_t b, float32x4_t v) {
        asm volatile("fmla %0.4s, %1.4s, %2.s[3]\n"
                     : "+w"(a)
                     : "w"(b), "w"(v)
                     :);
        return a;
    }
};

template <int lane>
struct Vfmsq_laneq_f32_armv8 {
    __ai float32x4_t impl(float32x4_t a, float32x4_t b, float32x4_t v);
};
template <>
struct Vfmsq_laneq_f32_armv8<0> {
    __ai float32x4_t impl(float32x4_t a, float32x4_t b, float32x4_t v) {
        asm volatile("fmls %0.4s, %1.4s, %2.s[0]\n"
                     : "+w"(a)
                     : "w"(b), "w"(v)
                     :);
        return a;
    }
};
template <>
struct Vfmsq_laneq_f32_armv8<1> {
    __ai float32x4_t impl(float32x4_t a, float32x4_t b, float32x4_t v) {
        asm volatile("fmls %0.4s, %1.4s, %2.s[1]\n"
                     : "+w"(a)
                     : "w"(b), "w"(v)
                     :);
        return a;
    }
};
template <>
struct Vfmsq_laneq_f32_armv8<2> {
    __ai float32x4_t impl(float32x4_t a, float32x4_t b, float32x4_t v) {
        asm volatile("fmls %0.4s, %1.4s, %2.s[2]\n"
                     : "+w"(a)
                     : "w"(b), "w"(v)
                     :);
        return a;
    }
};
template <>
struct Vfmsq_laneq_f32_armv8<3> {
    __ai float32x4_t impl(float32x4_t a, float32x4_t b, float32x4_t v) {
        asm volatile("fmls %0.4s, %1.4s, %2.s[3]\n"
                     : "+w"(a)
                     : "w"(b), "w"(v)
                     :);
        return a;
    }
};
}  // namespace
#undef vfmaq_laneq_f32
#define vfmaq_laneq_f32(a, b, v, lane) \
    Vfmaq_laneq_f32_armv8<lane>::impl(a, b, v)

#undef vfmsq_laneq_f32
#define vfmsq_laneq_f32(a, b, v, lane) \
    Vfmsq_laneq_f32_armv8<lane>::impl(a, b, v)
#endif

__ai int8x16_t vld_dup_tbl_s32(const int8_t* ptr, uint8x16_t& idx) {
    int8x16_t result = vreinterpretq_s8_s32(vld1q_dup_s32((const int32_t*)ptr));
    result = vqtbl1q_s8(result, idx);
    return result;
}
__ai int8x16_t vldq_tbl_s8(const int8_t* ptr, uint8x16_t& idx) {
    int8x16_t result = vld1q_s8(ptr);
    result = vqtbl1q_s8(result, idx);
    return result;
}
__ai int32x4_t vdotq_s32_h(int8x16_t& a, int8x16_t& b, int32x4_t& c,
                           int16x8_t& temp) {
    temp = vmull_s8(vget_low_s8(a), vget_low_s8(b));
    temp = vmlal_high_s8(temp, a, b);
    c = vpadalq_s16(c, temp);
    return c;
}
__ai int32x4_t vdot2_s32_h(int8x8_t& a, int8x8_t& b, int32x4_t& c,
                           int16x8_t& temp) {
    temp = vmull_s8(a, b);
    c = vpadalq_s16(c, temp);
    return c;
}

__ai int32x4_t vmlal_s16(int32x4_t& a, int16x8_t& b, int16x8_t& c) {
    return vmlal_s16(a, vget_low_s16(b), vget_low_s16(c));
}

__ai int16x8_t vldq_dup_4s8_8s16(const int8_t* ptr) {
    return vmovl_s8(vreinterpret_s8_s32(
            vld1_dup_s32(reinterpret_cast<const int32_t*>(ptr))));
}
__ai int8x8_t vldq_tbl_low_s8(const int8_t* ptr, uint8x16_t idx) {
    return vget_low_s8(vldq_tbl_s8(ptr, idx));
}
__ai int16x8_t vld1_dup_s8_s16(const int8_t* ptr) {
    return vmovl_s8(vld1_dup_s8(ptr));
}

//! we add this because we found that cpu=aarch64_android cann't compile fmsq into fmls.
//! it use dup+fmla instead
__ai float32x4_t Vfmsq_f32(float32x4_t& a, float32x4_t& b, float32x4_t& v) {
    asm volatile("fmls %0.4s, %1.4s, %2.4s\n"
                    : "+w"(a)
                    : "w"(b), "w"(v)
                    :);
    return a;
}

#undef __ai
#pragma GCC diagnostic pop

// vim: syntax=cpp.doxygen
