/**
 * \file dnn/src/arm_common/simd_macro/neon_helper.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/arm_common/simd_macro/marm_neon.h"

#define MEGDNN_SIMD_NAME NEON
#define MEGDNN_SIMD_TARGET neon
#define MEGDNN_SIMD_ATTRIBUTE_TARGET
#define MEGDNN_SIMD_LAMBDA_ATTRIBUTE_TARGET
#define MEGDNN_SIMD_WIDTH 4
#define MEGDNN_SIMD_TYPE float32x4_t
#define MEGDNN_SIMD_TYPE2 float32x4x2_t
#define MEGDNN_SIMD_LOADU(addr) vld1q_f32(addr)
#define MEGDNN_SIMD_LOADU_2(addr) vcombine_f32(vld1_f32(addr), vdup_n_f32(0.f))
#define MEGDNN_SIMD_LOADU_3(addr) vld1q_lane_f32(addr + 2, vcombine_f32(vld1_f32(addr), vdup_n_f32(0.f)), 2)
#define MEGDNN_SIMD_STOREU(addr, reg) vst1q_f32(addr, reg)
#define MEGDNN_SIMD_SETZERO() vdupq_n_f32(0.0f)
#define MEGDNN_SIMD_SET1(num) vdupq_n_f32(num)
// XXX The order of a, b, c
#define MEGDNN_SIMD_FMADD(a, b, c) vmlaq_f32(c, a, b)
#define MEGDNN_SIMD_MAX(a, b) vmaxq_f32(a, b)
#define MEGDNN_SIMD_UZP(s0, s1, d0, d1) do { \
    auto tmp__ = vuzpq_f32(s0, s1); \
    d0 = tmp__.val[0]; \
    d1 = tmp__.val[1]; \
} while (0)
#define MEGDNN_SIMD_LOAD2(addr) vld2q_f32(addr)
#define MEGDNN_SIMD_EXT(a, b, c) vextq_f32(a, b, c)
#define MEGDNN_SIMD_MUL(a, b) vmulq_f32(a, b)
#define MEGDNN_SIMD_ADD(a, b) vaddq_f32(a, b)
#define MEGDNN_SIMD_SET_LANE(a, b, c) vsetq_lane_f32(a, b, c)
#define MEGDNN_SIMD_GET_LOW(a) vget_low_f32(a)
#define MEGDNN_SIMD_GET_HIGH(a) vget_high_f32(a)
#define MEGDNN_SIMD_VMLAQ_LANE(a, b, c, d) vmlaq_lane_f32(a, b, c, d)
#if MEGDNN_ARMV7
#define MEGDNN_SIMD_FMA_LANE(a, b, c, d) ({ \
    auto ret__ = vdupq_n_f32(vgetq_lane_f32(c, d)); \
    ret__ = vmlaq_f32(a, b, ret__); \
    ret__;})
#define MEGDNN_SIMD_ADD_VEC(a) ({ \
    auto tmp__ = vadd_f32(vget_low_f32(a), vget_high_f32(a)); \
    tmp__ = vpadd_f32(tmp__, tmp__); \
    auto ret__ = vget_lane_f32(tmp__, 0); \
    ret__;})
#else
// MEGDNN_AARCH64
#define MEGDNN_SIMD_FMA_LANE(a, b, c, d) vfmaq_laneq_f32(a, b, c, d)
#define MEGDNN_SIMD_ADD_VEC(a) vaddvq_f32(a)
#endif

