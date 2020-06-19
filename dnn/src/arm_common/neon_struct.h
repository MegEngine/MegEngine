/**
 * \file dnn/src/arm_common/neon_struct.h
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
#include "src/arm_common/simd_macro/marm_neon.h"

#define __ai inline __attribute__((__always_inline__))
namespace megdnn {
namespace {
struct Vdotq_s32_h {
    static __ai int32x4_t impl(int8x16_t& a, int8x16_t& b, int32x4_t& c,
                               int16x8_t& temp) {
        return vdotq_s32_h(a, b, c, temp);
    }
};
struct Vdot2_s32_h {
    static __ai int32x4_t impl(int8x8_t a, int8x8_t b, int32x4_t c,
                               int16x8_t temp) {
        return vdot2_s32_h(a, b, c, temp);
    }
};

struct Vmlal_s16 {
    static __ai int32x4_t impl(int16x8_t a, int16x8_t b, int32x4_t c) {
        return vmlal_s16(c, vget_low_s16(a), vget_low_s16(b));
    }
};

struct Vld1q_s8 {
    static __ai int8x16_t impl(const int8_t* ptr) { return vld1q_s8(ptr); }
};
struct Vld1q_f32 {
    static __ai float32x4_t impl(const float32_t* ptr) {
        return vld1q_f32(ptr);
    }
};
struct Vld1_s8 {
    static __ai int8x8_t impl(const int8_t* ptr) { return vld1_s8(ptr); }
};
struct Vldq_dup_4s8_8s16 {
    static __ai int16x8_t impl(const int8_t* ptr) {
        return vldq_dup_4s8_8s16(ptr);
    }
};

struct Vldq_tbl_low_s8 {
    static __ai int8x8_t impl(const int8_t* ptr, uint8x16_t idx) {
        return vldq_tbl_low_s8(ptr, idx);
    }
};

struct Vld1_dup_s8_s16 {
    static __ai int16x8_t impl(const int8_t* ptr) {
        return vld1_dup_s8_s16(ptr);
    }
};

struct Vfmaq_laneq_f32 {
    template <const int lane>
    static __ai float32x4_t impl(float32x4_t a, float32x4_t b, float32x4_t v) {
        return vfmaq_laneq_f32(a, b, v, lane);
    }
};
#if __ARM_FEATURE_DOTPROD
struct Vdotq_laneq_s32 {
    template <const int lane>
    static __ai int32x4_t impl(int32x4_t a, int8x16_t b, int8x16_t v) {
        return vdotq_laneq_s32(a, b, v, lane);
    }
};
#endif

}  // namespace
}  // namespace megdnn

#undef __ai
// vim: syntax=cpp.doxygen