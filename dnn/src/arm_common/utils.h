/**
 * \file dnn/src/arm_common/utils.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include <cstring>
#include "src/arm_common/simd_macro/marm_neon.h"
#include "src/common/utils.h"

namespace megdnn {
namespace arm_common {

template <typename ctype, size_t len>
struct Vector;
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
template <>
struct Vector<__fp16, 4> {
    float16x4_t value;
    Vector() {}
    Vector(const __fp16 v) { value = vdup_n_f16(v); }
    Vector(const Vector& lr) { value = lr.value; }
    Vector(const Vector&& lr) { value = std::move(lr.value); }
    Vector(const float16x4_t& v) { value = v; }
    static Vector load(const __fp16* addr) {
        Vector v;
        v.value = vld1_f16(addr);
        return v;
    }
    static void save(__fp16* addr, const Vector& v) { vst1_f16(addr, v.value); }
    void save(__fp16* addr) { save(addr, *this); }
    Vector operator+(const Vector& lr) {
        Vector dst;
        dst.value = value + lr.value;
        return dst;
    }
    Vector& operator+=(const Vector& lr) {
        value = value + lr.value;
        return *this;
    }
    Vector operator-(const Vector& lr) {
        Vector dst;
        dst.value = value - lr.value;
        return dst;
    }
    Vector& operator-=(const Vector& lr) {
        value = value - lr.value;
        return *this;
    }
    Vector operator*(__fp16 lr) {
        Vector dst;
#if MEGDNN_AARCH64
        dst.value = vmul_n_f16(value, lr);
#else
        dst.value = vmul_n_fix_f16(value, lr);
#endif
        return dst;
    }
    Vector operator*(const Vector& lr) {
        Vector dst;
        dst.value = vmul_f16(value, lr.value);  // value * lr.value;
        return dst;
    }
    Vector& operator*=(const Vector& lr) {
        value = vmul_f16(value, lr.value);
        return *this;
    }
    Vector& operator=(const Vector& lr) {
        value = lr.value;
        return *this;
    }
    Vector& operator=(const Vector&& lr) {
        value = std::move(lr.value);
        return *this;
    }
    Vector operator-() {
        Vector dst;
        dst.value = -value;
        return dst;
    }
};
template <>
struct Vector<__fp16, 8> {
    float16x8_t value;
    Vector() {}
    Vector(const __fp16 v) { value = vdupq_n_f16(v); }
    Vector(const Vector& lr) { value = lr.value; }
    Vector(const Vector&& lr) { value = std::move(lr.value); }
    Vector(const float16x8_t& v) { value = v; }
    static Vector load(const __fp16* addr) {
        Vector v;
        v.value = vld1q_f16(addr);
        return v;
    }
    static void save(__fp16* addr, const Vector& v) {
        vst1q_f16(addr, v.value);
    }
    void save(__fp16* addr) { save(addr, *this); }
    Vector operator+(const Vector& lr) {
        Vector dst;
        dst.value = value + lr.value;
        return dst;
    }
    Vector& operator+=(const Vector& lr) {
        value = value + lr.value;
        return *this;
    }
    Vector operator-(const Vector& lr) {
        Vector dst;
        dst.value = value - lr.value;
        return dst;
    }
    Vector& operator-=(const Vector& lr) {
        value = value - lr.value;
        return *this;
    }
    Vector operator*(__fp16 lr) {
        Vector dst;
#if MEGDNN_AARCH64
        dst.value = vmulq_n_f16(value, lr);
#else
        dst.value = vmulq_n_fix_f16(value, lr);
#endif
        return dst;
    }
    Vector operator*(const Vector& lr) {
        Vector dst;
        dst.value = value * lr.value;
        return dst;
    }
    Vector& operator*=(const Vector& lr) {
        value = value * lr.value;
        return *this;
    }
    Vector& operator=(const Vector& lr) {
        value = lr.value;
        return *this;
    }
    Vector& operator=(const Vector&& lr) {
        value = std::move(lr.value);
        return *this;
    }
    Vector operator-() {
        Vector dst;
        dst.value = -value;
        return dst;
    }
};
#endif

template <>
struct Vector<float, 4> {
    float32x4_t value;
    Vector() {}
    Vector(const float v) { value = vdupq_n_f32(v); }
    Vector(const Vector& lr) { value = lr.value; }
    Vector(const Vector&& lr) { value = std::move(lr.value); }
    Vector(const float32x4_t& v) { value = v; }
    static Vector load(const float* addr) {
        Vector v;
        v.value = vld1q_f32(addr);
        return v;
    }
    static void save(float* addr, const Vector& v) { vst1q_f32(addr, v.value); }
    void save(float* addr) { save(addr, *this); }
    Vector operator+(const Vector& lr) {
        Vector dst;
        dst.value = vaddq_f32(value, lr.value);
        return dst;
    }
    Vector& operator+=(const Vector& lr) {
        value = vaddq_f32(value, lr.value);
        return *this;
    }
    Vector operator-(const Vector& lr) {
        Vector dst;
        dst.value = vsubq_f32(value, lr.value);
        return dst;
    }
    Vector& operator-=(const Vector& lr) {
        value = vsubq_f32(value, lr.value);
        return *this;
    }
    Vector operator*(float lr) {
        Vector dst;
        dst.value = vmulq_n_f32(value, lr);
        return dst;
    }
    Vector operator*(const Vector& lr) {
        Vector dst;
        dst.value = vmulq_f32(value, lr.value);
        return dst;
    }
    Vector& operator*=(const Vector& lr) {
        value = vmulq_f32(value, lr.value);
        return *this;
    }
    Vector& operator=(const Vector& lr) {
        value = lr.value;
        return *this;
    }
    Vector& operator=(const Vector&& lr) {
        value = std::move(lr.value);
        return *this;
    }
    Vector operator-() {
        Vector dst;
        dst.value = -value;
        return dst;
    }
};

template <>
struct Vector<float, 8> {
    float32x4x2_t value;
    Vector() {}
    Vector(const float v) {
        value.val[0] = vdupq_n_f32(v);
        value.val[1] = vdupq_n_f32(v);
    }
    Vector(const Vector& lr) { value = lr.value; }
    Vector(const Vector&& lr) { value = std::move(lr.value); }
    Vector(const float32x4x2_t& v) { value = v; }
    static Vector load(const float* addr) {
        Vector v;
        v.value = vld1q_f32_x2(addr);
        return v;
    }
    static void save(float* addr, const Vector& v) {
        vst1q_f32_x2(addr, v.value);
    }

    void save(float* addr) { save(addr, *this); }
    Vector operator+(const Vector& lr) {
        Vector dst;
        dst.value.val[0] = vaddq_f32(value.val[0], lr.value.val[0]);
        dst.value.val[1] = vaddq_f32(value.val[1], lr.value.val[1]);
        return dst;
    }
    Vector& operator+=(const Vector& lr) {
        value.val[0] = vaddq_f32(value.val[0], lr.value.val[0]);
        value.val[1] = vaddq_f32(value.val[1], lr.value.val[1]);
        return *this;
    }
    Vector& add(const Vector& lr) {
        value.val[0] = vaddq_f32(value.val[0], lr.value.val[0]);
        value.val[1] = vaddq_f32(value.val[1], lr.value.val[1]);
        return *this;
    }
    Vector operator-(const Vector& lr) {
        Vector dst;
        dst.value.val[0] = vsubq_f32(value.val[0], lr.value.val[0]);
        dst.value.val[1] = vsubq_f32(value.val[1], lr.value.val[1]);
        return dst;
    }
    Vector& operator-=(const Vector& lr) {
        value.val[0] = vsubq_f32(value.val[0], lr.value.val[0]);
        value.val[1] = vsubq_f32(value.val[1], lr.value.val[1]);
        return *this;
    }
    Vector operator*(float lr) {
        Vector dst;
        dst.value.val[0] = vmulq_n_f32(value.val[0], lr);
        dst.value.val[1] = vmulq_n_f32(value.val[1], lr);
        return dst;
    }
    //! val + lr * n
    Vector& mla(const Vector& lr, float n) {
        value.val[0] = vmlaq_n_f32(value.val[0], lr.value.val[0], n);
        value.val[1] = vmlaq_n_f32(value.val[1], lr.value.val[1], n);
        return *this;
    }

    Vector operator*(const Vector& lr) {
        Vector dst;
        dst.value.val[0] = vmulq_f32(value.val[0], lr.value.val[0]);
        dst.value.val[1] = vmulq_f32(value.val[1], lr.value.val[1]);
        return dst;
    }
    Vector& operator*=(const Vector& lr) {
        value.val[0] = vmulq_f32(value.val[0], lr.value.val[0]);
        value.val[1] = vmulq_f32(value.val[1], lr.value.val[1]);
        return *this;
    }
    Vector& operator=(const Vector& lr) {
        value = lr.value;
        return *this;
    }
    Vector& operator=(const Vector&& lr) {
        value = std::move(lr.value);
        return *this;
    }
    Vector operator-() {
        Vector dst;
        dst.value.val[0] = -value.val[0];
        dst.value.val[1] = -value.val[1];
        return dst;
    }
};

template <>
struct Vector<int16_t, 8> {
    int16x8_t value;
    Vector() {}
    Vector(const int16_t v) { value = vdupq_n_s16(v); }
    Vector(const Vector& lr) { value = lr.value; }
    Vector(const Vector&& lr) { value = std::move(lr.value); }
    Vector(const int16x8_t& v) { value = v; }
    static Vector load(const int16_t* addr) {
        Vector v;
        v.value = vld1q_s16(addr);
        return v;
    }
    static void save(int16_t* addr, const Vector& v) {
        vst1q_s16(addr, v.value);
    }
    void save(int16_t* addr) { save(addr, *this); }
    Vector operator+(const Vector& lr) {
        Vector dst;
        dst.value = vaddq_s16(value, lr.value);
        return dst;
    }
    Vector& operator+=(const Vector& lr) {
        value = vaddq_s16(value, lr.value);
        return *this;
    }
    Vector operator-(const Vector& lr) {
        Vector dst;
        dst.value = vsubq_s16(value, lr.value);
        return dst;
    }
    Vector& operator-=(const Vector& lr) {
        value = vsubq_s16(value, lr.value);
        return *this;
    }
    Vector operator*(int16_t lr) {
        Vector dst;
        dst.value = vmulq_n_s16(value, lr);
        return dst;
    }
    Vector operator*(const Vector& lr) {
        Vector dst;
        dst.value = vmulq_s16(value, lr.value);
        return dst;
    }
    Vector& operator*=(const Vector& lr) {
        value = vmulq_s16(value, lr.value);
        return *this;
    }
    Vector& operator=(const Vector& lr) {
        value = lr.value;
        return *this;
    }
    Vector& operator=(const Vector&& lr) {
        value = std::move(lr.value);
        return *this;
    }
    Vector operator-() {
        Vector dst;
        dst.value = -value;
        return dst;
    }
};

template <>
struct Vector<int16_t, 4> {
    int16x4_t value;
    Vector() {}
    Vector(const int16_t v) { value = vdup_n_s16(v); }
    Vector(const Vector& lr) { value = lr.value; }
    Vector(const Vector&& lr) { value = std::move(lr.value); }
    Vector(const int16x4_t& v) { value = v; }
    static Vector load(const int16_t* addr) {
        Vector v;
        v.value = vld1_s16(addr);
        return v;
    }
    static void save(int16_t* addr, const Vector& v) {
        vst1_s16(addr, v.value);
    }
    void save(int16_t* addr) { save(addr, *this); }
    Vector operator+(const Vector& lr) {
        Vector dst;
        dst.value = vadd_s16(value, lr.value);
        return dst;
    }
    Vector& operator+=(const Vector& lr) {
        value = vadd_s16(value, lr.value);
        return *this;
    }
    Vector operator-(const Vector& lr) {
        Vector dst;
        dst.value = vsub_s16(value, lr.value);
        return dst;
    }
    Vector& operator-=(const Vector& lr) {
        value = vsub_s16(value, lr.value);
        return *this;
    }
    Vector operator*(int16_t lr) {
        Vector dst;
        dst.value = vmul_n_s16(value, lr);
        return dst;
    }
    Vector operator*(const Vector& lr) {
        Vector dst;
        dst.value = vmul_s16(value, lr.value);
        return dst;
    }
    Vector& operator*=(const Vector& lr) {
        value = vmul_s16(value, lr.value);
        return *this;
    }
    Vector& operator=(const Vector& lr) {
        value = lr.value;
        return *this;
    }
    Vector& operator=(const Vector&& lr) {
        value = std::move(lr.value);
        return *this;
    }
    Vector operator-() {
        Vector dst;
        dst.value = -value;
        return dst;
    }
};



template <>
struct Vector<int32_t, 8> {
    int32x4x2_t value;
    Vector() {}
    Vector(const int32_t v) {
        value.val[0] = vdupq_n_s32(v);
        value.val[1] = vdupq_n_s32(v);
    }
    Vector(const Vector& lr) { value = lr.value; }
    Vector(const Vector&& lr) { value = std::move(lr.value); }
    Vector(const int32x4x2_t& v) { value = v; }
    static Vector load(const int32_t* addr) {
        Vector v;
        v.value.val[0] = vld1q_s32(addr);
        v.value.val[1] = vld1q_s32(addr + 4);
        return v;
    }
    static void save(int32_t* addr, const Vector& v) {
        vst1q_s32(addr, v.value.val[0]);
        vst1q_s32(addr + 4, v.value.val[1]);
    }

    void save(int32_t* addr) { save(addr, *this); }
    Vector operator+(const Vector& lr) {
        Vector dst;
        dst.value.val[0] = vaddq_s32(value.val[0], lr.value.val[0]);
        dst.value.val[1] = vaddq_s32(value.val[1], lr.value.val[1]);
        return dst;
    }
    Vector& operator+=(const Vector& lr) {
        value.val[0] = vaddq_s32(value.val[0], lr.value.val[0]);
        value.val[1] = vaddq_s32(value.val[1], lr.value.val[1]);
        return *this;
    }
    Vector& add(const Vector& lr) {
        value.val[0] = vaddq_s32(value.val[0], lr.value.val[0]);
        value.val[1] = vaddq_s32(value.val[1], lr.value.val[1]);
        return *this;
    }
    Vector operator-(const Vector& lr) {
        Vector dst;
        dst.value.val[0] = vsubq_s32(value.val[0], lr.value.val[0]);
        dst.value.val[1] = vsubq_s32(value.val[1], lr.value.val[1]);
        return dst;
    }
    Vector& operator-=(const Vector& lr) {
        value.val[0] = vsubq_s32(value.val[0], lr.value.val[0]);
        value.val[1] = vsubq_s32(value.val[1], lr.value.val[1]);
        return *this;
    }
    Vector operator*(int32_t lr) {
        Vector dst;
        dst.value.val[0] = vmulq_n_s32(value.val[0], lr);
        dst.value.val[1] = vmulq_n_s32(value.val[1], lr);
        return dst;
    }
    //! val + lr * n
    Vector& mla(const Vector& lr, int32_t n) {
        value.val[0] = vmlaq_n_s32(value.val[0], lr.value.val[0], n);
        value.val[1] = vmlaq_n_s32(value.val[1], lr.value.val[1], n);
        return *this;
    }
    Vector operator*(const Vector& lr) {
        Vector dst;
        dst.value.val[0] = vmulq_s32(value.val[0], lr.value.val[0]);
        dst.value.val[1] = vmulq_s32(value.val[1], lr.value.val[1]);
        return dst;
    }
    Vector& operator*=(const Vector& lr) {
        value.val[0] = vmulq_s32(value.val[0], lr.value.val[0]);
        value.val[1] = vmulq_s32(value.val[1], lr.value.val[1]);
        return *this;
    }
    Vector& operator=(const Vector& lr) {
        value = lr.value;
        return *this;
    }
    Vector& operator=(const Vector&& lr) {
        value = std::move(lr.value);
        return *this;
    }
    Vector operator-() {
        Vector dst;
        dst.value.val[0] = -value.val[0];
        dst.value.val[1] = -value.val[1];
        return dst;
    }
};

}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
