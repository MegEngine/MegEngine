#pragma once
#include "src/arm_common/simd_macro/marm_neon.h"

namespace megdnn {
namespace arm_common {
namespace resize {

using InterpolationMode = Resize::InterpolationMode;

template <typename ctype>
struct SIMDHelper {};

template <>
struct SIMDHelper<float> {
    using simd_type = float32x4_t;
    using simd_type_x2 = float32x4x2_t;
    using ctype = float;
    static constexpr size_t simd_width = 4;

    static inline simd_type load(const ctype* src_ptr) { return vld1q_f32(src_ptr); }
    static inline void store(ctype* dst_ptr, const simd_type& rdst) {
        vst1q_f32(dst_ptr, rdst);
    }
    static inline void store2_interleave(
            ctype* dst_ptr, const simd_type& rdst1, const simd_type& rdst2) {
        simd_type_x2 rdst;
        rdst.val[0] = rdst1;
        rdst.val[1] = rdst2;
        vst2q_f32(dst_ptr, rdst);
    }
    static inline simd_type fma(const simd_type& a, const simd_type& b, ctype n) {
#if defined(__ARM_FEATURE_FMA) && defined(__aarch64__)
        return vfmaq_n_f32(a, b, n);
#else
        return vmlaq_n_f32(a, b, n);
#endif
    }
    static inline simd_type fma(
            const simd_type& a, const simd_type& b, const simd_type& c) {
#if defined(__ARM_FEATURE_FMA)
        return vfmaq_f32(a, b, c);
#else
        return vmlaq_f32(a, b, c);
#endif
    }
    static inline simd_type dup(float val) { return vdupq_n_f32(val); }
};

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

template <>
struct SIMDHelper<__fp16> {
    using simd_type = float16x8_t;
    using simd_type_x2 = float16x8x2_t;
    using ctype = __fp16;
    static constexpr size_t simd_width = 8;

    static inline simd_type load(const ctype* src_ptr) { return vld1q_f16(src_ptr); }
    static inline void store(ctype* dst_ptr, const simd_type& rdst) {
        vst1q_f16(dst_ptr, rdst);
    }
    static inline void store2_interleave(
            ctype* dst_ptr, const simd_type& rdst1, const simd_type& rdst2) {
        simd_type_x2 rdst;
        rdst.val[0] = rdst1;
        rdst.val[1] = rdst2;
        vst2q_f16(dst_ptr, rdst);
    }
    static inline simd_type fma(const simd_type& a, const simd_type& b, ctype n) {
#if defined(__ARM_FEATURE_FMA) && defined(__aarch64__)
        return vfmaq_n_f16(a, b, n);
#else
        return vaddq_f16(a, vmulq_n_f16(b, n));
#endif
    }
    static inline simd_type fma(
            const simd_type& a, const simd_type& b, const simd_type& c) {
        return vfmaq_f16(a, b, c);
    }
    static inline simd_type dup(float val) { return vdupq_n_f16(val); }
};

#endif

static inline int get_nearest_src(float scale, int size, int idx) {
    return std::min(static_cast<int>(idx / scale), size - 1);
}

static inline std::tuple<float, int, float, int> get_nearest_linear_coord(
        InterpolationMode imode, float scale, int size, int idx) {
    if (size == 1) {
        return std::make_tuple(1.0f, 0, 0.0f, 0);
    }

    float alpha = (idx + 0.5f) / scale - 0.5f;
    int origin_idx = static_cast<int>(floor(alpha));
    alpha -= origin_idx;

    if (imode == InterpolationMode::INTER_NEAREST) {
        origin_idx = get_nearest_src(scale, size, idx);
        alpha = 0;
    }

    if (origin_idx < 0) {
        origin_idx = 0;
        alpha = 0;
    } else if (origin_idx + 1 >= size) {
        origin_idx = size - 2;
        alpha = 1;
    }

    return std::make_tuple(1 - alpha, origin_idx, alpha, origin_idx + 1);
}
};  // namespace resize
};  // namespace arm_common
};  // namespace megdnn
