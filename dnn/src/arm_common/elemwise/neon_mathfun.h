#pragma once

#include "src/arm_common/simd_macro/marm_neon.h"

namespace megdnn {
namespace arm_common {

typedef float32x4_t v4sf;  // vector of 4 float
typedef uint32x4_t v4su;   // vector of 4 uint32
typedef int32x4_t v4si;    // vector of 4 uint32

/**
 * \brief natural logarithm computed for 4 simultaneous float
 *   return NaN for x <= 0
 */
v4sf log_ps_f32(v4sf x);

//! exp() computed for 4 float at once
v4sf exp_ps_f32(v4sf x);

/**
 * \brief evaluation of 4 sines & cosines at once.
 *
 * The code is the exact rewriting of the cephes sinf function.
 * Precision is excellent as long as x < 8192 (I did not bother to
 * take into account the special handling they have for greater values
 * -- it does not return garbage for arguments over 8192, though, but
 * the extra precision is missing).
 *
 * Note that it is such that sinf((float)M_PI) = 8.74e-8, which is the
 * surprising but correct result.
 *
 * Note also that when you compute sin(x), cos(x) is available at
 * almost no extra price so both sin_ps_f32 and cos_ps_f32 make use of
 * sincos_ps_f32..
 */
void sincos_ps_f32(v4sf x, v4sf* ysin, v4sf* ycos);

v4sf sin_ps_f32(v4sf x);

v4sf cos_ps_f32(v4sf x);

v4sf tan_ps_f32(v4sf x);

static inline v4sf div_ps_f32(v4sf& x, v4sf& y) {
#if MEGDNN_AARCH64
    return vdivq_f32(x, y);
#else
    //! armv7 not support vdiv, so compute the reciprocal and iterate again
    float32x4_t recp = vrecpeq_f32(y);
    recp = vmulq_f32(vrecpsq_f32(y, recp), recp);
    return vmulq_f32(x, recp);
#endif
}

#if defined(__ARM_FEATURE_FMA)
#define fma_ps_f32(c, b, a) vfmaq_f32((c), (a), (b))
#else
#define fma_ps_f32(c, b, a) vmlaq_f32((c), (a), (b))
#endif

static const struct {
    float lower_range;
    float upper_range;
    float alpha_9;
    float alpha_7;
    float alpha_5;
    float alpha_3;
    float alpha_1;
    float beta_10;
    float beta_8;
    float beta_6;
    float beta_4;
    float beta_2;
    float beta_0;
    float one_half;
} sigmoid_constants = {
        -18.0f,
        18.0f,
        4.37031012579801e-11f,
        1.15627324459942e-07f,
        6.08574864600143e-05f,
        8.51377133304701e-03f,
        2.48287947061529e-01f,
        6.10247389755681e-13f,
        5.76102136993427e-09f,
        6.29106785017040e-06f,
        1.70198817374094e-03f,
        1.16817656904453e-01f,
        9.93151921023180e-01f,
        0.5f,
};
//! for compiler inline, do not move this func to cpp
static inline v4sf sigmoid_ps_f32(v4sf src) {
    auto val = vmaxq_f32(vdupq_n_f32(sigmoid_constants.lower_range), src);
    val = vminq_f32(vdupq_n_f32(sigmoid_constants.upper_range), val);
    auto squared = vmulq_f32(val, val);
    auto p = fma_ps_f32(
            vdupq_n_f32(sigmoid_constants.alpha_7), squared,
            vdupq_n_f32(sigmoid_constants.alpha_9));
    p = fma_ps_f32(vdupq_n_f32(sigmoid_constants.alpha_5), p, squared);
    p = fma_ps_f32(vdupq_n_f32(sigmoid_constants.alpha_3), p, squared);
    p = fma_ps_f32(vdupq_n_f32(sigmoid_constants.alpha_1), p, squared);
    p = vmulq_f32(p, val);
    auto q = fma_ps_f32(
            vdupq_n_f32(sigmoid_constants.beta_8), squared,
            vdupq_n_f32(sigmoid_constants.beta_10));
    q = fma_ps_f32(vdupq_n_f32(sigmoid_constants.beta_6), q, squared);
    q = fma_ps_f32(vdupq_n_f32(sigmoid_constants.beta_4), q, squared);
    q = fma_ps_f32(vdupq_n_f32(sigmoid_constants.beta_2), q, squared);
    q = fma_ps_f32(vdupq_n_f32(sigmoid_constants.beta_0), q, squared);
    return vaddq_f32(div_ps_f32(p, q), vdupq_n_f32(sigmoid_constants.one_half));
}

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
static inline float16x8_t sigmoid_ps_f16(float16x8_t x) {
    float32x4_t low = vcvt_f32_f16(vget_low_f16(x));
    float32x4_t high = vcvt_f32_f16(vget_high_f16(x));
    low = sigmoid_ps_f32(low);
    high = sigmoid_ps_f32(high);
    return vcombine_f16(vcvt_f16_f32(low), vcvt_f16_f32(high));
}
#endif

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
/**
 * \brief compute for 8 half at once, the inner just invoke exp_ps_f32 twice
 */
float16x8_t exp_ps_f16(float16x8_t x);

static inline float16x8_t div_ps_f16(float16x8_t& x, float16x8_t& y) {
#if MEGDNN_AARCH64
    return vdivq_f16(x, y);
#else
    //! armv7 not support vdiv, so compute the reciprocal and iterate again
    float16x8_t recp = vrecpeq_f16(y);
    recp = vmulq_f16(vrecpsq_f16(y, recp), recp);
    return vmulq_f16(x, recp);
#endif
}

#endif

}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
