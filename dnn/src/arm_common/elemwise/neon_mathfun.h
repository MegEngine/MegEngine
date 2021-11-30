/**
 * \file dnn/src/arm_common/elemwise/neon_mathfun.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

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

v4sf sigmoid_ps_f32(v4sf x);

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

float16x8_t sigmoid_ps_f16(float16x8_t x);

#endif

}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
