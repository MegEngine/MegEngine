/**
 * \file dnn/src/fallback/elemwise/gi_impl/gi_mathfun.h
 */

#pragma once

#include "src/fallback/general_intrinsic/gi_float.h"
#include "src/fallback/general_intrinsic/gi_int.h"

namespace megdnn {
namespace fallback {

typedef GI_FLOAT32_t v4sf;  // vector of 4 float
typedef GI_INT32_t v4si;    // vector of 4 int32
typedef GI_UINT32_t v4su;   // vector of 4 uint32

/**
 * \brief natural logarithm computed for 4 simultaneous float
 *   return NaN for x <= 0
 */
v4sf GiLogPsFloat32(v4sf x);

//! exp() computed for 4 float at once
v4sf GiExpPsFloat32(v4sf x);

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
void GiSinCosPsFloat32(v4sf x, v4sf* ysin, v4sf* ycos);

v4sf GiSinPsFloat32(v4sf x);

v4sf GiCosPsFloat32(v4sf x);

v4sf GiTanPsFloat32(v4sf x);

v4sf GiSigmoidPsFloat32(v4sf x);

}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
