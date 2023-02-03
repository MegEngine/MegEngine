/**
 * \file dnn/src/fallback/elemwise/gi_mathfun.cpp
 *
 * This file has been modified by Megvii ("Megvii Modifications").
 * All Megvii Modifications are Copyright (C) 2014-2021 Megvii Inc. All rights
 * reserved.
 *
 */

/* NEON implementation of sin, cos, exp and log

   Inspired by Intel Approximate Math library, and based on the
   corresponding algorithms of the cephes math library
*/

/* Copyright (C) 2011  Julien Pommier

  This software is provided 'as-is', without any express or implied
  warranty.  In no event will the authors be held liable for any damages
  arising from the use of this software.

  Permission is granted to anyone to use this software for any purpose,
  including commercial applications, and to alter it and redistribute it
  freely, subject to the following restrictions:

  1. The origin of this software must not be misrepresented; you must not
     claim that you wrote the original software. If you use this software
     in a product, an acknowledgment in the product documentation would be
     appreciated but is not required.
  2. Altered source versions must be plainly marked as such, and must not be
     misrepresented as being the original software.
  3. This notice may not be removed or altered from any source distribution.

  (this is the zlib license)
*/

#include "./gi_mathfun.h"

namespace megdnn {
namespace fallback {

#define c_inv_mant_mask ~0x7f800000u
#define c_cephes_SQRTHF 0.707106781186547524
#define c_cephes_log_p0 7.0376836292E-2
#define c_cephes_log_p1 -1.1514610310E-1
#define c_cephes_log_p2 1.1676998740E-1
#define c_cephes_log_p3 -1.2420140846E-1
#define c_cephes_log_p4 +1.4249322787E-1
#define c_cephes_log_p5 -1.6668057665E-1
#define c_cephes_log_p6 +2.0000714765E-1
#define c_cephes_log_p7 -2.4999993993E-1
#define c_cephes_log_p8 +3.3333331174E-1
#define c_cephes_log_q1 -2.12194440e-4
#define c_cephes_log_q2 0.693359375

/**
 * natural logarithm computed for 4 simultaneous float return NaN for x <= 0
 */
v4sf GiLogPsFloat32(v4sf x) {
    v4sf one = GiBroadcastFloat32(1);

    x = GiMaximumFloat32(
            x, GiBroadcastFloat32(0)); /* force flush to zero on denormal values */
    v4su invalid_mask = GiLessThanEqFloat32(x, GiBroadcastFloat32(0));

    v4si ux = GiReinterpretAsInt32(x);

    v4si emm0 = GiShiftRight23Int32(ux);

    /* keep only the fractional part */
    ux = GiAndInt32(ux, GiBroadcastInt32(c_inv_mant_mask));
    ux = GiOrInt32(ux, GiReinterpretAsInt32(GiBroadcastFloat32(0.5f)));
    x = GiReintInt32ToFloat32(ux);

    emm0 = GiSubtractInt32(emm0, GiBroadcastInt32(0x7f));
    v4sf e = GiCastToFloat32(emm0);

    e = GiAddFloat32(e, one);

    /* part2:
     *     if( x < SQRTHF ) {
     *       e -= 1;
     *       x = x + x - 1.0;
     *     } else { x = x - 1.0; }
     */
    v4su mask = GiLessThanFloat32(x, GiBroadcastFloat32(c_cephes_SQRTHF));
    v4sf tmp = GiAndFloat32(x, GiReintUint32ToFloat32(mask));
    x = GiSubtractFloat32(x, one);
    e = GiSubtractFloat32(e, GiAndFloat32(one, GiReintUint32ToFloat32(mask)));
    x = GiAddFloat32(x, tmp);

    v4sf z = GiMultiplyFloat32(x, x);

    v4sf y = GiBroadcastFloat32(c_cephes_log_p0);
    y = GiMultiplyAddFloat32(GiBroadcastFloat32(c_cephes_log_p1), y, x);
    y = GiMultiplyAddFloat32(GiBroadcastFloat32(c_cephes_log_p2), y, x);
    y = GiMultiplyAddFloat32(GiBroadcastFloat32(c_cephes_log_p3), y, x);
    y = GiMultiplyAddFloat32(GiBroadcastFloat32(c_cephes_log_p4), y, x);
    y = GiMultiplyAddFloat32(GiBroadcastFloat32(c_cephes_log_p5), y, x);
    y = GiMultiplyAddFloat32(GiBroadcastFloat32(c_cephes_log_p6), y, x);
    y = GiMultiplyAddFloat32(GiBroadcastFloat32(c_cephes_log_p7), y, x);
    y = GiMultiplyAddFloat32(GiBroadcastFloat32(c_cephes_log_p8), y, x);
    y = GiMultiplyFloat32(y, x);

    y = GiMultiplyFloat32(y, z);

    y = GiMultiplyAddFloat32(y, e, GiBroadcastFloat32(c_cephes_log_q1));

    y = GiMultiplySubFloat32(y, z, GiBroadcastFloat32(0.5f));

    x = GiAddFloat32(x, y);
    x = GiMultiplyAddFloat32(x, e, GiBroadcastFloat32(c_cephes_log_q2));
    x = GiOrFloat32(
            x, GiReintUint32ToFloat32(invalid_mask));  // negative arg will be NAN
    return x;
}

#define c_exp_hi 88.3762626647949f
#define c_exp_lo -88.3762626647949f

#define c_cephes_LOG2EF 1.44269504088896341
#define c_cephes_exp_C1 0.693359375
#define c_cephes_exp_C2 -2.12194440e-4

#define c_cephes_exp_p0 1.9875691500E-4
#define c_cephes_exp_p1 1.3981999507E-3
#define c_cephes_exp_p2 8.3334519073E-3
#define c_cephes_exp_p3 4.1665795894E-2
#define c_cephes_exp_p4 1.6666665459E-1
#define c_cephes_exp_p5 5.0000001201E-1

/* exp() computed for 4 float at once */
v4sf GiExpPsFloat32(v4sf x) {
    v4sf tmp, fx;

    v4sf one = GiBroadcastFloat32(1);
    x = GiMinimumFloat32(x, GiBroadcastFloat32(c_exp_hi));
    x = GiMaximumFloat32(x, GiBroadcastFloat32(c_exp_lo));

    /* express exp(x) as exp(g + n*log(2)) */
    fx = GiMultiplyAddFloat32(
            GiBroadcastFloat32(0.5f), x, GiBroadcastFloat32(c_cephes_LOG2EF));

    /* perform a floorf */
    tmp = GiCastToFloat32(GiCastToInt32(fx));

    /* if greater, subtract 1 */
    v4su mask = GiGreaterThanFloat32(tmp, fx);
    v4sf mask_float = GiAndFloat32(GiReintUint32ToFloat32(mask), one);

    fx = GiSubtractFloat32(tmp, mask_float);

    tmp = GiMultiplyFloat32(fx, GiBroadcastFloat32(c_cephes_exp_C1));
    v4sf z = GiMultiplyFloat32(fx, GiBroadcastFloat32(c_cephes_exp_C2));
    x = GiSubtractFloat32(x, tmp);
    x = GiSubtractFloat32(x, z);

    z = GiMultiplyFloat32(x, x);

    v4sf y = GiBroadcastFloat32(c_cephes_exp_p0);
    y = GiMultiplyAddFloat32(GiBroadcastFloat32(c_cephes_exp_p1), y, x);
    y = GiMultiplyAddFloat32(GiBroadcastFloat32(c_cephes_exp_p2), y, x);
    y = GiMultiplyAddFloat32(GiBroadcastFloat32(c_cephes_exp_p3), y, x);
    y = GiMultiplyAddFloat32(GiBroadcastFloat32(c_cephes_exp_p4), y, x);
    y = GiMultiplyAddFloat32(GiBroadcastFloat32(c_cephes_exp_p5), y, x);

    y = GiMultiplyAddFloat32(x, y, z);
    y = GiAddFloat32(y, one);

    /* build 2^n */
    v4si mm;
    mm = GiCastToInt32(fx);
    mm = GiAddInt32(mm, GiBroadcastInt32(0x7f));
    mm = GiShiftLeft23Int32(mm);
    v4sf pow2n = GiReintInt32ToFloat32(mm);

    y = GiMultiplyFloat32(y, pow2n);
    return y;
}

#define c_minus_cephes_DP1 -0.78515625
#define c_minus_cephes_DP2 -2.4187564849853515625e-4
#define c_minus_cephes_DP3 -3.77489497744594108e-8
#define c_sincof_p0        -1.9515295891E-4
#define c_sincof_p1        8.3321608736E-3
#define c_sincof_p2        -1.6666654611E-1
#define c_coscof_p0        2.443315711809948E-005
#define c_coscof_p1        -1.388731625493765E-003
#define c_coscof_p2        4.166664568298827E-002
#define c_cephes_FOPI      1.27323954473516  // 4 / M_PI

/* evaluation of 4 sines & cosines at once.

   The code is the exact rewriting of the cephes sinf function.
   Precision is excellent as long as x < 8192 (I did not bother to
   take into account the special handling they have for greater values
   -- it does not return garbage for arguments over 8192, though, but
   the extra precision is missing).

   Note that it is such that sinf((float)M_PI) = 8.74e-8, which is the
   surprising but correct result.

   Note also that when you compute sin(x), cos(x) is available at
   almost no extra price so both sin_ps_f32 and cos_ps_f32 make use of
   sincos_ps_f32..
  */
void GiSinCosPsFloat32(v4sf x, v4sf* ysin, v4sf* ycos) {
    // any x
    v4sf y;

    v4su emm2;

    v4su sign_mask_sin, sign_mask_cos;
    sign_mask_sin = GiLessThanFloat32(x, GiBroadcastFloat32(0));
    x = GiAbsFloat32(x);

    /* scale by 4/Pi */
    y = GiMultiplyFloat32(x, GiBroadcastFloat32(c_cephes_FOPI));

    /* store the integer part of y in mm0 */
    emm2 = GiReinterpretAsUint32(y);
    /* j=(j+1) & (~1) (see the cephes sources) */
    emm2 = GiAddUint32(emm2, GiBroadcastUint32(1));
    emm2 = GiAddUint32(emm2, GiBroadcastUint32(~1));
    y = GiReintUint32ToFloat32(emm2);

    /* get the polynom selection mask
     *     there is one polynom for 0 <= x <= Pi/4
     *     and another one for Pi/4<x<=Pi/2
     *
     *     Both branches will be computed.
     */
    v4su poly_mask = GiTestAndSetUint32(emm2, GiBroadcastUint32(2));

    /* The magic pass: "Extended precision modular arithmetic"
     *     x = ((x - y * DP1) - y * DP2) - y * DP3; */
    x = GiMultiplyAddFloat32(x, y, GiBroadcastFloat32(c_minus_cephes_DP1));
    x = GiMultiplyAddFloat32(x, y, GiBroadcastFloat32(c_minus_cephes_DP2));
    x = GiMultiplyAddFloat32(x, y, GiBroadcastFloat32(c_minus_cephes_DP3));

    sign_mask_sin =
            GiEOrUint32(sign_mask_sin, GiTestAndSetUint32(emm2, GiBroadcastUint32(4)));
    sign_mask_cos = GiTestAndSetUint32(
            GiSubtractUint32(emm2, GiBroadcastUint32(2)), GiBroadcastUint32(4));

    /* Evaluate the first polynom  (0 <= x <= Pi/4) in y1,
     *     and the second polynom      (Pi/4 <= x <= 0) in y2 */
    v4sf z = GiMultiplyFloat32(x, x);
    v4sf y1, y2;

    y1 = GiMultiplyAddFloat32(
            GiBroadcastFloat32(c_coscof_p1), z, GiBroadcastFloat32(c_coscof_p0));
    y2 = GiMultiplyAddFloat32(
            GiBroadcastFloat32(c_sincof_p1), z, GiBroadcastFloat32(c_sincof_p0));
    y1 = GiMultiplyAddFloat32(GiBroadcastFloat32(c_coscof_p2), y1, z);
    y2 = GiMultiplyAddFloat32(GiBroadcastFloat32(c_sincof_p2), y2, z);
    y1 = GiMultiplyFloat32(y1, z);
    y2 = GiMultiplyFloat32(y2, z);
    y1 = GiMultiplyFloat32(y1, z);
    y1 = GiMultiplySubFloat32(y1, z, GiBroadcastFloat32(0.5f));
    y2 = GiMultiplyAddFloat32(x, y2, x);
    y1 = GiAddFloat32(y1, GiBroadcastFloat32(1));

    /* select the correct result from the two polynoms */
    v4sf ys = GiBSLFloat32(poly_mask, y1, y2);
    v4sf yc = GiBSLFloat32(poly_mask, y2, y1);
    *ysin = GiBSLFloat32(sign_mask_sin, GiNegFloat32(ys), ys);
    *ycos = GiBSLFloat32(sign_mask_cos, yc, GiNegFloat32(yc));
}

v4sf GiSinPsFloat32(v4sf x) {
    v4sf ysin, ycos;
    GiSinCosPsFloat32(x, &ysin, &ycos);
    return ysin;
}

v4sf GiCosPsFloat32(v4sf x) {
    v4sf ysin, ycos;
    GiSinCosPsFloat32(x, &ysin, &ycos);
    return ycos;
}

v4sf GiTanPsFloat32(v4sf x) {
    v4sf ysin, ycos;
    GiSinCosPsFloat32(x, &ysin, &ycos);
    return GiDivFloat32(ysin, ycos);
}

#undef c_exp_hi
#undef c_exp_lo
#undef c_cephes_LOG2EF
#undef c_cephes_exp_C1
#undef c_cephes_exp_C2
#undef c_cephes_exp_p0
#undef c_cephes_exp_p1
#undef c_cephes_exp_p2
#undef c_cephes_exp_p3
#undef c_cephes_exp_p4
#undef c_cephes_exp_p5

#undef c_minus_cephes_DP1
#undef c_minus_cephes_DP2
#undef c_minus_cephes_DP3
#undef c_sincof_p0
#undef c_sincof_p1
#undef c_sincof_p2
#undef c_coscof_p0
#undef c_coscof_p1
#undef c_coscof_p2
#undef c_cephes_FOPI

#undef c_inv_mant_mask
#undef c_cephes_SQRTHF
#undef c_cephes_log_p0
#undef c_cephes_log_p1
#undef c_cephes_log_p2
#undef c_cephes_log_p3
#undef c_cephes_log_p4
#undef c_cephes_log_p5
#undef c_cephes_log_p6
#undef c_cephes_log_p7
#undef c_cephes_log_p8
#undef c_cephes_log_q1
#undef c_cephes_log_q2

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

v4sf GiSigmoidPsFloat32(v4sf src) {
    auto val = GiMaximumFloat32(GiBroadcastFloat32(sigmoid_constants.lower_range), src);
    val = GiMinimumFloat32(GiBroadcastFloat32(sigmoid_constants.upper_range), val);
    auto squared = GiMultiplyFloat32(val, val);
    auto p = GiMultiplyAddFloat32(
            GiBroadcastFloat32(sigmoid_constants.alpha_7), squared,
            GiBroadcastFloat32(sigmoid_constants.alpha_9));
    p = GiMultiplyAddFloat32(GiBroadcastFloat32(sigmoid_constants.alpha_5), p, squared);
    p = GiMultiplyAddFloat32(GiBroadcastFloat32(sigmoid_constants.alpha_3), p, squared);
    p = GiMultiplyAddFloat32(GiBroadcastFloat32(sigmoid_constants.alpha_1), p, squared);
    p = GiMultiplyFloat32(p, val);
    auto q = GiMultiplyAddFloat32(
            GiBroadcastFloat32(sigmoid_constants.beta_8), squared,
            GiBroadcastFloat32(sigmoid_constants.beta_10));
    q = GiMultiplyAddFloat32(GiBroadcastFloat32(sigmoid_constants.beta_6), q, squared);
    q = GiMultiplyAddFloat32(GiBroadcastFloat32(sigmoid_constants.beta_4), q, squared);
    q = GiMultiplyAddFloat32(GiBroadcastFloat32(sigmoid_constants.beta_2), q, squared);
    q = GiMultiplyAddFloat32(GiBroadcastFloat32(sigmoid_constants.beta_0), q, squared);
    return GiAddFloat32(
            GiDivideFloat32(p, q), GiBroadcastFloat32(sigmoid_constants.one_half));
}

#if defined(GI_SUPPORT_F16)
//! Using fp16 to calculate sigmoid has the problem of lack of accuracy, so it is
//! converted to fp32 for calculation.
GI_FLOAT16_t GiSigmoidPsFloat16(GI_FLOAT16_t x) {
    auto&& fp32 = GiCastFloat16ToFloat32(x);
    GI_FLOAT32_t low = GiGetSubVectorFloat32V2(fp32, 0);
    GI_FLOAT32_t high = GiGetSubVectorFloat32V2(fp32, 1);
    low = GiSigmoidPsFloat32(low);
    high = GiSigmoidPsFloat32(high);
    return GiCastFloat32ToFloat16(low, high);
}
#endif
}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
