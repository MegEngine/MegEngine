/**
 * \file dnn/src/arm_common/conv_bias/f16/strategy_6x3.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/arm_common/conv_bias/f16/helper.h"
#include "src/arm_common/conv_bias/f16/strategy.h"
#include "src/arm_common/elemwise_helper/op_unary.h"
#include "src/arm_common/simd_macro/marm_neon.h"
#include "src/arm_common/utils.h"
#include "src/common/unroll_macro.h"
#include "src/common/utils.h"
#include "src/fallback/conv_bias/winograd/winograd.h"
#include "src/naive/matrix_mul/matrix_mul_helper.h"
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#include "midout.h"
MIDOUT_DECL(megdnn_arm_common_winograd_fp16_F63)
using namespace megdnn;
using namespace arm_common;
namespace {
struct FilterTransform6X3 {
    // 1.0000000	     0.0000000	 0.0000000
    //-0.2222222     	-0.2222222	-0.2222222
    //-0.2222222	     0.2222222	-0.2222222
    // 0.0111111	     0.0222222	 0.0444444
    // 0.0111111	    -0.0222222	 0.0444444
    // 0.7111111	     0.3555556	 0.1777778
    // 0.7111111	    -0.3555556	 0.1777778
    // 0.0000000	     0.0000000	 1.0000000
#define FILTER_TRANSFORM(d, wd)                   \
    do {                                          \
        wd##0 = d##0;                             \
        wd##1 = (d##0 + d##1 + d##2) * -0.222168; \
        wd##2 = (d##0 - d##1 + d##2) * -0.222168; \
        auto tmpd0 = d##0 * 0.011108;             \
        auto tmpd1 = d##1 * 0.022217;             \
        auto tmpd2 = d##2 * 0.044434;             \
        wd##3 = tmpd0 + tmpd1 + tmpd2;            \
        wd##4 = tmpd0 - tmpd1 + tmpd2;            \
        tmpd0 = d##0 * 0.710938;                  \
        tmpd1 = d##1 * 0.355469;                  \
        tmpd2 = d##2 * 0.177734;                  \
        wd##5 = tmpd0 + tmpd1 + tmpd2;            \
        wd##6 = tmpd0 - tmpd1 + tmpd2;            \
        wd##7 = d##2;                             \
    } while (0);

#define FILTER_TRANSFORM_FINAL(d, wd)                  \
    do {                                               \
        wd##0 = d##0;                                  \
        wd##1 = (d##0 + d##1 + d##2) * -0.222168;      \
        wd##2 = (d##0 - d##1 + d##2) * -0.222168;      \
        auto tmp0 = d##0 * 0.011108 + d##2 * 0.044434; \
        auto tmp1 = d##1 * 0.022217;                   \
        wd##3 = tmp0 + tmp1;                           \
        wd##4 = tmp0 - tmp1;                           \
        tmp0 = d##0 * 0.710938 + d##2 * 0.177734;      \
        tmp1 = d##1 * 0.355469;                        \
        wd##5 = tmp0 + tmp1;                           \
        wd##6 = tmp0 - tmp1;                           \
        wd##7 = d##2;                                  \
    } while (0);
    static void transform(const __fp16* filter, __fp16* filter_transform_buf,
                          __fp16* transform_mid_buf, size_t OC, size_t IC,
                          size_t oc_start, size_t oc_end) {
        // Gg * GT
        // G
        // 1.0000000	     0.0000000	 0.0000000
        //-0.2222222     	-0.2222222	-0.2222222
        //-0.2222222	     0.2222222	-0.2222222
        // 0.0111111	     0.0222222	 0.0444444
        // 0.0111111	    -0.0222222	 0.0444444
        // 0.7111111	     0.3555556	 0.1777778
        // 0.7111111	    -0.3555556	 0.1777778
        // 0.0000000	     0.0000000	 1.0000000
        constexpr size_t alpha = 6 + 3 - 1;
        for (size_t oc = oc_start; oc < oc_end; oc++) {
            rep(ic, IC) {
                const __fp16* fptr = filter + (oc * IC + ic) * 3 * 3;

                float16x4_t v0, v1, v2, v3;
                v0 = vld1_f16(fptr);      // 0 1 2 3
                v1 = vld1_f16(fptr + 3);  // 3 4 5 6
                v2 = vld1_f16(fptr + 5);  // 5 6 7 8
                v3 = vdup_n_f16(0);
                v2 = vext_f16(v2, v3, 1);
                v0 = vset_lane_f16(0, v0, 3);
                v1 = vset_lane_f16(0, v1, 3);
#define cb(i) Vector<__fp16, 4> g##i(v##i);
                UNROLL_CALL_NOWRAPPER(3, cb);
#undef cb

#define cb(i) Vector<__fp16, 4> Gg##i;
                UNROLL_CALL_NOWRAPPER(8, cb);
#undef cb

#define cb(i) Vector<__fp16, 8> Ggt##i;
                UNROLL_CALL_NOWRAPPER(4, cb);
#undef cb

#define cb(i) Vector<__fp16, 8> result##i;
                UNROLL_CALL_NOWRAPPER(8, cb);
#undef cb
                FILTER_TRANSFORM(g, Gg)
#if MEGDNN_AARCH64
                TRANSPOSE_8x4(Gg, Ggt);
                FILTER_TRANSFORM_FINAL(Ggt, result);
#define cb(i) result##i.save(transform_mid_buf + i * alpha);
                UNROLL_CALL_NOWRAPPER(8, cb);
#undef cb
                rep(i, alpha) rep(j, alpha) {
                    filter_transform_buf[(i * alpha + j) * OC * IC + ic * OC +
                                         oc] = transform_mid_buf[j * alpha + i];
                }
#else
                /*  1.0000000   -0.2222222  -0.2222222  0.0111111   0.0111111
                   0.7111111   0.7111111   0.0000000 0.0000000   -0.2222222
                   0.2222222   0.0222222   -0.0222222  0.3555556   -0.3555556
                   0.0000000 0.0000000   -0.2222222  -0.2222222  0.0444444
                   0.0444444   0.1777778   0.1777778   1.0000000*/

#define GET_VECTOR_FP16D_ELEM(s, i, idx) vget_lane_f16(CONCAT(s, i).value, idx)
#define cb(i)                                                                 \
    do {                                                                      \
        mid_buf1[0] = GET_VECTOR_FP16D_ELEM(Gg, i, 0);                        \
        auto tmp02 = GET_VECTOR_FP16D_ELEM(Gg, i, 0) +                        \
                     GET_VECTOR_FP16D_ELEM(Gg, i, 2);                         \
        mid_buf1[1] = (tmp02 + GET_VECTOR_FP16D_ELEM(Gg, i, 1)) * -0.2222222; \
        mid_buf1[2] = (tmp02 - GET_VECTOR_FP16D_ELEM(Gg, i, 1)) * -0.2222222; \
        auto tmp0 = GET_VECTOR_FP16D_ELEM(Gg, i, 0) * 0.0111111;              \
        auto tmp1 = GET_VECTOR_FP16D_ELEM(Gg, i, 1) * 0.0222222;              \
        auto tmp2 = GET_VECTOR_FP16D_ELEM(Gg, i, 2) * 0.0444444;              \
        tmp02 = tmp0 + tmp2;                                                  \
        mid_buf1[3] = tmp02 + tmp1;                                           \
        mid_buf1[4] = tmp02 - tmp1;                                           \
        tmp0 = GET_VECTOR_FP16D_ELEM(Gg, i, 0) * 0.7111111;                   \
        tmp1 = GET_VECTOR_FP16D_ELEM(Gg, i, 1) * 0.3555556;                   \
        tmp2 = GET_VECTOR_FP16D_ELEM(Gg, i, 2) * 0.1777778;                   \
        tmp02 = tmp0 + tmp2;                                                  \
        mid_buf1[5] = tmp02 + tmp1;                                           \
        mid_buf1[6] = tmp02 - tmp1;                                           \
        mid_buf1[7] = GET_VECTOR_FP16D_ELEM(Gg, i, 2);                        \
        mid_buf1 += 8;                                                        \
    } while (0);
                __fp16* mid_buf1 = transform_mid_buf;
                UNROLL_CALL_NOWRAPPER(8, cb);
                mid_buf1 = transform_mid_buf;
#undef cb
                rep(i, alpha) rep(j, alpha) {
                    filter_transform_buf[(i * alpha + j) * OC * IC + ic * OC +
                                         oc] = transform_mid_buf[i * alpha + j];
                }
#undef GET_VECTOR_FP16D_ELEM
#endif
            }
        }
    }
};
#undef FILTER_TRANSFORM
#undef FILTER_TRANSFORM_FINAL
/**
 * input transform
 *
 * wd0 = (d0 - d6) + 5.25 * (d4 - d2)
 * wd1 = (d6 + d2  - 4.25 * d4) + (d1 + d5 - 4.25 * d3)
 * wd2 = (d6 + d2  - 4.25 * d4) - (d1 + d5 - 4.25 * d3)
 * wd3 = (d6 + 0.25 * d2 - 1.25 * d4) + 2.0 * (d5 + 0.25 * d1 - 1.25 * d3)
 * wd4 = (d6 + 0.25 * d2 - 1.25 * d4) - 2.0 * (d5 + 0.25 * d1 - 1.25 * d3)
 * wd5 = (d6 - 5.0 * d4 + 4.0 * d2) + 2.0 * (d1 + 0.25 * d5 - 1.25 * d3)
 * wd6 = (d6 - 5.0 * d4 + 4.0 * d2) - 2.0 * (d1 + 0.25 * d5 - 1.25 * d3)
 * wd7 = (d7 - d1) + 5.25 * (d3 - d5)
 */
#define INPUT_TRANSFORM(d, wd)                           \
    do {                                                 \
        wd##0 = (d##0 - d##6) + (d##4 - d##2) * 5.25;    \
        auto tmp0 = d##6 + d##2 - d##4 * 4.25;           \
        auto tmp1 = d##1 + d##5 - d##3 * 4.25;           \
        wd##1 = tmp0 + tmp1;                             \
        wd##2 = tmp0 - tmp1;                             \
        tmp0 = d##6 + d##2 * 0.25 - d##4 * 1.25;         \
        tmp1 = (d##5 + d##1 * 0.25 - d##3 * 1.25) * 2.0; \
        wd##3 = tmp0 + tmp1;                             \
        wd##4 = tmp0 - tmp1;                             \
        tmp0 = d6 - d4 * 5.0 + d2 * 4.0;                 \
        tmp1 = (d1 + d5 * 0.25 - d3 * 1.25) * 2.0;       \
        wd##5 = tmp0 + tmp1;                             \
        wd##6 = tmp0 - tmp1;                             \
        wd##7 = (d##7 - d##1) + (d##3 - d##5) * 5.25;    \
    } while (0);

#define GET_VECTOR_FP16Q_ELEM(s, i, idx) vgetq_lane_f16(CONCAT(s, i).value, idx)
struct InputTransform6x3 {
    template <bool inner>
    static void transform(const __fp16* input, __fp16* input_transform_buf,
                          __fp16* transform_mid_buf, int ih_start, int iw_start,
                          size_t ic, size_t IH, size_t IW, size_t IC,
                          size_t unit_idx, size_t nr_units_in_tile) {
        // BTd * B
        // 1.000   0.000  -5.25  0.000  5.250  0.000  -1.0 0.00
        // -0.00   1.000  1.000  -4.25  -4.25  1.000  1.00 -0.0
        // -0.00   -1.00  1.000  4.250  -4.25  -1.00  1.00 -0.0
        // 0.000   0.500  0.250  -2.50  -1.25  2.000  1.00 0.00
        // 0.000   -0.50  0.250  2.500  -1.25  -2.00  1.00 0.00
        // 0.000   2.000  4.000  -2.50  -5.00  0.500  1.00 0.00
        // 0.000   -2.00  4.000  2.500  -5.00  -0.50  1.00 0.00
        // 0.000   -1.00  0.000  5.250  0.000  -5.25  0.00 1.00
        constexpr size_t alpha = 6 + 3 - 1;
        if (!inner) {
            memset(transform_mid_buf, 0, sizeof(__fp16) * alpha * alpha);
        }

#define cb(i) Vector<__fp16, 8> d##i;
        UNROLL_CALL_NOWRAPPER(8, cb);
#undef cb

        if (inner) {
            const __fp16* input_ptr =
                    input + ic * IH * IW + ih_start * IW + iw_start;
#define cb(i) d##i = Vector<__fp16, 8>::load(input_ptr + IW * i);
            UNROLL_CALL_NOWRAPPER(8, cb);
#undef cb
        } else {
            int ih0_act = std::max<int>(ih_start, 0),
                ih1_act = std::min<int>(ih_start + alpha, IH),
                iw0_act = std::max<int>(iw_start, 0),
                iw1_act = std::min<int>(iw_start + alpha, IW);
            for (int ih = ih0_act; ih < ih1_act; ++ih) {
                for (int iw = iw0_act; iw < iw1_act; ++iw) {
                    size_t iho = ih - ih_start, iwo = iw - iw_start;
                    transform_mid_buf[iho * alpha + iwo] =
                            input[ic * IH * IW + ih * IW + iw];
                }
            }
#define cb(i) d##i = Vector<__fp16, 8>::load(transform_mid_buf + alpha * i);
            UNROLL_CALL_NOWRAPPER(8, cb);
#undef cb
        }

#define cb(i) Vector<__fp16, 8> wd##i, ret##i;
        UNROLL_CALL_NOWRAPPER(8, cb);
#undef cb

        INPUT_TRANSFORM(d, wd);

#if MEGDNN_AARCH64
        TRANSPOSE_8x8(wd, d);
        INPUT_TRANSFORM(d, ret);

#define cb(i) ret##i.save(transform_mid_buf + i * alpha);
        UNROLL_CALL_NOWRAPPER(8, cb);
#undef cb

        rep(i, alpha) rep(j, alpha) {
            input_transform_buf[(i * alpha + j) * nr_units_in_tile * IC +
                                unit_idx * IC + ic] =
                    transform_mid_buf[j * alpha + i];
        }
#else
        //!     1     0     0     0     0    0    0     0
        //!     0     1    -1   0.5  -0.5    2   -2    -1
        //! -5.25     1     1  0.25  0.25    4    4     0
        //!     0 -4.25  4.25  -2.5   2.5 -2.5  2.5  5.25
        //!  5.25 -4.25 -4.25 -1.25 -1.25   -5   -5     0
        //!     0     1    -1     2    -2  0.5 -0.5 -5.25
        //!    -1     1     1     1     1    1    1     0
        //!     0     0     0     0     0    0    0     1
#define cb(i)                                                   \
    do {                                                        \
        mid_buf1[0] = GET_VECTOR_FP16Q_ELEM(wd, i, 0) -         \
                      GET_VECTOR_FP16Q_ELEM(wd, i, 6) +         \
                      5.25 * (GET_VECTOR_FP16Q_ELEM(wd, i, 4) - \
                              GET_VECTOR_FP16Q_ELEM(wd, i, 2)); \
        mid_buf1[7] = GET_VECTOR_FP16Q_ELEM(wd, i, 7) -         \
                      GET_VECTOR_FP16Q_ELEM(wd, i, 1) +         \
                      5.25 * (GET_VECTOR_FP16Q_ELEM(wd, i, 3) - \
                              GET_VECTOR_FP16Q_ELEM(wd, i, 5)); \
        auto tmp0 = GET_VECTOR_FP16Q_ELEM(wd, i, 2) +           \
                    GET_VECTOR_FP16Q_ELEM(wd, i, 6) -           \
                    4.25 * GET_VECTOR_FP16Q_ELEM(wd, i, 4);     \
        auto tmp1 = GET_VECTOR_FP16Q_ELEM(wd, i, 1) -           \
                    GET_VECTOR_FP16Q_ELEM(wd, i, 3) * 4.25 +    \
                    GET_VECTOR_FP16Q_ELEM(wd, i, 5);            \
        mid_buf1[1] = tmp0 + tmp1;                              \
        mid_buf1[2] = tmp0 - tmp1;                              \
        tmp0 = GET_VECTOR_FP16Q_ELEM(wd, i, 2) * 0.25 +         \
               GET_VECTOR_FP16Q_ELEM(wd, i, 6) -                \
               GET_VECTOR_FP16Q_ELEM(wd, i, 4) * 1.25;          \
        tmp1 = GET_VECTOR_FP16Q_ELEM(wd, i, 1) * 0.5 -          \
               GET_VECTOR_FP16Q_ELEM(wd, i, 3) * 2.5 +          \
               GET_VECTOR_FP16Q_ELEM(wd, i, 5) * 2;             \
        mid_buf1[3] = tmp0 + tmp1;                              \
        mid_buf1[4] = tmp0 - tmp1;                              \
        tmp0 = GET_VECTOR_FP16Q_ELEM(wd, i, 6) +                \
               GET_VECTOR_FP16Q_ELEM(wd, i, 2) * 4.0 -          \
               GET_VECTOR_FP16Q_ELEM(wd, i, 4) * 5.0;           \
        tmp1 = GET_VECTOR_FP16Q_ELEM(wd, i, 1) * 2 -            \
               GET_VECTOR_FP16Q_ELEM(wd, i, 3) * 2.5 +          \
               GET_VECTOR_FP16Q_ELEM(wd, i, 5) * 0.5;           \
        mid_buf1[5] = tmp0 + tmp1;                              \
        mid_buf1[6] = tmp0 - tmp1;                              \
        mid_buf1 += 8;                                          \
    } while (0);

        __fp16* mid_buf1 = transform_mid_buf;
        UNROLL_CALL_NOWRAPPER(8, cb);
        mid_buf1 = transform_mid_buf;

#undef cb
        rep(i, alpha) rep(j, alpha) {
            input_transform_buf[(i * alpha + j) * nr_units_in_tile * IC +
                                unit_idx * IC + ic] =
                    transform_mid_buf[i * alpha + j];
        }
#endif
    }
};

#undef INPUT_TRANSFORM

#define OUTPUT_TRANSFORM(m, r)                                      \
    do {                                                            \
        auto m1addm2 = m##1 + m##2;                                 \
        auto m1subm2 = m##1 - m##2;                                 \
        auto m3addm4 = m##3 + m##4;                                 \
        auto m3subm4 = m##3 - m##4;                                 \
        auto m5addm6 = m##5 + m##6;                                 \
        auto m5subm6 = m##5 - m##6;                                 \
        r##0 = m##0 + m1addm2 + m3addm4 + m5addm6;                  \
        r##1 = m1subm2 + m3subm4 * 2.0 + m5subm6 * 0.5;             \
        r##2 = m1addm2 + m3addm4 * 4.0 + m5addm6 * 0.25;            \
        r##3 = m1subm2 + m3subm4 * 8.0 + m5subm6 * 0.125;           \
        r##4 = m1addm2 + m3addm4 * 16.0 + m5addm6 * 0.0625;         \
        r##5 = m1subm2 + m3subm4 * 32.0 + m5subm6 * 0.03125 + m##7; \
    } while (0)
template <BiasMode bmode, typename Op>
struct OutputTransform6X3 {
    static void transform(const dt_float16* output_transform_buf,
                          const dt_float16* bias, dt_float16* output,
                          dt_float16* transform_mid_buf, size_t oh_start,
                          size_t ow_start, size_t OH, size_t OW,
                          size_t oc_start, size_t oc_end, size_t oc_index,
                          size_t unit_idx, size_t nr_units_in_tile,
                          const DType& src_dtype, const DType& dst_dtype) {
        Op op(src_dtype, dst_dtype);
        //! AT * m * A
        // AT f45
        // 1.0  1.0   1.0   1.0    1.0  1.00000  1.00000  0.0
        // 0.0  1.0  -1.0   2.0   -2.0  0.50000 -0.50000  0.0
        // 0.0  1.0   1.0   4.0    4.0  0.25000  0.25000  0.0
        // 0.0  1.0  -1.0   8.0   -8.0  0.12500 -0.12500  0.0
        // 0.0  1.0   1.0  16.0   16.0  0.06250  0.06250  0.0
        // 0.0  1.0  -1.0  32.0  -32.0  0.03125 -0.03125  1.0
        constexpr size_t alpha = 3 + 6 - 1;
        const __fp16* fp16_output_transform_buf =
                reinterpret_cast<const __fp16*>(output_transform_buf);
        const __fp16* fp16_bias = reinterpret_cast<const __fp16*>(bias);
        __fp16* fp16_output = reinterpret_cast<__fp16*>(output);
        __fp16* fp16_transform_mid_buf =
                reinterpret_cast<__fp16*>(transform_mid_buf);

        __fp16* mid_buf1 = fp16_transform_mid_buf;

        size_t OC = oc_end - oc_start;
        size_t oc = oc_start + oc_index;

#define cb(m, n)                                                           \
    fp16_transform_mid_buf[m * alpha + n] =                                \
            fp16_output_transform_buf[(m * alpha + n) * nr_units_in_tile * \
                                              OC +                         \
                                      unit_idx * OC + oc_index];
        UNROLL_CALL_NOWRAPPER_D2(8, 8, cb);
#undef cb

#define cb(i) \
    auto m##i = Vector<__fp16, 8>::load(fp16_transform_mid_buf + alpha * i);
        UNROLL_CALL_NOWRAPPER(8, cb);
#undef cb
#define cb(i) Vector<__fp16, 8> s##i;
        UNROLL_CALL_NOWRAPPER(6, cb);
#undef cb
        /*  1.0    0.0   0.00    0.000    0.0000   0.00000
            1.0    1.0   1.00    1.000    1.0000   1.00000
            1.0   -1.0   1.00   -1.000    1.0000  -1.00000
            1.0    2.0   4.00    8.000   16.000   32.00000
            1.0   -2.0   4.00   -8.000   16.000  -32.00000
            1.0    0.5   0.25    0.125    0.0625   0.03125
            1.0   -0.5   0.25   -0.125    0.0625  -0.03125
            0.0    0.0   0.00    0.000    0.0000   1.00000*/

        OUTPUT_TRANSFORM(m, s);
        mid_buf1 = fp16_transform_mid_buf;

#define cb(i)                                                                 \
    do {                                                                      \
        auto m1addm2 = GET_VECTOR_FP16Q_ELEM(s, i, 1) +                       \
                       GET_VECTOR_FP16Q_ELEM(s, i, 2);                        \
        auto m1subm2 = GET_VECTOR_FP16Q_ELEM(s, i, 1) -                       \
                       GET_VECTOR_FP16Q_ELEM(s, i, 2);                        \
        auto m3addm4 = GET_VECTOR_FP16Q_ELEM(s, i, 3) +                       \
                       GET_VECTOR_FP16Q_ELEM(s, i, 4);                        \
        auto m3subm4 = GET_VECTOR_FP16Q_ELEM(s, i, 3) -                       \
                       GET_VECTOR_FP16Q_ELEM(s, i, 4);                        \
        auto m5addm6 = GET_VECTOR_FP16Q_ELEM(s, i, 5) +                       \
                       GET_VECTOR_FP16Q_ELEM(s, i, 6);                        \
        auto m5subm6 = GET_VECTOR_FP16Q_ELEM(s, i, 5) -                       \
                       GET_VECTOR_FP16Q_ELEM(s, i, 6);                        \
        mid_buf1[0] =                                                         \
                GET_VECTOR_FP16Q_ELEM(s, i, 0) + m1addm2 + m3addm4 + m5addm6; \
        mid_buf1[1] = m1subm2 + m3subm4 * 2 + m5subm6 * 0.5;                  \
        mid_buf1[2] = m1addm2 + m3addm4 * 4 + m5addm6 * 0.25;                 \
        mid_buf1[3] = m1subm2 + m3subm4 * 8 + m5subm6 * 0.125;                \
        mid_buf1[4] = m1addm2 + m3addm4 * 16 + m5addm6 * 0.0625;              \
        mid_buf1[5] = m1subm2 + m3subm4 * 32 + m5subm6 * 0.03125 +            \
                      GET_VECTOR_FP16Q_ELEM(s, i, 7);                         \
        mid_buf1 += 6;                                                        \
    } while (0);
        mid_buf1 = fp16_transform_mid_buf;
        UNROLL_CALL_NOWRAPPER(6, cb);
        mid_buf1 = fp16_transform_mid_buf;

#undef cb

        if (oh_start + 6 <= OH && ow_start + 6 <= OW) {
            int index = (oc * OH + oh_start) * OW + ow_start;

#define cb(i) float16x4_t vr##i = vld1_f16(mid_buf1 + i * 6);

            UNROLL_CALL_NOWRAPPER(6, cb);
#undef cb
            float16x8_t vr0123_45 = {mid_buf1[4],  mid_buf1[5],  mid_buf1[10],
                                     mid_buf1[11], mid_buf1[16], mid_buf1[17],
                                     mid_buf1[22], mid_buf1[23]};
            float16x4_t vr45_45 = {mid_buf1[28], mid_buf1[29], mid_buf1[34],
                                   mid_buf1[35]};

            if (bmode == BiasMode::BROADCAST_CHANNEL_BIAS) {
                float16x4_t bias0 = vdup_n_f16(fp16_bias[oc]);
#define cb(i) vr##i = vadd_f16(vr##i, bias0);

                UNROLL_CALL_NOWRAPPER(6, cb);
#undef cb
                vr45_45 = vadd_f16(vr45_45, bias0);
                vr0123_45 = vaddq_f16(vr0123_45, vcombine_f16(bias0, bias0));

            } else if (bmode == BiasMode::BIAS) {
#define cb(i) float16x4_t bmbias##i = vld1_f16(fp16_bias + index + OW * i);

                UNROLL_CALL_NOWRAPPER(6, cb);
#undef cb

#define cb(i) vr##i = vadd_f16(vr##i, bmbias##i);

                UNROLL_CALL_NOWRAPPER(6, cb);
#undef cb
                float16x8_t vb0123_45 = {fp16_bias[index + 0 * OW + 4],
                                         fp16_bias[index + 0 * OW + 5],
                                         fp16_bias[index + 1 * OW + 4],
                                         fp16_bias[index + 1 * OW + 5],
                                         fp16_bias[index + 2 * OW + 4],
                                         fp16_bias[index + 2 * OW + 5],
                                         fp16_bias[index + 3 * OW + 4],
                                         fp16_bias[index + 3 * OW + 5]};
                float16x4_t vb45_45 = {fp16_bias[index + 4 * OW + 4],
                                       fp16_bias[index + 4 * OW + 5],
                                       fp16_bias[index + 5 * OW + 4],
                                       fp16_bias[index + 5 * OW + 5]};
                vr45_45 = vadd_f16(vr45_45, vb45_45);
                vr0123_45 = vaddq_f16(vr0123_45, vb0123_45);
            }

            float16x8_t item01 = op(vcombine_f16(vr0, vr1));
            float16x8_t item23 = op(vcombine_f16(vr2, vr3));
            float16x8_t item45 = op(vcombine_f16(vr4, vr5));

            vst1_f16(fp16_output + index, vget_low_f16(item01));
            vst1_f16(fp16_output + index + OW, vget_high_f16(item01));
            vst1_f16(fp16_output + index + OW * 2, vget_low_f16(item23));
            vst1_f16(fp16_output + index + OW * 3, vget_high_f16(item23));
            vst1_f16(fp16_output + index + OW * 4, vget_low_f16(item45));
            vst1_f16(fp16_output + index + OW * 5, vget_high_f16(item45));
            vr0123_45 = op(vr0123_45);
            float16x8_t vr45 = op(vcombine_f16(vr45_45, vr45_45));

            fp16_output[index + OW * 0 + 4] = vgetq_lane_f16(vr0123_45, 0);
            fp16_output[index + OW * 0 + 5] = vgetq_lane_f16(vr0123_45, 1);
            fp16_output[index + OW * 1 + 4] = vgetq_lane_f16(vr0123_45, 2);
            fp16_output[index + OW * 1 + 5] = vgetq_lane_f16(vr0123_45, 3);
            fp16_output[index + OW * 2 + 4] = vgetq_lane_f16(vr0123_45, 4);
            fp16_output[index + OW * 2 + 5] = vgetq_lane_f16(vr0123_45, 5);
            fp16_output[index + OW * 3 + 4] = vgetq_lane_f16(vr0123_45, 6);
            fp16_output[index + OW * 3 + 5] = vgetq_lane_f16(vr0123_45, 7);
            fp16_output[index + OW * 4 + 4] = vgetq_lane_f16(vr45, 0);
            fp16_output[index + OW * 4 + 5] = vgetq_lane_f16(vr45, 1);
            fp16_output[index + OW * 5 + 4] = vgetq_lane_f16(vr45, 2);
            fp16_output[index + OW * 5 + 5] = vgetq_lane_f16(vr45, 3);
        } else {
            for (size_t oho = 0; oho < 6 && oh_start + oho < OH; ++oho) {
                for (size_t owo = 0; owo < 6 && ow_start + owo < OW; ++owo) {
                    size_t oh = oh_start + oho;
                    size_t ow = ow_start + owo;
                    __fp16 res = mid_buf1[oho * 6 + owo];
                    if (bmode == BiasMode::BIAS) {
                        res += fp16_bias[oc * OH * OW + oh * OW + ow];
                    } else if (bmode == BiasMode::BROADCAST_CHANNEL_BIAS) {
                        res += fp16_bias[oc];
                    }
                    res = op(res);
                    fp16_output[oc * OH * OW + oh * OW + ow] = res;
                }
            }
        }
    }
};
#undef GET_VECTOR_FP16Q_ELEM
#undef OUTPUT_TRANSFORM
}  // namespace

namespace megdnn {
namespace arm_common {
namespace winograd {

MEGDNN_REG_WINOGRAD_STRATEGY_IMPL(winograd_6x3_1x1_f16)

void winograd_6x3_1x1_f16::filter(const dt_float16* filter,
                                  dt_float16* filter_transform_buf,
                                  dt_float16* transform_mid_buf, size_t OC,
                                  size_t IC, size_t oc_start, size_t oc_end) {
    FilterTransform6X3::transform(
            reinterpret_cast<const __fp16*>(filter),
            reinterpret_cast<__fp16*>(filter_transform_buf),
            reinterpret_cast<__fp16*>(transform_mid_buf), OC, IC, oc_start,
            oc_end);
}

void winograd_6x3_1x1_f16::input(const dt_float16* input,
                                 dt_float16* input_transform_buf,
                                 dt_float16* transform_mid_buf, size_t IH,
                                 size_t IW, size_t IC, size_t PH, size_t PW,
                                 size_t unit_start_idx,
                                 size_t nr_units_in_tile) {
    constexpr int alpha = 6 + 3 - 1;
    // OW = IW + 2 * PW - KERNEL_SIZE + 1
    auto units_w = div_ceil<size_t>(IW + 2 * PW - KERNEL_SIZE + 1, OUTPUT_BLOCK_SIZE);
    rep(ic, IC) {
        rep(unit_idx, nr_units_in_tile) {
            size_t index = unit_start_idx + unit_idx;
            size_t nh = index / units_w;
            size_t nw = index % units_w;
            int ih_start = nh * OUTPUT_BLOCK_SIZE - PH;
            int iw_start = nw * OUTPUT_BLOCK_SIZE - PW;
            if (ih_start >= 0 && ih_start + alpha <= static_cast<int>(IH) &&
                iw_start >= 0 && iw_start + alpha <= static_cast<int>(IW)) {
                InputTransform6x3::transform<true>(
                        reinterpret_cast<const __fp16*>(input),
                        reinterpret_cast<__fp16*>(input_transform_buf),
                        reinterpret_cast<__fp16*>(transform_mid_buf), ih_start,
                        iw_start, ic, IH, IW, IC, unit_idx, nr_units_in_tile);

            } else {
                InputTransform6x3::transform<false>(
                        reinterpret_cast<const __fp16*>(input),
                        reinterpret_cast<__fp16*>(input_transform_buf),
                        reinterpret_cast<__fp16*>(transform_mid_buf), ih_start,
                        iw_start, ic, IH, IW, IC, unit_idx, nr_units_in_tile);
            }
        }
    }
}

void winograd_6x3_1x1_f16::output(const dt_float16* output_transform_buf,
                                  const dt_float16* bias, dt_float16* output,
                                  dt_float16* transform_mid_buf, BiasMode bmode,
                                  NonlineMode nonline_mode, size_t OH, size_t OW,
                                  size_t oc_start, size_t oc_end,
                                  size_t unit_start_idx, size_t nr_units_in_tile) {
#define cb(_bmode, _nonline_op, ...) \
    OutputTransform6X3<_bmode MEGDNN_COMMA _nonline_op>::transform(__VA_ARGS__);

    auto units_w = div_ceil<size_t>(OW, OUTPUT_BLOCK_SIZE);

    for (size_t oc = oc_start; oc < oc_end; oc++) {
        size_t oc_index = oc - oc_start;
        rep(unit_idx, nr_units_in_tile) {
            size_t index = unit_start_idx + unit_idx;
            auto nh = index / units_w;
            auto nw = index % units_w;
            size_t oh_start = nh * OUTPUT_BLOCK_SIZE;
            size_t ow_start = nw * OUTPUT_BLOCK_SIZE;
            DISPATCH_CONV_WINOGRAD_BIAS(
                    megdnn_arm_common_winograd_fp16_F63, cb, __fp16, __fp16, bmode,
                    nonline_mode, output_transform_buf, bias, output, transform_mid_buf,
                    oh_start, ow_start, OH, OW, oc_start, oc_end, oc_index, unit_idx,
                    nr_units_in_tile, src_dtype, dst_dtype);
        }
    }
#undef cb
}
}  // namespace winograd
}  // namespace arm_common
}  // namespace megdnn
#endif
// vim: syntax=cpp.doxygen
