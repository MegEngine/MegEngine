/**
 * \file dnn/src/arm_common/conv_bias/fp32/strategy_4x5.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/arm_common/conv_bias/fp32/strategy.h"
#include "src/arm_common/simd_macro/marm_neon.h"
#include "src/arm_common/utils.h"
#include "src/common/unroll_macro.h"
#include "src/common/utils.h"
#include "src/fallback/conv_bias/winograd/winograd.h"

#include "src/arm_common/conv_bias/fp32/helper.h"
#include "src/arm_common/elemwise_helper/op_unary.h"
#include "src/naive/matrix_mul/matrix_mul_helper.h"

#include "midout.h"
MIDOUT_DECL(megdnn_arm_common_winograd_fp32_F45)

using namespace megdnn;
using namespace arm_common;
namespace {

struct FilterTransform4X5 {
#define FILTER_TRANSFORM(d, wd)                                        \
    do {                                                               \
        wd##0 = d##0;                                                  \
        wd##r0 = d##r0;                                                \
        wd##1 = (d##0 + d##1 + d##2 + d##3 + d##4) * -0.2222222;       \
        wd##r1 = (d##r0 + d##r1 + d##r2 + d##r3 + d##r4) * -0.2222222; \
        wd##2 = (d##0 - d##1 + d##2 - d##3 + d##4) * -0.2222222;       \
        wd##r2 = (d##r0 - d##r1 + d##r2 - d##r3 + d##r4) * -0.2222222; \
        auto tmpd0 = d##0 * 0.7111111;                                 \
        auto tmpd1 = d##1 * 0.3555556;                                 \
        auto tmpd2 = d##2 * 0.1777778;                                 \
        auto tmpd3 = d##3 * 0.0888889;                                 \
        auto tmpd4 = d##4 * 0.0444444;                                 \
        auto tmpdr0 = d##r0 * 0.7111111;                               \
        auto tmpdr1 = d##r1 * 0.3555556;                               \
        auto tmpdr2 = d##r2 * 0.1777778;                               \
        auto tmpdr3 = d##r3 * 0.0888889;                               \
        auto tmpdr4 = d##r4 * 0.0444444;                               \
        wd##3 = tmpd0 + tmpd1 + tmpd2 + tmpd3 + tmpd4;                 \
        wd##r3 = tmpdr0 + tmpdr1 + tmpdr2 + tmpdr3 + tmpdr4;           \
        wd##4 = tmpd0 - tmpd1 + tmpd2 - tmpd3 + tmpd4;                 \
        wd##r4 = tmpdr0 - tmpdr1 + tmpdr2 - tmpdr3 + tmpdr4;           \
        tmpd0 = d##0 * 0.0111111;                                      \
        tmpd1 = d##1 * 0.0222222;                                      \
        tmpd2 = d##2 * 0.0444444;                                      \
        tmpd3 = d##3 * 0.0888889;                                      \
        tmpd4 = d##4 * 0.1777778;                                      \
        tmpdr0 = d##r0 * 0.0111111;                                    \
        tmpdr1 = d##r1 * 0.0222222;                                    \
        tmpdr2 = d##r2 * 0.0444444;                                    \
        tmpdr3 = d##r3 * 0.0888889;                                    \
        tmpdr4 = d##r4 * 0.1777778;                                    \
        wd##5 = tmpd0 + tmpd1 + tmpd2 + tmpd3 + tmpd4;                 \
        wd##r5 = tmpdr0 + tmpdr1 + tmpdr2 + tmpdr3 + tmpdr4;           \
        wd##6 = tmpd0 - tmpd1 + tmpd2 - tmpd3 + tmpd4;                 \
        wd##r6 = tmpdr0 - tmpdr1 + tmpdr2 - tmpdr3 + tmpdr4;           \
        wd##7 = d##4;                                                  \
        wd##r7 = d##r4;                                                \
    } while (0);

#define FILTER_TRANSFORM_FINAL(d, wd)                                       \
    do {                                                                    \
        wd##0 = d##0;                                                       \
        wd##1 = (d##0 + d##1 + d##2 + d##3 + d##4) * -0.2222222;            \
        wd##2 = (d##0 - d##1 + d##2 - d##3 + d##4) * -0.2222222;            \
        auto tmp0 = d##0 * 0.7111111 + d##2 * 0.1777778 + d##4 * 0.0444444; \
        auto tmp1 = d##1 * 0.3555556 + d##3 * 0.0888889;                    \
        wd##3 = tmp0 + tmp1;                                                \
        wd##4 = tmp0 - tmp1;                                                \
        tmp0 = d##0 * 0.0111111 + d##2 * 0.0444444 + d##4 * 0.1777778;      \
        tmp1 = d##1 * 0.0222222 + d##3 * 0.0888889;                         \
        wd##5 = tmp0 + tmp1;                                                \
        wd##6 = tmp0 - tmp1;                                                \
        wd##7 = d##4;                                                       \
    } while (0);
    static void transform(const float* filter, float* filter_transform_buf,
                          float* transform_mid_buf, size_t OC, size_t IC,
                          size_t oc_start, size_t oc_end) {
        // Gg * GT
        // G
        //[[ 1.         0.         0.         0.         0.       ]
        // [-0.2222222 -0.2222222 -0.2222222 -0.2222222 -0.2222222]
        // [-0.2222222  0.2222222 -0.2222222  0.2222222 -0.2222222]
        // [ 0.7111111  0.3555556  0.1777778  0.0888889  0.0444444]
        // [ 0.7111111 -0.3555556  0.1777778 -0.0888889  0.0444444]
        // [ 0.0111111  0.0222222  0.0444444  0.0888889  0.1777778]
        // [ 0.0111111 -0.0222222  0.0444444 -0.0888889  0.1777778]
        // [ 0.         0.         0.         0.         1.       ]]
        constexpr size_t alpha = 4 + 5 - 1;
        for (size_t oc = oc_start; oc < oc_end; oc++)
            rep(ic, IC) {
                const float* fptr = filter + (oc * IC + ic) * 5 * 5;

#define cb(i) Vector<float, 4> g##i = Vector<float, 4>::load(fptr + 5 * i);
                UNROLL_CALL_NOWRAPPER(5, cb);
#undef cb

#define cb(i) float gr##i = *(fptr + 5 * i + 4);
                UNROLL_CALL_NOWRAPPER(5, cb);

#undef cb
#define cb(i) Vector<float, 4> Gg##i;
                UNROLL_CALL_NOWRAPPER(8, cb);
#undef cb

#define cb(i) float Ggr##i;
                UNROLL_CALL_NOWRAPPER(8, cb);
#undef cb

#define cb(i) Vector<float, 8> Ggt##i;
                UNROLL_CALL_NOWRAPPER(4, cb);
#undef cb

#define cb(i) Vector<float, 8> result##i;
                UNROLL_CALL_NOWRAPPER(8, cb);
#undef cb

                FILTER_TRANSFORM(g, Gg)
                float32x4x2_t vgr;
                float32x4_t vgr0 = {Ggr0, Ggr1, Ggr2, Ggr3};
                float32x4_t vgr1 = {Ggr4, Ggr5, Ggr6, Ggr7};
                vgr.val[0] = vgr0;  //{Ggr0, Ggr1, Ggr2, Ggr3};
                vgr.val[1] = vgr1;  //{Ggr4, Ggr5, Ggr6, Ggr7};
                Vector<float, 8> Ggt4(vgr);
                TRANSPOSE_8x4(Gg, Ggt);
                FILTER_TRANSFORM_FINAL(Ggt, result);

#define cb(i) result##i.save(transform_mid_buf + i * alpha);
                UNROLL_CALL_NOWRAPPER(8, cb);
#undef cb
                rep(i, alpha) rep(j, alpha) {
                    filter_transform_buf[(i * alpha + j) * OC * IC + ic * OC +
                                         oc] = transform_mid_buf[j * alpha + i];
                }
            }
    }
};
#undef FILTER_TRANSFORM
#undef FILTER_TRANSFORM_FINAL

struct InputTransform4X5 {
#define INPUT_TRANSFORM(d, wd)                          \
    do {                                                \
        wd##0 = (d##0 - d##6) + (d##4 - d##2) * 5.25f;  \
        auto tmp0 = d##2 - d##4 * 4.25f + d##6;         \
        auto tmp1 = d##1 - d##3 * 4.25f + d##5;         \
        wd##1 = tmp0 + tmp1;                            \
        wd##2 = tmp0 - tmp1;                            \
        tmp0 = d##2 * 4.0f - d##4 * 5.0f + d##6;        \
        tmp1 = d##1 * 2.0f - d##3 * 2.5f + d##5 * 0.5f; \
        wd##3 = tmp0 + tmp1;                            \
        wd##4 = tmp0 - tmp1;                            \
        tmp0 = d##2 * 0.25f - d##4 * 1.25f + d##6;      \
        tmp1 = d##1 * 0.5f - d##3 * 2.5f + d##5 * 2.0f; \
        wd##5 = tmp0 + tmp1;                            \
        wd##6 = tmp0 - tmp1;                            \
        wd##7 = (d##7 - d##1) + (d##3 - d##5) * 5.25f;  \
    } while (0)

#define GET_VECTOR_HIGH_ELEM(s, i, idx) \
    vgetq_lane_f32(CONCAT(s, i).value.val[1], idx)
#define GET_VECTOR_LOW_ELEM(s, i, idx) \
    vgetq_lane_f32(CONCAT(s, i).value.val[0], idx)

    template <bool inner>
    static void transform(const float* input, float* input_transform_buf,
                          float* transform_mid_buf, int ih_start, int iw_start,
                          size_t ic, size_t IH, size_t IW, size_t IC,
                          size_t unit_idx, size_t nr_units_in_tile) {
    // BTd * B
    //([[ 1.  ,  0.  , -5.25,  0.  ,  5.25,  0.  , -1.  ,  0.  ],
    //  [ 0.  ,  1.  ,  1.  , -4.25, -4.25,  1.  ,  1.  ,  0.  ],
    //  [ 0.  , -1.  ,  1.  ,  4.25, -4.25, -1.  ,  1.  ,  0.  ],
    //  [ 0.  ,  2.  ,  4.  , -2.5 , -5.  ,  0.5 ,  1.  ,  0.  ],
    //  [ 0.  , -2.  ,  4.  ,  2.5 , -5.  , -0.5 ,  1.  ,  0.  ],
    //  [ 0.  ,  0.5 ,  0.25, -2.5 , -1.25,  2.  ,  1.  ,  0.  ],
    //  [ 0.  , -0.5 ,  0.25,  2.5 , -1.25, -2.  ,  1.  ,  0.  ],
    //  [ 0.  , -1.  ,  0.  ,  5.25,  0.  , -5.25,  0.  ,  1.  ]]))

        constexpr size_t alpha = 4 + 5 - 1;
        if (!inner) {
            memset(transform_mid_buf, 0, sizeof(float) * alpha * alpha);
        }

#define cb(i) Vector<float, 8> d##i;
        UNROLL_CALL_NOWRAPPER(8, cb);
#undef cb

        if (inner) {
            const float* input_ptr =
                    input + ic * IH * IW + ih_start * IW + iw_start;
#define cb(i) d##i = Vector<float, 8>::load(input_ptr + IW * i);
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
#define cb(i) d##i = Vector<float, 8>::load(transform_mid_buf + alpha * i);
            UNROLL_CALL_NOWRAPPER(8, cb);
#undef cb
        }

#define cb(i) Vector<float, 8> wd##i, ret##i;
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
#define cb(i)                                                  \
    do {                                                       \
        mid_buf1[0] = GET_VECTOR_LOW_ELEM(wd, i, 0) -          \
                      GET_VECTOR_HIGH_ELEM(wd, i, 2) +         \
                      5.25 * (GET_VECTOR_HIGH_ELEM(wd, i, 0) - \
                              GET_VECTOR_LOW_ELEM(wd, i, 2));  \
        mid_buf1[7] = GET_VECTOR_HIGH_ELEM(wd, i, 3) -         \
                      GET_VECTOR_LOW_ELEM(wd, i, 1) +          \
                      5.25 * (GET_VECTOR_LOW_ELEM(wd, i, 3) -  \
                              GET_VECTOR_HIGH_ELEM(wd, i, 1)); \
        auto tmp0 = 4 * GET_VECTOR_LOW_ELEM(wd, i, 2) +        \
                    -5 * GET_VECTOR_HIGH_ELEM(wd, i, 0) +      \
                    GET_VECTOR_HIGH_ELEM(wd, i, 2);            \
        auto tmp1 = 2 * GET_VECTOR_LOW_ELEM(wd, i, 1) +        \
                    -2.5 * GET_VECTOR_LOW_ELEM(wd, i, 3) +     \
                    0.5 * GET_VECTOR_HIGH_ELEM(wd, i, 1);      \
        mid_buf1[3] = tmp0 + tmp1;                             \
        mid_buf1[4] = tmp0 - tmp1;                             \
        tmp0 = GET_VECTOR_LOW_ELEM(wd, i, 2) +                 \
               -4.25 * GET_VECTOR_HIGH_ELEM(wd, i, 0) +        \
               GET_VECTOR_HIGH_ELEM(wd, i, 2);                 \
        tmp1 = GET_VECTOR_LOW_ELEM(wd, i, 1) +                 \
               GET_VECTOR_LOW_ELEM(wd, i, 3) * -4.25 +         \
               GET_VECTOR_HIGH_ELEM(wd, i, 1);                 \
        mid_buf1[1] = tmp0 + tmp1;                             \
        mid_buf1[2] = tmp0 - tmp1;                             \
        tmp0 = GET_VECTOR_LOW_ELEM(wd, i, 2) * 0.25 +          \
               GET_VECTOR_HIGH_ELEM(wd, i, 0) * -1.25 +        \
               GET_VECTOR_HIGH_ELEM(wd, i, 2);                 \
        tmp1 = GET_VECTOR_LOW_ELEM(wd, i, 1) * 0.5 +           \
               GET_VECTOR_LOW_ELEM(wd, i, 3) * -2.5 +          \
               GET_VECTOR_HIGH_ELEM(wd, i, 1) * 2;             \
        mid_buf1[5] = tmp0 + tmp1;                             \
        mid_buf1[6] = tmp0 - tmp1;                             \
        mid_buf1 += 8;                                         \
    } while (0);

        float* mid_buf1 = transform_mid_buf;
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

#define OUTPUT_TRANSFORM(m, s)                                             \
    do {                                                                   \
        s0 = m0 + m1 + m2 + m3 + m4 + m5 + m6;                             \
        s1 = m1 - m2 + m3 * 0.5 - m4 * 0.5 + m5 * 2.0 - m6 * 2.0;          \
        s2 = m1 + m2 + m3 * 0.25 + m4 * 0.25 + m5 * 4.0 + m6 * 4.0;        \
        s3 = m1 - m2 + m3 * 0.125 - m4 * 0.125 + m5 * 8.0 - m6 * 8.0 + m7; \
    } while (0)
template <BiasMode bmode, typename Op>
struct OutputTransform4X5 {
    static void transform(const float* output_transform_buf, const float* bias,
                          float* output, float* transform_mid_buf,
                          size_t oh_start, size_t ow_start, size_t OH,
                          size_t OW, size_t oc_start, size_t oc_end,
                          size_t oc_index, size_t unit_idx,
                          size_t nr_units_in_tile, const DType& src_dtype,
                          const DType& dst_dtype) {
        Op op(src_dtype, dst_dtype);
        //! AT * m * A
        // AT f45
        // 1.0	1.0	 1.0	1.000	 1.000	1.0	 1.0	0.0
        // 0.0	1.0	-1.0	0.500	-0.500	2.0	-2.0	0.0
        // 0.0	1.0	 1.0	0.250	 0.250	4.0	 4.0	0.0
        // 0.0	1.0	-1.0	0.125	-0.125	8.0	-8.0	1.0
        constexpr size_t alpha = 5 + 4 - 1;
        float* mid_buf1 = transform_mid_buf;

        size_t OC = oc_end - oc_start;
        size_t oc = oc_start + oc_index;

#define cb(m, n)                                                           \
    transform_mid_buf[m * alpha + n] =                                     \
            output_transform_buf[(m * alpha + n) * nr_units_in_tile * OC + \
                                 unit_idx * OC + oc_index];
        UNROLL_CALL_NOWRAPPER_D2(8, 8, cb);
#undef cb

#define cb(i) auto m##i = Vector<float, 8>::load(transform_mid_buf + alpha * i);
        UNROLL_CALL_NOWRAPPER(8, cb);
#undef cb
#define cb(i) Vector<float, 8> s##i;
        UNROLL_CALL_NOWRAPPER(4, cb);
#undef cb

        OUTPUT_TRANSFORM(m, s);
#define cb(i)                                                                  \
    do {                                                                       \
        auto add12 =                                                           \
                GET_VECTOR_LOW_ELEM(s, i, 1) + GET_VECTOR_LOW_ELEM(s, i, 2);   \
        auto add34 =                                                           \
                GET_VECTOR_LOW_ELEM(s, i, 3) + GET_VECTOR_HIGH_ELEM(s, i, 0);  \
        auto add56 =                                                           \
                GET_VECTOR_HIGH_ELEM(s, i, 1) + GET_VECTOR_HIGH_ELEM(s, i, 2); \
        auto sub12 =                                                           \
                GET_VECTOR_LOW_ELEM(s, i, 1) - GET_VECTOR_LOW_ELEM(s, i, 2);   \
        auto sub34 =                                                           \
                GET_VECTOR_LOW_ELEM(s, i, 3) - GET_VECTOR_HIGH_ELEM(s, i, 0);  \
        auto sub56 =                                                           \
                GET_VECTOR_HIGH_ELEM(s, i, 1) - GET_VECTOR_HIGH_ELEM(s, i, 2); \
        mid_buf1[0] = GET_VECTOR_LOW_ELEM(s, i, 0) + add12 + add34 + add56;    \
        mid_buf1[1] = sub12 + sub34 * 0.5 + sub56 * 2.0;                       \
        mid_buf1[2] = add12 + add34 * 0.25 + add56 * 4.0;                      \
        mid_buf1[3] = sub12 + sub34 * 0.125 + sub56 * 8.0 +                    \
                      GET_VECTOR_HIGH_ELEM(s, i, 3);                           \
        mid_buf1 += 4;                                                         \
    } while (0);

        mid_buf1 = transform_mid_buf;
        UNROLL_CALL_NOWRAPPER(4, cb);
        mid_buf1 = transform_mid_buf;
#undef cb

        if (oh_start + 4 <= OH && ow_start + 4 <= OW) {
            float32x4_t bias0;
            if (bmode == BiasMode::BROADCAST_CHANNEL_BIAS) {
                bias0 = vdupq_n_f32(bias[oc]);
            }
            rep(i, 4) {
                size_t oh = oh_start + i;
                float32x4_t item0 = vld1q_f32(mid_buf1);

                if (bmode == BiasMode::BROADCAST_CHANNEL_BIAS) {
                    item0 = vaddq_f32(item0, bias0);
                } else if (bmode == BiasMode::BIAS) {
                    bias0 = vld1q_f32(bias + oc * OH * OW + oh * OW + ow_start);
                    item0 = vaddq_f32(item0, bias0);
                }
                item0 = op(item0);
                vst1q_f32(output + oc * OH * OW + oh * OW + ow_start, item0);
                mid_buf1 += 4;
            }
        } else {
            for (size_t oho = 0; oho < 4 && oh_start + oho < OH; ++oho) {
                for (size_t owo = 0; owo < 4 && ow_start + owo < OW; ++owo) {
                    size_t oh = oh_start + oho;
                    size_t ow = ow_start + owo;
                    float res = mid_buf1[oho * 4 + owo];
                    if (bmode == BiasMode::BIAS) {
                        res += bias[oc * OH * OW + oh * OW + ow];
                    } else if (bmode == BiasMode::BROADCAST_CHANNEL_BIAS) {
                        res += bias[oc];
                    }
                    res = op(res);
                    output[oc * OH * OW + oh * OW + ow] = res;
                }
            }
        }
    }
};
#undef OUTPUT_TRANSFORM
#undef GET_VECTOR_HIGH_ELEM
#undef GET_VECTOR_LOW_ELEM

}  // namespace

namespace megdnn {
namespace arm_common {
namespace winograd {

MEGDNN_REG_WINOGRAD_STRATEGY_IMPL(winograd_4x5_1x1_f)

void winograd_4x5_1x1_f::filter(const float* filter,
                                float* filter_transform_buf,
                                float* transform_mid_buf, size_t OC, size_t IC,
                                size_t oc_start, size_t oc_end) {
    FilterTransform4X5::transform(filter, filter_transform_buf,
                                  transform_mid_buf, OC, IC, oc_start, oc_end);
}

void winograd_4x5_1x1_f::input(const float* input, float* input_transform_buf,
                               float* transform_mid_buf, size_t IH, size_t IW,
                               size_t IC, size_t PH, size_t PW,
                               size_t unit_start_idx, size_t nr_units_in_tile) {
    constexpr int alpha = 4 + 5 - 1;

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
                InputTransform4X5::transform<true>(
                        input, input_transform_buf, transform_mid_buf, ih_start,
                        iw_start, ic, IH, IW, IC, unit_idx, nr_units_in_tile);

            } else {
                InputTransform4X5::transform<false>(
                        input, input_transform_buf, transform_mid_buf, ih_start,
                        iw_start, ic, IH, IW, IC, unit_idx, nr_units_in_tile);
            }
        }
    }
}

void winograd_4x5_1x1_f::output(const float* output_transform_buf,
                                const float* bias, float* output,
                                float* transform_mid_buf, BiasMode bmode,
                                NonlineMode nonline_mode, size_t OH, size_t OW,
                                size_t oc_start, size_t oc_end, size_t unit_start_idx,
                                size_t nr_units_in_tile) {
#define cb(_bmode, _nonline_op, ...) \
    OutputTransform4X5<_bmode MEGDNN_COMMA _nonline_op>::transform(__VA_ARGS__);

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
                    megdnn_arm_common_winograd_fp32_F45, cb, float, float, bmode,
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

// vim: syntax=cpp.doxygen
