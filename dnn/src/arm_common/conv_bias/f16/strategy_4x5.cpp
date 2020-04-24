/**
 * \file dnn/src/arm_common/conv_bias/f16/strategy_4x5.cpp
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
MIDOUT_DECL(megdnn_arm_common_winograd_fp16_F45)
using namespace megdnn;
using namespace arm_common;
namespace {

struct FilterTransform4X5 {
#define FILTER_TRANSFORM(d, wd)                                       \
    do {                                                              \
        wd##0 = d##0;                                                 \
        wd##r0 = d##r0;                                               \
        wd##1 = (d##0 + d##1 + d##2 + d##3 + d##4) * -0.222168;       \
        wd##r1 = (d##r0 + d##r1 + d##r2 + d##r3 + d##r4) * -0.222168; \
        wd##2 = (d##0 - d##1 + d##2 - d##3 + d##4) * -0.222168;       \
        wd##r2 = (d##r0 - d##r1 + d##r2 - d##r3 + d##r4) * -0.222168; \
        auto tmpd0 = d##0 * 0.710938;                                 \
        auto tmpd1 = d##1 * 0.355469;                                 \
        auto tmpd2 = d##2 * 0.177734;                                 \
        auto tmpd3 = d##3 * 0.088867;                                 \
        auto tmpd4 = d##4 * 0.044434;                                 \
        auto tmpdr0 = d##r0 * 0.710938;                               \
        auto tmpdr1 = d##r1 * 0.355469;                               \
        auto tmpdr2 = d##r2 * 0.177734;                               \
        auto tmpdr3 = d##r3 * 0.088867;                               \
        auto tmpdr4 = d##r4 * 0.044434;                               \
        wd##3 = tmpd0 + tmpd1 + tmpd2 + tmpd3 + tmpd4;                \
        wd##r3 = tmpdr0 + tmpdr1 + tmpdr2 + tmpdr3 + tmpdr4;          \
        wd##4 = tmpd0 - tmpd1 + tmpd2 - tmpd3 + tmpd4;                \
        wd##r4 = tmpdr0 - tmpdr1 + tmpdr2 - tmpdr3 + tmpdr4;          \
        tmpd0 = d##0 * 0.011108;                                      \
        tmpd1 = d##1 * 0.022217;                                      \
        tmpd2 = d##2 * 0.044434;                                      \
        tmpd3 = d##3 * 0.088867;                                      \
        tmpd4 = d##4 * 0.177734;                                      \
        tmpdr0 = d##r0 * 0.011108;                                    \
        tmpdr1 = d##r1 * 0.022217;                                    \
        tmpdr2 = d##r2 * 0.044434;                                    \
        tmpdr3 = d##r3 * 0.088867;                                    \
        tmpdr4 = d##r4 * 0.177734;                                    \
        wd##5 = tmpd0 + tmpd1 + tmpd2 + tmpd3 + tmpd4;                \
        ;                                                             \
        wd##r5 = tmpdr0 + tmpdr1 + tmpdr2 + tmpdr3 + tmpdr4;          \
        ;                                                             \
        wd##6 = tmpd0 - tmpd1 + tmpd2 - tmpd3 + tmpd4;                \
        ;                                                             \
        wd##r6 = tmpdr0 - tmpdr1 + tmpdr2 - tmpdr3 + tmpdr4;          \
        ;                                                             \
        wd##7 = d##4;                                                 \
        wd##r7 = d##r4;                                               \
    } while (0);

#define FILTER_TRANSFORM_FINAL(d, wd)                                    \
    do {                                                                 \
        wd##0 = d##0;                                                    \
        wd##1 = (d##0 + d##1 + d##2 + d##3 + d##4) * -0.222168;          \
        wd##2 = (d##0 - d##1 + d##2 - d##3 + d##4) * -0.222168;          \
        auto tmp0 = d##0 * 0.710938 + d##2 * 0.177734 + d##4 * 0.044434; \
        auto tmp1 = d##1 * 0.355469 + d##3 * 0.088867;                   \
        wd##3 = tmp0 + tmp1;                                             \
        wd##4 = tmp0 - tmp1;                                             \
        tmp0 = d##0 * 0.011108 + d##2 * 0.044434 + d##4 * 0.177734;      \
        tmp1 = d##1 * 0.022217 + d##3 * 0.088867;                        \
        wd##5 = tmp0 + tmp1;                                             \
        wd##6 = tmp0 - tmp1;                                             \
        wd##7 = d##4;                                                    \
    } while (0);
    static void transform(const __fp16* filter, __fp16* filter_transform_buf,
                          __fp16* transform_mid_buf, size_t OC, size_t IC,
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
        for (size_t oc = oc_start; oc < oc_end; oc++) {
            rep(ic, IC) {
                const __fp16* fptr = filter + (oc * IC + ic) * 5 * 5;

#define cb(i) Vector<__fp16, 4> g##i = Vector<__fp16, 4>::load(fptr + 5 * i);
                UNROLL_CALL_NOWRAPPER(5, cb);
#undef cb

#define cb(i) __fp16 gr##i = *(fptr + 5 * i + 4);
                UNROLL_CALL_NOWRAPPER(5, cb);

#undef cb
#define cb(i) Vector<__fp16, 4> Gg##i;
                UNROLL_CALL_NOWRAPPER(8, cb);
#undef cb

#define cb(i) __fp16 Ggr##i;
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
                float16x8_t vgr = {Ggr0, Ggr1, Ggr2, Ggr3,
                                   Ggr4, Ggr5, Ggr6, Ggr7};
                Vector<__fp16, 8> Ggt4(vgr);
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

#define GET_VECTOR_FP16D_ELEM(s, i, idx) vget_lane_f16(CONCAT(s, i).value, idx)

#define cb(i)                                                    \
    do {                                                         \
        mid_buf1[0] = GET_VECTOR_FP16D_ELEM(Gg, i, 0);           \
        auto tmp024 = GET_VECTOR_FP16D_ELEM(Gg, i, 0) +          \
                      GET_VECTOR_FP16D_ELEM(Gg, i, 2) + Ggr##i;  \
        auto tmp13 = GET_VECTOR_FP16D_ELEM(Gg, i, 1) +           \
                     GET_VECTOR_FP16D_ELEM(Gg, i, 3);            \
        mid_buf1[1] = (tmp024 + tmp13) * -0.2222222;             \
        mid_buf1[2] = (tmp024 - tmp13) * -0.2222222;             \
        auto tmp0 = GET_VECTOR_FP16D_ELEM(Gg, i, 0) * 0.7111111; \
        auto tmp1 = GET_VECTOR_FP16D_ELEM(Gg, i, 1) * 0.3555556; \
        auto tmp2 = GET_VECTOR_FP16D_ELEM(Gg, i, 2) * 0.1777778; \
        auto tmp3 = GET_VECTOR_FP16D_ELEM(Gg, i, 3) * 0.0888889; \
        auto tmp4 = Ggr##i * 0.0444444;                          \
        tmp024 = tmp0 + tmp2 + tmp4;                             \
        tmp13 = tmp1 + tmp3;                                     \
        mid_buf1[3] = tmp024 + tmp13;                            \
        mid_buf1[4] = tmp024 - tmp13;                            \
        tmp0 = GET_VECTOR_FP16D_ELEM(Gg, i, 0) * 0.0111111;      \
        tmp1 = GET_VECTOR_FP16D_ELEM(Gg, i, 1) * 0.0222222;      \
        tmp2 = GET_VECTOR_FP16D_ELEM(Gg, i, 2) * 0.0444444;      \
        tmp3 = GET_VECTOR_FP16D_ELEM(Gg, i, 3) * 0.0888889;      \
        tmp4 = Ggr##i * 0.1777778;                               \
        tmp024 = tmp0 + tmp2 + tmp4;                             \
        tmp13 = tmp1 + tmp3;                                     \
        mid_buf1[5] = tmp024 + tmp13;                            \
        mid_buf1[6] = tmp024 - tmp13;                            \
        mid_buf1[7] = Ggr##i;                                    \
        mid_buf1 += 8;                                           \
    } while (0);
                __fp16* mid_buf1 = transform_mid_buf;
                UNROLL_CALL_NOWRAPPER(8, cb);
                mid_buf1 = transform_mid_buf;
#undef cb
#undef GET_VECTOR_FP16D_ELEM
                rep(i, alpha) rep(j, alpha) {
                    filter_transform_buf[(i * alpha + j) * OC * IC + ic * OC +
                                         oc] = transform_mid_buf[i * alpha + j];
                }
#endif
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

#define GET_VECTOR_FP16Q_ELEM(s, i, idx) vgetq_lane_f16(CONCAT(s, i).value, idx)

    template <bool inner>
    static void transform(const __fp16* input, __fp16* input_transform_buf,
                          __fp16* transform_mid_buf, int ih_start, int iw_start,
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
    }
};
#undef INPUT_TRANSFORM

#define OUTPUT_TRANSFORM(m, s)                                          \
    do {                                                                \
        s##0 = m##0 + m##1 + m##2 + m##3 + m##4 + m##5 + m##6;          \
        s##1 = m##1 - m##2 + m##3 * 0.5 - m##4 * 0.5 + m##5 * 2.0 -     \
               m##6 * 2.0;                                              \
        s##2 = m##1 + m##2 + m##3 * 0.25 + m##4 * 0.25 + m##5 * 4.0 +   \
               m##6 * 4.0;                                              \
        s##3 = m##1 - m##2 + m##3 * 0.125 - m##4 * 0.125 + m##5 * 8.0 - \
               m##6 * 8.0 + m##7;                                       \
    } while (0)
template <BiasMode bmode, typename Op>
struct OutputTransform4X5 {
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
        // 1.0	1.0	 1.0	1.000	 1.000	1.0	 1.0	0.0
        // 0.0	1.0	-1.0	0.500	-0.500	2.0	-2.0	0.0
        // 0.0	1.0	 1.0	0.250	 0.250	4.0	 4.0	0.0
        // 0.0	1.0	-1.0	0.125	-0.125	8.0	-8.0	1.0
        constexpr size_t alpha = 5 + 4 - 1;
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
        UNROLL_CALL_NOWRAPPER(4, cb);
#undef cb
#define cb(i) Vector<__fp16, 4> st##i;
        UNROLL_CALL_NOWRAPPER(8, cb);
#undef cb
#define cb(i) Vector<__fp16, 4> result##i;
        UNROLL_CALL_NOWRAPPER(4, cb);
#undef cb

        OUTPUT_TRANSFORM(m, s);
        TRANSPOSE_4x8(s, st);
        OUTPUT_TRANSFORM(st, result);
        TRANSPOSE_4x4(result, result);

        if (oh_start + 4 <= OH && ow_start + 4 <= OW) {
            int index = (oc * OH + oh_start) * OW + ow_start;
            if (bmode == BiasMode::BROADCAST_CHANNEL_BIAS) {
                float16x4_t bias0 = vdup_n_f16(fp16_bias[oc]);
                result0.value = vadd_f16(result0.value, bias0);
                result1.value = vadd_f16(result1.value, bias0);
                result2.value = vadd_f16(result2.value, bias0);
                result3.value = vadd_f16(result3.value, bias0);
            } else if (bmode == BiasMode::BIAS) {
                float16x4_t bmbias0 = vld1_f16(fp16_bias + index);
                float16x4_t bmbias1 = vld1_f16(fp16_bias + index + OW);
                float16x4_t bmbias2 = vld1_f16(fp16_bias + index + OW * 2);
                float16x4_t bmbias3 = vld1_f16(fp16_bias + index + OW * 3);
                result0.value = vadd_f16(result0.value, bmbias0);
                result1.value = vadd_f16(result1.value, bmbias1);
                result2.value = vadd_f16(result2.value, bmbias2);
                result3.value = vadd_f16(result3.value, bmbias3);
            }

            float16x8_t item01 = op(vcombine_f16(result0.value, result1.value));
            float16x8_t item23 = op(vcombine_f16(result2.value, result3.value));

            vst1_f16(fp16_output + index, vget_low_f16(item01));
            vst1_f16(fp16_output + index + OW, vget_high_f16(item01));
            vst1_f16(fp16_output + index + OW * 2, vget_low_f16(item23));
            vst1_f16(fp16_output + index + OW * 3, vget_high_f16(item23));
        } else {
#define cb(i) result##i.save(mid_buf1 + i * 4);
            mid_buf1 = fp16_transform_mid_buf;
            UNROLL_CALL_NOWRAPPER(4, cb);
            mid_buf1 = fp16_transform_mid_buf;
#undef cb
            for (size_t oho = 0; oho < 4 && oh_start + oho < OH; ++oho) {
                for (size_t owo = 0; owo < 4 && ow_start + owo < OW; ++owo) {
                    size_t oh = oh_start + oho;
                    size_t ow = ow_start + owo;
                    __fp16 res = mid_buf1[oho * 4 + owo];
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
#undef OUTPUT_TRANSFORM
#undef GET_VECTOR_FP16Q_ELEM
}  // namespace

namespace megdnn {
namespace arm_common {
namespace winograd {

MEGDNN_REG_WINOGRAD_STRATEGY_IMPL(winograd_4x5_1x1_f16)

void winograd_4x5_1x1_f16::filter(const dt_float16* filter,
                                  dt_float16* filter_transform_buf,
                                  dt_float16* transform_mid_buf, size_t OC,
                                  size_t IC, size_t oc_start, size_t oc_end) {
    FilterTransform4X5::transform(
            reinterpret_cast<const __fp16*>(filter),
            reinterpret_cast<__fp16*>(filter_transform_buf),
            reinterpret_cast<__fp16*>(transform_mid_buf), OC, IC, oc_start,
            oc_end);
}

void winograd_4x5_1x1_f16::input(const dt_float16* input,
                                 dt_float16* input_transform_buf,
                                 dt_float16* transform_mid_buf, size_t IH,
                                 size_t IW, size_t IC, size_t PH, size_t PW,
                                 size_t unit_start_idx,
                                 size_t nr_units_in_tile) {
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
                        reinterpret_cast<const __fp16*>(input),
                        reinterpret_cast<__fp16*>(input_transform_buf),
                        reinterpret_cast<__fp16*>(transform_mid_buf), ih_start,
                        iw_start, ic, IH, IW, IC, unit_idx, nr_units_in_tile);

            } else {
                InputTransform4X5::transform<false>(
                        reinterpret_cast<const __fp16*>(input),
                        reinterpret_cast<__fp16*>(input_transform_buf),
                        reinterpret_cast<__fp16*>(transform_mid_buf), ih_start,
                        iw_start, ic, IH, IW, IC, unit_idx, nr_units_in_tile);
            }
        }
    }
}

void winograd_4x5_1x1_f16::output(const dt_float16* output_transform_buf,
                                  const dt_float16* bias, dt_float16* output,
                                  dt_float16* transform_mid_buf, BiasMode bmode,
                                  NonlineMode nonline_mode, size_t OH, size_t OW,
                                  size_t oc_start, size_t oc_end,
                                  size_t unit_start_idx, size_t nr_units_in_tile) {
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
                    megdnn_arm_common_winograd_fp16_F45, cb, __fp16, __fp16, bmode,
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
