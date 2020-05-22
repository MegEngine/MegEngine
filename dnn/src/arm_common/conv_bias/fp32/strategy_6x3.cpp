/**
 * \file dnn/src/arm_common/conv_bias/fp32/strategy_6x3.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/arm_common/conv_bias/fp32/filter_transform.h"
#include "src/arm_common/conv_bias/fp32/helper.h"
#include "src/arm_common/conv_bias/fp32/strategy.h"
#include "src/arm_common/elemwise_helper/op_unary.h"
#include "src/arm_common/simd_macro/marm_neon.h"
#include "src/arm_common/utils.h"
#include "src/common/unroll_macro.h"
#include "src/common/utils.h"
#include "src/fallback/conv_bias/winograd/winograd.h"
#include "src/naive/matrix_mul/matrix_mul_helper.h"

#include "midout.h"
MIDOUT_DECL(megdnn_arm_common_winograd_fp32_F63)

using namespace megdnn;
using namespace arm_common;
namespace {

/**
 * input transform
 *
 * wd0 = (d0 - d6) + 5.25 * (d4 - d2)
 * wd1 = (d6 + d2 - 4.25 * d4) + (d1 + d5 - 4.25 * d3)
 * wd2 = (d6 + d2 - 4.25 * d4) - (d1 + d5 - 4.25 * d3)
 * wd3 = (d6 + 0.25 * d2 - 1.25 * d4) + 2.0 * (d5 + 0.25 * d1 - 1.25 * d3)
 * wd4 = (d6 + 0.25 * d2 - 1.25 * d4) - 2.0 * (d5 + 0.25 * d1 - 1.25 * d3)
 * wd5 = (d6 - 5.0 * d4 + 4.0 * d2) + 2.0 * (d1 + 0.25 * d5 - 1.25 * d3)
 * wd6 = (d6 - 5.0 * d4 + 4.0 * d2) - 2.0 * (d1 + 0.25 * d5 - 1.25 * d3)
 * wd7 = (d7 - d1) + 5.25 * (d3 - d5)
 */
#define INPUT_TRANSFORM(d, wd)                              \
    do {                                                    \
        wd##0 = (d##0 - d##6) + (d##4 - d##2) * 5.25f;      \
        auto tmp0 = d##6 + d##2 - d##4 * 4.25f;             \
        auto tmp1 = d##1 + d##5 - d##3 * 4.25f;             \
        wd##1 = tmp0 + tmp1;                                \
        wd##2 = tmp0 - tmp1;                                \
        tmp0 = d##6 + d##2 * 0.25f - d##4 * 1.25f;          \
        tmp1 = (d##5 + d##1 * 0.25f - d##3 * 1.25f) * 2.0f; \
        wd##3 = tmp0 + tmp1;                                \
        wd##4 = tmp0 - tmp1;                                \
        tmp0 = d6 - d4 * 5.0f + d2 * 4.0f;                  \
        tmp1 = (d1 + d5 * 0.25f - d3 * 1.25f) * 2.0f;       \
        wd##5 = tmp0 + tmp1;                                \
        wd##6 = tmp0 - tmp1;                                \
        wd##7 = (d##7 - d##1) + (d##3 - d##5) * 5.25f;      \
    } while (0);

#define GET_VECTOR_HIGH_ELEM(s, i, idx) \
    vgetq_lane_f32(CONCAT(s, i).value.val[1], idx)
#define GET_VECTOR_LOW_ELEM(s, i, idx) \
    vgetq_lane_f32(CONCAT(s, i).value.val[0], idx)
struct InputTransform6X3 {
    template <bool inner>
    static void transform(const float* input, float* input_transform_buf,
                          float* transform_mid_buf, int ih_start, int iw_start,
                          size_t ic, size_t IH, size_t IW, size_t IC,
                          size_t unit_idx, size_t nr_units_in_tile) {
        constexpr size_t alpha = 6 + 3 - 1;
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
        mid_buf1[0] = GET_VECTOR_LOW_ELEM(wd, i, 0) -           \
                      GET_VECTOR_HIGH_ELEM(wd, i, 2) +          \
                      5.25f * (GET_VECTOR_HIGH_ELEM(wd, i, 0) - \
                               GET_VECTOR_LOW_ELEM(wd, i, 2));  \
        mid_buf1[7] = GET_VECTOR_HIGH_ELEM(wd, i, 3) -          \
                      GET_VECTOR_LOW_ELEM(wd, i, 1) +           \
                      5.25f * (GET_VECTOR_LOW_ELEM(wd, i, 3) -  \
                               GET_VECTOR_HIGH_ELEM(wd, i, 1)); \
        auto tmp0 = GET_VECTOR_LOW_ELEM(wd, i, 2) +             \
                    GET_VECTOR_HIGH_ELEM(wd, i, 2) -            \
                    4.25f * GET_VECTOR_HIGH_ELEM(wd, i, 0);     \
        auto tmp1 = GET_VECTOR_LOW_ELEM(wd, i, 1) +             \
                    GET_VECTOR_HIGH_ELEM(wd, i, 1) -            \
                    4.25f * GET_VECTOR_LOW_ELEM(wd, i, 3);      \
        mid_buf1[1] = tmp0 + tmp1;                              \
        mid_buf1[2] = tmp0 - tmp1;                              \
        tmp0 = GET_VECTOR_HIGH_ELEM(wd, i, 2) +                 \
               0.25f * GET_VECTOR_LOW_ELEM(wd, i, 2) -          \
               GET_VECTOR_HIGH_ELEM(wd, i, 0) * 1.25f;          \
        tmp1 = GET_VECTOR_LOW_ELEM(wd, i, 1) * 0.5f -           \
               GET_VECTOR_LOW_ELEM(wd, i, 3) * 2.5f +           \
               GET_VECTOR_HIGH_ELEM(wd, i, 1) * 2.f;            \
        mid_buf1[3] = tmp0 + tmp1;                              \
        mid_buf1[4] = tmp0 - tmp1;                              \
        tmp0 = GET_VECTOR_HIGH_ELEM(wd, i, 2) +                 \
               (GET_VECTOR_LOW_ELEM(wd, i, 2) -                 \
                GET_VECTOR_HIGH_ELEM(wd, i, 0) * 1.25f) *       \
                       4;                                       \
        tmp1 = GET_VECTOR_LOW_ELEM(wd, i, 1) * 2.f -            \
               GET_VECTOR_LOW_ELEM(wd, i, 3) * 2.5f +           \
               GET_VECTOR_HIGH_ELEM(wd, i, 1) * 0.5f;           \
        mid_buf1[5] = tmp0 + tmp1;                              \
        mid_buf1[6] = tmp0 - tmp1;                              \
        mid_buf1 += 8;                                          \
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

/**
 * Output Transform: use fma
 *
 * s0 = m0 + (m1 + m2) +      (m3 + m4) + 32 * (m5 + m6) / 32
 * s1 =      (m1 - m2) +  2 * (m3 - m4) + 16 * (m5 - m6) / 32
 * s2 =      (m1 + m2) +  4 * (m3 + m4) +  8 * (m5 + m6) / 32
 * s3 =      (m1 - m2) +  8 * (m3 - m4) +  4 * (m5 - m6) / 32
 * s4 =      (m1 + m2) + 16 * (m3 + m4) +  2 * (m5 + m6) / 32
 * s5 =      (m1 - m2) + 32 * (m3 - m4) +      (m5 - m6) / 32 + m7
 */
#define OUTPUT_TRANSFORM(m, s)                                     \
    do {                                                           \
        auto m1addm2 = m##1 + m##2;                                \
        auto m1subm2 = m##1 - m##2;                                \
        auto m3addm4 = m##3 + m##4;                                \
        auto m3subm4 = m##3 - m##4;                                \
        auto m5addm6 = (m##5 + m##6) * 0.03125f;                   \
        auto m5subm6 = (m##5 - m##6) * 0.03125f;                   \
        s##0 = m##0;                                               \
        CONCAT(s, 0).mla(m5addm6, 32.f).add(m3addm4).add(m1addm2); \
        CONCAT(s, 1) = m1subm2;                                    \
        CONCAT(s, 1).mla(m3subm4, 2.f).mla(m5subm6, 16.f);         \
        CONCAT(s, 2) = m1addm2;                                    \
        CONCAT(s, 2).mla(m3addm4, 4.f).mla(m5addm6, 8.f);          \
        CONCAT(s, 3) = m1subm2;                                    \
        CONCAT(s, 3).mla(m3subm4, 8.f).mla(m5subm6, 4.f);          \
        CONCAT(s, 4) = m1addm2;                                    \
        CONCAT(s, 4).mla(m3addm4, 16.f).mla(m5addm6, 2.f);         \
        CONCAT(s, 5) = m1subm2;                                    \
        CONCAT(s, 5).mla(m3subm4, 32.f).add(m5subm6).add(m##7);    \
    } while (0);

template <BiasMode bmode, typename Op>
struct OutputTransform6X3 {
    static void transform(const float* output_transform_buf, const float* bias,
                          float* output, float* transform_mid_buf,
                          size_t oh_start, size_t ow_start, size_t OH,
                          size_t OW, size_t oc_start, size_t oc_end,
                          size_t oc_index, size_t unit_idx,
                          size_t nr_units_in_tile, const DType& src_dtype,
                          const DType& dst_dtype) {
        constexpr size_t alpha = 6 + 3 - 1;
        Op op(src_dtype, dst_dtype);
        float* mid_buf1 = transform_mid_buf;

        //! AT * m * A
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
#define cb(i) Vector<float, 8> s##i, ret##i;
        UNROLL_CALL_NOWRAPPER(8, cb);
#undef cb

        OUTPUT_TRANSFORM(m, s);
        /**
         * Output transform: m * A
         *
         * 1    0    0      0       0         0
         * 1    1    1      1       1         1
         * 1   -1    1     -1       1        -1
         * 1    2    4      8      16        32
         * 1   -2    4     -8      16       -32
         * 1  0.5 0.25  0.125  0.0625   0.03125
         * 1 -0.5 0.25 -0.125  0.0625  -0.03125
         * 0  0.0    0      0       0         1
         */
#define cb(i)                                                                  \
    do {                                                                       \
        auto m1addm2 =                                                         \
                GET_VECTOR_LOW_ELEM(s, i, 1) + GET_VECTOR_LOW_ELEM(s, i, 2);   \
        auto m1subm2 =                                                         \
                GET_VECTOR_LOW_ELEM(s, i, 1) - GET_VECTOR_LOW_ELEM(s, i, 2);   \
        auto m3addm4 =                                                         \
                GET_VECTOR_LOW_ELEM(s, i, 3) + GET_VECTOR_HIGH_ELEM(s, i, 0);  \
        auto m3subm4 =                                                         \
                GET_VECTOR_LOW_ELEM(s, i, 3) - GET_VECTOR_HIGH_ELEM(s, i, 0);  \
        auto m5addm6 =                                                         \
                GET_VECTOR_HIGH_ELEM(s, i, 1) + GET_VECTOR_HIGH_ELEM(s, i, 2); \
        auto m5subm6 =                                                         \
                GET_VECTOR_HIGH_ELEM(s, i, 1) - GET_VECTOR_HIGH_ELEM(s, i, 2); \
        mid_buf1[0] =                                                          \
                GET_VECTOR_LOW_ELEM(s, i, 0) + m1addm2 + m3addm4 + m5addm6;    \
        mid_buf1[1] = m1subm2 + 2.f * m3subm4 + 0.5f * m5subm6;                \
        mid_buf1[2] = m1addm2 + 4.f * m3addm4 + 0.25f * m5addm6;               \
        mid_buf1[3] = m1subm2 + 8.f * m3subm4 + 0.125f * m5subm6;              \
        mid_buf1[4] = m1addm2 + 16.f * m3addm4 + 0.0625f * m5addm6;            \
        mid_buf1[5] = m1subm2 + 32.f * m3subm4 + 0.03125f * m5subm6 +          \
                      GET_VECTOR_HIGH_ELEM(s, i, 3);                           \
        mid_buf1 += 6;                                                         \
    } while (0);

        mid_buf1 = transform_mid_buf;
        UNROLL_CALL_NOWRAPPER(6, cb);
        mid_buf1 = transform_mid_buf;
#undef cb

        if (oh_start + 6 <= OH && ow_start + 6 <= OW) {
            float32x4_t bias0;
            float32x2_t bias1;
            if (bmode == BiasMode::BROADCAST_CHANNEL_BIAS) {
                bias0 = vdupq_n_f32(bias[oc]);
                bias1 = vdup_n_f32(bias[oc]);
            }
            rep(i, 6) {
                size_t oh = oh_start + i;
                float32x4_t item0 = vld1q_f32(mid_buf1);
                float32x2_t item1 = vld1_f32(mid_buf1 + 4);

                if (bmode == BiasMode::BROADCAST_CHANNEL_BIAS) {
                    item0 = vaddq_f32(item0, bias0);
                    item1 = vadd_f32(item1, bias1);
                } else if (bmode == BiasMode::BIAS) {
                    bias0 = vld1q_f32(bias + oc * OH * OW + oh * OW + ow_start);
                    bias1 = vld1_f32(bias + oc * OH * OW + oh * OW + ow_start +
                                     4);
                    item0 = vaddq_f32(item0, bias0);
                    item1 = vadd_f32(item1, bias1);
                }
                item0 = op(item0);
                item1 = vset_lane_f32(op(vget_lane_f32(item1, 0)), item1, 0);
                item1 = vset_lane_f32(op(vget_lane_f32(item1, 1)), item1, 1);
                vst1q_f32(output + oc * OH * OW + oh * OW + ow_start, item0);
                vst1_f32(output + oc * OH * OW + oh * OW + ow_start + 4, item1);

                mid_buf1 += 6;
            }
        } else {
            for (size_t oho = 0; oho < 6 && oh_start + oho < OH; ++oho) {
                for (size_t owo = 0; owo < 6 && ow_start + owo < OW; ++owo) {
                    size_t oh = oh_start + oho;
                    size_t ow = ow_start + owo;
                    float res = mid_buf1[oho * 6 + owo];
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

#undef GET_VECTOR_HIGH_ELEM
#undef GET_VECTOR_LOW_ELEM
#undef OUTPUT_TRANSFORM

}  // namespace

namespace megdnn {
namespace arm_common {
namespace winograd {

MEGDNN_REG_WINOGRAD_STRATEGY_IMPL(winograd_6x3_1x1_f)

void winograd_6x3_1x1_f::filter(const float* filter,
                                float* filter_transform_buf,
                                float* transform_mid_buf, size_t OC, size_t IC,
                                size_t oc_start, size_t oc_end) {
    FilterTransform6X3<param::MatrixMul::Format::DEFAULT>::transform(
            filter, filter_transform_buf, transform_mid_buf, OC, IC, oc_start,
            oc_end);
}

void winograd_6x3_1x1_f::input(const float* input, float* input_transform_buf,
                               float* transform_mid_buf, size_t IH, size_t IW,
                               size_t IC, size_t PH, size_t PW,
                               size_t unit_start_idx, size_t nr_units_in_tile) {
    constexpr int alpha = 3 + 6 - 1;

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
                InputTransform6X3::transform<true>(
                        input, input_transform_buf, transform_mid_buf, ih_start,
                        iw_start, ic, IH, IW, IC, unit_idx, nr_units_in_tile);

            } else {
                InputTransform6X3::transform<false>(
                        input, input_transform_buf, transform_mid_buf, ih_start,
                        iw_start, ic, IH, IW, IC, unit_idx, nr_units_in_tile);
            }
        }
    }
}

void winograd_6x3_1x1_f::output(const float* output_transform_buf,
                                const float* bias, float* output,
                                float* transform_mid_buf, BiasMode bmode,
                                NonlineMode nonline_mode, size_t OH, size_t OW,
                                size_t oc_start, size_t oc_end,
                                size_t unit_start_idx,
                                size_t nr_units_in_tile) {
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
                    megdnn_arm_common_winograd_fp32_F63, cb, float, float, bmode,
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
