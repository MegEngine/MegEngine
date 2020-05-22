/**
 * \file dnn/src/arm_common/conv_bias/fp32/strategy_5x4.cpp
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
MIDOUT_DECL(megdnn_arm_common_winograd_fp32_F54)

using namespace megdnn;
using namespace arm_common;
namespace {

struct FilterTransform5X4 {
#define FILTER_TRANSFORM(d, wd)                            \
    do {                                                   \
        wd##0 = d##0;                                      \
        auto tmp0 = d##0 * 0.7111111f + d##2 * 0.1777778f; \
        auto tmp1 = d##1 * 0.3555556f + d##3 * 0.0888889f; \
        wd##1 = tmp0 + tmp1;                               \
        wd##2 = tmp0 - tmp1;                               \
        tmp0 = (d##0 + d##2) * -0.2222222f;                \
        tmp1 = (d##1 + d##3) * -0.2222222f;                \
        wd##3 = tmp0 + tmp1;                               \
        wd##4 = tmp0 - tmp1;                               \
        tmp0 = d##0 * 0.0111111f + d##2 * 0.0444444f;      \
        tmp1 = d##1 * 0.0222222f + d##3 * 0.0888889f;      \
        wd##5 = tmp0 + tmp1;                               \
        wd##6 = tmp0 - tmp1;                               \
        wd##7 = d##3;                                      \
    } while (0)

    static void transform(const float* filter, float* filter_transform_buf,
                          float* transform_mid_buf, size_t OC, size_t IC,
                          size_t oc_start, size_t oc_end) {
        // Gg * GT
        // G
        // 1            0           0           0
        // 0.7111111    0.3555556   0.1777778   0.0888889
        // 0.7111111   -0.3555556   0.1777778  -0.0888889
        // -0.2222222  -0.2222222  -0.2222222  -0.2222222
        // -0.2222222   0.2222222  -0.2222222   0.2222222
        // 0.0111111    0.0222222   0.0444444   0.0888889
        // 0.0111111   -0.0222222   0.0444444  -0.0888889
        // 0            0           0           1

        constexpr size_t alpha = 4 + 5 - 1;
        for (size_t oc = oc_start; oc < oc_end; oc++) {
            rep(ic, IC) {
                const float* fptr = filter + (oc * IC + ic) * 4 * 4;

#define cb(i) Vector<float, 4> g##i = Vector<float, 4>::load(fptr + 4 * i);
                UNROLL_CALL_NOWRAPPER(4, cb);
#undef cb

#define cb(i) Vector<float, 4> wd##i;
                UNROLL_CALL_NOWRAPPER(8, cb);
#undef cb

#define cb(i) Vector<float, 8> wdt##i;
                UNROLL_CALL_NOWRAPPER(4, cb);
#undef cb

#define cb(i) Vector<float, 8> ret##i;
                UNROLL_CALL_NOWRAPPER(8, cb);
#undef cb

                FILTER_TRANSFORM(g, wd);
#if MEGDNN_AARCH64
                TRANSPOSE_8x4(wd, wdt);
                FILTER_TRANSFORM(wdt, ret);

#define cb(i) ret##i.save(transform_mid_buf + i * alpha);
                UNROLL_CALL_NOWRAPPER(8, cb);
#undef cb
                rep(i, alpha) rep(j, alpha) {
                    filter_transform_buf[(i * alpha + j) * OC * IC + ic * OC +
                                         oc] = transform_mid_buf[j * alpha + i];
                }
#else

#define cb(i)                                                            \
    do {                                                                 \
        mid_buf1[0] = GET_VECTOR_ELEM(wd, i, 0);                         \
        auto tmp0 = GET_VECTOR_ELEM(wd, i, 0) * 0.7111111f +             \
                    GET_VECTOR_ELEM(wd, i, 2) * 0.1777778f;              \
        auto tmp1 = GET_VECTOR_ELEM(wd, i, 1) * 0.3555556f +             \
                    GET_VECTOR_ELEM(wd, i, 3) * 0.0888889f;              \
        mid_buf1[1] = tmp0 + tmp1;                                       \
        mid_buf1[2] = tmp0 - tmp1;                                       \
        tmp0 = (GET_VECTOR_ELEM(wd, i, 0) + GET_VECTOR_ELEM(wd, i, 2)) * \
               -0.2222222f;                                              \
        tmp1 = (GET_VECTOR_ELEM(wd, i, 1) + GET_VECTOR_ELEM(wd, i, 3)) * \
               -0.2222222f;                                              \
        mid_buf1[3] = tmp0 + tmp1;                                       \
        mid_buf1[4] = tmp0 - tmp1;                                       \
        tmp0 = GET_VECTOR_ELEM(wd, i, 0) * 0.0111111f +                  \
               GET_VECTOR_ELEM(wd, i, 2) * 0.0444444f;                   \
        tmp1 = GET_VECTOR_ELEM(wd, i, 1) * 0.0222222f +                  \
               GET_VECTOR_ELEM(wd, i, 3) * 0.0888889f;                   \
        mid_buf1[5] = tmp0 + tmp1;                                       \
        mid_buf1[6] = tmp0 - tmp1;                                       \
        mid_buf1[7] = GET_VECTOR_ELEM(wd, i, 3);                         \
        mid_buf1 += 8;                                                   \
    } while (0);
#define GET_VECTOR_ELEM(s, i, idx) vgetq_lane_f32(CONCAT(s, i).value, idx)

                float* mid_buf1 = transform_mid_buf;
                UNROLL_CALL_NOWRAPPER(8, cb);
                mid_buf1 = transform_mid_buf;
#undef cb
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
#undef GET_VECTOR_ELEM

struct InputTransform5X4 {
#define INPUT_TRANSFORM(d, wd)                               \
    do {                                                     \
        wd##0 = (d##0 - d##6) + (d##4 - d##2) * 5.25f;       \
        auto tmp0 = d##2 * 4.0f - d##4 * 5.0f + d##6;        \
        auto tmp1 = d##1 * 2.0f - d##3 * 2.5f + d##5 * 0.5f; \
        wd##1 = tmp0 + tmp1;                                 \
        wd##2 = tmp0 - tmp1;                                 \
        tmp0 = d##2 - d##4 * 4.25f + d##6;                   \
        tmp1 = d##1 - d##3 * 4.25f + d##5;                   \
        wd##3 = tmp0 + tmp1;                                 \
        wd##4 = tmp0 - tmp1;                                 \
        tmp0 = d##2 * 0.25f - d##4 * 1.25f + d##6;           \
        tmp1 = d##1 * 0.5f - d##3 * 2.5f + d##5 * 2.0f;      \
        wd##5 = tmp0 + tmp1;                                 \
        wd##6 = tmp0 - tmp1;                                 \
        wd##7 = (d##7 - d##1) + (d##3 - d##5) * 5.25f;       \
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
        // BT
        // 1   0    -5.25   0      5.25   0     -1  0
        // 0   2    4      -2.5   -5      0.5    1  0
        // 0  -2    4       2.5   -5     -0.5    1  0
        // 0   1    1      -4.25  -4.25   1      1  0
        // 0  -1    1       4.25  -4.25  -1      1  0
        // 0   0.5  0.25   -2.5   -1.25   2      1  0
        // 0  -0.5  0.25    2.5   -1.25  -2      1  0
        // 0  -1    0       5.25   0     -5.25   0  1

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
        mid_buf1[1] = tmp0 + tmp1;                             \
        mid_buf1[2] = tmp0 - tmp1;                             \
        tmp0 = GET_VECTOR_LOW_ELEM(wd, i, 2) +                 \
               -4.25 * GET_VECTOR_HIGH_ELEM(wd, i, 0) +        \
               GET_VECTOR_HIGH_ELEM(wd, i, 2);                 \
        tmp1 = GET_VECTOR_LOW_ELEM(wd, i, 1) +                 \
               GET_VECTOR_LOW_ELEM(wd, i, 3) * -4.25 +         \
               GET_VECTOR_HIGH_ELEM(wd, i, 1);                 \
        mid_buf1[3] = tmp0 + tmp1;                             \
        mid_buf1[4] = tmp0 - tmp1;                             \
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

#define OUTPUT_TRANSFORM(m, s)                                               \
    do {                                                                     \
        auto m1addm2 = m##1 + m##2;                                          \
        auto m1subm2 = m##1 - m##2;                                          \
        auto m3addm4 = m##3 + m##4;                                          \
        auto m3subm4 = m##3 - m##4;                                          \
        auto m5addm6 = (m##5 + m##6);                                        \
        auto m5subm6 = (m##5 - m##6);                                        \
        s##0 = m##0;                                                         \
        CONCAT(s, 0).add(m1addm2).add(m3addm4).add(m5addm6);                 \
        CONCAT(s, 1) = m3subm4;                                              \
        CONCAT(s, 1).mla(m1subm2, 0.5f).mla(m5subm6, 2.0f);                  \
        CONCAT(s, 2) = m3addm4;                                              \
        CONCAT(s, 2).mla(m1addm2, 0.25f).mla(m5addm6, 4.0f);                 \
        CONCAT(s, 3) = m3subm4;                                              \
        CONCAT(s, 3).mla(m1subm2, 0.125f).mla(m5subm6, 8.0f);                \
        CONCAT(s, 4) = m##7;                                                 \
        CONCAT(s, 4).mla(m1addm2, 0.0625f).add(m3addm4).mla(m5addm6, 16.0f); \
    } while (0)

template <BiasMode bmode, typename Op>
struct OutputTransform5X4 {
    static void transform(const float* output_transform_buf, const float* bias,
                          float* output, float* transform_mid_buf,
                          size_t oh_start, size_t ow_start, size_t OH,
                          size_t OW, size_t oc_start, size_t oc_end,
                          size_t oc_index, size_t unit_idx,
                          size_t nr_units_in_tile, const DType& src_dtype,
                          const DType& dst_dtype) {
        Op op(src_dtype, dst_dtype);
        //! AT * m * A
        // AT
        // 1  1       1       1  1  1   1   0
        // 0  0.5    -0.5     1 -1  2  -2   0
        // 0  0.25    0.25    1  1  4   4   0
        // 0  0.125  -0.125   1 -1  8  -8   0
        // 0  0.0625  0.0625  1  1  16  16  1
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
#define cb(i) Vector<float, 8> s##i, ret##i;
        UNROLL_CALL_NOWRAPPER(8, cb);
#undef cb

        OUTPUT_TRANSFORM(m, s);
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
        mid_buf1[1] = 0.5f * m1subm2 + m3subm4 + 2.0f * m5subm6;               \
        mid_buf1[2] = 0.25f * m1addm2 + m3addm4 + 4.0f * m5addm6;              \
        mid_buf1[3] = 0.125f * m1subm2 + m3subm4 + 8.0f * m5subm6;             \
        mid_buf1[4] = 0.0625f * m1addm2 + m3addm4 + 16.0f * m5addm6 +          \
                      GET_VECTOR_HIGH_ELEM(s, i, 3);                           \
        mid_buf1 += 5;                                                         \
    } while (0);

        mid_buf1 = transform_mid_buf;
        UNROLL_CALL_NOWRAPPER(5, cb);
        mid_buf1 = transform_mid_buf;
#undef cb

        if (oh_start + 5 <= OH && ow_start + 5 <= OW) {
            float32x4_t bias0;
            float32_t bias1;
            if (bmode == BiasMode::BROADCAST_CHANNEL_BIAS) {
                bias0 = vdupq_n_f32(bias[oc]);
                bias1 = bias[oc];
            }
            rep(i, 5) {
                size_t oh = oh_start + i;
                float32x4_t item0 = vld1q_f32(mid_buf1);
                float32_t item1 = mid_buf1[4];

                if (bmode == BiasMode::BROADCAST_CHANNEL_BIAS) {
                    item0 = vaddq_f32(item0, bias0);
                    item1 = item1 + bias1;
                } else if (bmode == BiasMode::BIAS) {
                    bias0 = vld1q_f32(bias + oc * OH * OW + oh * OW + ow_start);
                    bias1 = bias[oc * OH * OW + oh * OW + ow_start + 4];
                    item0 = vaddq_f32(item0, bias0);
                    item1 = item1 + bias1;
                }
                item0 = op(item0);
                item1 = op(item1);
                vst1q_f32(output + oc * OH * OW + oh * OW + ow_start, item0);
                output[oc * OH * OW + oh * OW + ow_start + 4] = item1;

                mid_buf1 += 5;
            }
        } else {
            for (size_t oho = 0; oho < 5 && oh_start + oho < OH; ++oho) {
                for (size_t owo = 0; owo < 5 && ow_start + owo < OW; ++owo) {
                    size_t oh = oh_start + oho;
                    size_t ow = ow_start + owo;
                    float res = mid_buf1[oho * 5 + owo];
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

MEGDNN_REG_WINOGRAD_STRATEGY_IMPL(winograd_5x4_1x1_f)

void winograd_5x4_1x1_f::filter(const float* filter,
                                float* filter_transform_buf,
                                float* transform_mid_buf, size_t OC, size_t IC,
                                size_t oc_start, size_t oc_end) {
    FilterTransform5X4::transform(filter, filter_transform_buf,
                                  transform_mid_buf, OC, IC, oc_start, oc_end);
}

void winograd_5x4_1x1_f::input(const float* input, float* input_transform_buf,
                               float* transform_mid_buf, size_t IH, size_t IW,
                               size_t IC, size_t PH, size_t PW,
                               size_t unit_start_idx, size_t nr_units_in_tile) {
    constexpr int alpha = 5 + 4 - 1;

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
                InputTransform5X4::transform<true>(
                        input, input_transform_buf, transform_mid_buf, ih_start,
                        iw_start, ic, IH, IW, IC, unit_idx, nr_units_in_tile);

            } else {
                InputTransform5X4::transform<false>(
                        input, input_transform_buf, transform_mid_buf, ih_start,
                        iw_start, ic, IH, IW, IC, unit_idx, nr_units_in_tile);
            }
        }
    }
}

void winograd_5x4_1x1_f::output(const float* output_transform_buf,
                                const float* bias, float* output,
                                float* transform_mid_buf, BiasMode bmode,
                                NonlineMode nonline_mode, size_t OH, size_t OW,
                                size_t oc_start, size_t oc_end,
                                size_t unit_start_idx,
                                size_t nr_units_in_tile) {
#define cb(_bmode, _nonline_op, ...) \
    OutputTransform5X4<_bmode MEGDNN_COMMA _nonline_op>::transform(__VA_ARGS__);

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
                    megdnn_arm_common_winograd_fp32_F54, cb, float, float, bmode,
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
