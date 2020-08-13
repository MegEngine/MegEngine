/**
 * \file
 * dnn/src/arm_common/conv_bias/int8x8x16/direct_kernels/int8_direct_nchw44_s1_armv7.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#include "src/common/utils.h"
#if MEGDNN_ARMV7
#include "src/arm_common/conv_bias/int8x8x16/direct_8x8x16_nchw44_kern.h"
#include "src/arm_common/simd_macro/marm_neon.h"
#include "src/fallback/conv_bias/common.h"

namespace megdnn {
namespace arm_common {
namespace {

#define INIT_SUM()                        \
    int16x4_t init_sum;                   \
    if (bias_mode == BiasMode::NO_BIAS) { \
        init_sum = vdup_n_s16(0);         \
    } else {                              \
        init_sum = vld1_s16(bias_ptr);    \
    }

#define STORE_1_LINE_RESULT()                                        \
    switch (remain_w) {                                              \
        case 8:                                                      \
            vst1q_s16(dst_ptr, vcombine_s16(c[0][0], c[0][1]));      \
            vst1q_s16(dst_ptr + 8, vcombine_s16(c[0][2], c[0][3]));  \
            vst1q_s16(dst_ptr + 16, vcombine_s16(c[0][4], c[0][5])); \
            vst1q_s16(dst_ptr + 24, vcombine_s16(c[0][6], c[0][7])); \
            break;                                                   \
        case 1:                                                      \
            vst1_s16(dst_ptr, c[0][0]);                              \
            break;                                                   \
        case 2:                                                      \
            vst1q_s16(dst_ptr, vcombine_s16(c[0][0], c[0][1]));      \
            break;                                                   \
        case 3:                                                      \
            vst1q_s16(dst_ptr, vcombine_s16(c[0][0], c[0][1]));      \
            vst1_s16(dst_ptr + 8, c[0][2]);                          \
            break;                                                   \
        case 4:                                                      \
            vst1q_s16(dst_ptr, vcombine_s16(c[0][0], c[0][1]));      \
            vst1q_s16(dst_ptr + 8, vcombine_s16(c[0][2], c[0][3]));  \
            break;                                                   \
        case 5:                                                      \
            vst1q_s16(dst_ptr, vcombine_s16(c[0][0], c[0][1]));      \
            vst1q_s16(dst_ptr + 8, vcombine_s16(c[0][2], c[0][3]));  \
            vst1_s16(dst_ptr + 16, c[0][4]);                         \
            break;                                                   \
        case 6:                                                      \
            vst1q_s16(dst_ptr, vcombine_s16(c[0][0], c[0][1]));      \
            vst1q_s16(dst_ptr + 8, vcombine_s16(c[0][2], c[0][3]));  \
            vst1q_s16(dst_ptr + 16, vcombine_s16(c[0][4], c[0][5])); \
            break;                                                   \
        case 7:                                                      \
            vst1q_s16(dst_ptr, vcombine_s16(c[0][0], c[0][1]));      \
            vst1q_s16(dst_ptr + 8, vcombine_s16(c[0][2], c[0][3]));  \
            vst1q_s16(dst_ptr + 16, vcombine_s16(c[0][4], c[0][5])); \
            vst1_s16(dst_ptr + 24, c[0][6]);                         \
            break;                                                   \
        default:                                                     \
            megdnn_assert(0, "oc 1 error remainw");                  \
    };

#define STORE_2_LINE_RESULT()                                               \
    switch (remain_w) {                                                     \
        case 8:                                                             \
            vst1q_s16(dst_ptr, vcombine_s16(c[0][0], c[0][1]));             \
            vst1q_s16(dst_ptr + 8, vcombine_s16(c[0][2], c[0][3]));         \
            vst1q_s16(dst_ptr + 16, vcombine_s16(c[0][4], c[0][5]));        \
            vst1q_s16(dst_ptr + 24, vcombine_s16(c[0][6], c[0][7]));        \
            vst1q_s16(dst_ptr + ld_dst_oc, vcombine_s16(c[1][0], c[1][1])); \
            vst1q_s16(dst_ptr + ld_dst_oc + 8,                              \
                      vcombine_s16(c[1][2], c[1][3]));                      \
            vst1q_s16(dst_ptr + ld_dst_oc + 16,                             \
                      vcombine_s16(c[1][4], c[1][5]));                      \
            vst1q_s16(dst_ptr + ld_dst_oc + 24,                             \
                      vcombine_s16(c[1][6], c[1][7]));                      \
            break;                                                          \
        case 1:                                                             \
            vst1_s16(dst_ptr, c[0][0]);                                     \
            vst1_s16(dst_ptr + ld_dst_oc, c[1][0]);                         \
            break;                                                          \
        case 2:                                                             \
            vst1q_s16(dst_ptr, vcombine_s16(c[0][0], c[0][1]));             \
            vst1q_s16(dst_ptr + ld_dst_oc, vcombine_s16(c[1][0], c[1][1])); \
            break;                                                          \
        case 3:                                                             \
            vst1q_s16(dst_ptr, vcombine_s16(c[0][0], c[0][1]));             \
            vst1_s16(dst_ptr + 8, c[0][2]);                                 \
            vst1q_s16(dst_ptr + ld_dst_oc, vcombine_s16(c[1][0], c[1][1])); \
            vst1_s16(dst_ptr + ld_dst_oc + 8, c[1][2]);                     \
            break;                                                          \
        case 4:                                                             \
            vst1q_s16(dst_ptr, vcombine_s16(c[0][0], c[0][1]));             \
            vst1q_s16(dst_ptr + 8, vcombine_s16(c[0][2], c[0][3]));         \
            vst1q_s16(dst_ptr + ld_dst_oc, vcombine_s16(c[1][0], c[1][1])); \
            vst1q_s16(dst_ptr + ld_dst_oc + 8,                              \
                      vcombine_s16(c[1][2], c[1][3]));                      \
            break;                                                          \
        case 5:                                                             \
            vst1q_s16(dst_ptr, vcombine_s16(c[0][0], c[0][1]));             \
            vst1q_s16(dst_ptr + 8, vcombine_s16(c[0][2], c[0][3]));         \
            vst1_s16(dst_ptr + 16, c[0][4]);                                \
            vst1q_s16(dst_ptr + ld_dst_oc, vcombine_s16(c[1][0], c[1][1])); \
            vst1q_s16(dst_ptr + ld_dst_oc + 8,                              \
                      vcombine_s16(c[1][2], c[1][3]));                      \
            vst1_s16(dst_ptr + ld_dst_oc + 16, c[1][4]);                    \
            break;                                                          \
        case 6:                                                             \
            vst1q_s16(dst_ptr, vcombine_s16(c[0][0], c[0][1]));             \
            vst1q_s16(dst_ptr + 8, vcombine_s16(c[0][2], c[0][3]));         \
            vst1q_s16(dst_ptr + 16, vcombine_s16(c[0][4], c[0][5]));        \
            vst1q_s16(dst_ptr + ld_dst_oc, vcombine_s16(c[1][0], c[1][1])); \
            vst1q_s16(dst_ptr + ld_dst_oc + 8,                              \
                      vcombine_s16(c[1][2], c[1][3]));                      \
            vst1q_s16(dst_ptr + ld_dst_oc + 16,                             \
                      vcombine_s16(c[1][4], c[1][5]));                      \
            break;                                                          \
        case 7:                                                             \
            vst1q_s16(dst_ptr, vcombine_s16(c[0][0], c[0][1]));             \
            vst1q_s16(dst_ptr + 8, vcombine_s16(c[0][2], c[0][3]));         \
            vst1q_s16(dst_ptr + 16, vcombine_s16(c[0][4], c[0][5]));        \
            vst1_s16(dst_ptr + 24, c[0][6]);                                \
            vst1q_s16(dst_ptr + ld_dst_oc, vcombine_s16(c[1][0], c[1][1])); \
            vst1q_s16(dst_ptr + ld_dst_oc + 8,                              \
                      vcombine_s16(c[1][2], c[1][3]));                      \
            vst1q_s16(dst_ptr + ld_dst_oc + 16,                             \
                      vcombine_s16(c[1][4], c[1][5]));                      \
            vst1_s16(dst_ptr + ld_dst_oc + 24, c[1][6]);                    \
            break;                                                          \
        default:                                                            \
            megdnn_assert(0, "oc 2 error remainw");                         \
            break;                                                          \
    }

template <BiasMode bias_mode, int remain_w, int filter_size>
static void ker_neon_dirctconv_2x2s1_oc8_ow8(const int8_t* src_ptr,
                                             const int8_t* weight_ptr,
                                             const int16_t* bias_ptr,
                                             int16_t* dst_ptr, int ic, int ih,
                                             int iw, int ld_dst_oc) {
    constexpr int fh = filter_size;
    constexpr int fw = filter_size;
    constexpr int ic_step = 4;
    constexpr int oc_step = 4;
    constexpr int loop_ic_step = 4;
    constexpr int ld_weight_ic4 = 16;
    constexpr int src_expand_size = 4;
    const int ic_stride = ih * iw * src_expand_size;
    const int ld_weight_oc4 = oc_step * fh * fw * ic;

    int16x4_t c[2][8];
    int8x16_t weight[2][2];
    int8x16_t src[4];

    INIT_SUM();
#define cb(_i)           \
    c[0][_i] = init_sum; \
    c[1][_i] = init_sum;

    UNROLL_CALL_RAW(8, cb);
#undef cb

    for (int ic_idx = 0; ic_idx < ic; ic_idx += loop_ic_step) {
        const int8_t* src_row0 = src_ptr + ic_idx * ic_stride +
                                 0 * iw * ic_step * src_expand_size;
        const int8_t* src_row1 = src_ptr + ic_idx * ic_stride +
                                 1 * iw * ic_step * src_expand_size;

        src[0] = vld1q_s8(src_row0);
        src[1] = vld1q_s8(src_row0 + 16);

        weight[0][0] = vld1q_s8(weight_ptr);
        weight[0][1] = vld1q_s8(weight_ptr + 16);
        weight[1][0] = vld1q_s8(weight_ptr + ld_weight_oc4);
        weight[1][1] = vld1q_s8(weight_ptr + ld_weight_oc4 + 16);

#define CALC_ONE_RESULT(_src0, _src1, _w0, _w1, _c)                           \
    do {                                                                      \
        tmp0 = vmull_s8(vget_low_s8(_src0), vget_low_s8(_w0));                \
        tmp0 = vmlal_s8(tmp0, vget_high_s8(_src0), vget_high_s8(_w0));        \
        tmp0 = vmlal_s8(tmp0, vget_low_s8(_src1), vget_low_s8(_w1));          \
        tmp0 = vmlal_s8(tmp0, vget_high_s8(_src1), vget_high_s8(_w1));        \
        _c = vadd_s16(_c, vadd_s16(vget_low_s16(tmp0), vget_high_s16(tmp0))); \
    } while (0);

        int16x8_t tmp0;
        src[2] = vld1q_s8(src_row0 + 2 * 16);
        src[3] = vld1q_s8(src_row0 + 3 * 16);
        CALC_ONE_RESULT(src[0], src[1], weight[0][0], weight[0][1], c[0][0]);
        CALC_ONE_RESULT(src[0], src[1], weight[1][0], weight[1][1], c[1][0]);

        CALC_ONE_RESULT(src[1], src[2], weight[0][0], weight[0][1], c[0][1]);
        CALC_ONE_RESULT(src[1], src[2], weight[1][0], weight[1][1], c[1][1]);

        src[0] = vld1q_s8(src_row0 + 4 * 16);

        CALC_ONE_RESULT(src[2], src[3], weight[0][0], weight[0][1], c[0][2]);
        CALC_ONE_RESULT(src[2], src[3], weight[1][0], weight[1][1], c[1][2]);

        src[1] = vld1q_s8(src_row0 + 5 * 16);
        CALC_ONE_RESULT(src[3], src[0], weight[0][0], weight[0][1], c[0][3]);
        CALC_ONE_RESULT(src[3], src[0], weight[1][0], weight[1][1], c[1][3]);

        src[2] = vld1q_s8(src_row0 + 6 * 16);

        CALC_ONE_RESULT(src[0], src[1], weight[0][0], weight[0][1], c[0][4]);
        CALC_ONE_RESULT(src[0], src[1], weight[1][0], weight[1][1], c[1][4]);

        src[3] = vld1q_s8(src_row0 + 7 * 16);
        CALC_ONE_RESULT(src[1], src[2], weight[0][0], weight[0][1], c[0][5]);
        CALC_ONE_RESULT(src[1], src[2], weight[1][0], weight[1][1], c[1][5]);

        src[0] = vld1q_s8(src_row0 + 8 * 16);

        CALC_ONE_RESULT(src[2], src[3], weight[0][0], weight[0][1], c[0][6]);
        CALC_ONE_RESULT(src[2], src[3], weight[1][0], weight[1][1], c[1][6]);

        src[1] = vld1q_s8(src_row1 + 0 * 16);
        src[2] = vld1q_s8(src_row1 + 1 * 16);

        CALC_ONE_RESULT(src[3], src[0], weight[0][0], weight[0][1], c[0][7]);
        CALC_ONE_RESULT(src[3], src[0], weight[1][0], weight[1][1], c[1][7]);

        weight[0][0] = vld1q_s8(weight_ptr + 32);
        weight[0][1] = vld1q_s8(weight_ptr + 48);
        src[3] = vld1q_s8(src_row1 + 2 * 16);

        CALC_ONE_RESULT(src[1], src[2], weight[0][0], weight[0][1], c[0][0]);
        weight[1][0] = vld1q_s8(weight_ptr + ld_weight_oc4 + 32);
        weight[1][1] = vld1q_s8(weight_ptr + ld_weight_oc4 + 48);
        src[0] = vld1q_s8(src_row1 + 3 * 16);
        CALC_ONE_RESULT(src[1], src[2], weight[1][0], weight[1][1], c[1][0]);

        CALC_ONE_RESULT(src[2], src[3], weight[0][0], weight[0][1], c[0][1]);
        CALC_ONE_RESULT(src[2], src[3], weight[1][0], weight[1][1], c[1][1]);

        src[1] = vld1q_s8(src_row1 + 4 * 16);

        CALC_ONE_RESULT(src[3], src[0], weight[0][0], weight[0][1], c[0][2]);
        CALC_ONE_RESULT(src[3], src[0], weight[1][0], weight[1][1], c[1][2]);

        src[2] = vld1q_s8(src_row1 + 5 * 16);

        CALC_ONE_RESULT(src[0], src[1], weight[0][0], weight[0][1], c[0][3]);
        CALC_ONE_RESULT(src[0], src[1], weight[1][0], weight[1][1], c[1][3]);

        src[3] = vld1q_s8(src_row1 + 6 * 16);

        CALC_ONE_RESULT(src[1], src[2], weight[0][0], weight[0][1], c[0][4]);
        CALC_ONE_RESULT(src[1], src[2], weight[1][0], weight[1][1], c[1][4]);

        src[0] = vld1q_s8(src_row1 + 7 * 16);

        CALC_ONE_RESULT(src[2], src[3], weight[0][0], weight[0][1], c[0][5]);
        CALC_ONE_RESULT(src[2], src[3], weight[1][0], weight[1][1], c[1][5]);

        src[1] = vld1q_s8(src_row1 + 8 * 16);
        CALC_ONE_RESULT(src[3], src[0], weight[0][0], weight[0][1], c[0][6]);
        CALC_ONE_RESULT(src[3], src[0], weight[1][0], weight[1][1], c[1][6]);

        CALC_ONE_RESULT(src[0], src[1], weight[0][0], weight[0][1], c[0][7]);
        CALC_ONE_RESULT(src[0], src[1], weight[1][0], weight[1][1], c[1][7]);

        weight_ptr += fh * fw * ld_weight_ic4;
    }
    STORE_2_LINE_RESULT();
}

template <BiasMode bias_mode, int remain_w, int filter_size>
static void ker_neon_dirctconv_2x2s1_oc4_ow8(const int8_t* src_ptr,
                                             const int8_t* weight_ptr,
                                             const int16_t* bias_ptr,
                                             int16_t* dst_ptr, int ic, int ih,
                                             int iw, int /*ld_dst_oc*/) {
    constexpr int fh = filter_size;
    constexpr int fw = filter_size;
    constexpr int ic_step = 4;
    constexpr int loop_ic_step = 4;
    constexpr int ld_weight_ic4 = 16;
    constexpr int src_expand_size = 4;

    const int ic_stride = ih * iw * src_expand_size;

    int16x4_t c[1][8];
    int8x16_t weight[1][2];
    int8x16_t src[4];

    INIT_SUM();
#define cb(_i) c[0][_i] = init_sum;

    UNROLL_CALL_RAW(8, cb);
#undef cb

    for (int ic_idx = 0; ic_idx < ic; ic_idx += loop_ic_step) {
        for (int fh_idx = 0; fh_idx < fh; ++fh_idx) {
            const int8_t* src_ic_0_3 = src_ptr + ic_idx * ic_stride +
                                       fh_idx * iw * ic_step * src_expand_size;
            src[0] = vld1q_s8(src_ic_0_3);
            src[1] = vld1q_s8(src_ic_0_3 + 16);
            const int8_t* read_weight_ptr =
                    weight_ptr + fh_idx * fw * ld_weight_ic4;

            weight[0][0] = vld1q_s8(read_weight_ptr);
            weight[0][1] = vld1q_s8(read_weight_ptr + 16);
            int16x8_t tmp0;
            src[2] = vld1q_s8(src_ic_0_3 + 2 * 16);
            src[3] = vld1q_s8(src_ic_0_3 + 3 * 16);
            CALC_ONE_RESULT(src[0], src[1], weight[0][0], weight[0][1],
                            c[0][0]);

            src[0] = vld1q_s8(src_ic_0_3 + 4 * 16);
            CALC_ONE_RESULT(src[1], src[2], weight[0][0], weight[0][1],
                            c[0][1]);

            src[1] = vld1q_s8(src_ic_0_3 + 5 * 16);
            CALC_ONE_RESULT(src[2], src[3], weight[0][0], weight[0][1],
                            c[0][2]);

            src[2] = vld1q_s8(src_ic_0_3 + 6 * 16);
            CALC_ONE_RESULT(src[3], src[0], weight[0][0], weight[0][1],
                            c[0][3]);

            src[3] = vld1q_s8(src_ic_0_3 + 7 * 16);
            CALC_ONE_RESULT(src[0], src[1], weight[0][0], weight[0][1],
                            c[0][4]);

            src[0] = vld1q_s8(src_ic_0_3 + 8 * 16);
            CALC_ONE_RESULT(src[1], src[2], weight[0][0], weight[0][1],
                            c[0][5]);

            CALC_ONE_RESULT(src[2], src[3], weight[0][0], weight[0][1],
                            c[0][6]);

            CALC_ONE_RESULT(src[3], src[0], weight[0][0], weight[0][1],
                            c[0][7]);
        }
        weight_ptr += fh * fw * ld_weight_ic4;
    }

    STORE_1_LINE_RESULT();
}
#undef CALC_ONE_RESULT

template <BiasMode bias_mode, int remain_w, int filter_size>
struct KerNeonDirectStride1Int8 {
    static void impl(const int8_t* src_ptr, const int8_t* weight_ptr,
                     const int16_t* bias_ptr, int16_t* dst_ptr, int ic, int ih,
                     int iw, int ld_dst_oc);
};

template <BiasMode bias_mode, int remain_w>
struct KerNeonDirectStride1Int8<bias_mode, remain_w, 3> {
    static void impl(const int8_t* src_ptr, const int8_t* weight_ptr,
                     const int16_t* bias_ptr, int16_t* dst_ptr, int ic, int ih,
                     int iw, int /*ld_dst_oc*/) {
        constexpr int filter_size = 3;
        constexpr int fh = filter_size;
        constexpr int fw = filter_size;
        constexpr int ic_step = 4;
        constexpr int loop_ic_step = 4;
        constexpr int ld_weight_ic4 = 16;
        constexpr int src_expand_size = 4;

        const int ic_stride = ih * iw * src_expand_size;

        int16x4_t c[1][8];
        int8x16_t weight[3];
        int8x16_t src[5];

        INIT_SUM();
#define cb(_i) c[0][_i] = init_sum;

        UNROLL_CALL_RAW(8, cb);
#undef cb

        for (int ic_idx = 0; ic_idx < ic; ic_idx += loop_ic_step) {
            for (int fh_idx = 0; fh_idx < fh; ++fh_idx) {
                const int8_t* src_ic_0_3 =
                        src_ptr + ic_idx * ic_stride +
                        fh_idx * iw * ic_step * src_expand_size;

                src[0] = vld1q_s8(src_ic_0_3 + 0 * 16);
                src[1] = vld1q_s8(src_ic_0_3 + 1 * 16);
                src[2] = vld1q_s8(src_ic_0_3 + 2 * 16);

                const int8_t* read_weight_ptr =
                        weight_ptr + fh_idx * fw * ld_weight_ic4;

                weight[0] = vld1q_s8(read_weight_ptr);
                weight[1] = vld1q_s8(read_weight_ptr + 16);
                weight[2] = vld1q_s8(read_weight_ptr + 2 * 16);

                src[3] = vld1q_s8(src_ic_0_3 + 3 * 16);

#define CALC_ONE_RESULT(_src0, _src1, _src2, _w0, _w1, _w2, _c)               \
    do {                                                                      \
        tmp0 = vmull_s8(vget_low_s8(_src0), vget_low_s8(_w0));                \
        tmp0 = vmlal_s8(tmp0, vget_high_s8(_src0), vget_high_s8(_w0));        \
        tmp0 = vmlal_s8(tmp0, vget_low_s8(_src1), vget_low_s8(_w1));          \
        tmp0 = vmlal_s8(tmp0, vget_high_s8(_src1), vget_high_s8(_w1));        \
        tmp0 = vmlal_s8(tmp0, vget_low_s8(_src2), vget_low_s8(_w2));          \
        tmp0 = vmlal_s8(tmp0, vget_high_s8(_src2), vget_high_s8(_w2));        \
        _c = vadd_s16(_c, vadd_s16(vget_low_s16(tmp0), vget_high_s16(tmp0))); \
    } while (0);

                int16x8_t tmp0;

                CALC_ONE_RESULT(src[0], src[1], src[2], weight[0], weight[1],
                                weight[2], c[0][0]);
                src[4] = vld1q_s8(src_ic_0_3 + 4 * 16);
                CALC_ONE_RESULT(src[1], src[2], src[3], weight[0], weight[1],
                                weight[2], c[0][1]);
                src[0] = vld1q_s8(src_ic_0_3 + 5 * 16);
                CALC_ONE_RESULT(src[2], src[3], src[4], weight[0], weight[1],
                                weight[2], c[0][2]);
                src[1] = vld1q_s8(src_ic_0_3 + 6 * 16);
                CALC_ONE_RESULT(src[3], src[4], src[0], weight[0], weight[1],
                                weight[2], c[0][3]);
                src[2] = vld1q_s8(src_ic_0_3 + 7 * 16);
                CALC_ONE_RESULT(src[4], src[0], src[1], weight[0], weight[1],
                                weight[2], c[0][4]);
                src[3] = vld1q_s8(src_ic_0_3 + 8 * 16);
                CALC_ONE_RESULT(src[0], src[1], src[2], weight[0], weight[1],
                                weight[2], c[0][5]);
                src[4] = vld1q_s8(src_ic_0_3 + 9 * 16);
                CALC_ONE_RESULT(src[1], src[2], src[3], weight[0], weight[1],
                                weight[2], c[0][6]);
                CALC_ONE_RESULT(src[2], src[3], src[4], weight[0], weight[1],
                                weight[2], c[0][7]);
            }
            weight_ptr += fh * fw * ld_weight_ic4;
        }
        STORE_1_LINE_RESULT();
    }
};

#undef CALC_ONE_RESULT
template <BiasMode bias_mode, int remain_w>
struct KerNeonDirectStride1Int8<bias_mode, remain_w, 5> {
    static void impl(const int8_t* src_ptr, const int8_t* weight_ptr,
                     const int16_t* bias_ptr, int16_t* dst_ptr, int ic, int ih,
                     int iw, int /*ld_dst_oc*/) {
        constexpr int filter_size = 5;
        constexpr int fh = filter_size;
        constexpr int fw = filter_size;
        constexpr int ic_step = 4;
        constexpr int loop_ic_step = 4;
        constexpr int ld_weight_ic4 = 16;
        constexpr int src_expand_size = 4;

        const int ic_stride = ih * iw * src_expand_size;
        int16x4_t c[1][8];
        int8x16_t weight[5];
        int8x16_t src[8 + 2];

        INIT_SUM();
#define cb(_i) c[0][_i] = init_sum;

        UNROLL_CALL_RAW(8, cb);
#undef cb

        for (int ic_idx = 0; ic_idx < ic; ic_idx += loop_ic_step) {
            for (int fh_idx = 0; fh_idx < fh; ++fh_idx) {
                const int8_t* src_ic_0_3 =
                        src_ptr + ic_idx * ic_stride +
                        fh_idx * iw * ic_step * src_expand_size;

                src[0] = vld1q_s8(src_ic_0_3 + 0 * 16);
                src[1] = vld1q_s8(src_ic_0_3 + 1 * 16);
                src[2] = vld1q_s8(src_ic_0_3 + 2 * 16);
                src[3] = vld1q_s8(src_ic_0_3 + 3 * 16);
                src[4] = vld1q_s8(src_ic_0_3 + 4 * 16);
                src[5] = vld1q_s8(src_ic_0_3 + 5 * 16);
                src[6] = vld1q_s8(src_ic_0_3 + 6 * 16);
                src[7] = vld1q_s8(src_ic_0_3 + 7 * 16);
                src[8] = vld1q_s8(src_ic_0_3 + 8 * 16);
                src[9] = vld1q_s8(src_ic_0_3 + 9 * 16);
                const int8_t* read_weight_ptr =
                        weight_ptr + fh_idx * fw * ld_weight_ic4;

                weight[0] = vld1q_s8(read_weight_ptr);
                weight[1] = vld1q_s8(read_weight_ptr + 16);
                weight[2] = vld1q_s8(read_weight_ptr + 2 * 16);
                weight[3] = vld1q_s8(read_weight_ptr + 3 * 16);
                weight[4] = vld1q_s8(read_weight_ptr + 4 * 16);
#define CALC_ONE_RESULT(_src0, _src1, _src2, _src3, _src4, _w0, _w1, _w2, _w3, \
                        _w4, _c)                                               \
    do {                                                                       \
        int16x8_t tmp0 = vmull_s8(vget_low_s8(_src0), vget_low_s8(_w0));       \
        int16x8_t tmp1 = vmull_s8(vget_high_s8(_src0), vget_high_s8(_w0));     \
        tmp0 = vmlal_s8(tmp0, vget_low_s8(_src1), vget_low_s8(_w1));           \
        tmp1 = vmlal_s8(tmp1, vget_high_s8(_src1), vget_high_s8(_w1));         \
        tmp0 = vmlal_s8(tmp0, vget_low_s8(_src2), vget_low_s8(_w2));           \
        tmp1 = vmlal_s8(tmp1, vget_high_s8(_src2), vget_high_s8(_w2));         \
        tmp0 = vmlal_s8(tmp0, vget_low_s8(_src3), vget_low_s8(_w3));           \
        tmp1 = vmlal_s8(tmp1, vget_high_s8(_src3), vget_high_s8(_w3));         \
        tmp0 = vmlal_s8(tmp0, vget_low_s8(_src4), vget_low_s8(_w4));           \
        tmp1 = vmlal_s8(tmp1, vget_high_s8(_src4), vget_high_s8(_w4));         \
        tmp0 = vaddq_s16(tmp0, tmp1);                                          \
        _c = vadd_s16(_c, vadd_s16(vget_low_s16(tmp0), vget_high_s16(tmp0)));  \
    } while (0);

                CALC_ONE_RESULT(src[0], src[1], src[2], src[3], src[4],
                                weight[0], weight[1], weight[2], weight[3],
                                weight[4], c[0][0]);
                CALC_ONE_RESULT(src[1], src[2], src[3], src[4], src[5],
                                weight[0], weight[1], weight[2], weight[3],
                                weight[4], c[0][1]);
                CALC_ONE_RESULT(src[2], src[3], src[4], src[5], src[6],
                                weight[0], weight[1], weight[2], weight[3],
                                weight[4], c[0][2]);
                CALC_ONE_RESULT(src[3], src[4], src[5], src[6], src[7],
                                weight[0], weight[1], weight[2], weight[3],
                                weight[4], c[0][3]);
                CALC_ONE_RESULT(src[4], src[5], src[6], src[7], src[8],
                                weight[0], weight[1], weight[2], weight[3],
                                weight[4], c[0][4]);
                CALC_ONE_RESULT(src[5], src[6], src[7], src[8], src[9],
                                weight[0], weight[1], weight[2], weight[3],
                                weight[4], c[0][5]);
                src[0] = vld1q_s8(src_ic_0_3 + 10 * 16);
                src[1] = vld1q_s8(src_ic_0_3 + 11 * 16);
                CALC_ONE_RESULT(src[6], src[7], src[8], src[9], src[0],
                                weight[0], weight[1], weight[2], weight[3],
                                weight[4], c[0][6]);
                CALC_ONE_RESULT(src[7], src[8], src[9], src[0], src[1],
                                weight[0], weight[1], weight[2], weight[3],
                                weight[4], c[0][7]);
            }
            weight_ptr += fh * fw * ld_weight_ic4;
        }

        STORE_1_LINE_RESULT();
    }
};
#undef CALC_ONE_RESULT
template <BiasMode bias_mode, int remain_w>
struct KerNeonDirectStride1Int8<bias_mode, remain_w, 7> {
    static void impl(const int8_t* src_ptr, const int8_t* weight_ptr,
                     const int16_t* bias_ptr, int16_t* dst_ptr, int ic, int ih,
                     int iw, int /*ld_dst_oc*/) {
        constexpr int filter_size = 7;
        constexpr int fh = filter_size;
        constexpr int fw = filter_size;
        constexpr int ic_step = 4;
        constexpr int loop_ic_step = 4;
        constexpr int ld_weight_ic4 = 16;
        constexpr int src_expand_size = 4;

        const int ic_stride = ih * iw * src_expand_size;

        int16x4_t c[1][8];
        int8x16_t weight[7];
        int8x16_t src[8 + 2];

        INIT_SUM();
#define cb(_i) c[0][_i] = init_sum;

        UNROLL_CALL_RAW(8, cb);
#undef cb

        for (int ic_idx = 0; ic_idx < ic; ic_idx += loop_ic_step) {
            for (int fh_idx = 0; fh_idx < fh; ++fh_idx) {
                const int8_t* src_ic_0_3 =
                        src_ptr + ic_idx * ic_stride +
                        fh_idx * iw * ic_step * src_expand_size;

                src[0] = vld1q_s8(src_ic_0_3 + 0 * 16);
                src[1] = vld1q_s8(src_ic_0_3 + 1 * 16);
                src[2] = vld1q_s8(src_ic_0_3 + 2 * 16);
                src[3] = vld1q_s8(src_ic_0_3 + 3 * 16);
                src[4] = vld1q_s8(src_ic_0_3 + 4 * 16);
                src[5] = vld1q_s8(src_ic_0_3 + 5 * 16);
                src[6] = vld1q_s8(src_ic_0_3 + 6 * 16);
                src[7] = vld1q_s8(src_ic_0_3 + 7 * 16);
                src[8] = vld1q_s8(src_ic_0_3 + 8 * 16);
                src[9] = vld1q_s8(src_ic_0_3 + 9 * 16);

                const int8_t* read_weight_ptr =
                        weight_ptr + fh_idx * fw * ld_weight_ic4;

                weight[0] = vld1q_s8(read_weight_ptr);
                weight[1] = vld1q_s8(read_weight_ptr + 16);
                weight[2] = vld1q_s8(read_weight_ptr + 2 * 16);
                weight[3] = vld1q_s8(read_weight_ptr + 3 * 16);
                weight[4] = vld1q_s8(read_weight_ptr + 4 * 16);
                weight[5] = vld1q_s8(read_weight_ptr + 5 * 16);
                weight[6] = vld1q_s8(read_weight_ptr + 6 * 16);

#define CALC_ONE_RESULT(_src0, _src1, _src2, _src3, _src4, _src5, _src6, _w,  \
                        _c)                                                   \
    do {                                                                      \
        tmp0 = vmull_s8(vget_low_s8(_src0), vget_low_s8(_w[0]));              \
        tmp0 = vmlal_s8(tmp0, vget_high_s8(_src0), vget_high_s8(_w[0]));      \
        tmp0 = vmlal_s8(tmp0, vget_low_s8(_src1), vget_low_s8(_w[1]));        \
        tmp0 = vmlal_s8(tmp0, vget_high_s8(_src1), vget_high_s8(_w[1]));      \
        tmp0 = vmlal_s8(tmp0, vget_low_s8(_src2), vget_low_s8(_w[2]));        \
        tmp0 = vmlal_s8(tmp0, vget_high_s8(_src2), vget_high_s8(_w[2]));      \
        tmp0 = vmlal_s8(tmp0, vget_low_s8(_src3), vget_low_s8(_w[3]));        \
        tmp0 = vmlal_s8(tmp0, vget_high_s8(_src3), vget_high_s8(_w[3]));      \
        tmp0 = vmlal_s8(tmp0, vget_low_s8(_src4), vget_low_s8(_w[4]));        \
        tmp0 = vmlal_s8(tmp0, vget_high_s8(_src4), vget_high_s8(_w[4]));      \
        tmp0 = vmlal_s8(tmp0, vget_low_s8(_src5), vget_low_s8(_w[5]));        \
        tmp0 = vmlal_s8(tmp0, vget_high_s8(_src5), vget_high_s8(_w[5]));      \
        tmp0 = vmlal_s8(tmp0, vget_low_s8(_src6), vget_low_s8(_w[6]));        \
        tmp0 = vmlal_s8(tmp0, vget_high_s8(_src6), vget_high_s8(_w[6]));      \
        _c = vadd_s16(_c, vadd_s16(vget_low_s16(tmp0), vget_high_s16(tmp0))); \
    } while (0);
                int16x8_t tmp0;
                CALC_ONE_RESULT(src[0], src[1], src[2], src[3], src[4], src[5],
                                src[6], weight, c[0][0]);
                CALC_ONE_RESULT(src[1], src[2], src[3], src[4], src[5], src[6],
                                src[7], weight, c[0][1]);
                src[0] = vld1q_s8(src_ic_0_3 + 10 * 16);
                src[1] = vld1q_s8(src_ic_0_3 + 11 * 16);

                CALC_ONE_RESULT(src[2], src[3], src[4], src[5], src[6], src[7],
                                src[8], weight, c[0][2]);
                CALC_ONE_RESULT(src[3], src[4], src[5], src[6], src[7], src[8],
                                src[9], weight, c[0][3]);

                src[2] = vld1q_s8(src_ic_0_3 + 12 * 16);
                src[3] = vld1q_s8(src_ic_0_3 + 13 * 16);
                CALC_ONE_RESULT(src[4], src[5], src[6], src[7], src[8], src[9],
                                src[0], weight, c[0][4]);
                CALC_ONE_RESULT(src[5], src[6], src[7], src[8], src[9], src[0],
                                src[1], weight, c[0][5]);
                CALC_ONE_RESULT(src[6], src[7], src[8], src[9], src[0], src[1],
                                src[2], weight, c[0][6]);
                CALC_ONE_RESULT(src[7], src[8], src[9], src[0], src[1], src[2],
                                src[3], weight, c[0][7]);
            }
            weight_ptr += fh * fw * ld_weight_ic4;
        }
        STORE_1_LINE_RESULT();
    }
};

template <BiasMode bias_mode>
void conv_direct_stride1_2x2_int8_oc8_ow8_nchw44(
        const int8_t* src, const int8_t* filter, const int16_t* bias,
        int16_t* dst, const size_t oc, const size_t ic, const size_t ih,
        const size_t iw, const size_t oh, const size_t ow) {
    constexpr size_t filter_size = 2;
    constexpr size_t fh = filter_size;
    constexpr size_t fw = filter_size;
    constexpr size_t ic_step = 4;
    constexpr size_t oc_step = 4;
    constexpr size_t big_oc_step = 8;
    constexpr size_t oh_step = 1;
    constexpr size_t ow_step = 8;
    constexpr size_t src_expand_size = 4;

    const size_t img_stride = oh * ow;
    const size_t ow_end = ow / ow_step * ow_step;
    const size_t ow_remain = ow - ow_end;
    const size_t oc_end = oc / big_oc_step * big_oc_step;
    const size_t oc_remain = oc - oc_end;
    const int ld_oc = oh * ow * oc_step;
    using remain_fun =
            std::function<void(const int8_t* src_ptr, const int8_t* weight_ptr,
                               const int16_t* bias_ptr, int16_t* dst_ptr,
                               int ic, int ih, int iw, int ld_dst_oc)>;
    remain_fun kern_big_oc_remain = nullptr;
    remain_fun kern_small_oc_remain = nullptr;

    switch (ow_remain) {
#define cb(step)                                                               \
    case step:                                                                 \
        kern_big_oc_remain = ker_neon_dirctconv_2x2s1_oc8_ow8<bias_mode, step, \
                                                              filter_size>;    \
        kern_small_oc_remain =                                                 \
                ker_neon_dirctconv_2x2s1_oc4_ow8<bias_mode, step,              \
                                                 filter_size>;                 \
        break;

        UNROLL_CALL_RAW(8, cb);
        default:
            megdnn_assert(0, "no remain %zu for kern", ow_remain);
    }
#undef cb
    for (size_t oc_idx = 0; oc_idx < oc_end; oc_idx += big_oc_step) {
        const size_t weight_offset = oc_idx * ic * fh * fw;
        size_t oh_idx = 0;
        for (; oh_idx < oh; oh_idx += oh_step) {
            for (size_t ow_idx = 0; ow_idx < ow_end; ow_idx += ow_step) {
                const size_t src_offset =
                        (oh_idx * iw + ow_idx) * ic_step * src_expand_size;
                const size_t dst_offset =
                        oc_idx * img_stride + (oh_idx * ow + ow_idx) * oc_step;
                ker_neon_dirctconv_2x2s1_oc8_ow8<bias_mode, ow_step, filter_size>(
                        src + src_offset, filter + weight_offset, bias + oc_idx,
                        dst + dst_offset, ic, ih, iw, ld_oc);
            }
            if (ow_remain > 0) {
                const size_t src_offset =
                        (oh_idx * iw + ow_end) * ic_step * src_expand_size;
                const size_t dst_offset =
                        oc_idx * img_stride + (oh_idx * ow + ow_end) * oc_step;
                kern_big_oc_remain(src + src_offset, filter + weight_offset,
                                   bias + oc_idx, dst + dst_offset, ic, ih, iw,
                                   ld_oc);
            }
        }
    }
    if (oc_remain > 0) {
        const size_t oc_idx = oc_end;
        const size_t weight_offset = oc_idx * ic * fh * fw;
        for (size_t oh_idx = 0; oh_idx < oh; oh_idx += oh_step) {
            for (size_t ow_idx = 0; ow_idx < ow_end; ow_idx += ow_step) {
                const size_t src_offset =
                        (oh_idx * iw + ow_idx) * ic_step * src_expand_size;
                const size_t dst_offset =
                        oc_idx * img_stride + (oh_idx * ow + ow_idx) * oc_step;
                ker_neon_dirctconv_2x2s1_oc4_ow8<bias_mode, ow_step, filter_size>(
                        src + src_offset, filter + weight_offset, bias + oc_idx,
                        dst + dst_offset, ic, ih, iw, ld_oc);
            }
            if (ow_remain > 0) {
                const size_t src_offset =
                        (oh_idx * iw + ow_end) * ic_step * src_expand_size;
                const size_t dst_offset =
                        oc_idx * img_stride + (oh_idx * ow + ow_end) * oc_step;
                kern_small_oc_remain(src + src_offset, filter + weight_offset,
                                     bias + oc_idx, dst + dst_offset, ic, ih,
                                     iw, ld_oc);
            }
        }
    }
}
#undef CALC_ONE_RESULT

template <BiasMode bias_mode, int filter_size>
void conv_direct_stride1_int8_nchw44_kern(const int8_t* src,
                                          const int8_t* filter,
                                          const int16_t* bias, int16_t* dst,
                                          const size_t oc, const size_t ic,
                                          const size_t ih, const size_t iw,
                                          const size_t oh, const size_t ow) {
    constexpr size_t fh = filter_size;
    constexpr size_t fw = filter_size;
    constexpr size_t ic_step = 4;
    constexpr size_t oc_step = 4;
    constexpr size_t oh_step = 1;
    constexpr size_t ow_step = 8;
    constexpr size_t src_expand_size = 4;

    const size_t img_stride = oh * ow;
    const int ld_dst_oc = oh * ow * oc_step;
    const size_t ow_end = ow / ow_step * ow_step;
    const size_t ow_remain = ow - ow_end;

    using remain_fun =
            std::function<void(const int8_t* src_ptr, const int8_t* weight_ptr,
                               const int16_t* bias_ptr, int16_t* dst_ptr,
                               int ic, int ih, int iw, int ld_dst_oc)>;

    remain_fun kern_small_oc_remain = nullptr;
    switch (ow_remain) {
#define cb(step)                                                               \
    case step:                                                                 \
        kern_small_oc_remain = KerNeonDirectStride1Int8<bias_mode, step,       \
                                                        filter_size>::impl;    \
        break;

        UNROLL_CALL_RAW(8, cb);
        default:
            megdnn_assert(0, "no remain %zu for kern", ow_remain);
    }
#undef cb

    for (size_t oc_idx = 0; oc_idx < oc; oc_idx += oc_step) {
        const size_t weight_offset = oc_idx * ic * fh * fw;
        for (size_t oh_idx = 0; oh_idx < oh; oh_idx += oh_step) {
            for (size_t ow_idx = 0; ow_idx < ow_end; ow_idx += ow_step) {
                const size_t src_offset =
                        (oh_idx * iw + ow_idx) * ic_step * src_expand_size;
                const size_t dst_offset =
                        oc_idx * img_stride + (oh_idx * ow + ow_idx) * oc_step;
                KerNeonDirectStride1Int8<bias_mode, ow_step, filter_size>::impl(
                        src + src_offset, filter + weight_offset, bias + oc_idx,
                        dst + dst_offset, ic, ih, iw, ld_dst_oc);
            }
            if (ow_remain > 0) {
                const size_t src_offset =
                        (oh_idx * iw + ow_end) * ic_step * src_expand_size;
                const size_t dst_offset =
                        oc_idx * img_stride + (oh_idx * ow + ow_end) * oc_step;
                kern_small_oc_remain(src + src_offset, filter + weight_offset,
                                     bias + oc_idx, dst + dst_offset, ic, ih,
                                     iw, ld_dst_oc);
            }
        }
    }
}
}  // namespace

namespace int8x8x16_direct_nchw44 {
template <BiasMode bias_mode, int filter_size>
struct ConvDirectInt8Nchw44Choose<bias_mode, filter_size, 1> {
    static void impl(const int8_t* src, const int8_t* filter,
                     const int16_t* bias, int16_t* dst, const size_t oc,
                     const size_t ic, const size_t ih, const size_t iw,
                     const size_t oh, const size_t ow) {
        conv_direct_stride1_int8_nchw44_kern<bias_mode, filter_size>(
                src, filter, bias, dst, oc, ic, ih, iw, oh, ow);
    }
};

template <BiasMode bias_mode>
struct ConvDirectInt8Nchw44Choose<bias_mode, 2, 1> {
    static void impl(const int8_t* src, const int8_t* filter,
                     const int16_t* bias, int16_t* dst, const size_t oc,
                     const size_t ic, const size_t ih, const size_t iw,
                     const size_t oh, const size_t ow) {
        conv_direct_stride1_2x2_int8_oc8_ow8_nchw44<bias_mode>(
                src, filter, bias, dst, oc, ic, ih, iw, oh, ow);
    }
};

#define DO_CONV_KERN_FUN(stride, filter_size, bias_mode) \
    template struct ConvDirectInt8Nchw44Choose<bias_mode, filter_size, stride>;

#define GET_OP_PARAM(stride, filter, bias_mode) \
    DO_CONV_KERN_FUN(stride, filter, bias_mode)

#define GET_BIAS_MODE_PARAM(stride, filter)         \
    GET_OP_PARAM(stride, filter, BiasMode::NO_BIAS) \
    GET_OP_PARAM(stride, filter, BiasMode::BROADCAST_CHANNEL_BIAS)

#define DISPATCH_CONV_KERN(stride) \
    GET_BIAS_MODE_PARAM(stride, 2) \
    GET_BIAS_MODE_PARAM(stride, 3) \
    GET_BIAS_MODE_PARAM(stride, 5) \
    GET_BIAS_MODE_PARAM(stride, 7)

DISPATCH_CONV_KERN(1);

}  // namespace int8x8x16_direct_nchw44
}  // namespace arm_common
}  // namespace megdnn
#endif

// vim: syntax=cpp.doxygen
