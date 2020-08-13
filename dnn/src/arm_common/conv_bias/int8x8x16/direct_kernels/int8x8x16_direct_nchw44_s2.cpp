/**
 * \file
 * dnn/src/arm_common/conv_bias/int8x8x16/direct_kernels/int8x8x16_direct_nchw44_s2.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/arm_common/conv_bias/int8x8x16/direct_8x8x16_nchw44_kern.h"
#include "src/arm_common/simd_macro/marm_neon.h"
#include "src/common/utils.h"
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
            break;                                                   \
    };

#define STORE_1_LINE_RESULT_OW4()                                   \
    switch (remain_w) {                                             \
        case 4:                                                     \
            vst1q_s16(dst_ptr, vcombine_s16(c[0][0], c[0][1]));     \
            vst1q_s16(dst_ptr + 8, vcombine_s16(c[0][2], c[0][3])); \
            break;                                                  \
        case 1:                                                     \
            vst1_s16(dst_ptr, c[0][0]);                             \
            break;                                                  \
        case 2:                                                     \
            vst1q_s16(dst_ptr, vcombine_s16(c[0][0], c[0][1]));     \
            break;                                                  \
        case 3:                                                     \
            vst1q_s16(dst_ptr, vcombine_s16(c[0][0], c[0][1]));     \
            vst1_s16(dst_ptr + 8, c[0][2]);                         \
            break;                                                  \
        default:                                                    \
            megdnn_assert(0, "oc 1 error remainw");                 \
            break;                                                  \
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

#define STORE_2_LINE_RESULT_OW4()                                           \
    switch (remain_w) {                                                     \
        case 4:                                                             \
            vst1q_s16(dst_ptr, vcombine_s16(c[0][0], c[0][1]));             \
            vst1q_s16(dst_ptr + 8, vcombine_s16(c[0][2], c[0][3]));         \
            vst1q_s16(dst_ptr + ld_dst_oc, vcombine_s16(c[1][0], c[1][1])); \
            vst1q_s16(dst_ptr + ld_dst_oc + 8,                              \
                      vcombine_s16(c[1][2], c[1][3]));                      \
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
        default:                                                            \
            megdnn_assert(0, "oc 2 error remainw");                         \
            break;                                                          \
    }

template <BiasMode bias_mode, int remain_w, int filter_size>
static void ker_neon_dirctconv_2x2s2_oc8_ow8(const int8_t* src_ptr,
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

    const int ic_stride = ih * iw;
    const int ld_weight_oc4 = oc_step * fh * fw * ic;

    static const uint8_t idx_buffer[16] = {0, 0, 0, 0, 1, 1, 1, 1,
                                           2, 2, 2, 2, 3, 3, 3, 3};
    static uint8x16_t idx = vld1q_u8(idx_buffer);
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
        for (int fh_idx = 0; fh_idx < fh; ++fh_idx) {
            const int8_t* src_ic_0_3 =
                    src_ptr + ic_idx * ic_stride + fh_idx * iw * ic_step;

            src[0] = vld_dup_tbl_s32(src_ic_0_3, idx);
            src[1] = vld_dup_tbl_s32(src_ic_0_3 + 4, idx);
            src[2] = vld_dup_tbl_s32(src_ic_0_3 + 8, idx);
            src[3] = vld_dup_tbl_s32(src_ic_0_3 + 12, idx);

            const int8_t* read_weight_ptr =
                    weight_ptr + fh_idx * fw * ld_weight_ic4;

            weight[0][0] = vld1q_s8(read_weight_ptr);
            weight[0][1] = vld1q_s8(read_weight_ptr + 16);
            weight[1][0] = vld1q_s8(read_weight_ptr + ld_weight_oc4);
            weight[1][1] = vld1q_s8(read_weight_ptr + ld_weight_oc4 + 16);

#define CALC_ONE_RESULT(_src0, _src1, _w0, _w1, _c)                           \
    do {                                                                      \
        tmp0 = vmull_s8(vget_low_s8(_src0), vget_low_s8(_w0));                \
        tmp0 = vmlal_s8(tmp0, vget_high_s8(_src0), vget_high_s8(_w0));        \
        tmp0 = vmlal_s8(tmp0, vget_low_s8(_src1), vget_low_s8(_w1));          \
        tmp0 = vmlal_s8(tmp0, vget_high_s8(_src1), vget_high_s8(_w1));        \
        _c = vadd_s16(_c, vadd_s16(vget_low_s16(tmp0), vget_high_s16(tmp0))); \
    } while (0);

            int16x8_t tmp0;
            CALC_ONE_RESULT(src[0], src[1], weight[0][0], weight[0][1],
                            c[0][0]);
            CALC_ONE_RESULT(src[0], src[1], weight[1][0], weight[1][1],
                            c[1][0]);

            src[0] = vld_dup_tbl_s32(src_ic_0_3 + 16, idx);
            src[1] = vld_dup_tbl_s32(src_ic_0_3 + 20, idx);

            CALC_ONE_RESULT(src[2], src[3], weight[0][0], weight[0][1],
                            c[0][1]);
            CALC_ONE_RESULT(src[2], src[3], weight[1][0], weight[1][1],
                            c[1][1]);

            src[2] = vld_dup_tbl_s32(src_ic_0_3 + 24, idx);
            src[3] = vld_dup_tbl_s32(src_ic_0_3 + 28, idx);

            CALC_ONE_RESULT(src[0], src[1], weight[0][0], weight[0][1],
                            c[0][2]);
            CALC_ONE_RESULT(src[0], src[1], weight[1][0], weight[1][1],
                            c[1][2]);

            src[0] = vld_dup_tbl_s32(src_ic_0_3 + 32, idx);
            src[1] = vld_dup_tbl_s32(src_ic_0_3 + 36, idx);

            CALC_ONE_RESULT(src[2], src[3], weight[0][0], weight[0][1],
                            c[0][3]);
            CALC_ONE_RESULT(src[2], src[3], weight[1][0], weight[1][1],
                            c[1][3]);

            src[2] = vld_dup_tbl_s32(src_ic_0_3 + 40, idx);
            src[3] = vld_dup_tbl_s32(src_ic_0_3 + 44, idx);

            CALC_ONE_RESULT(src[0], src[1], weight[0][0], weight[0][1],
                            c[0][4]);
            CALC_ONE_RESULT(src[0], src[1], weight[1][0], weight[1][1],
                            c[1][4]);

            src[0] = vld_dup_tbl_s32(src_ic_0_3 + 48, idx);
            src[1] = vld_dup_tbl_s32(src_ic_0_3 + 52, idx);

            CALC_ONE_RESULT(src[2], src[3], weight[0][0], weight[0][1],
                            c[0][5]);
            CALC_ONE_RESULT(src[2], src[3], weight[1][0], weight[1][1],
                            c[1][5]);

            src[2] = vld_dup_tbl_s32(src_ic_0_3 + 56, idx);
            src[3] = vld_dup_tbl_s32(src_ic_0_3 + 60, idx);

            CALC_ONE_RESULT(src[0], src[1], weight[0][0], weight[0][1],
                            c[0][6]);
            CALC_ONE_RESULT(src[0], src[1], weight[1][0], weight[1][1],
                            c[1][6]);
            CALC_ONE_RESULT(src[2], src[3], weight[0][0], weight[0][1],
                            c[0][7]);
            CALC_ONE_RESULT(src[2], src[3], weight[1][0], weight[1][1],
                            c[1][7]);
        }
        weight_ptr += fh * fw * ld_weight_ic4;
    }
    STORE_2_LINE_RESULT();
}

template <BiasMode bias_mode, int remain_w, int filter_size>
static void ker_neon_dirctconv_2x2s2_oc4_ow8(const int8_t* src_ptr,
                                             const int8_t* weight_ptr,
                                             const int16_t* bias_ptr,
                                             int16_t* dst_ptr, int ic, int ih,
                                             int iw, int /*ld_dst_oc*/) {
    constexpr int fh = filter_size;
    constexpr int fw = filter_size;
    constexpr int ic_step = 4;
    constexpr int loop_ic_step = 4;
    constexpr int ld_weight_ic4 = 16;

    const int ic_stride = ih * iw;
    static const uint8_t idx_buffer[16] = {0, 0, 0, 0, 1, 1, 1, 1,
                                           2, 2, 2, 2, 3, 3, 3, 3};
    static uint8x16_t idx = vld1q_u8(idx_buffer);

    int16x4_t c[1][8];
    int8x16_t weight[2];
    int8x16_t src[4];
    INIT_SUM();
#define cb(_i) c[0][_i] = init_sum;

    UNROLL_CALL_RAW(8, cb);

#undef cb

    for (int ic_idx = 0; ic_idx < ic; ic_idx += loop_ic_step) {
        for (int fh_idx = 0; fh_idx < fh; ++fh_idx) {
            const int8_t* src_ic_0_3 =
                    src_ptr + ic_idx * ic_stride + fh_idx * iw * ic_step;

            src[0] = vld_dup_tbl_s32(src_ic_0_3, idx);
            src[1] = vld_dup_tbl_s32(src_ic_0_3 + 4, idx);
            src[2] = vld_dup_tbl_s32(src_ic_0_3 + 8, idx);
            src[3] = vld_dup_tbl_s32(src_ic_0_3 + 12, idx);

            const int8_t* read_weight_ptr =
                    weight_ptr + fh_idx * fw * ld_weight_ic4;

            weight[0] = vld1q_s8(read_weight_ptr);
            weight[1] = vld1q_s8(read_weight_ptr + 16);
            int16x8_t tmp0;
            CALC_ONE_RESULT(src[0], src[1], weight[0], weight[1], c[0][0]);
            src[0] = vld_dup_tbl_s32(src_ic_0_3 + 16, idx);
            src[1] = vld_dup_tbl_s32(src_ic_0_3 + 20, idx);

            CALC_ONE_RESULT(src[2], src[3], weight[0], weight[1], c[0][1]);
            src[2] = vld_dup_tbl_s32(src_ic_0_3 + 24, idx);
            src[3] = vld_dup_tbl_s32(src_ic_0_3 + 28, idx);

            CALC_ONE_RESULT(src[0], src[1], weight[0], weight[1], c[0][2]);

            src[0] = vld_dup_tbl_s32(src_ic_0_3 + 32, idx);
            src[1] = vld_dup_tbl_s32(src_ic_0_3 + 36, idx);

            CALC_ONE_RESULT(src[2], src[3], weight[0], weight[1], c[0][3]);
            src[2] = vld_dup_tbl_s32(src_ic_0_3 + 40, idx);
            src[3] = vld_dup_tbl_s32(src_ic_0_3 + 44, idx);
            CALC_ONE_RESULT(src[0], src[1], weight[0], weight[1], c[0][4]);
            src[0] = vld_dup_tbl_s32(src_ic_0_3 + 48, idx);
            src[1] = vld_dup_tbl_s32(src_ic_0_3 + 52, idx);
            CALC_ONE_RESULT(src[2], src[3], weight[0], weight[1], c[0][5]);
            src[2] = vld_dup_tbl_s32(src_ic_0_3 + 56, idx);
            src[3] = vld_dup_tbl_s32(src_ic_0_3 + 60, idx);
            CALC_ONE_RESULT(src[0], src[1], weight[0], weight[1], c[0][6]);
            CALC_ONE_RESULT(src[2], src[3], weight[0], weight[1], c[0][7]);
        }
        weight_ptr += fh * fw * ld_weight_ic4;
    }
    STORE_1_LINE_RESULT();
}
#undef CALC_ONE_RESULT

#define CALC_ONE_RESULT(_src0, _src1, _src2, _w, _c)                          \
    do {                                                                      \
        tmp0 = vmull_s8(vget_low_s8(_src0), vget_low_s8(_w[0]));              \
        tmp1 = vmull_s8(vget_high_s8(_src0), vget_high_s8(_w[0]));            \
        tmp0 = vmlal_s8(tmp0, vget_low_s8(_src1), vget_low_s8(_w[1]));        \
        tmp1 = vmlal_s8(tmp1, vget_high_s8(_src1), vget_high_s8(_w[1]));      \
        tmp0 = vmlal_s8(tmp0, vget_low_s8(_src2), vget_low_s8(_w[2]));        \
        tmp1 = vmlal_s8(tmp1, vget_high_s8(_src2), vget_high_s8(_w[2]));      \
        tmp0 = vaddq_s16(tmp0, tmp1);                                         \
        _c = vadd_s16(_c, vadd_s16(vget_low_s16(tmp0), vget_high_s16(tmp0))); \
    } while (0);

template <BiasMode bias_mode, int remain_w, int filter_size>
static void ker_neon_dirctconv_3x3s2_oc8_ow4(const int8_t* src_ptr,
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
    const int ic_stride = ih * iw;
    const int ld_weight_oc4 = oc_step * fh * fw * ic;

    static const uint8_t idx_buffer[16] = {0, 0, 0, 0, 1, 1, 1, 1,
                                           2, 2, 2, 2, 3, 3, 3, 3};
    static uint8x16_t idx = vld1q_u8(idx_buffer);
    int16x4_t c[2][4];
    int8x16_t weight[2][3];
    int8x16_t src[5];

    INIT_SUM();
#define cb(_i)           \
    c[0][_i] = init_sum; \
    c[1][_i] = init_sum;

    UNROLL_CALL_RAW(4, cb);

#undef cb

    for (int ic_idx = 0; ic_idx < ic; ic_idx += loop_ic_step) {
        for (int fh_idx = 0; fh_idx < fh; ++fh_idx) {
            const int8_t* src_ic_0_3 =
                    src_ptr + ic_idx * ic_stride + fh_idx * iw * ic_step;

            src[0] = vld_dup_tbl_s32(src_ic_0_3, idx);
            src[1] = vld_dup_tbl_s32(src_ic_0_3 + 4, idx);
            src[2] = vld_dup_tbl_s32(src_ic_0_3 + 8, idx);
            src[3] = vld_dup_tbl_s32(src_ic_0_3 + 12, idx);
            src[4] = vld_dup_tbl_s32(src_ic_0_3 + 16, idx);
            const int8_t* read_weight_ptr =
                    weight_ptr + fh_idx * fw * ld_weight_ic4;

            weight[0][0] = vld1q_s8(read_weight_ptr);
            weight[0][1] = vld1q_s8(read_weight_ptr + 16);
            weight[0][2] = vld1q_s8(read_weight_ptr + 32);
            weight[1][0] = vld1q_s8(read_weight_ptr + ld_weight_oc4);
            weight[1][1] = vld1q_s8(read_weight_ptr + ld_weight_oc4 + 16);
            weight[1][2] = vld1q_s8(read_weight_ptr + ld_weight_oc4 + 32);

            int16x8_t tmp0, tmp1;
            CALC_ONE_RESULT(src[0], src[1], src[2], weight[0], c[0][0]);
            CALC_ONE_RESULT(src[0], src[1], src[2], weight[1], c[1][0]);

            src[0] = vld_dup_tbl_s32(src_ic_0_3 + 20, idx);
            src[1] = vld_dup_tbl_s32(src_ic_0_3 + 24, idx);
            CALC_ONE_RESULT(src[2], src[3], src[4], weight[0], c[0][1]);
            CALC_ONE_RESULT(src[2], src[3], src[4], weight[1], c[1][1]);

            src[2] = vld_dup_tbl_s32(src_ic_0_3 + 28, idx);
            src[3] = vld_dup_tbl_s32(src_ic_0_3 + 32, idx);

            CALC_ONE_RESULT(src[4], src[0], src[1], weight[0], c[0][2]);
            CALC_ONE_RESULT(src[4], src[0], src[1], weight[1], c[1][2]);

            CALC_ONE_RESULT(src[1], src[2], src[3], weight[0], c[0][3]);
            CALC_ONE_RESULT(src[1], src[2], src[3], weight[1], c[1][3]);
        }
        weight_ptr += fh * fw * ld_weight_ic4;
    }

    vst1q_s16(dst_ptr, vcombine_s16(c[0][0], c[0][1]));
    vst1q_s16(dst_ptr + 8, vcombine_s16(c[0][2], c[0][3]));
    vst1q_s16(dst_ptr + ld_dst_oc, vcombine_s16(c[1][0], c[1][1]));
    vst1q_s16(dst_ptr + ld_dst_oc + 8, vcombine_s16(c[1][2], c[1][3]));
}

template <BiasMode bias_mode, int remain_w, int filter_size>
static void ker_neon_dirctconv_3x3s2_oc8_ow4_remain(const int8_t* src_ptr,
                                                    const int8_t* weight_ptr,
                                                    const int16_t* bias_ptr,
                                                    int16_t* dst_ptr, int ic,
                                                    int ih, int iw,
                                                    int ld_dst_oc) {
    constexpr int fh = filter_size;
    constexpr int fw = filter_size;
    constexpr int ic_step = 4;
    constexpr int oc_step = 4;
    constexpr int loop_ic_step = 4;
    constexpr int ld_weight_ic4 = 16;
    const int ic_stride = ih * iw;
    const int ld_weight_oc4 = oc_step * fh * fw * ic;

    static const uint8_t idx_buffer[16] = {0, 0, 0, 0, 1, 1, 1, 1,
                                           2, 2, 2, 2, 3, 3, 3, 3};
    static uint8x16_t idx = vld1q_u8(idx_buffer);
    int16x4_t c[2][4];
    int8x16_t weight[2][3];
    int8x16_t src[5];

    INIT_SUM();
#define cb(_i)           \
    c[0][_i] = init_sum; \
    c[1][_i] = init_sum;

    UNROLL_CALL_RAW(4, cb);

#undef cb

    for (int ic_idx = 0; ic_idx < ic; ic_idx += loop_ic_step) {
        for (int fh_idx = 0; fh_idx < fh; ++fh_idx) {
            const int8_t* src_ic_0_3 =
                    src_ptr + ic_idx * ic_stride + fh_idx * iw * ic_step;

            src[0] = vld_dup_tbl_s32(src_ic_0_3, idx);
            src[1] = vld_dup_tbl_s32(src_ic_0_3 + 4, idx);
            src[2] = vld_dup_tbl_s32(src_ic_0_3 + 8, idx);
            src[3] = vld_dup_tbl_s32(src_ic_0_3 + 12, idx);
            src[4] = vld_dup_tbl_s32(src_ic_0_3 + 16, idx);
            const int8_t* read_weight_ptr =
                    weight_ptr + fh_idx * fw * ld_weight_ic4;

            weight[0][0] = vld1q_s8(read_weight_ptr);
            weight[0][1] = vld1q_s8(read_weight_ptr + 16);
            weight[0][2] = vld1q_s8(read_weight_ptr + 32);
            weight[1][0] = vld1q_s8(read_weight_ptr + ld_weight_oc4);
            weight[1][1] = vld1q_s8(read_weight_ptr + ld_weight_oc4 + 16);
            weight[1][2] = vld1q_s8(read_weight_ptr + ld_weight_oc4 + 32);

            int16x8_t tmp0, tmp1;
            CALC_ONE_RESULT(src[0], src[1], src[2], weight[0], c[0][0]);
            CALC_ONE_RESULT(src[0], src[1], src[2], weight[1], c[1][0]);

            src[0] = vld_dup_tbl_s32(src_ic_0_3 + 20, idx);
            src[1] = vld_dup_tbl_s32(src_ic_0_3 + 24, idx);
            CALC_ONE_RESULT(src[2], src[3], src[4], weight[0], c[0][1]);
            CALC_ONE_RESULT(src[2], src[3], src[4], weight[1], c[1][1]);

            src[2] = vld_dup_tbl_s32(src_ic_0_3 + 28, idx);
            src[3] = vld_dup_tbl_s32(src_ic_0_3 + 32, idx);

            CALC_ONE_RESULT(src[4], src[0], src[1], weight[0], c[0][2]);
            CALC_ONE_RESULT(src[4], src[0], src[1], weight[1], c[1][2]);

            CALC_ONE_RESULT(src[1], src[2], src[3], weight[0], c[0][3]);
            CALC_ONE_RESULT(src[1], src[2], src[3], weight[1], c[1][3]);
        }
        weight_ptr += fh * fw * ld_weight_ic4;
    }
    STORE_2_LINE_RESULT_OW4();
}

#undef CALC_ONE_RESULT

#define CALC_ONE_RESULT(_src0, _src1, _src2, _w, _c)                          \
    do {                                                                      \
        int16x8_t tmp0 = vmull_s8(vget_low_s8(_src0), vget_low_s8(_w[0]));    \
        tmp0 = vmlal_s8(tmp0, vget_high_s8(_src0), vget_high_s8(_w[0]));      \
        tmp0 = vmlal_s8(tmp0, vget_low_s8(_src1), vget_low_s8(_w[1]));        \
        tmp0 = vmlal_s8(tmp0, vget_high_s8(_src1), vget_high_s8(_w[1]));      \
        tmp0 = vmlal_s8(tmp0, vget_low_s8(_src2), vget_low_s8(_w[2]));        \
        tmp0 = vmlal_s8(tmp0, vget_high_s8(_src2), vget_high_s8(_w[2]));      \
        _c = vadd_s16(_c, vadd_s16(vget_low_s16(tmp0), vget_high_s16(tmp0))); \
    } while (0);

template <BiasMode bias_mode, int remain_w, int filter_size>
static void ker_neon_dirctconv_3x3s2_oc4_ow4(const int8_t* src_ptr,
                                             const int8_t* weight_ptr,
                                             const int16_t* bias_ptr,
                                             int16_t* dst_ptr, int ic, int ih,
                                             int iw, int /*ld_dst_oc*/) {
    constexpr int fh = filter_size;
    constexpr int fw = filter_size;
    constexpr int ic_step = 4;
    constexpr int loop_ic_step = 4;
    constexpr int ld_weight_ic4 = 16;

    const int ic_stride = ih * iw;
    static const uint8_t idx_buffer[16] = {0, 0, 0, 0, 1, 1, 1, 1,
                                           2, 2, 2, 2, 3, 3, 3, 3};
    static uint8x16_t idx = vld1q_u8(idx_buffer);

    int16x4_t c[1][4];
    int8x16_t weight[3];
    int8x16_t src[5];

    INIT_SUM();
#define cb(_i) c[0][_i] = init_sum;

    UNROLL_CALL_RAW(4, cb);

#undef cb

    for (int ic_idx = 0; ic_idx < ic; ic_idx += loop_ic_step) {
        for (int fh_idx = 0; fh_idx < fh; ++fh_idx) {
            const int8_t* src_ic_0_3 =
                    src_ptr + ic_idx * ic_stride + fh_idx * iw * ic_step;

            src[0] = vld_dup_tbl_s32(src_ic_0_3, idx);
            src[1] = vld_dup_tbl_s32(src_ic_0_3 + 4, idx);
            src[2] = vld_dup_tbl_s32(src_ic_0_3 + 8, idx);
            src[3] = vld_dup_tbl_s32(src_ic_0_3 + 12, idx);
            src[4] = vld_dup_tbl_s32(src_ic_0_3 + 16, idx);
            const int8_t* read_weight_ptr =
                    weight_ptr + fh_idx * fw * ld_weight_ic4;

            weight[0] = vld1q_s8(read_weight_ptr);
            weight[1] = vld1q_s8(read_weight_ptr + 16);
            weight[2] = vld1q_s8(read_weight_ptr + 2 * 16);

            CALC_ONE_RESULT(src[0], src[1], src[2], weight, c[0][0]);

            src[0] = vld_dup_tbl_s32(src_ic_0_3 + 20, idx);
            src[1] = vld_dup_tbl_s32(src_ic_0_3 + 24, idx);
            CALC_ONE_RESULT(src[2], src[3], src[4], weight, c[0][1]);

            src[2] = vld_dup_tbl_s32(src_ic_0_3 + 28, idx);
            src[3] = vld_dup_tbl_s32(src_ic_0_3 + 32, idx);

            CALC_ONE_RESULT(src[4], src[0], src[1], weight, c[0][2]);

            CALC_ONE_RESULT(src[1], src[2], src[3], weight, c[0][3]);
        }
        weight_ptr += fh * fw * ld_weight_ic4;
    }
    vst1q_s16(dst_ptr, vcombine_s16(c[0][0], c[0][1]));
    vst1q_s16(dst_ptr + 8, vcombine_s16(c[0][2], c[0][3]));
}
template <BiasMode bias_mode, int remain_w, int filter_size>
static void ker_neon_dirctconv_3x3s2_oc4_ow4_remain(const int8_t* src_ptr,
                                                    const int8_t* weight_ptr,
                                                    const int16_t* bias_ptr,
                                                    int16_t* dst_ptr, int ic,
                                                    int ih, int iw,
                                                    int /*ld_dst_oc*/) {
    constexpr int fh = filter_size;
    constexpr int fw = filter_size;
    constexpr int ic_step = 4;
    constexpr int loop_ic_step = 4;
    constexpr int ld_weight_ic4 = 16;

    const int ic_stride = ih * iw;
    static const uint8_t idx_buffer[16] = {0, 0, 0, 0, 1, 1, 1, 1,
                                           2, 2, 2, 2, 3, 3, 3, 3};
    static uint8x16_t idx = vld1q_u8(idx_buffer);

    int16x4_t c[1][4];
    int8x16_t weight[3];
    int8x16_t src[5];
    INIT_SUM();
#define cb(_i) c[0][_i] = init_sum;

    UNROLL_CALL_RAW(4, cb);

#undef cb
    for (int ic_idx = 0; ic_idx < ic; ic_idx += loop_ic_step) {
        for (int fh_idx = 0; fh_idx < fh; ++fh_idx) {
            const int8_t* src_ic_0_3 =
                    src_ptr + ic_idx * ic_stride + fh_idx * iw * ic_step;

            src[0] = vld_dup_tbl_s32(src_ic_0_3, idx);
            src[1] = vld_dup_tbl_s32(src_ic_0_3 + 4, idx);
            src[2] = vld_dup_tbl_s32(src_ic_0_3 + 8, idx);
            src[3] = vld_dup_tbl_s32(src_ic_0_3 + 12, idx);
            src[4] = vld_dup_tbl_s32(src_ic_0_3 + 16, idx);
            const int8_t* read_weight_ptr =
                    weight_ptr + fh_idx * fw * ld_weight_ic4;

            weight[0] = vld1q_s8(read_weight_ptr);
            weight[1] = vld1q_s8(read_weight_ptr + 16);
            weight[2] = vld1q_s8(read_weight_ptr + 2 * 16);

            CALC_ONE_RESULT(src[0], src[1], src[2], weight, c[0][0]);

            src[0] = vld_dup_tbl_s32(src_ic_0_3 + 20, idx);
            src[1] = vld_dup_tbl_s32(src_ic_0_3 + 24, idx);
            CALC_ONE_RESULT(src[2], src[3], src[4], weight, c[0][1]);

            src[2] = vld_dup_tbl_s32(src_ic_0_3 + 28, idx);
            src[3] = vld_dup_tbl_s32(src_ic_0_3 + 32, idx);

            CALC_ONE_RESULT(src[4], src[0], src[1], weight, c[0][2]);

            CALC_ONE_RESULT(src[1], src[2], src[3], weight, c[0][3]);
        }
        weight_ptr += fh * fw * ld_weight_ic4;
    }
    STORE_1_LINE_RESULT_OW4();
}

#undef CALC_ONE_RESULT

template <BiasMode bias_mode>
void conv_direct_stride2_2x2_int8_nchw44(const int8_t* src,
                                         const int8_t* filter,
                                         const int16_t* bias, int16_t* dst,
                                         const size_t oc, const size_t ic,
                                         const size_t ih, const size_t iw,
                                         const size_t oh, const size_t ow) {
    constexpr size_t filter_size = 2;
    constexpr size_t fh = filter_size;
    constexpr size_t fw = filter_size;
    constexpr size_t ic_step = 4;
    constexpr size_t oc_step = 4;
    constexpr size_t big_oc_step = 8;
    constexpr size_t oh_step = 1;
    constexpr size_t ow_step = 8;
    constexpr size_t stride_h = 2;
    constexpr size_t stride_w = 2;

    const size_t out_img_stride = oh * ow;
    const size_t ow_end = ow / ow_step * ow_step;
    const size_t ow_remain = ow - ow_end;
    const size_t oc_end = oc / big_oc_step * big_oc_step;
    const size_t oc_remain = oc - oc_end;
    const int ld_dst_oc = oh * ow * oc_step;

    using remain_fun =
            std::function<void(const int8_t* src_ptr, const int8_t* weight_ptr,
                               const int16_t* bias_ptr, int16_t* dst_ptr,
                               int ic, int ih, int iw, int ld_dst_oc)>;
    remain_fun kern_big_oc_remain = nullptr;
    remain_fun kern_small_oc_remain = nullptr;

    switch (ow_remain) {
#define cb(step)                                                               \
    case step:                                                                 \
        kern_big_oc_remain = ker_neon_dirctconv_2x2s2_oc8_ow8<bias_mode, step, \
                                                              filter_size>;    \
        kern_small_oc_remain =                                                 \
                ker_neon_dirctconv_2x2s2_oc4_ow8<bias_mode, step,              \
                                                 filter_size>;                 \
        break;

        UNROLL_CALL_RAW(8, cb);
        default:
            megdnn_assert(0, "no remain %zu for kern", ow_remain);
    }
#undef cb

    for (size_t oc_idx = 0; oc_idx < oc_end; oc_idx += big_oc_step) {
        const size_t weight_offset = oc_idx * ic * fh * fw;
        for (size_t oh_idx = 0; oh_idx < oh; oh_idx += oh_step) {
            for (size_t ow_idx = 0; ow_idx < ow_end; ow_idx += ow_step) {
                const size_t src_offset =
                        (oh_idx * stride_h * iw + ow_idx * stride_w) * ic_step;
                const size_t dst_offset = oc_idx * out_img_stride +
                                          (oh_idx * ow + ow_idx) * oc_step;
                ker_neon_dirctconv_2x2s2_oc8_ow8<bias_mode, ow_step,
                                                 filter_size>(
                        src + src_offset, filter + weight_offset, bias + oc_idx,
                        dst + dst_offset, ic, ih, iw, ld_dst_oc);
            }
            if (ow_remain > 0) {
                const size_t src_offset =
                        (oh_idx * stride_h * iw + ow_end * stride_w) * ic_step;
                const size_t dst_offset = oc_idx * out_img_stride +
                                          (oh_idx * ow + ow_end) * oc_step;
                kern_big_oc_remain(src + src_offset, filter + weight_offset,
                                   bias + oc_idx, dst + dst_offset, ic, ih, iw,
                                   ld_dst_oc);
            }
        }
    }

    if (oc_remain > 0) {
        const size_t oc_idx = oc_end;
        const size_t weight_offset = oc_idx * ic * fh * fw;
        for (size_t oh_idx = 0; oh_idx < oh; oh_idx += oh_step) {
            for (size_t ow_idx = 0; ow_idx < ow_end; ow_idx += ow_step) {
                const size_t src_offset =
                        (oh_idx * stride_h * iw + ow_idx * stride_w) * ic_step;
                const size_t dst_offset = oc_idx * out_img_stride +
                                          (oh_idx * ow + ow_idx) * oc_step;
                ker_neon_dirctconv_2x2s2_oc4_ow8<bias_mode, ow_step,
                                                 filter_size>(
                        src + src_offset, filter + weight_offset, bias + oc_idx,
                        dst + dst_offset, ic, ih, iw, ld_dst_oc);
            }
            if (ow_remain > 0) {
                const size_t src_offset =
                        (oh_idx * stride_h * iw + ow_end * stride_w) * ic_step;
                const size_t dst_offset = oc_idx * out_img_stride +
                                          (oh_idx * ow + ow_end) * oc_step;
                kern_small_oc_remain(src + src_offset, filter + weight_offset,
                                     bias + oc_idx, dst + dst_offset, ic, ih,
                                     iw, ld_dst_oc);
            }
        }
    }
}

template <BiasMode bias_mode>
void conv_direct_stride2_3x3_int8_nchw44(const int8_t* src,
                                         const int8_t* filter,
                                         const int16_t* bias, int16_t* dst,
                                         const size_t oc, const size_t ic,
                                         const size_t ih, const size_t iw,
                                         const size_t oh, const size_t ow) {
    constexpr size_t filter_size = 3;
    constexpr size_t fh = filter_size;
    constexpr size_t fw = filter_size;
    constexpr size_t ic_step = 4;
    constexpr size_t oc_step = 4;
    constexpr size_t big_oc_step = 8;
    constexpr size_t oh_step = 1;
    constexpr size_t ow_step = 4;
    constexpr size_t ow_step4 = 4;
    constexpr size_t stride_h = 2;
    constexpr size_t stride_w = 2;

    const size_t out_img_stride = oh * ow;
    const size_t ow_end = ow / ow_step * ow_step;
    const size_t ow_remain = ow - ow_end;
    const size_t oc_end = oc / big_oc_step * big_oc_step;
    const size_t oc_remain = oc - oc_end;
    const int ld_dst_oc = oh * ow * oc_step;

    using remain_fun =
            std::function<void(const int8_t* src_ptr, const int8_t* weight_ptr,
                               const int16_t* bias_ptr, int16_t* dst_ptr,
                               int ic, int ih, int iw, int ld_dst_oc)>;
    remain_fun kern_big_oc_remain = nullptr;
    remain_fun kern_small_oc_remain = nullptr;

    switch (ow_remain) {
#define cb(step)                                                         \
    case step:                                                           \
        kern_big_oc_remain =                                             \
                ker_neon_dirctconv_3x3s2_oc8_ow4_remain<bias_mode, step, \
                                                        filter_size>;    \
        kern_small_oc_remain =                                           \
                ker_neon_dirctconv_3x3s2_oc4_ow4_remain<bias_mode, step, \
                                                        filter_size>;    \
        break;

        UNROLL_CALL_RAW(8, cb);
        default:
            megdnn_assert(0, "no remain %zu for kern", ow_remain);
    }
#undef cb

    for (size_t oc_idx = 0; oc_idx < oc_end; oc_idx += big_oc_step) {
        const size_t weight_offset = oc_idx * ic * fh * fw;
        for (size_t oh_idx = 0; oh_idx < oh; oh_idx += oh_step) {
            for (size_t ow_idx = 0; ow_idx < ow_end; ow_idx += ow_step4) {
                const size_t src_offset =
                        (oh_idx * stride_h * iw + ow_idx * stride_w) * ic_step;
                const size_t dst_offset = oc_idx * out_img_stride +
                                          (oh_idx * ow + ow_idx) * oc_step;
                ker_neon_dirctconv_3x3s2_oc8_ow4<bias_mode, ow_step,
                                                 filter_size>(
                        src + src_offset, filter + weight_offset, bias + oc_idx,
                        dst + dst_offset, ic, ih, iw, ld_dst_oc);
            }
            if (ow_remain > 0) {
                const size_t src_offset =
                        (oh_idx * stride_h * iw + ow_end * stride_w) * ic_step;
                const size_t dst_offset = oc_idx * out_img_stride +
                                          (oh_idx * ow + ow_end) * oc_step;
                kern_big_oc_remain(src + src_offset, filter + weight_offset,
                                   bias + oc_idx, dst + dst_offset, ic, ih, iw,
                                   ld_dst_oc);
            }
        }
    }

    if (oc_remain > 0) {
        const size_t oc_idx = oc_end;
        const size_t weight_offset = oc_idx * ic * fh * fw;
        for (size_t oh_idx = 0; oh_idx < oh; oh_idx += oh_step) {
            for (size_t ow_idx = 0; ow_idx < ow_end; ow_idx += ow_step) {
                const size_t src_offset =
                        (oh_idx * stride_h * iw + ow_idx * stride_w) * ic_step;
                const size_t dst_offset = oc_idx * out_img_stride +
                                          (oh_idx * ow + ow_idx) * oc_step;
                ker_neon_dirctconv_3x3s2_oc4_ow4<bias_mode, ow_step,
                                                 filter_size>(
                        src + src_offset, filter + weight_offset, bias + oc_idx,
                        dst + dst_offset, ic, ih, iw, ld_dst_oc);
            }
            if (ow_remain > 0) {
                const size_t src_offset =
                        (oh_idx * stride_h * iw + ow_end * stride_w) * ic_step;
                const size_t dst_offset = oc_idx * out_img_stride +
                                          (oh_idx * ow + ow_end) * oc_step;
                kern_small_oc_remain(src + src_offset, filter + weight_offset,
                                     bias + oc_idx, dst + dst_offset, ic, ih,
                                     iw, ld_dst_oc);
            }
        }
    }
}
#undef CALC_ONE_RESULT
#undef LOAD_SRC
template <BiasMode bias_mode>
void conv_direct_stride2_5x5_int8x8x16_nchw44(
        const int8_t* src, const int8_t* filter, const int16_t* bias,
        int16_t* dst, const size_t oc, const size_t ic, const size_t ih,
        const size_t iw, const size_t oh, const size_t ow) {
    constexpr size_t filter_size = 5;
    constexpr size_t fh = filter_size;
    constexpr size_t fw = filter_size;
    constexpr size_t ic_step = 4;
    constexpr size_t oc_step = 4;
    constexpr size_t oh_step1 = 1;
    constexpr size_t ow_step = 4;
    constexpr size_t stride_h = 2;
    constexpr size_t stride_w = 2;
    const size_t remain_w = ow & 3;

    const size_t out_img_stride = oh * ow;
    static const uint8_t idx_buffer[16] = {0, 0, 0, 0, 1, 1, 1, 1,
                                           2, 2, 2, 2, 3, 3, 3, 3};
    static uint8x16_t idx = vld1q_u8(idx_buffer);
    size_t oc_idx = 0;

    for (; oc_idx + 3 < oc; oc_idx += oc_step) {
        const size_t weight_offset = oc_idx * ic * fh * fw;
        const int16_t* bias_ptr = bias + oc_idx;

        int16x4_t init_sum;

        if (bias_mode == BiasMode::NO_BIAS) {
            init_sum = vdup_n_s16(0);
        } else {
            init_sum = vld1_s16(bias_ptr);
        }
        size_t oh_idx = 0;

#define CALC_ONE_RESULT(_src0, _src1, _src2, _src3, _src4, _w, _c)            \
    do {                                                                      \
        tmp0 = vmull_s8(vget_low_s8(_src0), vget_low_s8(_w[0]));              \
        tmp1 = vmull_s8(vget_high_s8(_src0), vget_high_s8(_w[0]));            \
        tmp0 = vmlal_s8(tmp0, vget_low_s8(_src1), vget_low_s8(_w[1]));        \
        tmp1 = vmlal_s8(tmp1, vget_high_s8(_src1), vget_high_s8(_w[1]));      \
        tmp0 = vmlal_s8(tmp0, vget_low_s8(_src2), vget_low_s8(_w[2]));        \
        tmp1 = vmlal_s8(tmp1, vget_high_s8(_src2), vget_high_s8(_w[2]));      \
        tmp0 = vmlal_s8(tmp0, vget_low_s8(_src3), vget_low_s8(_w[3]));        \
        tmp1 = vmlal_s8(tmp1, vget_high_s8(_src3), vget_high_s8(_w[3]));      \
        tmp0 = vmlal_s8(tmp0, vget_low_s8(_src4), vget_low_s8(_w[4]));        \
        tmp1 = vmlal_s8(tmp1, vget_high_s8(_src4), vget_high_s8(_w[4]));      \
        tmp0 = vaddq_s16(tmp0, tmp1);                                         \
        _c = vadd_s16(_c, vadd_s16(vget_low_s16(tmp0), vget_high_s16(tmp0))); \
    } while (0);

        for (; oh_idx < oh; oh_idx += oh_step1) {
            size_t ow_idx = 0;
            for (; ow_idx + ow_step - 1 < ow; ow_idx += ow_step) {
                const size_t src_offset =
                        (oh_idx * stride_h * iw + ow_idx * stride_w) * ic_step;
                const size_t dst_offset = oc_idx * out_img_stride +
                                          (oh_idx * ow + ow_idx) * oc_step;

                constexpr int ld_weight_ic4 = 16;

                const int ic_stride = ih * iw;
                int16x4_t c[1][4];
                const int8_t* src_ptr = src + src_offset;
                int16_t* dst_ptr = dst + dst_offset;
                const int8_t* weight_ptr = filter + weight_offset;

                c[0][0] = init_sum;
                c[0][1] = init_sum;
                c[0][2] = init_sum;
                c[0][3] = init_sum;
#if MEGDNN_AARCH64 
                int8x16_t weight[3][5];
                int8x16_t ssrc[2][5];
#else
                int8x16_t weight[1][5];
                int8x16_t ssrc[1][9];
#endif
                for (size_t ic_idx = 0; ic_idx < ic; ic_idx += ic_step) {
                    const int8_t* src_row0 =
                            src_ptr + ic_idx * ic_stride + 0 * iw * ic_step;
                    const int8_t* src_row1 =
                            src_ptr + ic_idx * ic_stride + 1 * iw * ic_step;
                    const int8_t* src_row2 =
                            src_ptr + ic_idx * ic_stride + 2 * iw * ic_step;
                    const int8_t* src_row3 =
                            src_ptr + ic_idx * ic_stride + 3 * iw * ic_step;
                    const int8_t* src_row4 =
                            src_ptr + ic_idx * ic_stride + 4 * iw * ic_step;
#if MEGDNN_AARCH64

#define LOAD_SRC(_src, _src_ptr)                   \
    _src[0] = vld_dup_tbl_s32(_src_ptr, idx);      \
    _src[1] = vld_dup_tbl_s32(_src_ptr + 4, idx);  \
    _src[2] = vld_dup_tbl_s32(_src_ptr + 8, idx);  \
    _src[3] = vld_dup_tbl_s32(_src_ptr + 12, idx); \
    _src[4] = vld_dup_tbl_s32(_src_ptr + 16, idx);

#define LOAD_WEIGHT(_w, _w_ptr, _id0, _id1, _id2, _id3, _id4) \
    _w[0] = vld1q_s8(_w_ptr + _id0 * 16);                     \
    _w[1] = vld1q_s8(_w_ptr + _id1 * 16);                     \
    _w[2] = vld1q_s8(_w_ptr + _id2 * 16);                     \
    _w[3] = vld1q_s8(_w_ptr + _id3 * 16);                     \
    _w[4] = vld1q_s8(_w_ptr + _id4 * 16);

#define CALC_4_RESULT(_src, _w, _src_ptr)                                      \
    CALC_ONE_RESULT(_src[0], _src[1], _src[2], _src[3], _src[4], _w, c[0][0]); \
    _src[0] = vld_dup_tbl_s32(_src_ptr + 20, idx);                             \
    _src[1] = vld_dup_tbl_s32(_src_ptr + 24, idx);                             \
    CALC_ONE_RESULT(_src[2], _src[3], _src[4], _src[0], _src[1], _w, c[0][1]); \
    _src[2] = vld_dup_tbl_s32(_src_ptr + 28, idx);                             \
    _src[3] = vld_dup_tbl_s32(_src_ptr + 32, idx);                             \
    CALC_ONE_RESULT(_src[4], _src[0], _src[1], _src[2], _src[3], _w, c[0][2]); \
    _src[4] = vld_dup_tbl_s32(_src_ptr + 36, idx);                             \
    _src[0] = vld_dup_tbl_s32(_src_ptr + 40, idx);                             \
    CALC_ONE_RESULT(_src[1], _src[2], _src[3], _src[4], _src[0], _w, c[0][3]);

                    int16x8_t tmp0, tmp1;

                    LOAD_SRC(ssrc[0], src_row0);
                    LOAD_WEIGHT(weight[0], weight_ptr, 0, 1, 2, 3, 4);
                    LOAD_WEIGHT(weight[1], weight_ptr, 5, 6, 7, 8, 9);
                    CALC_4_RESULT(ssrc[0], weight[0], src_row0);
                    
                    LOAD_SRC(ssrc[1], src_row1);
                    LOAD_WEIGHT(weight[2], weight_ptr, 10, 11, 12, 13, 14);
                    LOAD_SRC(ssrc[0], src_row2);
                    CALC_4_RESULT(ssrc[1], weight[1], src_row1);

                    LOAD_SRC(ssrc[1], src_row3);
                    LOAD_WEIGHT(weight[0], weight_ptr, 15, 16, 17, 18, 19);
                    CALC_4_RESULT(ssrc[0], weight[2], src_row2);
                    
                    LOAD_SRC(ssrc[0], src_row4);
                    LOAD_WEIGHT(weight[1], weight_ptr, 20, 21, 22, 23, 24);
                    CALC_4_RESULT(ssrc[1], weight[0], src_row3);
                    CALC_4_RESULT(ssrc[0], weight[1], src_row4);
#else

#define LOAD_SRC(_src_ptr)                            \
    ssrc[0][0] = vld_dup_tbl_s32(_src_ptr, idx);      \
    ssrc[0][1] = vld_dup_tbl_s32(_src_ptr + 4, idx);  \
    ssrc[0][2] = vld_dup_tbl_s32(_src_ptr + 8, idx);  \
    ssrc[0][3] = vld_dup_tbl_s32(_src_ptr + 12, idx); \
    ssrc[0][4] = vld_dup_tbl_s32(_src_ptr + 16, idx); \
    ssrc[0][5] = vld_dup_tbl_s32(_src_ptr + 20, idx); \
    ssrc[0][6] = vld_dup_tbl_s32(_src_ptr + 24, idx); \
    ssrc[0][7] = vld_dup_tbl_s32(_src_ptr + 28, idx); \
    ssrc[0][8] = vld_dup_tbl_s32(_src_ptr + 32, idx);

#define LOAD_WEIGHT(_w_ptr, _id0, _id1, _id2, _id3, _id4) \
    weight[0][0] = vld1q_s8(_w_ptr + _id0 * 16);          \
    weight[0][1] = vld1q_s8(_w_ptr + _id1 * 16);          \
    weight[0][2] = vld1q_s8(_w_ptr + _id2 * 16);          \
    weight[0][3] = vld1q_s8(_w_ptr + _id3 * 16);          \
    weight[0][4] = vld1q_s8(_w_ptr + _id4 * 16);

#define CALC_4_RESULT(_src_ptr)                                     \
    CALC_ONE_RESULT(ssrc[0][0], ssrc[0][1], ssrc[0][2], ssrc[0][3], \
                    ssrc[0][4], weight[0], c[0][0]);                \
    ssrc[0][0] = vld_dup_tbl_s32(_src_ptr + 36, idx);               \
    ssrc[0][1] = vld_dup_tbl_s32(_src_ptr + 40, idx);               \
    CALC_ONE_RESULT(ssrc[0][2], ssrc[0][3], ssrc[0][4], ssrc[0][5], \
                    ssrc[0][6], weight[0], c[0][1]);                \
    CALC_ONE_RESULT(ssrc[0][4], ssrc[0][5], ssrc[0][6], ssrc[0][7], \
                    ssrc[0][8], weight[0], c[0][2]);                \
    CALC_ONE_RESULT(ssrc[0][6], ssrc[0][7], ssrc[0][8], ssrc[0][0], \
                    ssrc[0][1], weight[0], c[0][3]);

                    int16x8_t tmp0, tmp1;
                    
                    LOAD_WEIGHT(weight_ptr, 0, 1, 2, 3, 4);
                    LOAD_SRC(src_row0);
                    CALC_4_RESULT(src_row0);

                    LOAD_WEIGHT(weight_ptr, 5, 6, 7, 8, 9);
                    LOAD_SRC(src_row1);
                    CALC_4_RESULT(src_row1);
                    
                    LOAD_WEIGHT(weight_ptr, 10, 11, 12, 13, 14);
                    LOAD_SRC(src_row2);
                    CALC_4_RESULT(src_row2);
                    
                    LOAD_WEIGHT(weight_ptr, 15, 16, 17, 18, 19);
                    LOAD_SRC(src_row3);
                    CALC_4_RESULT(src_row3);
                    
                    LOAD_WEIGHT(weight_ptr, 20, 21, 22, 23, 24);
                    LOAD_SRC(src_row4);
                    CALC_4_RESULT(src_row4);
#endif
                    weight_ptr += fh * fw * ld_weight_ic4;
                }

                vst1q_s16(dst_ptr, vcombine_s16(c[0][0], c[0][1]));
                vst1q_s16(dst_ptr + 8, vcombine_s16(c[0][2], c[0][3]));
            }
            if (remain_w > 0) {
                const size_t src_offset =
                        (oh_idx * stride_h * iw + ow_idx * stride_w) * ic_step;
                const size_t dst_offset = oc_idx * out_img_stride +
                                          (oh_idx * ow + ow_idx) * oc_step;

                constexpr int ld_weight_ic4 = 16;

                const int ic_stride = ih * iw;
                int16x4_t c[1][3];
                const int8_t* src_ptr = src + src_offset;
                int16_t* dst_ptr = dst + dst_offset;
                const int8_t* weight_ptr = filter + weight_offset;

                c[0][0] = init_sum;
                c[0][1] = init_sum;
                c[0][2] = init_sum;
#if MEGDNN_AARCH64
                int8x16_t weight[3][5];
                int8x16_t ssrc[2][5];
#else
                int8x16_t weight[1][5];
                int8x16_t ssrc[1][9];
#endif
                for (size_t ic_idx = 0; ic_idx < ic; ic_idx += ic_step) {
                    const int8_t* src_row0 =
                            src_ptr + ic_idx * ic_stride + 0 * iw * ic_step;
                    const int8_t* src_row1 =
                            src_ptr + ic_idx * ic_stride + 1 * iw * ic_step;
                    const int8_t* src_row2 =
                            src_ptr + ic_idx * ic_stride + 2 * iw * ic_step;
                    const int8_t* src_row3 =
                            src_ptr + ic_idx * ic_stride + 3 * iw * ic_step;
                    const int8_t* src_row4 =
                            src_ptr + ic_idx * ic_stride + 4 * iw * ic_step;
#if MEGDNN_AARCH64

#define LOAD_SRC(_src, _src_ptr)                   \
    _src[0] = vld_dup_tbl_s32(_src_ptr, idx);      \
    _src[1] = vld_dup_tbl_s32(_src_ptr + 4, idx);  \
    _src[2] = vld_dup_tbl_s32(_src_ptr + 8, idx);  \
    _src[3] = vld_dup_tbl_s32(_src_ptr + 12, idx); \
    _src[4] = vld_dup_tbl_s32(_src_ptr + 16, idx);

#define LOAD_WEIGHT(_w, _w_ptr, _id0, _id1, _id2, _id3, _id4) \
    _w[0] = vld1q_s8(_w_ptr + _id0 * 16);                     \
    _w[1] = vld1q_s8(_w_ptr + _id1 * 16);                     \
    _w[2] = vld1q_s8(_w_ptr + _id2 * 16);                     \
    _w[3] = vld1q_s8(_w_ptr + _id3 * 16);                     \
    _w[4] = vld1q_s8(_w_ptr + _id4 * 16);

#define CALC_3_RESULT(_src, _w, _src_ptr)                                      \
    CALC_ONE_RESULT(_src[0], _src[1], _src[2], _src[3], _src[4], _w, c[0][0]); \
    _src[0] = vld_dup_tbl_s32(_src_ptr + 20, idx);                             \
    _src[1] = vld_dup_tbl_s32(_src_ptr + 24, idx);                             \
    CALC_ONE_RESULT(_src[2], _src[3], _src[4], _src[0], _src[1], _w, c[0][1]); \
    _src[2] = vld_dup_tbl_s32(_src_ptr + 28, idx);                             \
    _src[3] = vld_dup_tbl_s32(_src_ptr + 32, idx);                             \
    CALC_ONE_RESULT(_src[4], _src[0], _src[1], _src[2], _src[3], _w, c[0][2]);

                    int16x8_t tmp0, tmp1;

                    LOAD_SRC(ssrc[0], src_row0);
                    LOAD_WEIGHT(weight[0], weight_ptr, 0, 1, 2, 3, 4);
                    LOAD_WEIGHT(weight[1], weight_ptr, 5, 6, 7, 8, 9);
                    CALC_3_RESULT(ssrc[0], weight[0], src_row0);

                    LOAD_SRC(ssrc[1], src_row1);
                    LOAD_WEIGHT(weight[2], weight_ptr, 10, 11, 12, 13, 14);
                    LOAD_SRC(ssrc[0], src_row2);
                    CALC_3_RESULT(ssrc[1], weight[1], src_row1);

                    LOAD_SRC(ssrc[1], src_row3);
                    LOAD_WEIGHT(weight[0], weight_ptr, 15, 16, 17, 18, 19);
                    CALC_3_RESULT(ssrc[0], weight[2], src_row2);

                    LOAD_SRC(ssrc[0], src_row4);
                    LOAD_WEIGHT(weight[1], weight_ptr, 20, 21, 22, 23, 24);
                    CALC_3_RESULT(ssrc[1], weight[0], src_row3);
                    CALC_3_RESULT(ssrc[0], weight[1], src_row4);
#else

#define LOAD_SRC(_src_ptr)                            \
    ssrc[0][0] = vld_dup_tbl_s32(_src_ptr, idx);      \
    ssrc[0][1] = vld_dup_tbl_s32(_src_ptr + 4, idx);  \
    ssrc[0][2] = vld_dup_tbl_s32(_src_ptr + 8, idx);  \
    ssrc[0][3] = vld_dup_tbl_s32(_src_ptr + 12, idx); \
    ssrc[0][4] = vld_dup_tbl_s32(_src_ptr + 16, idx); \
    ssrc[0][5] = vld_dup_tbl_s32(_src_ptr + 20, idx); \
    ssrc[0][6] = vld_dup_tbl_s32(_src_ptr + 24, idx); \
    ssrc[0][7] = vld_dup_tbl_s32(_src_ptr + 28, idx); \
    ssrc[0][8] = vld_dup_tbl_s32(_src_ptr + 32, idx);

#define LOAD_WEIGHT(_w_ptr, _id0, _id1, _id2, _id3, _id4) \
    weight[0][0] = vld1q_s8(_w_ptr + _id0 * 16);          \
    weight[0][1] = vld1q_s8(_w_ptr + _id1 * 16);          \
    weight[0][2] = vld1q_s8(_w_ptr + _id2 * 16);          \
    weight[0][3] = vld1q_s8(_w_ptr + _id3 * 16);          \
    weight[0][4] = vld1q_s8(_w_ptr + _id4 * 16);

#define CALC_3_RESULT(_src_ptr)                                     \
    CALC_ONE_RESULT(ssrc[0][0], ssrc[0][1], ssrc[0][2], ssrc[0][3], \
                    ssrc[0][4], weight[0], c[0][0]);                \
    CALC_ONE_RESULT(ssrc[0][2], ssrc[0][3], ssrc[0][4], ssrc[0][5], \
                    ssrc[0][6], weight[0], c[0][1]);                \
    CALC_ONE_RESULT(ssrc[0][4], ssrc[0][5], ssrc[0][6], ssrc[0][7], \
                    ssrc[0][8], weight[0], c[0][2]);

                    int16x8_t tmp0, tmp1;

                    LOAD_WEIGHT(weight_ptr, 0, 1, 2, 3, 4);
                    LOAD_SRC(src_row0);
                    CALC_3_RESULT(src_row0);

                    LOAD_WEIGHT(weight_ptr, 5, 6, 7, 8, 9);
                    LOAD_SRC(src_row1);
                    CALC_3_RESULT(src_row1);

                    LOAD_WEIGHT(weight_ptr, 10, 11, 12, 13, 14);
                    LOAD_SRC(src_row2);
                    CALC_3_RESULT(src_row2);

                    LOAD_WEIGHT(weight_ptr, 15, 16, 17, 18, 19);
                    LOAD_SRC(src_row3);
                    CALC_3_RESULT(src_row3);

                    LOAD_WEIGHT(weight_ptr, 20, 21, 22, 23, 24);
                    LOAD_SRC(src_row4);
                    CALC_3_RESULT(src_row4);
#endif
                    weight_ptr += fh * fw * ld_weight_ic4;
                }
                switch (remain_w) {
                    case 1:
                        vst1_s16(dst_ptr, c[0][0]);
                        break;
                    case 2:
                        vst1q_s16(dst_ptr, vcombine_s16(c[0][0], c[0][1]));
                        break;
                    case 3:
                        vst1q_s16(dst_ptr, vcombine_s16(c[0][0], c[0][1]));
                        vst1_s16(dst_ptr + 8, c[0][2]);
                        break;
                    default:
                        megdnn_throw("invalid remain_w");
                        break;
                }
            }
        }
    }
}
#undef CALC_4_RESULT
#undef LOAD_SRC
#undef LOAD_WEIGHT
#undef CALC_ONE_RESULT

template <BiasMode bias_mode>
void conv_direct_stride2_7x7_int8x8x16_nchw44(
        const int8_t* src, const int8_t* filter, const int16_t* bias,
        int16_t* dst, const size_t oc, const size_t ic, const size_t ih,
        const size_t iw, const size_t oh, const size_t ow) {
    constexpr size_t filter_size = 7;
    constexpr size_t fh = filter_size;
    constexpr size_t fw = filter_size;
    constexpr size_t ic_step = 4;
    constexpr size_t oc_step = 4;
    constexpr size_t oh_step1 = 1;
    constexpr size_t ow_step = 4;
    constexpr size_t stride_h = 2;
    constexpr size_t stride_w = 2;

    const size_t out_img_stride = oh * ow;
    static const uint8_t idx_buffer[16] = {0, 0, 0, 0, 1, 1, 1, 1,
                                           2, 2, 2, 2, 3, 3, 3, 3};
    static uint8x16_t idx = vld1q_u8(idx_buffer);
    size_t oc_idx = 0;

    for (; oc_idx + 3 < oc; oc_idx += oc_step) {
        const size_t weight_offset = oc_idx * ic * fh * fw;
        const int16_t* bias_ptr = bias + oc_idx;

        int16x4_t init_sum;

        if (bias_mode == BiasMode::NO_BIAS) {
            init_sum = vdup_n_s16(0);
        } else {
            init_sum = vld1_s16(bias_ptr);
        }
        size_t oh_idx = 0;

#define CALC_ONE_RESULT(_src0, _src1, _src2, _src3, _src4, _src5, _src6, _w,  \
                        _c)                                                   \
    do {                                                                      \
        tmp0 = vmull_s8(vget_low_s8(_src0), vget_low_s8(_w[0]));              \
        tmp1 = vmull_s8(vget_high_s8(_src0), vget_high_s8(_w[0]));            \
        tmp0 = vmlal_s8(tmp0, vget_low_s8(_src1), vget_low_s8(_w[1]));        \
        tmp1 = vmlal_s8(tmp1, vget_high_s8(_src1), vget_high_s8(_w[1]));      \
        tmp0 = vmlal_s8(tmp0, vget_low_s8(_src2), vget_low_s8(_w[2]));        \
        tmp1 = vmlal_s8(tmp1, vget_high_s8(_src2), vget_high_s8(_w[2]));      \
        tmp0 = vmlal_s8(tmp0, vget_low_s8(_src3), vget_low_s8(_w[3]));        \
        tmp1 = vmlal_s8(tmp1, vget_high_s8(_src3), vget_high_s8(_w[3]));      \
        tmp0 = vmlal_s8(tmp0, vget_low_s8(_src4), vget_low_s8(_w[4]));        \
        tmp1 = vmlal_s8(tmp1, vget_high_s8(_src4), vget_high_s8(_w[4]));      \
        tmp0 = vmlal_s8(tmp0, vget_low_s8(_src5), vget_low_s8(_w[5]));        \
        tmp1 = vmlal_s8(tmp1, vget_high_s8(_src5), vget_high_s8(_w[5]));      \
        tmp0 = vmlal_s8(tmp0, vget_low_s8(_src6), vget_low_s8(_w[6]));        \
        tmp1 = vmlal_s8(tmp1, vget_high_s8(_src6), vget_high_s8(_w[6]));      \
        tmp0 = vaddq_s16(tmp0, tmp1);                                         \
        _c = vadd_s16(_c, vadd_s16(vget_low_s16(tmp0), vget_high_s16(tmp0))); \
    } while (0);

        for (; oh_idx < oh; oh_idx += oh_step1) {
            size_t ow_idx = 0;
            for (; ow_idx + ow_step - 1 < ow; ow_idx += ow_step) {
                const size_t src_offset =
                        (oh_idx * stride_h * iw + ow_idx * stride_w) * ic_step;
                const size_t dst_offset = oc_idx * out_img_stride +
                                          (oh_idx * ow + ow_idx) * oc_step;

                constexpr int ld_weight_ic4 = 16;

                const int ic_stride = ih * iw;
                int16x4_t c[1][4];
                int8x16_t weight[1][7];
                int8x16_t ssrc[1][9];
                const int8_t* src_ptr = src + src_offset;
                int16_t* dst_ptr = dst + dst_offset;
                const int8_t* weight_ptr = filter + weight_offset;

                c[0][0] = init_sum;
                c[0][1] = init_sum;
                c[0][2] = init_sum;
                c[0][3] = init_sum;
                for (size_t ic_idx = 0; ic_idx < ic; ic_idx += ic_step) {
                    const int8_t* src_row0 =
                            src_ptr + ic_idx * ic_stride + 0 * iw * ic_step;

                    const int8_t* src_row1 =
                            src_ptr + ic_idx * ic_stride + 1 * iw * ic_step;
                    const int8_t* src_row2 =
                            src_ptr + ic_idx * ic_stride + 2 * iw * ic_step;
                    const int8_t* src_row3 =
                            src_ptr + ic_idx * ic_stride + 3 * iw * ic_step;
                    const int8_t* src_row4 =
                            src_ptr + ic_idx * ic_stride + 4 * iw * ic_step;
                    const int8_t* src_row5 =
                            src_ptr + ic_idx * ic_stride + 5 * iw * ic_step;
                    const int8_t* src_row6 =
                            src_ptr + ic_idx * ic_stride + 6 * iw * ic_step;

#define LOAD_SRC(_src)                            \
    ssrc[0][0] = vld_dup_tbl_s32(_src, idx);      \
    ssrc[0][1] = vld_dup_tbl_s32(_src + 4, idx);  \
    ssrc[0][2] = vld_dup_tbl_s32(_src + 8, idx);  \
    ssrc[0][3] = vld_dup_tbl_s32(_src + 12, idx); \
    ssrc[0][4] = vld_dup_tbl_s32(_src + 16, idx); \
    ssrc[0][5] = vld_dup_tbl_s32(_src + 20, idx); \
    ssrc[0][6] = vld_dup_tbl_s32(_src + 24, idx);

#define LOAD_WEIGHT(_id0, _id1, _id2, _id3, _id4, _id5, _id6) \
    weight[0][0] = vld1q_s8(weight_ptr + _id0 * 16);          \
    weight[0][1] = vld1q_s8(weight_ptr + _id1 * 16);          \
    weight[0][2] = vld1q_s8(weight_ptr + _id2 * 16);          \
    weight[0][3] = vld1q_s8(weight_ptr + _id3 * 16);          \
    weight[0][4] = vld1q_s8(weight_ptr + _id4 * 16);          \
    weight[0][5] = vld1q_s8(weight_ptr + _id5 * 16);          \
    weight[0][6] = vld1q_s8(weight_ptr + _id6 * 16);

#define CALC_4_RESULT(_row)                                                  \
    CALC_ONE_RESULT(ssrc[0][0], ssrc[0][1], ssrc[0][2], ssrc[0][3],          \
                    ssrc[0][4], ssrc[0][5], ssrc[0][6], weight[0], c[0][0]); \
                                                                             \
    ssrc[0][7] = vld_dup_tbl_s32(_row + 28, idx);                            \
    ssrc[0][8] = vld_dup_tbl_s32(_row + 32, idx);                            \
    CALC_ONE_RESULT(ssrc[0][2], ssrc[0][3], ssrc[0][4], ssrc[0][5],          \
                    ssrc[0][6], ssrc[0][7], ssrc[0][8], weight[0], c[0][1]); \
                                                                             \
    ssrc[0][0] = vld_dup_tbl_s32(_row + 36, idx);                            \
    ssrc[0][1] = vld_dup_tbl_s32(_row + 40, idx);                            \
                                                                             \
    CALC_ONE_RESULT(ssrc[0][4], ssrc[0][5], ssrc[0][6], ssrc[0][7],          \
                    ssrc[0][8], ssrc[0][0], ssrc[0][1], weight[0], c[0][2]); \
    ssrc[0][2] = vld_dup_tbl_s32(_row + 44, idx);                            \
    ssrc[0][3] = vld_dup_tbl_s32(_row + 48, idx);                            \
                                                                             \
    CALC_ONE_RESULT(ssrc[0][6], ssrc[0][7], ssrc[0][8], ssrc[0][0],          \
                    ssrc[0][1], ssrc[0][2], ssrc[0][3], weight[0], c[0][3]);

                    int16x8_t tmp0, tmp1;

                    LOAD_SRC(src_row0);
                    LOAD_WEIGHT(0, 1, 2, 3, 4, 5, 6);
                    CALC_4_RESULT(src_row0);

                    LOAD_SRC(src_row1);
                    LOAD_WEIGHT(7, 8, 9, 10, 11, 12, 13);
                    CALC_4_RESULT(src_row1);

                    LOAD_SRC(src_row2);
                    LOAD_WEIGHT(14, 15, 16, 17, 18, 19, 20);
                    CALC_4_RESULT(src_row2);

                    LOAD_SRC(src_row3);
                    LOAD_WEIGHT(21, 22, 23, 24, 25, 26, 27);
                    CALC_4_RESULT(src_row3);

                    LOAD_SRC(src_row4);
                    LOAD_WEIGHT(28, 29, 30, 31, 32, 33, 34);
                    CALC_4_RESULT(src_row4);

                    LOAD_SRC(src_row5);
                    LOAD_WEIGHT(35, 36, 37, 38, 39, 40, 41);
                    CALC_4_RESULT(src_row5);

                    LOAD_SRC(src_row6);
                    LOAD_WEIGHT(42, 43, 44, 45, 46, 47, 48);
                    CALC_4_RESULT(src_row6);
                    weight_ptr += fh * fw * ld_weight_ic4;
                }

                vst1q_s16(dst_ptr, vcombine_s16(c[0][0], c[0][1]));
                vst1q_s16(dst_ptr + 8, vcombine_s16(c[0][2], c[0][3]));
            }
            for (; ow_idx < ow; ow_idx++) {
                const size_t src_offset =
                        (oh_idx * stride_h * iw + ow_idx * stride_w) * ic_step;
                const size_t dst_offset = oc_idx * out_img_stride +
                                          (oh_idx * ow + ow_idx) * oc_step;

                constexpr int ld_weight_ic4 = 16;

                const int ic_stride = ih * iw;
                int16x4_t c = init_sum;
                int8x16_t weight[1][7];
                int8x16_t ssrc[1][7];
                const int8_t* src_ptr = src + src_offset;
                int16_t* dst_ptr = dst + dst_offset;
                const int8_t* weight_ptr = filter + weight_offset;

                for (size_t ic_idx = 0; ic_idx < ic; ic_idx += ic_step) {
                    const int8_t* src_row0 =
                            src_ptr + ic_idx * ic_stride + 0 * iw * ic_step;

                    const int8_t* src_row1 =
                            src_ptr + ic_idx * ic_stride + 1 * iw * ic_step;
                    const int8_t* src_row2 =
                            src_ptr + ic_idx * ic_stride + 2 * iw * ic_step;
                    const int8_t* src_row3 =
                            src_ptr + ic_idx * ic_stride + 3 * iw * ic_step;
                    const int8_t* src_row4 =
                            src_ptr + ic_idx * ic_stride + 4 * iw * ic_step;
                    const int8_t* src_row5 =
                            src_ptr + ic_idx * ic_stride + 5 * iw * ic_step;
                    const int8_t* src_row6 =
                            src_ptr + ic_idx * ic_stride + 6 * iw * ic_step;
#define CALC_1_RESULT(_row)                                         \
    CALC_ONE_RESULT(ssrc[0][0], ssrc[0][1], ssrc[0][2], ssrc[0][3], \
                    ssrc[0][4], ssrc[0][5], ssrc[0][6], weight[0], c);

                    int16x8_t tmp0, tmp1;
                    LOAD_SRC(src_row0);
                    LOAD_WEIGHT(0, 1, 2, 3, 4, 5, 6);
                    CALC_1_RESULT(src_row0);

                    LOAD_SRC(src_row1);
                    LOAD_WEIGHT(7, 8, 9, 10, 11, 12, 13);
                    CALC_1_RESULT(src_row1);

                    LOAD_SRC(src_row2);
                    LOAD_WEIGHT(14, 15, 16, 17, 18, 19, 20);
                    CALC_1_RESULT(src_row2);

                    LOAD_SRC(src_row3);
                    LOAD_WEIGHT(21, 22, 23, 24, 25, 26, 27);
                    CALC_1_RESULT(src_row3);
                    LOAD_SRC(src_row4);
                    LOAD_WEIGHT(28, 29, 30, 31, 32, 33, 34);
                    CALC_1_RESULT(src_row4);
                    LOAD_SRC(src_row5);
                    LOAD_WEIGHT(35, 36, 37, 38, 39, 40, 41);
                    CALC_1_RESULT(src_row5);
                    LOAD_SRC(src_row6);
                    LOAD_WEIGHT(42, 43, 44, 45, 46, 47, 48);
                    CALC_1_RESULT(src_row6);

                    weight_ptr += fh * fw * ld_weight_ic4;
                }
                vst1_s16(dst_ptr, c);
            }
        }
    }
}
#undef CALC_ONE_RESULT
#undef CALC_1_RESULT
#undef CALC_4_RESULT
#undef LOAD_SRC
#undef LOAD_WEIGHT
}  // namespace

namespace int8x8x16_direct_nchw44 {

template <BiasMode bias_mode>
struct ConvDirectInt8Nchw44Choose<bias_mode, 2, 2> {
    static void impl(const int8_t* src, const int8_t* filter,
                     const int16_t* bias, int16_t* dst, const size_t oc,
                     const size_t ic, const size_t ih, const size_t iw,
                     const size_t oh, const size_t ow) {
        conv_direct_stride2_2x2_int8_nchw44<bias_mode>(src, filter, bias, dst,
                                                       oc, ic, ih, iw, oh, ow);
    }
};

template <BiasMode bias_mode>
struct ConvDirectInt8Nchw44Choose<bias_mode, 3, 2> {
    static void impl(const int8_t* src, const int8_t* filter,
                     const int16_t* bias, int16_t* dst, const size_t oc,
                     const size_t ic, const size_t ih, const size_t iw,
                     const size_t oh, const size_t ow) {
        conv_direct_stride2_3x3_int8_nchw44<bias_mode>(src, filter, bias, dst,
                                                       oc, ic, ih, iw, oh, ow);
    }
};

template <BiasMode bias_mode>
struct ConvDirectInt8Nchw44Choose<bias_mode, 5, 2> {
    static void impl(const int8_t* src, const int8_t* filter,
                     const int16_t* bias, int16_t* dst, const size_t oc,
                     const size_t ic, const size_t ih, const size_t iw,
                     const size_t oh, const size_t ow) {
        conv_direct_stride2_5x5_int8x8x16_nchw44<bias_mode>(
                src, filter, bias, dst, oc, ic, ih, iw, oh, ow);
    }
};
template <BiasMode bias_mode>
struct ConvDirectInt8Nchw44Choose<bias_mode, 7, 2> {
    static void impl(const int8_t* src, const int8_t* filter,
                     const int16_t* bias, int16_t* dst, const size_t oc,
                     const size_t ic, const size_t ih, const size_t iw,
                     const size_t oh, const size_t ow) {
        conv_direct_stride2_7x7_int8x8x16_nchw44<bias_mode>(
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

DISPATCH_CONV_KERN(2);

}  // namespace int8x8x16_direct_nchw44
}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
