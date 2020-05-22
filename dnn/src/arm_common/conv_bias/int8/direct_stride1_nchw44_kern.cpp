/**
 * \file dnn/src/arm_common/conv_bias/int8/direct_stride1_nchw44_kern.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/arm_common/conv_bias/int8/direct.h"
#include "src/arm_common/conv_bias/intrinsic_helper.h"
#include "src/arm_common/elemwise_op.h"
#include "src/arm_common/simd_macro/marm_neon.h"
#include "src/common/utils.h"
#include "src/fallback/conv_bias/common.h"

using namespace megdnn;
using namespace arm_common;
namespace {

/**
dot like impl. dot 4 ic to 1 oc, accumale to c <ow, oc>
example: (format like weight<oc, ic>)
packed weight
low 64 bit  <0, 0> <0, 1> <1, 2> <1, 3> | <2, 0> <2, 1> <3, 2> <3, 3>
---------------------------------------------------------------------
high 64 bit <0, 3> <0, 2> <1, 1> <1, 0> | <2, 3> <2, 2> <3, 1> <3, 0>
dot: (<0, 0> + <0, 3>) + (<0, 1> + <0, 2>) -> <0>
**/
// TODO: can try oh = 2 impl, oc = 8 impl
template <BiasMode bias_mode, typename Op, int remain_w, int filter_size>
static void ker_neon_dirctconv_3x3s1_oc4_ow8(const int8_t* src_ptr,
                                             const int8_t* weight_ptr,
                                             const int32_t* bias_ptr,
                                             int8_t* dst_ptr, int ic, int ih,
                                             int iw, const Op& op) {
    constexpr int fh = filter_size;
    constexpr int fw = filter_size;
    constexpr int ic_step = 4;
    constexpr int loop_ic_step = 4;
    constexpr int ld_weight_ic4 = 16;
    constexpr int pack_iw_len = 4;

    const int ic_stride = ih * iw * pack_iw_len;

    int32x4_t c[2 * 4];
    int8x16_t weight[3];
    int8x16_t src[8 + 2];
    int16x8_t temp_c[2];
    init_oc4_ow8<bias_mode>(c, bias_ptr);

    for (int ic_idx = 0; ic_idx < ic; ic_idx += loop_ic_step) {
        for (int fh_idx = 0; fh_idx < fh; ++fh_idx) {
            const int8_t* src_ic_0_3 = src_ptr + ic_idx * ic_stride +
                                       fh_idx * iw * ic_step * pack_iw_len;

            src[0] = vld1q_s8(src_ic_0_3);
            src[1] = vld1q_s8((src_ic_0_3 + 16));
            src[2] = vld1q_s8((src_ic_0_3 + 2 * 16));
            src[3] = vld1q_s8((src_ic_0_3 + 3 * 16));
            src[4] = vld1q_s8((src_ic_0_3 + 4 * 16));
            src[5] = vld1q_s8((src_ic_0_3 + 5 * 16));
            src[6] = vld1q_s8((src_ic_0_3 + 6 * 16));
            src[7] = vld1q_s8((src_ic_0_3 + 7 * 16));
            src[8] = vld1q_s8((src_ic_0_3 + 8 * 16));
            src[9] = vld1q_s8((src_ic_0_3 + 9 * 16));

            // oc == 0
            const int8_t* read_weight_ptr =
                    weight_ptr + fh_idx * fw * ld_weight_ic4;

            weight[0] = vld1q_s8(read_weight_ptr);
            weight[1] = vld1q_s8(read_weight_ptr + 16);
            weight[2] = vld1q_s8(read_weight_ptr + 2 * 16);

            c[0] = vdotq_s32_h(weight[0], src[0], c[0], temp_c[0]);
            c[1] = vdotq_s32_h(weight[0], src[1], c[1], temp_c[1]);
            c[0] = vdotq_s32_h(weight[1], src[1], c[0], temp_c[0]);
            c[1] = vdotq_s32_h(weight[1], src[2], c[1], temp_c[1]);
            c[0] = vdotq_s32_h(weight[2], src[2], c[0], temp_c[0]);
            c[1] = vdotq_s32_h(weight[2], src[3], c[1], temp_c[1]);

            c[2] = vdotq_s32_h(weight[0], src[2], c[2], temp_c[0]);
            c[3] = vdotq_s32_h(weight[0], src[3], c[3], temp_c[1]);
            c[2] = vdotq_s32_h(weight[1], src[3], c[2], temp_c[0]);
            c[3] = vdotq_s32_h(weight[1], src[4], c[3], temp_c[1]);
            c[2] = vdotq_s32_h(weight[2], src[4], c[2], temp_c[0]);
            c[3] = vdotq_s32_h(weight[2], src[5], c[3], temp_c[1]);

            c[4] = vdotq_s32_h(weight[0], src[4], c[4], temp_c[0]);
            c[5] = vdotq_s32_h(weight[0], src[5], c[5], temp_c[1]);
            c[4] = vdotq_s32_h(weight[1], src[5], c[4], temp_c[0]);
            c[5] = vdotq_s32_h(weight[1], src[6], c[5], temp_c[1]);
            c[4] = vdotq_s32_h(weight[2], src[6], c[4], temp_c[0]);
            c[5] = vdotq_s32_h(weight[2], src[7], c[5], temp_c[1]);

            c[6] = vdotq_s32_h(weight[0], src[6], c[6], temp_c[0]);
            c[7] = vdotq_s32_h(weight[0], src[7], c[7], temp_c[1]);
            c[6] = vdotq_s32_h(weight[1], src[7], c[6], temp_c[0]);
            c[7] = vdotq_s32_h(weight[1], src[8], c[7], temp_c[1]);
            c[6] = vdotq_s32_h(weight[2], src[8], c[6], temp_c[0]);
            c[7] = vdotq_s32_h(weight[2], src[9], c[7], temp_c[1]);
        }
        weight_ptr += fh * fw * ld_weight_ic4;
    }

    store_oc4_ow8_remain_static<remain_w, Op>(c, op, dst_ptr);
}

template <BiasMode bias_mode, typename Op, int remain_w, int filter_size>
static void ker_neon_dirctconv_2x2s1_oc8_ow8(const int8_t* src_ptr,
                                             const int8_t* weight_ptr,
                                             const int32_t* bias_ptr,
                                             int8_t* dst_ptr, int ic, int ih,
                                             int iw, int ld_dst_oc,
                                             const Op& op) {
    constexpr int fh = filter_size;
    constexpr int fw = filter_size;
    constexpr int ic_step = 4;
    constexpr int oc_step = 4;
    constexpr int loop_ic_step = 4;
    constexpr int ld_weight_ic4 = 16;
    constexpr int pack_iw_len = 4;

    const int ic_stride = ih * iw * pack_iw_len;
    const int ld_weight_oc4 = oc_step * fh * fw * ic;

    int32x4_t c[2][8];
    int8x16_t weight[2][2];
    int8x16_t src[8 + 1];
    int16x8_t temp_c[4];

    init_oc8_ow8<bias_mode>(c, bias_ptr, oc_step);

    for (int ic_idx = 0; ic_idx < ic; ic_idx += loop_ic_step) {
        for (int fh_idx = 0; fh_idx < fh; ++fh_idx) {
            const int8_t* src_ic_0_3 = src_ptr + ic_idx * ic_stride +
                                       fh_idx * iw * ic_step * pack_iw_len;

            src[0] = vld1q_s8(src_ic_0_3);
            src[1] = vld1q_s8((src_ic_0_3 + 16));
            src[2] = vld1q_s8((src_ic_0_3 + 2 * 16));
            src[3] = vld1q_s8((src_ic_0_3 + 3 * 16));
            src[4] = vld1q_s8((src_ic_0_3 + 4 * 16));
            src[5] = vld1q_s8((src_ic_0_3 + 5 * 16));
            src[6] = vld1q_s8((src_ic_0_3 + 6 * 16));
            src[7] = vld1q_s8((src_ic_0_3 + 7 * 16));
            src[8] = vld1q_s8((src_ic_0_3 + 8 * 16));

            // oc == 0
            const int8_t* read_weight_ptr =
                    weight_ptr + fh_idx * fw * ld_weight_ic4;

            weight[0][0] = vld1q_s8(read_weight_ptr);
            weight[0][1] = vld1q_s8(read_weight_ptr + 16);
            weight[1][0] = vld1q_s8(read_weight_ptr + ld_weight_oc4);
            weight[1][1] = vld1q_s8(read_weight_ptr + ld_weight_oc4 + 16);

            c[0][0] = vdotq_s32_h(weight[0][0], src[0], c[0][0], temp_c[0]);
            c[1][0] = vdotq_s32_h(weight[1][0], src[0], c[1][0], temp_c[1]);
            c[0][1] = vdotq_s32_h(weight[0][0], src[1], c[0][1], temp_c[2]);
            c[1][1] = vdotq_s32_h(weight[1][0], src[1], c[1][1], temp_c[3]);
            c[0][0] = vdotq_s32_h(weight[0][1], src[1], c[0][0], temp_c[0]);
            c[1][0] = vdotq_s32_h(weight[1][1], src[1], c[1][0], temp_c[1]);
            c[0][1] = vdotq_s32_h(weight[0][1], src[2], c[0][1], temp_c[2]);
            c[1][1] = vdotq_s32_h(weight[1][1], src[2], c[1][1], temp_c[3]);

            c[0][2] = vdotq_s32_h(weight[0][0], src[2], c[0][2], temp_c[0]);
            c[1][2] = vdotq_s32_h(weight[1][0], src[2], c[1][2], temp_c[1]);
            c[0][3] = vdotq_s32_h(weight[0][0], src[3], c[0][3], temp_c[2]);
            c[1][3] = vdotq_s32_h(weight[1][0], src[3], c[1][3], temp_c[3]);
            c[0][2] = vdotq_s32_h(weight[0][1], src[3], c[0][2], temp_c[0]);
            c[1][2] = vdotq_s32_h(weight[1][1], src[3], c[1][2], temp_c[1]);
            c[0][3] = vdotq_s32_h(weight[0][1], src[4], c[0][3], temp_c[2]);
            c[1][3] = vdotq_s32_h(weight[1][1], src[4], c[1][3], temp_c[3]);

            c[0][4] = vdotq_s32_h(weight[0][0], src[4], c[0][4], temp_c[0]);
            c[1][4] = vdotq_s32_h(weight[1][0], src[4], c[1][4], temp_c[1]);
            c[0][5] = vdotq_s32_h(weight[0][0], src[5], c[0][5], temp_c[2]);
            c[1][5] = vdotq_s32_h(weight[1][0], src[5], c[1][5], temp_c[3]);
            c[0][4] = vdotq_s32_h(weight[0][1], src[5], c[0][4], temp_c[0]);
            c[1][4] = vdotq_s32_h(weight[1][1], src[5], c[1][4], temp_c[1]);
            c[0][5] = vdotq_s32_h(weight[0][1], src[6], c[0][5], temp_c[2]);
            c[1][5] = vdotq_s32_h(weight[1][1], src[6], c[1][5], temp_c[3]);

            c[0][6] = vdotq_s32_h(weight[0][0], src[6], c[0][6], temp_c[0]);
            c[1][6] = vdotq_s32_h(weight[1][0], src[6], c[1][6], temp_c[1]);
            c[0][7] = vdotq_s32_h(weight[0][0], src[7], c[0][7], temp_c[2]);
            c[1][7] = vdotq_s32_h(weight[1][0], src[7], c[1][7], temp_c[3]);
            c[0][6] = vdotq_s32_h(weight[0][1], src[7], c[0][6], temp_c[0]);
            c[1][6] = vdotq_s32_h(weight[1][1], src[7], c[1][6], temp_c[1]);
            c[0][7] = vdotq_s32_h(weight[0][1], src[8], c[0][7], temp_c[2]);
            c[1][7] = vdotq_s32_h(weight[1][1], src[8], c[1][7], temp_c[3]);
        }
        weight_ptr += fh * fw * ld_weight_ic4;
    }
    store_oc8_ow8_remain_static<remain_w>(c, op, dst_ptr, ld_dst_oc);
}

template <BiasMode bias_mode, typename Op, int remain_w, int filter_size>
static void ker_neon_dirctconv_2x2s1_oc4_ow8(const int8_t* src_ptr,
                                             const int8_t* weight_ptr,
                                             const int32_t* bias_ptr,
                                             int8_t* dst_ptr, int ic, int ih,
                                             int iw, const Op& op) {
    constexpr int fh = filter_size;
    constexpr int fw = filter_size;
    constexpr int ic_step = 4;
    constexpr int loop_ic_step = 4;
    constexpr int ld_weight_ic4 = 16;
    constexpr int pack_iw_len = 4;

    const int ic_stride = ih * iw * pack_iw_len;

    int32x4_t c[2 * 4];
    int8x16_t weight[2];
    int8x16_t src[8 + 1];
    int16x8_t temp_c[2];
    init_oc4_ow8<bias_mode>(c, bias_ptr);

    for (int ic_idx = 0; ic_idx < ic; ic_idx += loop_ic_step) {
        for (int fh_idx = 0; fh_idx < fh; ++fh_idx) {
            const int8_t* src_ic_0_3 = src_ptr + ic_idx * ic_stride +
                                       fh_idx * iw * ic_step * pack_iw_len;

            src[0] = vld1q_s8(src_ic_0_3);
            src[1] = vld1q_s8((src_ic_0_3 + 16));
            src[2] = vld1q_s8((src_ic_0_3 + 2 * 16));
            src[3] = vld1q_s8((src_ic_0_3 + 3 * 16));
            src[4] = vld1q_s8((src_ic_0_3 + 4 * 16));
            src[5] = vld1q_s8((src_ic_0_3 + 5 * 16));
            src[6] = vld1q_s8((src_ic_0_3 + 6 * 16));
            src[7] = vld1q_s8((src_ic_0_3 + 7 * 16));
            src[8] = vld1q_s8((src_ic_0_3 + 8 * 16));

            // oc == 0
            const int8_t* read_weight_ptr =
                    weight_ptr + fh_idx * fw * ld_weight_ic4;

            weight[0] = vld1q_s8(read_weight_ptr);
            weight[1] = vld1q_s8(read_weight_ptr + 16);

            c[0] = vdotq_s32_h(weight[0], src[0], c[0], temp_c[0]);
            c[1] = vdotq_s32_h(weight[0], src[1], c[1], temp_c[1]);
            c[0] = vdotq_s32_h(weight[1], src[1], c[0], temp_c[0]);
            c[1] = vdotq_s32_h(weight[1], src[2], c[1], temp_c[1]);

            c[2] = vdotq_s32_h(weight[0], src[2], c[2], temp_c[0]);
            c[3] = vdotq_s32_h(weight[0], src[3], c[3], temp_c[1]);
            c[2] = vdotq_s32_h(weight[1], src[3], c[2], temp_c[0]);
            c[3] = vdotq_s32_h(weight[1], src[4], c[3], temp_c[1]);

            c[4] = vdotq_s32_h(weight[0], src[4], c[4], temp_c[0]);
            c[5] = vdotq_s32_h(weight[0], src[5], c[5], temp_c[1]);
            c[4] = vdotq_s32_h(weight[1], src[5], c[4], temp_c[0]);
            c[5] = vdotq_s32_h(weight[1], src[6], c[5], temp_c[1]);

            c[6] = vdotq_s32_h(weight[0], src[6], c[6], temp_c[0]);
            c[7] = vdotq_s32_h(weight[0], src[7], c[7], temp_c[1]);
            c[6] = vdotq_s32_h(weight[1], src[7], c[6], temp_c[0]);
            c[7] = vdotq_s32_h(weight[1], src[8], c[7], temp_c[1]);
        }
        weight_ptr += fh * fw * ld_weight_ic4;
    }

    store_oc4_ow8_remain_static<remain_w, Op>(c, op, dst_ptr);
}

template <BiasMode bias_mode, typename Op, int remain_w, int filter_size>
static void ker_neon_dirctconv_5x5s1_oc4_ow8(const int8_t* src_ptr,
                                             const int8_t* weight_ptr,
                                             const int32_t* bias_ptr,
                                             int8_t* dst_ptr, int ic, int ih,
                                             int iw, const Op& op) {
    constexpr int fh = filter_size;
    constexpr int fw = filter_size;
    constexpr int ic_step = 4;
    constexpr int loop_ic_step = 4;
    constexpr int ld_weight_ic4 = 16;
    constexpr int pack_iw_len = 4;

    const int ic_stride = ih * iw * pack_iw_len;

    int32x4_t c[2 * 4];
    int8x16_t weight[5];
    int8x16_t src[8 + 2];
    int16x8_t temp_c[2];
    init_oc4_ow8<bias_mode>(c, bias_ptr);

    for (int ic_idx = 0; ic_idx < ic; ic_idx += loop_ic_step) {
        for (int fh_idx = 0; fh_idx < fh; ++fh_idx) {
            const int8_t* src_ic_0_3 = src_ptr + ic_idx * ic_stride +
                                       fh_idx * iw * ic_step * pack_iw_len;

            src[0] = vld1q_s8(src_ic_0_3);
            src[1] = vld1q_s8((src_ic_0_3 + 16));
            src[2] = vld1q_s8((src_ic_0_3 + 2 * 16));
            src[3] = vld1q_s8((src_ic_0_3 + 3 * 16));
            src[4] = vld1q_s8((src_ic_0_3 + 4 * 16));
            src[5] = vld1q_s8((src_ic_0_3 + 5 * 16));
            src[6] = vld1q_s8((src_ic_0_3 + 6 * 16));
            src[7] = vld1q_s8((src_ic_0_3 + 7 * 16));
            src[8] = vld1q_s8((src_ic_0_3 + 8 * 16));
            src[9] = vld1q_s8((src_ic_0_3 + 9 * 16));

            // oc == 0
            const int8_t* read_weight_ptr =
                    weight_ptr + fh_idx * fw * ld_weight_ic4;

            weight[0] = vld1q_s8(read_weight_ptr);
            weight[1] = vld1q_s8(read_weight_ptr + 16);
            weight[2] = vld1q_s8(read_weight_ptr + 2 * 16);
            weight[3] = vld1q_s8(read_weight_ptr + 3 * 16);
            weight[4] = vld1q_s8(read_weight_ptr + 4 * 16);

            c[0] = vdotq_s32_h(weight[0], src[0], c[0], temp_c[0]);
            c[1] = vdotq_s32_h(weight[0], src[1], c[1], temp_c[1]);
            c[0] = vdotq_s32_h(weight[1], src[1], c[0], temp_c[0]);
            c[1] = vdotq_s32_h(weight[1], src[2], c[1], temp_c[1]);
            c[0] = vdotq_s32_h(weight[2], src[2], c[0], temp_c[0]);
            c[1] = vdotq_s32_h(weight[2], src[3], c[1], temp_c[1]);
            c[0] = vdotq_s32_h(weight[3], src[3], c[0], temp_c[0]);
            c[1] = vdotq_s32_h(weight[3], src[4], c[1], temp_c[1]);
            c[0] = vdotq_s32_h(weight[4], src[4], c[0], temp_c[0]);
            c[1] = vdotq_s32_h(weight[4], src[5], c[1], temp_c[1]);

            c[2] = vdotq_s32_h(weight[0], src[2], c[2], temp_c[0]);
            c[3] = vdotq_s32_h(weight[0], src[3], c[3], temp_c[1]);
            c[2] = vdotq_s32_h(weight[1], src[3], c[2], temp_c[0]);
            c[3] = vdotq_s32_h(weight[1], src[4], c[3], temp_c[1]);
            c[2] = vdotq_s32_h(weight[2], src[4], c[2], temp_c[0]);
            c[3] = vdotq_s32_h(weight[2], src[5], c[3], temp_c[1]);
            c[2] = vdotq_s32_h(weight[3], src[5], c[2], temp_c[0]);
            c[3] = vdotq_s32_h(weight[3], src[6], c[3], temp_c[1]);
            c[2] = vdotq_s32_h(weight[4], src[6], c[2], temp_c[0]);
            c[3] = vdotq_s32_h(weight[4], src[7], c[3], temp_c[1]);

            c[4] = vdotq_s32_h(weight[0], src[4], c[4], temp_c[0]);
            c[5] = vdotq_s32_h(weight[0], src[5], c[5], temp_c[1]);
            c[4] = vdotq_s32_h(weight[1], src[5], c[4], temp_c[0]);
            c[5] = vdotq_s32_h(weight[1], src[6], c[5], temp_c[1]);
            c[4] = vdotq_s32_h(weight[2], src[6], c[4], temp_c[0]);
            c[5] = vdotq_s32_h(weight[2], src[7], c[5], temp_c[1]);
            c[4] = vdotq_s32_h(weight[3], src[7], c[4], temp_c[0]);
            c[5] = vdotq_s32_h(weight[3], src[8], c[5], temp_c[1]);
            c[4] = vdotq_s32_h(weight[4], src[8], c[4], temp_c[0]);
            c[5] = vdotq_s32_h(weight[4], src[9], c[5], temp_c[1]);

            src[0] = vld1q_s8(src_ic_0_3 + 10 * 16);
            src[1] = vld1q_s8((src_ic_0_3 + 11 * 16));

            c[6] = vdotq_s32_h(weight[0], src[6], c[6], temp_c[0]);
            c[7] = vdotq_s32_h(weight[0], src[7], c[7], temp_c[1]);
            c[6] = vdotq_s32_h(weight[1], src[7], c[6], temp_c[0]);
            c[7] = vdotq_s32_h(weight[1], src[8], c[7], temp_c[1]);
            c[6] = vdotq_s32_h(weight[2], src[8], c[6], temp_c[0]);
            c[7] = vdotq_s32_h(weight[2], src[9], c[7], temp_c[1]);
            c[6] = vdotq_s32_h(weight[3], src[9], c[6], temp_c[0]);
            c[7] = vdotq_s32_h(weight[3], src[0], c[7], temp_c[1]);
            c[6] = vdotq_s32_h(weight[4], src[0], c[6], temp_c[0]);
            c[7] = vdotq_s32_h(weight[4], src[1], c[7], temp_c[1]);
        }
        weight_ptr += fh * fw * ld_weight_ic4;
    }

    store_oc4_ow8_remain_static<remain_w, Op>(c, op, dst_ptr);
}

template <BiasMode bias_mode, typename Op, int remain_w, int filter_size>
static void ker_neon_dirctconv_7x7s1_oc4_ow8(const int8_t* src_ptr,
                                             const int8_t* weight_ptr,
                                             const int32_t* bias_ptr,
                                             int8_t* dst_ptr, int ic, int ih,
                                             int iw, const Op& op) {
    constexpr int fh = filter_size;
    constexpr int fw = filter_size;
    constexpr int ic_step = 4;
    constexpr int loop_ic_step = 4;
    constexpr int ld_weight_ic4 = 16;
    constexpr int pack_iw_len = 4;

    const int ic_stride = ih * iw * pack_iw_len;

    int32x4_t c[2 * 4];
    int8x16_t weight[7];
    int8x16_t src[8 + 2];
    int16x8_t temp_c[2];
    init_oc4_ow8<bias_mode>(c, bias_ptr);
    for (int ic_idx = 0; ic_idx < ic; ic_idx += loop_ic_step) {
        for (int fh_idx = 0; fh_idx < fh; ++fh_idx) {
            const int8_t* src_ic_0_3 = src_ptr + ic_idx * ic_stride +
                                       fh_idx * iw * ic_step * pack_iw_len;

            src[0] = vld1q_s8(src_ic_0_3);
            src[1] = vld1q_s8((src_ic_0_3 + 16));
            src[2] = vld1q_s8((src_ic_0_3 + 2 * 16));
            src[3] = vld1q_s8((src_ic_0_3 + 3 * 16));
            src[4] = vld1q_s8((src_ic_0_3 + 4 * 16));
            src[5] = vld1q_s8((src_ic_0_3 + 5 * 16));
            src[6] = vld1q_s8((src_ic_0_3 + 6 * 16));
            src[7] = vld1q_s8((src_ic_0_3 + 7 * 16));
            src[8] = vld1q_s8((src_ic_0_3 + 8 * 16));
            src[9] = vld1q_s8((src_ic_0_3 + 9 * 16));

            // oc == 0
            const int8_t* read_weight_ptr =
                    weight_ptr + fh_idx * fw * ld_weight_ic4;

            weight[0] = vld1q_s8(read_weight_ptr);
            weight[1] = vld1q_s8(read_weight_ptr + 16);
            weight[2] = vld1q_s8(read_weight_ptr + 2 * 16);
            weight[3] = vld1q_s8(read_weight_ptr + 3 * 16);
            weight[4] = vld1q_s8(read_weight_ptr + 4 * 16);
            weight[5] = vld1q_s8(read_weight_ptr + 5 * 16);
            weight[6] = vld1q_s8(read_weight_ptr + 6 * 16);

            c[0] = vdotq_s32_h(weight[0], src[0], c[0], temp_c[0]);
            c[1] = vdotq_s32_h(weight[0], src[1], c[1], temp_c[1]);
            c[0] = vdotq_s32_h(weight[1], src[1], c[0], temp_c[0]);
            c[1] = vdotq_s32_h(weight[1], src[2], c[1], temp_c[1]);
            c[0] = vdotq_s32_h(weight[2], src[2], c[0], temp_c[0]);
            c[1] = vdotq_s32_h(weight[2], src[3], c[1], temp_c[1]);
            c[0] = vdotq_s32_h(weight[3], src[3], c[0], temp_c[0]);
            c[1] = vdotq_s32_h(weight[3], src[4], c[1], temp_c[1]);
            c[0] = vdotq_s32_h(weight[4], src[4], c[0], temp_c[0]);
            c[1] = vdotq_s32_h(weight[4], src[5], c[1], temp_c[1]);
            c[0] = vdotq_s32_h(weight[5], src[5], c[0], temp_c[0]);
            c[1] = vdotq_s32_h(weight[5], src[6], c[1], temp_c[1]);
            c[0] = vdotq_s32_h(weight[6], src[6], c[0], temp_c[0]);
            c[1] = vdotq_s32_h(weight[6], src[7], c[1], temp_c[1]);

            c[2] = vdotq_s32_h(weight[0], src[2], c[2], temp_c[0]);
            c[3] = vdotq_s32_h(weight[0], src[3], c[3], temp_c[1]);
            c[2] = vdotq_s32_h(weight[1], src[3], c[2], temp_c[0]);
            c[3] = vdotq_s32_h(weight[1], src[4], c[3], temp_c[1]);
            c[2] = vdotq_s32_h(weight[2], src[4], c[2], temp_c[0]);
            c[3] = vdotq_s32_h(weight[2], src[5], c[3], temp_c[1]);
            c[2] = vdotq_s32_h(weight[3], src[5], c[2], temp_c[0]);
            c[3] = vdotq_s32_h(weight[3], src[6], c[3], temp_c[1]);
            c[2] = vdotq_s32_h(weight[4], src[6], c[2], temp_c[0]);
            c[3] = vdotq_s32_h(weight[4], src[7], c[3], temp_c[1]);
            c[2] = vdotq_s32_h(weight[5], src[7], c[2], temp_c[0]);
            c[3] = vdotq_s32_h(weight[5], src[8], c[3], temp_c[1]);
            c[2] = vdotq_s32_h(weight[6], src[8], c[2], temp_c[0]);
            c[3] = vdotq_s32_h(weight[6], src[9], c[3], temp_c[1]);

            src[0] = vld1q_s8(src_ic_0_3 + 10 * 16);
            src[1] = vld1q_s8((src_ic_0_3 + 11 * 16));

            c[4] = vdotq_s32_h(weight[0], src[4], c[4], temp_c[0]);
            c[5] = vdotq_s32_h(weight[0], src[5], c[5], temp_c[1]);
            c[4] = vdotq_s32_h(weight[1], src[5], c[4], temp_c[0]);
            c[5] = vdotq_s32_h(weight[1], src[6], c[5], temp_c[1]);
            c[4] = vdotq_s32_h(weight[2], src[6], c[4], temp_c[0]);
            c[5] = vdotq_s32_h(weight[2], src[7], c[5], temp_c[1]);
            c[4] = vdotq_s32_h(weight[3], src[7], c[4], temp_c[0]);
            c[5] = vdotq_s32_h(weight[3], src[8], c[5], temp_c[1]);
            c[4] = vdotq_s32_h(weight[4], src[8], c[4], temp_c[0]);
            c[5] = vdotq_s32_h(weight[4], src[9], c[5], temp_c[1]);
            c[4] = vdotq_s32_h(weight[5], src[9], c[4], temp_c[0]);
            c[5] = vdotq_s32_h(weight[5], src[0], c[5], temp_c[1]);
            c[4] = vdotq_s32_h(weight[6], src[0], c[4], temp_c[0]);
            c[5] = vdotq_s32_h(weight[6], src[1], c[5], temp_c[1]);

            src[2] = vld1q_s8(src_ic_0_3 + 12 * 16);
            src[3] = vld1q_s8((src_ic_0_3 + 13 * 16));

            c[6] = vdotq_s32_h(weight[0], src[6], c[6], temp_c[0]);
            c[7] = vdotq_s32_h(weight[0], src[7], c[7], temp_c[1]);
            c[6] = vdotq_s32_h(weight[1], src[7], c[6], temp_c[0]);
            c[7] = vdotq_s32_h(weight[1], src[8], c[7], temp_c[1]);
            c[6] = vdotq_s32_h(weight[2], src[8], c[6], temp_c[0]);
            c[7] = vdotq_s32_h(weight[2], src[9], c[7], temp_c[1]);
            c[6] = vdotq_s32_h(weight[3], src[9], c[6], temp_c[0]);
            c[7] = vdotq_s32_h(weight[3], src[0], c[7], temp_c[1]);
            c[6] = vdotq_s32_h(weight[4], src[0], c[6], temp_c[0]);
            c[7] = vdotq_s32_h(weight[4], src[1], c[7], temp_c[1]);
            c[6] = vdotq_s32_h(weight[5], src[1], c[6], temp_c[0]);
            c[7] = vdotq_s32_h(weight[5], src[2], c[7], temp_c[1]);
            c[6] = vdotq_s32_h(weight[6], src[2], c[6], temp_c[0]);
            c[7] = vdotq_s32_h(weight[6], src[3], c[7], temp_c[1]);
        }
        weight_ptr += fh * fw * ld_weight_ic4;
    }

    store_oc4_ow8_remain_static<remain_w, Op>(c, op, dst_ptr);
}

}  // namespace

/**
origin weight shape <oc/4, ic/4, fh, fw, 4, 4>
packed weight shape <oc/4, ic/4, fh, fw, 16>
example: (format like weight<oc, ic>)
origin
<0, 0>  <1, 0>  <2, 0>  <3, 0>
<0, 1>  <1, 1>  <2, 1>  <3, 1>
<0, 2>  <1, 2>  <2, 2>  <3, 2>
<0, 3>  <1, 3>  <2, 3>  <3, 3>
packed
low 64 bit  <0, 0> <0, 1> <1, 2> <1, 3> | <2, 0> <2, 1> <3, 2> <3, 3>
---------------------------------------------------------------------
high 64 bit <0, 3> <0, 2> <1, 1> <1, 0> | <2, 3> <2, 2> <3, 1> <3, 0>
**/
void conv_bias::nchw44_pack_filter(const int8_t* src, int8_t* dst, int length) {
    static const uint8_t weight_idx_buffer[16] = {0,  4, 9, 13, 2,  6,  11, 15,
                                                  12, 8, 5, 1,  14, 10, 7,  3};
    constexpr int simd_len = 16;
    uint8x16_t weight_idx = vld1q_u8(weight_idx_buffer);
    for (int i = 0; i < length; i++) {
        int8x16_t result = vldq_tbl_s8(src + i * simd_len, weight_idx);
        vst1q_s8(dst + i * simd_len, result);
    }
}
/**
origin src shape <n, ic/4, h, w, 4>
packed src shape <n, ic/4, h, w, 16>
example: (format like <ic>)
origin
<0>  <0>  <0>  <0>
packed
low 64 bit  <0> <1> <2> <3> | <0> <1> <2> <3>
---------------------------------------------------------------------
high 64 bit <3> <2> <1> <0> | <3> <2> <1> <0>
**/
void conv_bias::nchw44_pack_src(const int8_t* src, int8_t* dst, int length) {
    static const uint8_t src_idx_buffer[16] = {0, 1, 2, 3, 0, 1, 2, 3,
                                               3, 2, 1, 0, 3, 2, 1, 0};
    constexpr int pack_ic = 4;
    constexpr int simd_len = 16;
    uint8x16_t src_idx = vld1q_u8(src_idx_buffer);
    for (int i = 0; i < length; i++) {
        int8x16_t result = vld_dup_tbl_s32(src + i * pack_ic, src_idx);
        vst1q_s8(dst + i * simd_len, result);
    }
}

template <BiasMode bias_mode, typename Op, int remain_w>
void conv_bias::conv_direct_stride1_2x2_int8_nchw44(
        const int8_t* src, const int8_t* filter, const int32_t* bias,
        int32_t* temp, int8_t* dst, const size_t oc, const size_t ic,
        const size_t ih, const size_t iw, const size_t oh, const size_t ow,
        const Op& op) {
    MEGDNN_MARK_USED_VAR(temp);
    constexpr size_t filter_size = 2;
    constexpr size_t fh = filter_size;
    constexpr size_t fw = filter_size;
    constexpr size_t ic_step = 4;
    constexpr size_t oc_step = 4;
    constexpr size_t big_oc_step = 8;
    constexpr size_t oh_step = 1;
    constexpr size_t ow_step = 8;
    constexpr int pack_iw_len = 4;

    const size_t img_stride = oh * ow;
    const size_t ow_end = ow / ow_step * ow_step;
    const size_t ow_remain = ow - ow_end;
    const size_t oc_end = oc / big_oc_step * big_oc_step;
    const size_t oc_remain = oc - oc_end;
    const int ld_oc = oh * ow * ic_step;
    for (size_t oc_idx = 0; oc_idx < oc_end; oc_idx += big_oc_step) {
        const size_t weight_offset = oc_idx * ic * fh * fw;
        for (size_t oh_idx = 0; oh_idx < oh; oh_idx += oh_step) {
            for (size_t ow_idx = 0; ow_idx < ow_end; ow_idx += ow_step) {
                const size_t src_offset =
                        (oh_idx * iw + ow_idx) * ic_step * pack_iw_len;
                const size_t dst_offset =
                        oc_idx * img_stride + (oh_idx * ow + ow_idx) * oc_step;
                ker_neon_dirctconv_2x2s1_oc8_ow8<bias_mode, Op, 0, filter_size>(
                        src + src_offset, filter + weight_offset, bias + oc_idx,
                        dst + dst_offset, ic, ih, iw, ld_oc, op);
            }
            if (ow_remain > 0) {
                const size_t src_offset =
                        (oh_idx * iw + ow_end) * ic_step * pack_iw_len;
                const size_t dst_offset =
                        oc_idx * img_stride + (oh_idx * ow + ow_end) * oc_step;
                ker_neon_dirctconv_2x2s1_oc8_ow8<bias_mode, Op, remain_w,
                                                 filter_size>(
                        src + src_offset, filter + weight_offset, bias + oc_idx,
                        dst + dst_offset, ic, ih, iw, ld_oc, op);
            }
        }
    }
    if (oc_remain > 0) {
        const size_t oc_idx = oc_end;
        const size_t weight_offset = oc_idx * ic * fh * fw;
        for (size_t oh_idx = 0; oh_idx < oh; oh_idx += oh_step) {
            for (size_t ow_idx = 0; ow_idx < ow_end; ow_idx += ow_step) {
                const size_t src_offset =
                        (oh_idx * iw + ow_idx) * ic_step * pack_iw_len;
                const size_t dst_offset =
                        oc_idx * img_stride + (oh_idx * ow + ow_idx) * oc_step;
                ker_neon_dirctconv_2x2s1_oc4_ow8<bias_mode, Op, 0, filter_size>(
                        src + src_offset, filter + weight_offset, bias + oc_idx,
                        dst + dst_offset, ic, ih, iw, op);
            }
            if (ow_remain > 0) {
                const size_t src_offset =
                        (oh_idx * iw + ow_end) * ic_step * pack_iw_len;
                const size_t dst_offset =
                        oc_idx * img_stride + (oh_idx * ow + ow_end) * oc_step;
                ker_neon_dirctconv_2x2s1_oc4_ow8<bias_mode, Op, remain_w,
                                                 filter_size>(
                        src + src_offset, filter + weight_offset, bias + oc_idx,
                        dst + dst_offset, ic, ih, iw, op);
            }
        }
    }
}
template <BiasMode bias_mode, typename Op, int remain_w>
void conv_bias::conv_direct_stride1_3x3_int8_nchw44(
        const int8_t* src, const int8_t* filter, const int32_t* bias,
        int32_t* temp, int8_t* dst, const size_t oc, const size_t ic,
        const size_t ih, const size_t iw, const size_t oh, const size_t ow,
        const Op& op) {
    MEGDNN_MARK_USED_VAR(temp);
    constexpr size_t filter_size = 3;
    constexpr size_t fh = filter_size;
    constexpr size_t fw = filter_size;
    constexpr size_t ic_step = 4;
    constexpr size_t oc_step = 4;
    constexpr size_t oh_step = 1;
    constexpr size_t ow_step = 8;
    constexpr int pack_iw_len = 4;

    const size_t img_stride = oh * ow;
    const size_t ow_end = ow / ow_step * ow_step;
    const size_t ow_remain = ow - ow_end;
    for (size_t oc_idx = 0; oc_idx < oc; oc_idx += oc_step) {
        const size_t weight_offset = oc_idx * ic * fh * fw;
        for (size_t oh_idx = 0; oh_idx < oh; oh_idx += oh_step) {
            for (size_t ow_idx = 0; ow_idx < ow_end; ow_idx += ow_step) {
                const size_t src_offset =
                        (oh_idx * iw + ow_idx) * ic_step * pack_iw_len;
                const size_t dst_offset =
                        oc_idx * img_stride + (oh_idx * ow + ow_idx) * oc_step;
                ker_neon_dirctconv_3x3s1_oc4_ow8<bias_mode, Op, 0, filter_size>(
                        src + src_offset, filter + weight_offset, bias + oc_idx,
                        dst + dst_offset, ic, ih, iw, op);
            }
            if (ow_remain > 0) {
                const size_t src_offset =
                        (oh_idx * iw + ow_end) * ic_step * pack_iw_len;
                const size_t dst_offset =
                        oc_idx * img_stride + (oh_idx * ow + ow_end) * oc_step;
                ker_neon_dirctconv_3x3s1_oc4_ow8<bias_mode, Op, remain_w,
                                                 filter_size>(
                        src + src_offset, filter + weight_offset, bias + oc_idx,
                        dst + dst_offset, ic, ih, iw, op);
            }
        }
    }
}
template <BiasMode bias_mode, typename Op, int remain_w>
void conv_bias::conv_direct_stride1_5x5_int8_nchw44(
        const int8_t* src, const int8_t* filter, const int32_t* bias,
        int32_t* temp, int8_t* dst, const size_t oc, const size_t ic,
        const size_t ih, const size_t iw, const size_t oh, const size_t ow,
        const Op& op) {
    MEGDNN_MARK_USED_VAR(temp);
    constexpr size_t filter_size = 5;
    constexpr size_t fh = filter_size;
    constexpr size_t fw = filter_size;
    constexpr size_t ic_step = 4;
    constexpr size_t oc_step = 4;
    constexpr size_t oh_step = 1;
    constexpr size_t ow_step = 8;
    constexpr int pack_iw_len = 4;

    const size_t img_stride = oh * ow;
    const size_t ow_end = ow / ow_step * ow_step;
    const size_t ow_remain = ow - ow_end;
    for (size_t oc_idx = 0; oc_idx < oc; oc_idx += oc_step) {
        const size_t weight_offset = oc_idx * ic * fh * fw;
        for (size_t oh_idx = 0; oh_idx < oh; oh_idx += oh_step) {
            for (size_t ow_idx = 0; ow_idx < ow_end; ow_idx += ow_step) {
                const size_t src_offset =
                        (oh_idx * iw + ow_idx) * ic_step * pack_iw_len;
                const size_t dst_offset =
                        oc_idx * img_stride + (oh_idx * ow + ow_idx) * oc_step;
                ker_neon_dirctconv_5x5s1_oc4_ow8<bias_mode, Op, 0, filter_size>(
                        src + src_offset, filter + weight_offset, bias + oc_idx,
                        dst + dst_offset, ic, ih, iw, op);
            }
            if (ow_remain > 0) {
                const size_t src_offset =
                        (oh_idx * iw + ow_end) * ic_step * pack_iw_len;
                const size_t dst_offset =
                        oc_idx * img_stride + (oh_idx * ow + ow_end) * oc_step;
                ker_neon_dirctconv_5x5s1_oc4_ow8<bias_mode, Op, remain_w,
                                                 filter_size>(
                        src + src_offset, filter + weight_offset, bias + oc_idx,
                        dst + dst_offset, ic, ih, iw, op);
            }
        }
    }
}

template <BiasMode bias_mode, typename Op, int remain_w>
void conv_bias::conv_direct_stride1_7x7_int8_nchw44(
        const int8_t* src, const int8_t* filter, const int32_t* bias,
        int32_t* temp, int8_t* dst, const size_t oc, const size_t ic,
        const size_t ih, const size_t iw, const size_t oh, const size_t ow,
        const Op& op) {
    MEGDNN_MARK_USED_VAR(temp);
    constexpr size_t filter_size = 7;
    constexpr size_t fh = filter_size;
    constexpr size_t fw = filter_size;
    constexpr size_t ic_step = 4;
    constexpr size_t oc_step = 4;
    constexpr size_t oh_step = 1;
    constexpr size_t ow_step = 8;
    constexpr int pack_iw_len = 4;

    const size_t img_stride = oh * ow;
    const size_t ow_end = ow / ow_step * ow_step;
    const size_t ow_remain = ow - ow_end;
    for (size_t oc_idx = 0; oc_idx < oc; oc_idx += oc_step) {
        const size_t weight_offset = oc_idx * ic * fh * fw;
        for (size_t oh_idx = 0; oh_idx < oh; oh_idx += oh_step) {
            for (size_t ow_idx = 0; ow_idx < ow_end; ow_idx += ow_step) {
                const size_t src_offset =
                        (oh_idx * iw + ow_idx) * ic_step * pack_iw_len;
                const size_t dst_offset =
                        oc_idx * img_stride + (oh_idx * ow + ow_idx) * oc_step;
                ker_neon_dirctconv_7x7s1_oc4_ow8<bias_mode, Op, 0, filter_size>(
                        src + src_offset, filter + weight_offset, bias + oc_idx,
                        dst + dst_offset, ic, ih, iw, op);
            }
            if (ow_remain > 0) {
                const size_t src_offset =
                        (oh_idx * iw + ow_end) * ic_step * pack_iw_len;
                const size_t dst_offset =
                        oc_idx * img_stride + (oh_idx * ow + ow_end) * oc_step;
                ker_neon_dirctconv_7x7s1_oc4_ow8<bias_mode, Op, remain_w,
                                                 filter_size>(
                        src + src_offset, filter + weight_offset, bias + oc_idx,
                        dst + dst_offset, ic, ih, iw, op);
            }
        }
    }
}

#define INSTANTIATION(stride, i, bias, remain_w, Op)                           \
    template void conv_bias::conv_direct_##stride##_##i##x##i##_int8_nchw44<   \
            bias, Op, remain_w>(const int8_t*, const int8_t*, const int32_t*,  \
                                int32_t*, int8_t*, const size_t, const size_t, \
                                const size_t, const size_t, const size_t,      \
                                const size_t, const Op&);

#define FOR_OP(stride, i, bias, remain_w)                     \
    INSTANTIATION(stride, i, bias, remain_w,                  \
                  TypeCvtOp<dt_qint32 MEGDNN_COMMA dt_qint8>) \
    INSTANTIATION(stride, i, bias, remain_w,                  \
                  ReluOp<dt_qint32 MEGDNN_COMMA dt_qint8>)    \
    INSTANTIATION(stride, i, bias, remain_w,                  \
                  HSwishOp<dt_qint32 MEGDNN_COMMA dt_qint8>)

#define FOR_REMAIN(stride, i, bias) \
    FOR_OP(stride, i, bias, 0)      \
    FOR_OP(stride, i, bias, 1)      \
    FOR_OP(stride, i, bias, 2)      \
    FOR_OP(stride, i, bias, 3)      \
    FOR_OP(stride, i, bias, 4)      \
    FOR_OP(stride, i, bias, 5)      \
    FOR_OP(stride, i, bias, 6)      \
    FOR_OP(stride, i, bias, 7)

#define FOR_BIAS(stride, i)                  \
    FOR_REMAIN(stride, i, BiasMode::NO_BIAS) \
    FOR_REMAIN(stride, i, BiasMode::BROADCAST_CHANNEL_BIAS)

#define FOR_FILTER(stride) \
    FOR_BIAS(stride, 2)    \
    FOR_BIAS(stride, 3)    \
    FOR_BIAS(stride, 5)    \
    FOR_BIAS(stride, 7)

FOR_FILTER(stride1)

#undef FOR_STRIDE
#undef FOR_FILTER
#undef FOR_IC
#undef FOR_BIAS
#undef FOR_NONLINEAR
#undef FOR_REMAIN
#undef INSTANTIATION
