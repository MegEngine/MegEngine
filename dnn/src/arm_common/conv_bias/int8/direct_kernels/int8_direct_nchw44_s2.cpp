/**
 * \file
 * dnn/src/arm_common/conv_bias/int8/direct_kernels/int8_direct_nchw44_s2.cpp
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
#include "src/arm_common/conv_bias/int8/direct_nchw44_kern.h"
#include "src/arm_common/conv_bias/intrinsic_helper.h"
#include "src/arm_common/elemwise_op.h"
#include "src/arm_common/simd_macro/marm_neon.h"
#include "src/common/utils.h"
#include "src/fallback/conv_bias/common.h"

namespace megdnn {
namespace arm_common {
namespace {
template <BiasMode bias_mode, typename Op, int remain_w, int filter_size,
          int c_dim, typename DstType>
struct KerNeonDirectStride2Int8 {
    static void impl(const int8_t* src_ptr, const int8_t* weight_ptr,
                     const int32_t* bias_ptr, DstType* dst_ptr, int ic, int ih,
                     int iw, const Op& op, int ld_dst_oc);
};

template <BiasMode bias_mode, typename Op, int remain_w, int filter_size,
          int c_dim, typename DstType>
static void ker_neon_dirctconv_2x2s2_oc8_ow8(const int8_t* src_ptr,
                                             const int8_t* weight_ptr,
                                             const int32_t* bias_ptr,
                                             DstType* dst_ptr, int ic, int ih,
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

    init_ocx_ow8<c_dim, bias_mode, remain_w>(c, bias_ptr, oc_step);

    for (int ic_idx = 0; ic_idx < ic; ic_idx += loop_ic_step) {
        for (int fh_idx = 0; fh_idx < fh; ++fh_idx) {
            const int8_t* src_ic_0_3 = src_ptr + ic_idx * ic_stride +
                                       fh_idx * iw * ic_step * pack_iw_len;

            src[0] = vld1q_s8(src_ic_0_3);
            src[1] = vld1q_s8(src_ic_0_3 + 16);
            src[2] = vld1q_s8(src_ic_0_3 + 2 * 16);
            src[3] = vld1q_s8(src_ic_0_3 + 3 * 16);
            src[4] = vld1q_s8(src_ic_0_3 + 4 * 16);
            src[5] = vld1q_s8(src_ic_0_3 + 5 * 16);
            src[6] = vld1q_s8(src_ic_0_3 + 6 * 16);
            src[7] = vld1q_s8(src_ic_0_3 + 7 * 16);
            src[8] = vld1q_s8(src_ic_0_3 + 8 * 16);

            // oc == 0
            const int8_t* read_weight_ptr =
                    weight_ptr + fh_idx * fw * ld_weight_ic4;

            weight[0][0] = vld1q_s8(read_weight_ptr);
            weight[0][1] = vld1q_s8(read_weight_ptr + 16);
            weight[1][0] = vld1q_s8(read_weight_ptr + ld_weight_oc4);
            weight[1][1] = vld1q_s8(read_weight_ptr + ld_weight_oc4 + 16);

            c[0][0] = vdotq_s32_h(weight[0][0], src[0], c[0][0], temp_c[0]);
            c[1][0] = vdotq_s32_h(weight[1][0], src[0], c[1][0], temp_c[1]);
            c[0][1] = vdotq_s32_h(weight[0][0], src[2], c[0][1], temp_c[2]);
            c[1][1] = vdotq_s32_h(weight[1][0], src[2], c[1][1], temp_c[3]);
            c[0][0] = vdotq_s32_h(weight[0][1], src[1], c[0][0], temp_c[0]);
            c[1][0] = vdotq_s32_h(weight[1][1], src[1], c[1][0], temp_c[1]);
            c[0][1] = vdotq_s32_h(weight[0][1], src[3], c[0][1], temp_c[2]);
            c[1][1] = vdotq_s32_h(weight[1][1], src[3], c[1][1], temp_c[3]);

            c[0][2] = vdotq_s32_h(weight[0][0], src[4], c[0][2], temp_c[0]);
            c[1][2] = vdotq_s32_h(weight[1][0], src[4], c[1][2], temp_c[1]);
            c[0][3] = vdotq_s32_h(weight[0][0], src[6], c[0][3], temp_c[2]);
            c[1][3] = vdotq_s32_h(weight[1][0], src[6], c[1][3], temp_c[3]);
            c[0][2] = vdotq_s32_h(weight[0][1], src[5], c[0][2], temp_c[0]);
            c[1][2] = vdotq_s32_h(weight[1][1], src[5], c[1][2], temp_c[1]);
            c[0][3] = vdotq_s32_h(weight[0][1], src[7], c[0][3], temp_c[2]);
            c[1][3] = vdotq_s32_h(weight[1][1], src[7], c[1][3], temp_c[3]);

            src[0] = vld1q_s8(src_ic_0_3 + 9 * 16);
            src[1] = vld1q_s8(src_ic_0_3 + 10 * 16);
            src[2] = vld1q_s8(src_ic_0_3 + 11 * 16);
            c[0][4] = vdotq_s32_h(weight[0][0], src[8], c[0][4], temp_c[0]);
            c[1][4] = vdotq_s32_h(weight[1][0], src[8], c[1][4], temp_c[1]);
            c[0][5] = vdotq_s32_h(weight[0][0], src[1], c[0][5], temp_c[2]);
            c[1][5] = vdotq_s32_h(weight[1][0], src[1], c[1][5], temp_c[3]);
            c[0][4] = vdotq_s32_h(weight[0][1], src[0], c[0][4], temp_c[0]);
            c[1][4] = vdotq_s32_h(weight[1][1], src[0], c[1][4], temp_c[1]);
            c[0][5] = vdotq_s32_h(weight[0][1], src[2], c[0][5], temp_c[2]);
            c[1][5] = vdotq_s32_h(weight[1][1], src[2], c[1][5], temp_c[3]);

            src[3] = vld1q_s8(src_ic_0_3 + 12 * 16);
            src[4] = vld1q_s8(src_ic_0_3 + 13 * 16);
            src[5] = vld1q_s8(src_ic_0_3 + 14 * 16);
            src[6] = vld1q_s8(src_ic_0_3 + 15 * 16);
            c[0][6] = vdotq_s32_h(weight[0][0], src[3], c[0][6], temp_c[0]);
            c[1][6] = vdotq_s32_h(weight[1][0], src[3], c[1][6], temp_c[1]);
            c[0][7] = vdotq_s32_h(weight[0][0], src[5], c[0][7], temp_c[2]);
            c[1][7] = vdotq_s32_h(weight[1][0], src[5], c[1][7], temp_c[3]);
            c[0][6] = vdotq_s32_h(weight[0][1], src[4], c[0][6], temp_c[0]);
            c[1][6] = vdotq_s32_h(weight[1][1], src[4], c[1][6], temp_c[1]);
            c[0][7] = vdotq_s32_h(weight[0][1], src[6], c[0][7], temp_c[2]);
            c[1][7] = vdotq_s32_h(weight[1][1], src[6], c[1][7], temp_c[3]);
        }
        weight_ptr += fh * fw * ld_weight_ic4;
    }
    store_ocx_ow8_remain_static_dt<c_dim, remain_w, Op, DstType*>(
            c, op, dst_ptr, ld_dst_oc);
}

template <BiasMode bias_mode, typename Op, int remain_w, int filter_size,
          int c_dim, typename DstType>
static void ker_neon_dirctconv_2x2s2_oc4_ow8(const int8_t* src_ptr,
                                             const int8_t* weight_ptr,
                                             const int32_t* bias_ptr,
                                             DstType* dst_ptr, int ic, int ih,
                                             int iw, int ld_dst_oc,
                                             const Op& op) {
    constexpr int fh = filter_size;
    constexpr int fw = filter_size;
    constexpr int oc_step = 4;
    constexpr int ic_step = 4;
    constexpr int loop_ic_step = 4;
    constexpr int ld_weight_ic4 = 16;
    constexpr int pack_iw_len = 4;

    const int ic_stride = ih * iw * pack_iw_len;

    int32x4_t c[c_dim][8];
    int8x16_t weight[2];
    int8x16_t src[8 + 1];
    int16x8_t temp_c[2];
    init_ocx_ow8<c_dim, bias_mode, remain_w>(c, bias_ptr, oc_step);

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

            c[0][0] = vdotq_s32_h(weight[0], src[0], c[0][0], temp_c[0]);
            c[0][1] = vdotq_s32_h(weight[0], src[2], c[0][1], temp_c[1]);
            c[0][0] = vdotq_s32_h(weight[1], src[1], c[0][0], temp_c[0]);
            c[0][1] = vdotq_s32_h(weight[1], src[3], c[0][1], temp_c[1]);

            c[0][2] = vdotq_s32_h(weight[0], src[4], c[0][2], temp_c[0]);
            c[0][3] = vdotq_s32_h(weight[0], src[6], c[0][3], temp_c[1]);
            c[0][2] = vdotq_s32_h(weight[1], src[5], c[0][2], temp_c[0]);
            c[0][3] = vdotq_s32_h(weight[1], src[7], c[0][3], temp_c[1]);

            src[0] = vld1q_s8(src_ic_0_3 + 9 * 16);
            src[1] = vld1q_s8(src_ic_0_3 + 10 * 16);
            src[2] = vld1q_s8(src_ic_0_3 + 11 * 16);
            c[0][4] = vdotq_s32_h(weight[0], src[8], c[0][4], temp_c[0]);
            c[0][5] = vdotq_s32_h(weight[0], src[1], c[0][5], temp_c[1]);
            c[0][4] = vdotq_s32_h(weight[1], src[0], c[0][4], temp_c[0]);
            c[0][5] = vdotq_s32_h(weight[1], src[2], c[0][5], temp_c[1]);

            src[3] = vld1q_s8(src_ic_0_3 + 12 * 16);
            src[4] = vld1q_s8(src_ic_0_3 + 13 * 16);
            src[5] = vld1q_s8(src_ic_0_3 + 14 * 16);
            src[6] = vld1q_s8(src_ic_0_3 + 15 * 16);
            c[0][6] = vdotq_s32_h(weight[0], src[3], c[0][6], temp_c[0]);
            c[0][7] = vdotq_s32_h(weight[0], src[5], c[0][7], temp_c[1]);
            c[0][6] = vdotq_s32_h(weight[1], src[4], c[0][6], temp_c[0]);
            c[0][7] = vdotq_s32_h(weight[1], src[6], c[0][7], temp_c[1]);
        }
        weight_ptr += fh * fw * ld_weight_ic4;
    }

    store_ocx_ow8_remain_static_dt<c_dim, remain_w, Op, DstType*>(
            c, op, dst_ptr, ld_dst_oc);
}
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
template <BiasMode bias_mode, typename Op, int remain_w, int c_dim,
          typename DstType>
struct KerNeonDirectStride2Int8<bias_mode, Op, remain_w, 3, c_dim, DstType> {
    static void impl(const int8_t* src_ptr, const int8_t* weight_ptr,
                     const int32_t* bias_ptr, DstType* dst_ptr, int ic, int ih,
                     int iw, const Op& op, int ld_dst_oc) {
        constexpr int filter_size = 3;
        constexpr int fh = filter_size;
        constexpr int fw = filter_size;
        constexpr int oc_step = 4;
        constexpr int ic_step = 4;
        constexpr int loop_ic_step = 4;
        constexpr int ld_weight_ic4 = 16;
        constexpr int pack_iw_len = 4;

        const int ic_stride = ih * iw * pack_iw_len;

        int32x4_t c[c_dim][8];
        int8x16_t weight[3];
        int8x16_t src[8 + 2];
        int16x8_t temp_c[4];
        init_ocx_ow8<c_dim, bias_mode, remain_w>(c, bias_ptr, oc_step);

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

                c[0][0] = vdotq_s32_h(weight[0], src[0], c[0][0], temp_c[0]);
                c[0][1] = vdotq_s32_h(weight[0], src[2], c[0][1], temp_c[1]);
                c[0][0] = vdotq_s32_h(weight[1], src[1], c[0][0], temp_c[2]);
                c[0][1] = vdotq_s32_h(weight[1], src[3], c[0][1], temp_c[3]);
                c[0][0] = vdotq_s32_h(weight[2], src[2], c[0][0], temp_c[0]);
                c[0][1] = vdotq_s32_h(weight[2], src[4], c[0][1], temp_c[1]);

                c[0][2] = vdotq_s32_h(weight[0], src[4], c[0][2], temp_c[2]);
                c[0][3] = vdotq_s32_h(weight[0], src[6], c[0][3], temp_c[3]);
                c[0][2] = vdotq_s32_h(weight[1], src[5], c[0][2], temp_c[0]);
                c[0][3] = vdotq_s32_h(weight[1], src[7], c[0][3], temp_c[1]);
                c[0][2] = vdotq_s32_h(weight[2], src[6], c[0][2], temp_c[2]);
                c[0][3] = vdotq_s32_h(weight[2], src[8], c[0][3], temp_c[3]);

                src[0] = vld1q_s8(src_ic_0_3 + 10 * 16);
                src[1] = vld1q_s8((src_ic_0_3 + 11 * 16));
                src[2] = vld1q_s8((src_ic_0_3 + 12 * 16));
                c[0][4] = vdotq_s32_h(weight[0], src[8], c[0][4], temp_c[0]);
                c[0][5] = vdotq_s32_h(weight[0], src[0], c[0][5], temp_c[1]);
                c[0][4] = vdotq_s32_h(weight[1], src[9], c[0][4], temp_c[2]);
                c[0][5] = vdotq_s32_h(weight[1], src[1], c[0][5], temp_c[3]);
                c[0][4] = vdotq_s32_h(weight[2], src[0], c[0][4], temp_c[0]);
                c[0][5] = vdotq_s32_h(weight[2], src[2], c[0][5], temp_c[1]);

                src[3] = vld1q_s8((src_ic_0_3 + 13 * 16));
                src[4] = vld1q_s8((src_ic_0_3 + 14 * 16));
                src[5] = vld1q_s8((src_ic_0_3 + 15 * 16));
                src[6] = vld1q_s8((src_ic_0_3 + 16 * 16));
                c[0][6] = vdotq_s32_h(weight[0], src[2], c[0][6], temp_c[2]);
                c[0][7] = vdotq_s32_h(weight[0], src[4], c[0][7], temp_c[3]);
                c[0][6] = vdotq_s32_h(weight[1], src[3], c[0][6], temp_c[0]);
                c[0][7] = vdotq_s32_h(weight[1], src[5], c[0][7], temp_c[1]);
                c[0][6] = vdotq_s32_h(weight[2], src[4], c[0][6], temp_c[2]);
                c[0][7] = vdotq_s32_h(weight[2], src[6], c[0][7], temp_c[3]);
            }
            weight_ptr += fh * fw * ld_weight_ic4;
        }
        store_ocx_ow8_remain_static_dt<c_dim, remain_w, Op, DstType*>(
                c, op, dst_ptr, ld_dst_oc);
    }
};
template <BiasMode bias_mode, typename Op, int remain_w, int c_dim,
          typename DstType>
struct KerNeonDirectStride2Int8<bias_mode, Op, remain_w, 5, c_dim, DstType> {
    static void impl(const int8_t* src_ptr, const int8_t* weight_ptr,
                     const int32_t* bias_ptr, DstType* dst_ptr, int ic, int ih,
                     int iw, const Op& op, int ld_dst_oc) {
        constexpr int filter_size = 5;
        constexpr int fh = filter_size;
        constexpr int fw = filter_size;
        constexpr int oc_step = 4;
        constexpr int ic_step = 4;
        constexpr int loop_ic_step = 4;
        constexpr int ld_weight_ic4 = 16;
        constexpr int pack_iw_len = 4;

        const int ic_stride = ih * iw * pack_iw_len;

        int32x4_t c[c_dim][8];
        int8x16_t weight[5];
        int8x16_t src[8 + 2];
        int16x8_t temp_c[4];
        init_ocx_ow8<c_dim, bias_mode, remain_w>(c, bias_ptr, oc_step);
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

                c[0][0] = vdotq_s32_h(weight[0], src[0], c[0][0], temp_c[0]);
                c[0][1] = vdotq_s32_h(weight[0], src[2], c[0][1], temp_c[1]);
                c[0][0] = vdotq_s32_h(weight[1], src[1], c[0][0], temp_c[2]);
                c[0][1] = vdotq_s32_h(weight[1], src[3], c[0][1], temp_c[3]);
                c[0][0] = vdotq_s32_h(weight[2], src[2], c[0][0], temp_c[0]);
                c[0][1] = vdotq_s32_h(weight[2], src[4], c[0][1], temp_c[1]);
                c[0][0] = vdotq_s32_h(weight[3], src[3], c[0][0], temp_c[2]);
                c[0][1] = vdotq_s32_h(weight[3], src[5], c[0][1], temp_c[3]);
                c[0][0] = vdotq_s32_h(weight[4], src[4], c[0][0], temp_c[0]);
                c[0][1] = vdotq_s32_h(weight[4], src[6], c[0][1], temp_c[1]);

                src[0] = vld1q_s8(src_ic_0_3 + 10 * 16);
                c[0][2] = vdotq_s32_h(weight[0], src[4], c[0][2], temp_c[2]);
                c[0][3] = vdotq_s32_h(weight[0], src[6], c[0][3], temp_c[3]);
                c[0][2] = vdotq_s32_h(weight[1], src[5], c[0][2], temp_c[0]);
                c[0][3] = vdotq_s32_h(weight[1], src[7], c[0][3], temp_c[1]);
                c[0][2] = vdotq_s32_h(weight[2], src[6], c[0][2], temp_c[2]);
                c[0][3] = vdotq_s32_h(weight[2], src[8], c[0][3], temp_c[3]);
                c[0][2] = vdotq_s32_h(weight[3], src[7], c[0][2], temp_c[0]);
                c[0][3] = vdotq_s32_h(weight[3], src[9], c[0][3], temp_c[1]);
                c[0][2] = vdotq_s32_h(weight[4], src[8], c[0][2], temp_c[2]);
                c[0][3] = vdotq_s32_h(weight[4], src[0], c[0][3], temp_c[3]);

                src[1] = vld1q_s8((src_ic_0_3 + 11 * 16));
                src[2] = vld1q_s8((src_ic_0_3 + 12 * 16));
                src[3] = vld1q_s8((src_ic_0_3 + 13 * 16));
                src[4] = vld1q_s8((src_ic_0_3 + 14 * 16));
                c[0][4] = vdotq_s32_h(weight[0], src[8], c[0][4], temp_c[0]);
                c[0][5] = vdotq_s32_h(weight[0], src[0], c[0][5], temp_c[1]);
                c[0][4] = vdotq_s32_h(weight[1], src[9], c[0][4], temp_c[2]);
                c[0][5] = vdotq_s32_h(weight[1], src[1], c[0][5], temp_c[3]);
                c[0][4] = vdotq_s32_h(weight[2], src[0], c[0][4], temp_c[0]);
                c[0][5] = vdotq_s32_h(weight[2], src[2], c[0][5], temp_c[1]);
                c[0][4] = vdotq_s32_h(weight[3], src[1], c[0][4], temp_c[2]);
                c[0][5] = vdotq_s32_h(weight[3], src[3], c[0][5], temp_c[3]);
                c[0][4] = vdotq_s32_h(weight[4], src[2], c[0][4], temp_c[0]);
                c[0][5] = vdotq_s32_h(weight[4], src[4], c[0][5], temp_c[1]);

                src[5] = vld1q_s8((src_ic_0_3 + 15 * 16));
                src[6] = vld1q_s8((src_ic_0_3 + 16 * 16));
                src[7] = vld1q_s8((src_ic_0_3 + 17 * 16));
                src[8] = vld1q_s8((src_ic_0_3 + 18 * 16));
                c[0][6] = vdotq_s32_h(weight[0], src[2], c[0][6], temp_c[2]);
                c[0][7] = vdotq_s32_h(weight[0], src[4], c[0][7], temp_c[3]);
                c[0][6] = vdotq_s32_h(weight[1], src[3], c[0][6], temp_c[0]);
                c[0][7] = vdotq_s32_h(weight[1], src[5], c[0][7], temp_c[1]);
                c[0][6] = vdotq_s32_h(weight[2], src[4], c[0][6], temp_c[2]);
                c[0][7] = vdotq_s32_h(weight[2], src[6], c[0][7], temp_c[3]);
                c[0][6] = vdotq_s32_h(weight[3], src[5], c[0][6], temp_c[0]);
                c[0][7] = vdotq_s32_h(weight[3], src[7], c[0][7], temp_c[1]);
                c[0][6] = vdotq_s32_h(weight[4], src[6], c[0][6], temp_c[2]);
                c[0][7] = vdotq_s32_h(weight[4], src[8], c[0][7], temp_c[3]);
            }
            weight_ptr += fh * fw * ld_weight_ic4;
        }

        store_ocx_ow8_remain_static_dt<c_dim, remain_w, Op, DstType*>(
                c, op, dst_ptr, ld_dst_oc);
    }
};
template <BiasMode bias_mode, typename Op, int remain_w, int c_dim,
          typename DstType>
struct KerNeonDirectStride2Int8<bias_mode, Op, remain_w, 7, c_dim, DstType> {
    static void impl(const int8_t* src_ptr, const int8_t* weight_ptr,
                     const int32_t* bias_ptr, DstType* dst_ptr, int ic, int ih,
                     int iw, const Op& op, int ld_dst_oc) {
        constexpr int filter_size = 7;
        constexpr int fh = filter_size;
        constexpr int fw = filter_size;
        constexpr int oc_step = 4;
        constexpr int ic_step = 4;
        constexpr int loop_ic_step = 4;
        constexpr int ld_weight_ic4 = 16;
        constexpr int pack_iw_len = 4;

        const int ic_stride = ih * iw * pack_iw_len;

        int32x4_t c[c_dim][8];
        int8x16_t weight[7];
        int8x16_t src[8 + 2];
        int16x8_t temp_c[4];
        init_ocx_ow8<c_dim, bias_mode, remain_w>(c, bias_ptr, oc_step);
        for (int ic_idx = 0; ic_idx < ic; ic_idx += loop_ic_step) {
            for (int fh_idx = 0; fh_idx < fh; ++fh_idx) {
                const int8_t* src_ic_0_3 = src_ptr + ic_idx * ic_stride +
                                           fh_idx * iw * ic_step * pack_iw_len;

                src[0] = vld1q_s8(src_ic_0_3);
                src[1] = vld1q_s8(src_ic_0_3 + 1 * 16);
                src[2] = vld1q_s8(src_ic_0_3 + 2 * 16);
                src[3] = vld1q_s8(src_ic_0_3 + 3 * 16);
                src[4] = vld1q_s8(src_ic_0_3 + 4 * 16);
                src[5] = vld1q_s8(src_ic_0_3 + 5 * 16);
                src[6] = vld1q_s8(src_ic_0_3 + 6 * 16);
                src[7] = vld1q_s8(src_ic_0_3 + 7 * 16);
                src[8] = vld1q_s8(src_ic_0_3 + 8 * 16);
                src[9] = vld1q_s8(src_ic_0_3 + 9 * 16);

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

                c[0][0] = vdotq_s32_h(weight[0], src[0], c[0][0], temp_c[0]);
                c[0][1] = vdotq_s32_h(weight[0], src[2], c[0][1], temp_c[1]);
                c[0][0] = vdotq_s32_h(weight[1], src[1], c[0][0], temp_c[2]);
                c[0][1] = vdotq_s32_h(weight[1], src[3], c[0][1], temp_c[3]);
                c[0][0] = vdotq_s32_h(weight[2], src[2], c[0][0], temp_c[0]);
                c[0][1] = vdotq_s32_h(weight[2], src[4], c[0][1], temp_c[1]);
                c[0][0] = vdotq_s32_h(weight[3], src[3], c[0][0], temp_c[2]);
                c[0][1] = vdotq_s32_h(weight[3], src[5], c[0][1], temp_c[3]);
                c[0][0] = vdotq_s32_h(weight[4], src[4], c[0][0], temp_c[0]);
                c[0][1] = vdotq_s32_h(weight[4], src[6], c[0][1], temp_c[1]);
                c[0][0] = vdotq_s32_h(weight[5], src[5], c[0][0], temp_c[2]);
                c[0][1] = vdotq_s32_h(weight[5], src[7], c[0][1], temp_c[3]);
                c[0][0] = vdotq_s32_h(weight[6], src[6], c[0][0], temp_c[0]);
                c[0][1] = vdotq_s32_h(weight[6], src[8], c[0][1], temp_c[1]);

                src[0] = vld1q_s8(src_ic_0_3 + 10 * 16);
                src[1] = vld1q_s8(src_ic_0_3 + 11 * 16);
                src[2] = vld1q_s8(src_ic_0_3 + 12 * 16);
                c[0][2] = vdotq_s32_h(weight[0], src[4], c[0][2], temp_c[2]);
                c[0][3] = vdotq_s32_h(weight[0], src[6], c[0][3], temp_c[3]);
                c[0][2] = vdotq_s32_h(weight[1], src[5], c[0][2], temp_c[0]);
                c[0][3] = vdotq_s32_h(weight[1], src[7], c[0][3], temp_c[1]);
                c[0][2] = vdotq_s32_h(weight[2], src[6], c[0][2], temp_c[2]);
                c[0][3] = vdotq_s32_h(weight[2], src[8], c[0][3], temp_c[3]);
                c[0][2] = vdotq_s32_h(weight[3], src[7], c[0][2], temp_c[0]);
                c[0][3] = vdotq_s32_h(weight[3], src[9], c[0][3], temp_c[1]);
                c[0][2] = vdotq_s32_h(weight[4], src[8], c[0][2], temp_c[2]);
                c[0][3] = vdotq_s32_h(weight[4], src[0], c[0][3], temp_c[3]);
                c[0][2] = vdotq_s32_h(weight[5], src[9], c[0][2], temp_c[0]);
                c[0][3] = vdotq_s32_h(weight[5], src[1], c[0][3], temp_c[1]);
                c[0][2] = vdotq_s32_h(weight[6], src[0], c[0][2], temp_c[2]);
                c[0][3] = vdotq_s32_h(weight[6], src[2], c[0][3], temp_c[3]);

                src[3] = vld1q_s8(src_ic_0_3 + 13 * 16);
                src[4] = vld1q_s8(src_ic_0_3 + 14 * 16);
                src[5] = vld1q_s8(src_ic_0_3 + 15 * 16);
                src[6] = vld1q_s8(src_ic_0_3 + 16 * 16);
                c[0][4] = vdotq_s32_h(weight[0], src[8], c[0][4], temp_c[0]);
                c[0][5] = vdotq_s32_h(weight[0], src[0], c[0][5], temp_c[1]);
                c[0][4] = vdotq_s32_h(weight[1], src[9], c[0][4], temp_c[2]);
                c[0][5] = vdotq_s32_h(weight[1], src[1], c[0][5], temp_c[3]);
                c[0][4] = vdotq_s32_h(weight[2], src[0], c[0][4], temp_c[0]);
                c[0][5] = vdotq_s32_h(weight[2], src[2], c[0][5], temp_c[1]);
                c[0][4] = vdotq_s32_h(weight[3], src[1], c[0][4], temp_c[2]);
                c[0][5] = vdotq_s32_h(weight[3], src[3], c[0][5], temp_c[3]);
                c[0][4] = vdotq_s32_h(weight[4], src[2], c[0][4], temp_c[0]);
                c[0][5] = vdotq_s32_h(weight[4], src[4], c[0][5], temp_c[1]);
                c[0][4] = vdotq_s32_h(weight[5], src[3], c[0][4], temp_c[2]);
                c[0][5] = vdotq_s32_h(weight[5], src[5], c[0][5], temp_c[3]);
                c[0][4] = vdotq_s32_h(weight[6], src[4], c[0][4], temp_c[0]);
                c[0][5] = vdotq_s32_h(weight[6], src[6], c[0][5], temp_c[1]);

                src[7] = vld1q_s8(src_ic_0_3 + 17 * 16);
                src[8] = vld1q_s8(src_ic_0_3 + 18 * 16);
                src[9] = vld1q_s8(src_ic_0_3 + 19 * 16);
                src[0] = vld1q_s8(src_ic_0_3 + 20 * 16);
                c[0][6] = vdotq_s32_h(weight[0], src[2], c[0][6], temp_c[2]);
                c[0][7] = vdotq_s32_h(weight[0], src[4], c[0][7], temp_c[3]);
                c[0][6] = vdotq_s32_h(weight[1], src[3], c[0][6], temp_c[0]);
                c[0][7] = vdotq_s32_h(weight[1], src[5], c[0][7], temp_c[1]);
                c[0][6] = vdotq_s32_h(weight[2], src[4], c[0][6], temp_c[2]);
                c[0][7] = vdotq_s32_h(weight[2], src[6], c[0][7], temp_c[3]);
                c[0][6] = vdotq_s32_h(weight[3], src[5], c[0][6], temp_c[0]);
                c[0][7] = vdotq_s32_h(weight[3], src[7], c[0][7], temp_c[1]);
                c[0][6] = vdotq_s32_h(weight[4], src[6], c[0][6], temp_c[2]);
                c[0][7] = vdotq_s32_h(weight[4], src[8], c[0][7], temp_c[3]);
                c[0][6] = vdotq_s32_h(weight[5], src[7], c[0][6], temp_c[0]);
                c[0][7] = vdotq_s32_h(weight[5], src[9], c[0][7], temp_c[1]);
                c[0][6] = vdotq_s32_h(weight[6], src[8], c[0][6], temp_c[2]);
                c[0][7] = vdotq_s32_h(weight[6], src[0], c[0][7], temp_c[3]);
            }
            weight_ptr += fh * fw * ld_weight_ic4;
        }
        store_ocx_ow8_remain_static_dt<c_dim, remain_w, Op, DstType*>(
                c, op, dst_ptr, ld_dst_oc);
    }
};

template <BiasMode bias_mode, typename Op, typename DstType>
void conv_direct_stride2_2x2_int8_nchw44(
        const int8_t* src, const int8_t* filter, const int32_t* bias, int32_t*,
        DstType* dst, const size_t oc, const size_t ic, const size_t ih,
        const size_t iw, const size_t oh, const size_t ow, const Op& op) {
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
    constexpr int pack_iw_len = 4;

    const size_t out_img_stride = oh * ow;
    const size_t ow_end = ow / ow_step * ow_step;
    const size_t ow_remain = ow - ow_end;
    const size_t oc_end = oc / big_oc_step * big_oc_step;
    const size_t oc_remain = oc - oc_end;
    const int ld_dst_oc = oh * ow * oc_step;

    using remain_fun = std::function<void(
            const int8_t* src_ptr, const int8_t* weight_ptr,
            const int32_t* bias_ptr, DstType* dst_ptr, int ic, int ih, int iw,
            int ld_dst_oc, const Op& op)>;
    remain_fun kern_big_oc_remain = nullptr;
    remain_fun kern_small_oc_remain = nullptr;

    switch (ow_remain) {
#define cb(step)                                                           \
    case step:                                                             \
        kern_big_oc_remain =                                               \
                ker_neon_dirctconv_2x2s2_oc8_ow8<bias_mode, Op, step,      \
                                                 filter_size, 2, DstType>; \
        kern_small_oc_remain =                                             \
                ker_neon_dirctconv_2x2s2_oc4_ow8<bias_mode, Op, step,      \
                                                 filter_size, 1, DstType>; \
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
                        (oh_idx * stride_h * iw + ow_idx * stride_w) * ic_step *
                        pack_iw_len;
                const size_t dst_offset = oc_idx * out_img_stride +
                                          (oh_idx * ow + ow_idx) * oc_step;
                ker_neon_dirctconv_2x2s2_oc8_ow8<bias_mode, Op, ow_step,
                                                 filter_size, 2, DstType>(
                        src + src_offset, filter + weight_offset, bias + oc_idx,
                        dst + dst_offset, ic, ih, iw, ld_dst_oc, op);
            }
            if (ow_remain > 0) {
                const size_t src_offset =
                        (oh_idx * stride_h * iw + ow_end * stride_w) * ic_step *
                        pack_iw_len;
                const size_t dst_offset = oc_idx * out_img_stride +
                                          (oh_idx * ow + ow_end) * oc_step;
                kern_big_oc_remain(src + src_offset, filter + weight_offset,
                                   bias + oc_idx, dst + dst_offset, ic, ih, iw,
                                   ld_dst_oc, op);
            }
        }
    }

    if (oc_remain > 0) {
        const size_t oc_idx = oc_end;
        const size_t weight_offset = oc_idx * ic * fh * fw;
        for (size_t oh_idx = 0; oh_idx < oh; oh_idx += oh_step) {
            for (size_t ow_idx = 0; ow_idx < ow_end; ow_idx += ow_step) {
                const size_t src_offset =
                        (oh_idx * stride_h * iw + ow_idx * stride_w) * ic_step *
                        pack_iw_len;
                const size_t dst_offset = oc_idx * out_img_stride +
                                          (oh_idx * ow + ow_idx) * oc_step;
                ker_neon_dirctconv_2x2s2_oc4_ow8<bias_mode, Op, ow_step,
                                                 filter_size, 1, DstType>(
                        src + src_offset, filter + weight_offset, bias + oc_idx,
                        dst + dst_offset, ic, ih, iw, ld_dst_oc, op);
            }
            if (ow_remain > 0) {
                const size_t src_offset =
                        (oh_idx * stride_h * iw + ow_end * stride_w) * ic_step *
                        pack_iw_len;
                const size_t dst_offset = oc_idx * out_img_stride +
                                          (oh_idx * ow + ow_end) * oc_step;
                kern_small_oc_remain(src + src_offset, filter + weight_offset,
                                     bias + oc_idx, dst + dst_offset, ic, ih,
                                     iw, ld_dst_oc, op);
            }
        }
    }
}

template <BiasMode bias_mode, typename Op, int filter_size, typename DstType>
void conv_direct_stride2_int8_nchw44_kern(
        const int8_t* src, const int8_t* filter, const int32_t* bias, int32_t*,
        DstType* dst, const size_t oc, const size_t ic, const size_t ih,
        const size_t iw, const size_t oh, const size_t ow, const Op& op) {
    constexpr size_t fh = filter_size;
    constexpr size_t fw = filter_size;
    constexpr size_t ic_step = 4;
    constexpr size_t oc_step = 4;
    constexpr size_t oh_step = 1;
    constexpr size_t ow_step = 8;
    constexpr size_t stride_h = 2;
    constexpr size_t stride_w = 2;
    constexpr int pack_iw_len = 4;

    const size_t img_stride = oh * ow;
    const size_t ow_end = ow / ow_step * ow_step;
    const size_t ow_remain = ow - ow_end;
    const int ld_dst_oc = oh * ow * oc_step;

    using remain_fun = std::function<void(
            const int8_t* src_ptr, const int8_t* weight_ptr,
            const int32_t* bias_ptr, DstType* dst_ptr, int ic, int ih, int iw,
            const Op& op, int ld_dst_oc)>;

    remain_fun kern_small_oc_remain = nullptr;
    switch (ow_remain) {
#define cb(step)                                                              \
    case step:                                                                \
        kern_small_oc_remain =                                                \
                KerNeonDirectStride2Int8<bias_mode, Op, step, filter_size, 1, \
                                         DstType>::impl;                      \
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
                        (oh_idx * stride_h * iw + ow_idx * stride_w) * ic_step *
                        pack_iw_len;
                const size_t dst_offset =
                        oc_idx * img_stride + (oh_idx * ow + ow_idx) * oc_step;
                KerNeonDirectStride2Int8<bias_mode, Op, ow_step, filter_size, 1,
                                         DstType>::impl(src + src_offset,
                                                        filter + weight_offset,
                                                        bias + oc_idx,
                                                        dst + dst_offset, ic,
                                                        ih, iw, op, ld_dst_oc);
            }
            if (ow_remain > 0) {
                const size_t src_offset =
                        (oh_idx * stride_h * iw + ow_end * stride_w) * ic_step *
                        pack_iw_len;
                const size_t dst_offset =
                        oc_idx * img_stride + (oh_idx * ow + ow_end) * oc_step;
                kern_small_oc_remain(src + src_offset, filter + weight_offset,
                                     bias + oc_idx, dst + dst_offset, ic, ih,
                                     iw, op, ld_dst_oc);
            }
        }
    }
}
}  // namespace

namespace int8_direct_nchw44 {
template <BiasMode bias_mode, typename Op, int filter_size, typename DstType>
struct ConvDirectInt8Nchw44Choose<bias_mode, Op, filter_size, DstType, 2> {
    static void impl(const int8_t* src, const int8_t* filter,
                     const int32_t* bias, int32_t* temp, DstType* dst,
                     const size_t oc, const size_t ic, const size_t ih,
                     const size_t iw, const size_t oh, const size_t ow,
                     const Op& op) {
        conv_direct_stride2_int8_nchw44_kern<bias_mode, Op, filter_size,
                                             DstType>(
                src, filter, bias, temp, dst, oc, ic, ih, iw, oh, ow, op);
    }
};

template <BiasMode bias_mode, typename Op, typename DstType>
struct ConvDirectInt8Nchw44Choose<bias_mode, Op, 2, DstType, 2> {
    static void impl(const int8_t* src, const int8_t* filter,
                     const int32_t* bias, int32_t* temp, DstType* dst,
                     const size_t oc, const size_t ic, const size_t ih,
                     const size_t iw, const size_t oh, const size_t ow,
                     const Op& op) {
        conv_direct_stride2_2x2_int8_nchw44<bias_mode, Op, DstType>(
                src, filter, bias, temp, dst, oc, ic, ih, iw, oh, ow, op);
    }
};

#define DO_CONV_KERN_FUN(stride, DstType, filter_size, bias_mode, Op)      \
    template struct ConvDirectInt8Nchw44Choose<bias_mode, Op, filter_size, \
                                               DstType, stride>;

#define GET_OP_PARAM(stride, filter, bias_mode)                  \
    DO_CONV_KERN_FUN(stride, dt_qint8, filter, bias_mode,        \
                                                                 \
                     TypeCvtOp<dt_qint32 MEGDNN_COMMA dt_qint8>) \
    DO_CONV_KERN_FUN(stride, dt_qint8, filter, bias_mode,        \
                                                                 \
                     ReluOp<dt_qint32 MEGDNN_COMMA dt_qint8>)    \
    DO_CONV_KERN_FUN(stride, dt_qint8, filter, bias_mode,        \
                                                                 \
                     HSwishOp<dt_qint32 MEGDNN_COMMA dt_qint8>)  \
    DO_CONV_KERN_FUN(stride, dt_int32, filter, bias_mode, NoneOp<dt_int32>)

#define GET_BIAS_MODE_PARAM(stride, filter)         \
    GET_OP_PARAM(stride, filter, BiasMode::NO_BIAS) \
    GET_OP_PARAM(stride, filter, BiasMode::BROADCAST_CHANNEL_BIAS)

#define DISPATCH_CONV_KERN(stride) \
    GET_BIAS_MODE_PARAM(stride, 2) \
    GET_BIAS_MODE_PARAM(stride, 3) \
    GET_BIAS_MODE_PARAM(stride, 5) \
    GET_BIAS_MODE_PARAM(stride, 7)

DISPATCH_CONV_KERN(2);

}  // namespace int8_direct_nchw44
}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen