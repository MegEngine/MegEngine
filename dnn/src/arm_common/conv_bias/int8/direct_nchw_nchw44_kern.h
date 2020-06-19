/**
 * \file dnn/src/arm_common/conv_bias/int8/direct_nchw_nchw44_kern.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#pragma once
#include "megdnn/arch.h"
#include "src/arm_common/conv_bias/intrinsic_helper.h"
#include "src/arm_common/conv_bias/opr_impl.h"
#include "src/arm_common/elemwise_op.h"
#include "src/arm_common/simd_macro/marm_neon.h"
#include "src/common/unroll_macro.h"
#include "src/common/utils.h"
#include "src/fallback/conv_bias/common.h"

namespace megdnn {
namespace arm_common {
namespace {
/**
 * @brief core code for calculation patten
 *
 * @tparam src_idx is offset of src reg
 * @tparam weight_idx is offset of weight reg
 * @tparam c_dim is output channel
 * @tparam Func mla operation funcion
 * @tparam stride
 * @tparam T outpur regs type
 * @tparam T2 src regs type
 * @tparam T3 weight regs type
 * @tparam T4 temp regs type
 */
template <int src_idx, int weight_idx, int c_dim, typename Func, int stride,
          typename T, typename T2, typename T3, typename T4>
struct ShiftCalHelper {
    static MEGDNN_ALWAYS_INLINE void impl(T& c, T2& src, T3& weight, T4& temp);
    static MEGDNN_ALWAYS_INLINE void impl(T& c, T2& src, T3& weight);
};
template <int src_idx, int weight_idx, typename Func, typename T, typename T2,
          typename T3, typename T4>
struct ShiftCalHelper<src_idx, weight_idx, 2, Func, 2, T, T2, T3, T4> {
    static MEGDNN_ALWAYS_INLINE void impl(T& c, T2& src, T3& weight, T4& temp) {
        c[0][0] = Func::impl(src[0 + src_idx], weight[0][weight_idx], c[0][0],
                             temp[0]);
        c[1][0] = Func::impl(src[0 + src_idx], weight[1][weight_idx], c[1][0],
                             temp[1]);
        c[0][1] = Func::impl(src[1 + src_idx], weight[0][weight_idx], c[0][1],
                             temp[2]);
        c[1][1] = Func::impl(src[1 + src_idx], weight[1][weight_idx], c[1][1],
                             temp[3]);
        c[0][2] = Func::impl(src[2 + src_idx], weight[0][weight_idx], c[0][2],
                             temp[0]);
        c[1][2] = Func::impl(src[2 + src_idx], weight[1][weight_idx], c[1][2],
                             temp[1]);
        c[0][3] = Func::impl(src[3 + src_idx], weight[0][weight_idx], c[0][3],
                             temp[2]);
        c[1][3] = Func::impl(src[3 + src_idx], weight[1][weight_idx], c[1][3],
                             temp[3]);
    }
    static MEGDNN_ALWAYS_INLINE void impl(T& c, T2& src, T3& weight) {
        c[0][0] = Func::impl(src[0 + src_idx], weight[0][weight_idx], c[0][0]);
        c[1][0] = Func::impl(src[0 + src_idx], weight[1][weight_idx], c[1][0]);
        c[0][1] = Func::impl(src[1 + src_idx], weight[0][weight_idx], c[0][1]);
        c[1][1] = Func::impl(src[1 + src_idx], weight[1][weight_idx], c[1][1]);
        c[0][2] = Func::impl(src[2 + src_idx], weight[0][weight_idx], c[0][2]);
        c[1][2] = Func::impl(src[2 + src_idx], weight[1][weight_idx], c[1][2]);
        c[0][3] = Func::impl(src[3 + src_idx], weight[0][weight_idx], c[0][3]);
        c[1][3] = Func::impl(src[3 + src_idx], weight[1][weight_idx], c[1][3]);
    }
};
template <int src_idx, int weight_idx, typename Func, typename T, typename T2,
          typename T3, typename T4>
struct ShiftCalHelper<src_idx, weight_idx, 1, Func, 2, T, T2, T3, T4> {
    static MEGDNN_ALWAYS_INLINE void impl(T& c, T2& src, T3& weight, T4& temp) {
        c[0][0] = Func::impl(src[0 + src_idx], weight[0][weight_idx], c[0][0],
                             temp[0]);
        c[0][1] = Func::impl(src[1 + src_idx], weight[0][weight_idx], c[0][1],
                             temp[2]);
        c[0][2] = Func::impl(src[2 + src_idx], weight[0][weight_idx], c[0][2],
                             temp[0]);
        c[0][3] = Func::impl(src[3 + src_idx], weight[0][weight_idx], c[0][3],
                             temp[2]);
    }
    static MEGDNN_ALWAYS_INLINE void impl(T& c, T2& src, T3& weight) {
        c[0][0] = Func::impl(src[0 + src_idx], weight[0][weight_idx], c[0][0]);
        c[0][1] = Func::impl(src[1 + src_idx], weight[0][weight_idx], c[0][1]);
        c[0][2] = Func::impl(src[2 + src_idx], weight[0][weight_idx], c[0][2]);
        c[0][3] = Func::impl(src[3 + src_idx], weight[0][weight_idx], c[0][3]);
    }
};

template <int src_idx, int weight_idx, typename Func, typename T, typename T2,
          typename T3, typename T4>
struct ShiftCalHelper<src_idx, weight_idx, 2, Func, 1, T, T2, T3, T4> {
    static MEGDNN_ALWAYS_INLINE void impl(T& c, T2& src, T3& weight, T4& temp) {
        c[0][0] = Func::impl(src[(0 + src_idx) % 8], weight[0][weight_idx],
                             c[0][0], temp[0]);
        c[1][0] = Func::impl(src[(0 + src_idx) % 8], weight[1][weight_idx],
                             c[1][0], temp[1]);
        c[0][1] = Func::impl(src[(1 + src_idx) % 8], weight[0][weight_idx],
                             c[0][1], temp[2]);
        c[1][1] = Func::impl(src[(1 + src_idx) % 8], weight[1][weight_idx],
                             c[1][1], temp[3]);
        c[0][2] = Func::impl(src[(2 + src_idx) % 8], weight[0][weight_idx],
                             c[0][2], temp[0]);
        c[1][2] = Func::impl(src[(2 + src_idx) % 8], weight[1][weight_idx],
                             c[1][2], temp[1]);
        c[0][3] = Func::impl(src[(3 + src_idx) % 8], weight[0][weight_idx],
                             c[0][3], temp[2]);
        c[1][3] = Func::impl(src[(3 + src_idx) % 8], weight[1][weight_idx],
                             c[1][3], temp[3]);

        c[0][4] = Func::impl(src[(4 + src_idx) % 8], weight[0][weight_idx],
                             c[0][4], temp[0]);
        c[1][4] = Func::impl(src[(4 + src_idx) % 8], weight[1][weight_idx],
                             c[1][4], temp[1]);
        c[0][5] = Func::impl(src[(5 + src_idx) % 8], weight[0][weight_idx],
                             c[0][5], temp[2]);
        c[1][5] = Func::impl(src[(5 + src_idx) % 8], weight[1][weight_idx],
                             c[1][5], temp[3]);
        c[0][6] = Func::impl(src[(6 + src_idx) % 8], weight[0][weight_idx],
                             c[0][6], temp[0]);
        c[1][6] = Func::impl(src[(6 + src_idx) % 8], weight[1][weight_idx],
                             c[1][6], temp[1]);
        c[0][7] = Func::impl(src[(7 + src_idx) % 8], weight[0][weight_idx],
                             c[0][7], temp[2]);
        c[1][7] = Func::impl(src[(7 + src_idx) % 8], weight[1][weight_idx],
                             c[1][7], temp[3]);
    }
    static MEGDNN_ALWAYS_INLINE void impl(T&, T2&, T3&);
};
template <int src_idx, int weight_idx, typename Func, typename T, typename T2,
          typename T3, typename T4>
struct ShiftCalHelper<src_idx, weight_idx, 1, Func, 1, T, T2, T3, T4> {
    static MEGDNN_ALWAYS_INLINE void impl(T& c, T2& src, T3& weight, T4& temp) {
        c[0][0] = Func::impl(src[(0 + src_idx) % 8], weight[0][weight_idx],
                             c[0][0], temp[0]);
        c[0][1] = Func::impl(src[(1 + src_idx) % 8], weight[0][weight_idx],
                             c[0][1], temp[1]);
        c[0][2] = Func::impl(src[(2 + src_idx) % 8], weight[0][weight_idx],
                             c[0][2], temp[2]);
        c[0][3] = Func::impl(src[(3 + src_idx) % 8], weight[0][weight_idx],
                             c[0][3], temp[3]);
        c[0][4] = Func::impl(src[(4 + src_idx) % 8], weight[0][weight_idx],
                             c[0][4], temp[0]);
        c[0][5] = Func::impl(src[(5 + src_idx) % 8], weight[0][weight_idx],
                             c[0][5], temp[1]);
        c[0][6] = Func::impl(src[(6 + src_idx) % 8], weight[0][weight_idx],
                             c[0][6], temp[2]);
        c[0][7] = Func::impl(src[(7 + src_idx) % 8], weight[0][weight_idx],
                             c[0][7], temp[3]);
    }
    static MEGDNN_ALWAYS_INLINE void impl(T&, T2&, T3&);
};

template <int src_idx, int weight_idx, int c_dim, typename FUNC, int stride,
          typename T, typename T2, typename T3, typename T4>
MEGDNN_ALWAYS_INLINE void cal_helper(T& c, T2& src, T3& weight, T4& temp) {
    ShiftCalHelper<src_idx, weight_idx, c_dim, FUNC, stride, T, T2, T3,
                   T4>::impl(c, src, weight, temp);
}
template <int src_idx, int weight_idx, int c_dim, typename FUNC, int stride,
          typename T, typename T2, typename T3>
MEGDNN_ALWAYS_INLINE void cal_helper(T& c, T2& src, T3& weight) {
    ShiftCalHelper<src_idx, weight_idx, c_dim, FUNC, stride, T, T2, T3,
                   int>::impl(c, src, weight);
};

template <int oc>
struct OCHelper {
public:
    static const int val = 0;
};
template <>
struct OCHelper<4> {
public:
    static const int val = 1;
};
template <>
struct OCHelper<8> {
public:
    static const int val = 2;
};

template <BiasMode bias_mode, typename Op, int remain_w, int filter_size,
          int oc_block, int stride>
struct KerNeonXXs2NchwNchw44 {
    static void impl(const int8_t* src_ptr, const int8_t* weight_ptr,
                     const int32_t* bias_ptr, int8_t* dst_ptr, int ic, int ih,
                     int iw, int ld_dst_oc, const Op& op);
};
/**
 * filter shape = (oc/4, ic, 7, 7, 4), first 4 oc is f0 = filter[0, 0, :, :, :]
 * calculate sequence    \
 * f0[0:1, 0:1, 4] dot4, \
 * f0[0:1, 2:3, 4] dot4, \
 * f0[0:1, 4:5, 4] dot4, \
 * f0[0:1, 6, 4] dot2,   \
 * ...
 * f0[6, 0:1, 4] dot2,   \
 * f0[6, 2:3, 4] dot2,   \
 * f0[6, 4:5, 4] dot2,   \
 * f0[6, 6, 4] dot1,     \
 * look like:
 *       |---|---|---|-|
 *       |x x|x x|x x|x|
 *       |x x|x x|x x|x|
 *       |---|---|---|-|
 *       |x x|x x|x x|x|
 *       |x x|x x|x x|x|
 *       |---|---|---|-|
 *       |x x|x x|x x|x|
 *       |x x|x x|x x|x|
 *       |---|---|---|-|
 *       |x x|x x|x x|x|
 *       |---|---|---|-|
 **/
template <BiasMode bias_mode, typename Op, int remain_w, int oc_block,
          int stride>
struct KerNeonXXs2NchwNchw44<bias_mode, Op, remain_w, 7, oc_block, stride> {
    static void impl(const int8_t* src_ptr, const int8_t* weight_ptr,
                     const int32_t* bias_ptr, int8_t* dst_ptr, int ic, int ih,
                     int iw, int ld_dst_oc, const Op& op) {
        static const uint8_t src_idx_buffer[16] = {0, 8, 0, 8, 0, 8, 0, 8,
                                                   0, 8, 0, 8, 0, 8, 0, 8};
        constexpr int filter_size = 7;
        constexpr int ic_step = 1;
        constexpr int oc_step = 4;
        constexpr int pack_iw_len = 4;
        constexpr int fh_step = 2;
        constexpr int fh_end = filter_size / fh_step * fh_step;
        constexpr int c_dim = OCHelper<oc_block>::val;

        const int ic_stride = ih * iw * pack_iw_len;
        const int ld_dot4_weight_oc = oc_step * filter_size * filter_size * ic;

        int32x4_t c[c_dim][4];

        init_ocx_ow4<c_dim, bias_mode>(c, bias_ptr, oc_step);

        for (int ic_idx = 0; ic_idx < ic; ic_idx += ic_step) {
            for (int fh_idx = 0; fh_idx < fh_end; fh_idx += fh_step) {
                const int8_t* nchw_src_ptr =
                        src_ptr + ic_idx * ic_stride +
                        fh_idx * iw * ic_step * pack_iw_len;
                int8x16_t src[6];
                int8x16_t dot4_weight[c_dim][3];
                int16x8_t temp_c[4];
                load_helper<3, 0, 16, c_dim, Vld1q_s8>(dot4_weight, weight_ptr,
                                                       ld_dot4_weight_oc);
                load_helper<6, 0, 16, 0, Vld1q_s8>(src, nchw_src_ptr, 0);
                cal_helper<0, 0, c_dim, Vdotq_s32_h, stride>(
                        c, src, dot4_weight, temp_c);
                cal_helper<1, 1, c_dim, Vdotq_s32_h, stride>(
                        c, src, dot4_weight, temp_c);
                cal_helper<2, 2, c_dim, Vdotq_s32_h, stride>(
                        c, src, dot4_weight, temp_c);

                int8x8_t src_dot2[4];
                int8x8_t dot2_weight[c_dim][1];
                load_helper<1, 3 * 16, 8, c_dim, Vld1_s8>(
                        dot2_weight, weight_ptr, ld_dot4_weight_oc);
                load_helper<4, 3 * 16, 16, 0, Vld1_s8>(src_dot2, nchw_src_ptr,
                                                       0);
                cal_helper<0, 0, c_dim, Vdot2_s32_h, stride>(
                        c, src_dot2, dot2_weight, temp_c);
                weight_ptr += filter_size * pack_iw_len * fh_step;
            }
            const int8_t* nchw_src_ptr = src_ptr + ic_idx * ic_stride +
                                         6 * iw * ic_step * pack_iw_len;

            int8x8_t dot2_weight[c_dim][3];
            int16x8_t temp_c[4];
            int8x8_t src_dot2[6];
            uint8x16_t tbl = vld1q_u8(src_idx_buffer);
            load_helper<3, 0, 8, c_dim, Vld1_s8>(dot2_weight, weight_ptr,
                                                 ld_dot4_weight_oc);
            load_helper_x<6, 0, 16, 0, Vldq_tbl_low_s8>(src_dot2, nchw_src_ptr,
                                                        0, tbl);
            cal_helper<0, 0, c_dim, Vdot2_s32_h, stride>(c, src_dot2,
                                                         dot2_weight, temp_c);
            cal_helper<1, 1, c_dim, Vdot2_s32_h, stride>(c, src_dot2,
                                                         dot2_weight, temp_c);
            cal_helper<2, 2, c_dim, Vdot2_s32_h, stride>(c, src_dot2,
                                                         dot2_weight, temp_c);

            int16x8_t dot1_weight[c_dim][1];
            int16x8_t src_dot1[4];
            load_helper<1, 3 * 8, 8, c_dim, Vldq_dup_4s8_8s16>(
                    dot1_weight, weight_ptr, ld_dot4_weight_oc);
            load_helper<4, 3 * 16, 16, 0, Vld1_dup_s8_s16>(src_dot1,
                                                           nchw_src_ptr, 0);
            cal_helper<0, 0, c_dim, Vmlal_s16, stride>(c, src_dot1,
                                                       dot1_weight);
            weight_ptr += filter_size * pack_iw_len;
        }
        store_ocx_ow4_remain_static<c_dim, remain_w>(c, op, dst_ptr, ld_dst_oc);
    }
};
template <BiasMode bias_mode, typename Op, int remain_w, int oc_block,
          int stride>
struct KerNeonXXs2NchwNchw44<bias_mode, Op, remain_w, 5, oc_block, stride> {
    static void impl(const int8_t* src_ptr, const int8_t* weight_ptr,
                     const int32_t* bias_ptr, int8_t* dst_ptr, int ic, int ih,
                     int iw, int ld_dst_oc, const Op& op) {
        constexpr int filter_size = 5;
        static const uint8_t src_idx_buffer[16] = {0, 8, 0, 8, 0, 8, 0, 8,
                                                   0, 8, 0, 8, 0, 8, 0, 8};
        constexpr int ih_step = 2;
        constexpr int ic_step = 1;
        constexpr int oc_step = 4;
        constexpr int pack_iw_len = 4;
        constexpr int fh_end = filter_size / ih_step * ih_step;

        const int ic_stride = ih * iw * pack_iw_len;
        const int ld_dot4_weight_oc = oc_step * filter_size * filter_size * ic;
        constexpr int c_dim = OCHelper<oc_block>::val;
        int32x4_t c[c_dim][4];

        init_ocx_ow4<c_dim, bias_mode>(c, bias_ptr, oc_step);

        for (int ic_idx = 0; ic_idx < ic; ic_idx += ic_step) {
            for (int fh_idx = 0; fh_idx < fh_end; fh_idx += ih_step) {
                const int8_t* nchw_src_ptr =
                        src_ptr + ic_idx * ic_stride +
                        fh_idx * iw * ic_step * pack_iw_len;
                int8x16_t src[5];
                int8x16_t dot4_weight[c_dim][2];
                int16x8_t temp_c[4];
                load_helper<2, 0, 16, c_dim, Vld1q_s8>(dot4_weight, weight_ptr,
                                                       ld_dot4_weight_oc);
                load_helper<5, 0, 16, 0, Vld1q_s8>(src, nchw_src_ptr, 0);
                cal_helper<0, 0, c_dim, Vdotq_s32_h, stride>(
                        c, src, dot4_weight, temp_c);
                cal_helper<1, 1, c_dim, Vdotq_s32_h, stride>(
                        c, src, dot4_weight, temp_c);

                int8x8_t src_dot2[4];
                int8x8_t dot2_weight[c_dim][1];
                load_helper<1, 2 * 16, 8, c_dim, Vld1_s8>(
                        dot2_weight, weight_ptr, ld_dot4_weight_oc);
                load_helper<4, 2 * 16, 16, 0, Vld1_s8>(src_dot2, nchw_src_ptr,
                                                       0);
                cal_helper<0, 0, c_dim, Vdot2_s32_h, stride>(
                        c, src_dot2, dot2_weight, temp_c);
                weight_ptr += filter_size * pack_iw_len * ih_step;
            }
            const int8_t* nchw_src_ptr = src_ptr + ic_idx * ic_stride +
                                         fh_end * iw * ic_step * pack_iw_len;

            int8x8_t dot2_weight[c_dim][2];
            int16x8_t temp_c[4];
            int8x8_t src_dot2[5];
            uint8x16_t tbl = vld1q_u8(src_idx_buffer);
            load_helper<2, 0, 8, c_dim, Vld1_s8>(dot2_weight, weight_ptr,
                                                 ld_dot4_weight_oc);
            load_helper_x<5, 0, 16, 0, Vldq_tbl_low_s8>(src_dot2, nchw_src_ptr,
                                                        0, tbl);

            cal_helper<0, 0, c_dim, Vdot2_s32_h, stride>(c, src_dot2,
                                                         dot2_weight, temp_c);
            cal_helper<1, 1, c_dim, Vdot2_s32_h, stride>(c, src_dot2,
                                                         dot2_weight, temp_c);

            int16x8_t dot1_weight[c_dim][1];
            int16x8_t src_dot1[4];
            load_helper<1, 2 * 8, 8, c_dim, Vldq_dup_4s8_8s16>(
                    dot1_weight, weight_ptr, ld_dot4_weight_oc);
            load_helper<4, 2 * 16, 16, 0, Vld1_dup_s8_s16>(src_dot1,
                                                           nchw_src_ptr, 0);

            cal_helper<0, 0, c_dim, Vmlal_s16, stride>(c, src_dot1,
                                                       dot1_weight);
            weight_ptr += filter_size * pack_iw_len;
        }
        store_ocx_ow4_remain_static<c_dim, remain_w>(c, op, dst_ptr, ld_dst_oc);
    }
};
/**
 * filter shape = (oc/4, ic, 3, 3, 4), first 4 oc is f0 = filter[0, 0, :, :, :]
 * calculate sequence    \
 * f0[0:1, 0:1, 4] dot4, \
 * f0[0:1, 2, 4] dot2,   \
 * f0[2, 0:1, 4] dot2,   \
 * f0[2, 2, 4] dot1      \
 * look like:
 *       |---|-|
 *       |x x|x|
 *       |x x|x|
 *       |-----|
 *       |x x|x|
 *       |-----|
 **/
template <BiasMode bias_mode, typename Op, int remain_w, int oc_block,
          int stride>
struct KerNeonXXs2NchwNchw44<bias_mode, Op, remain_w, 3, oc_block, stride> {
    static void impl(const int8_t* src_ptr, const int8_t* weight_ptr,
                     const int32_t* bias_ptr, int8_t* dst_ptr, int ic, int ih,
                     int iw, int ld_dst_oc, const Op& op) {
        constexpr int filter_size = 3;
        static const uint8_t src_idx_buffer[16] = {0, 8, 0, 8, 0, 8, 0, 8,
                                                   0, 8, 0, 8, 0, 8, 0, 8};
        constexpr int oc_step = 4;
        constexpr int ic_step = 1;
        constexpr int loop_ic_step = 1;
        constexpr int pack_iw_len = 4;

        const int ic_stride = ih * iw * pack_iw_len;
        const int ld_weight_oc = oc_step * filter_size * filter_size * ic;
        constexpr int c_dim = OCHelper<oc_block>::val;

        int32x4_t c[c_dim][4];
        init_ocx_ow4<c_dim, bias_mode>(c, bias_ptr, oc_step);

        for (int ic_idx = 0; ic_idx < ic; ic_idx += loop_ic_step) {
            // first 2 line
            {
                const int8_t* nchw_src_ptr = src_ptr + ic_idx * ic_stride;
                int8x16_t src[4];
                int8x16_t dot4_weight[c_dim][1];
                int16x8_t temp_c[4];
                load_helper<1, 0, 16, c_dim, Vld1q_s8>(dot4_weight, weight_ptr,
                                                       ld_weight_oc);
                load_helper<4, 0, 16, 0, Vld1q_s8>(src, nchw_src_ptr, 0);
                cal_helper<0, 0, c_dim, Vdotq_s32_h, stride>(
                        c, src, dot4_weight, temp_c);

                int8x8_t src_dot2[4];
                int8x8_t dot2_weight[c_dim][1];
                load_helper<1, 1 * 16, 8, c_dim, Vld1_s8>(
                        dot2_weight, weight_ptr, ld_weight_oc);
                load_helper<4, 1 * 16, 16, 0, Vld1_s8>(src_dot2, nchw_src_ptr,
                                                       0);
                cal_helper<0, 0, c_dim, Vdot2_s32_h, stride>(
                        c, src_dot2, dot2_weight, temp_c);
            }
            // last line
            {
                const int8_t* nchw_src_ptr = src_ptr + ic_idx * ic_stride +
                                             2 * iw * ic_step * pack_iw_len;
                int16x8_t temp_c[4];
                int8x8_t src_dot2[4];
                int8x8_t dot2_weight[c_dim][1];
                uint8x16_t tbl = vld1q_u8(src_idx_buffer);
                load_helper<1, 24, 8, c_dim, Vld1_s8>(dot2_weight, weight_ptr,
                                                      ld_weight_oc);
                load_helper_x<4, 0, 16, 0, Vldq_tbl_low_s8>(
                        src_dot2, nchw_src_ptr, 0, tbl);
                cal_helper<0, 0, c_dim, Vdot2_s32_h, stride>(
                        c, src_dot2, dot2_weight, temp_c);
                int16x8_t dot1_weight[c_dim][1];
                int16x8_t src_dot1[4];
                load_helper<1, 32, 8, c_dim, Vldq_dup_4s8_8s16>(
                        dot1_weight, weight_ptr, ld_weight_oc);
                load_helper<4, 1 * 16, 16, 0, Vld1_dup_s8_s16>(src_dot1,
                                                               nchw_src_ptr, 0);
                cal_helper<0, 0, c_dim, Vmlal_s16, stride>(c, src_dot1,
                                                           dot1_weight);
                weight_ptr += filter_size * filter_size * pack_iw_len;
            }
        }
        store_ocx_ow4_remain_static<c_dim, remain_w>(c, op, dst_ptr, ld_dst_oc);
    }
};
template <BiasMode bias_mode, typename Op, int remain_w, int oc_block,
          int stride>
struct KerNeonXXs2NchwNchw44<bias_mode, Op, remain_w, 2, oc_block, stride> {
    static void impl(const int8_t* src_ptr, const int8_t* weight_ptr,
                     const int32_t* bias_ptr, int8_t* dst_ptr, int ic, int ih,
                     int iw, int ld_dst_oc, const Op& op) {
        constexpr int filter_size = 2;
        constexpr int oc_step = 4;
        constexpr int loop_ic_step = 1;
        constexpr int pack_iw_len = 4;

        const int ic_stride = ih * iw * pack_iw_len;
        const int ld_weight_oc = oc_step * filter_size * filter_size * ic;
        constexpr int c_dim = OCHelper<oc_block>::val;

        int32x4_t c[c_dim][4];
        init_ocx_ow4<c_dim, bias_mode>(c, bias_ptr, oc_step);

        for (int ic_idx = 0; ic_idx < ic; ic_idx += loop_ic_step) {
            const int8_t* nchw_src_ptr = src_ptr + ic_idx * ic_stride;
            int8x16_t src[4];
            int8x16_t dot4_weight[c_dim][1];
            int16x8_t temp_c[4];
            load_helper<1, 0, 16, c_dim, Vld1q_s8>(dot4_weight, weight_ptr,
                                                   ld_weight_oc);
            load_helper<4, 0, 16, 0, Vld1q_s8>(src, nchw_src_ptr, 0);
            cal_helper<0, 0, c_dim, Vdotq_s32_h, stride>(c, src, dot4_weight,
                                                         temp_c);
            weight_ptr += oc_step * filter_size * filter_size;
        }
        store_ocx_ow4_remain_static<c_dim, remain_w>(c, op, dst_ptr, ld_dst_oc);
    }
};

template <BiasMode bias_mode, typename Op, int remain_w, int oc_block>
struct KerNeonXXs2NchwNchw44<bias_mode, Op, remain_w, 2, oc_block, 1> {
    static void impl(const int8_t* src_ptr, const int8_t* weight_ptr,
                     const int32_t* bias_ptr, int8_t* dst_ptr, int ic, int ih,
                     int iw, int ld_dst_oc, const Op& op) {
        constexpr int stride = 1;
        constexpr int filter_height = 2;
        constexpr int filter_width = 4;
        constexpr int oc_step = 4;
        constexpr int loop_ic_step = 1;
        constexpr int simd_len = 16;
        constexpr int pack_iw_len = 16;
        constexpr int src_reg = 8;
        constexpr int weight_reg = 1;

        const int ic_stride = ih * iw * pack_iw_len;
        const int ld_weight_oc = oc_step * filter_height * filter_width * ic;
        constexpr int c_dim = OCHelper<oc_block>::val;
        int32x4_t c[c_dim][8];
        init_ocx_ow8<c_dim, bias_mode, 8>(c, bias_ptr, oc_step);

        for (int ic_idx = 0; ic_idx < ic; ic_idx += loop_ic_step) {
            const int8_t* nchw_src_ptr = src_ptr + ic_idx * ic_stride;
            int8x16_t src[src_reg];
            int8x16_t dot4_weight[c_dim][weight_reg];
            int16x8_t temp_c[4];
            load_helper<weight_reg, 0, simd_len, c_dim, Vld1q_s8>(
                    dot4_weight, weight_ptr, ld_weight_oc);
            load_helper<src_reg, 0, simd_len, 0, Vld1q_s8>(
                    src, nchw_src_ptr + 0 * iw * pack_iw_len, 0);
            cal_helper<0, 0, c_dim, Vdotq_s32_h, stride>(c, src, dot4_weight,
                                                         temp_c);

            load_helper<weight_reg, 0, simd_len, c_dim, Vld1q_s8>(
                    dot4_weight, weight_ptr + 1 * filter_width * oc_step,
                    ld_weight_oc);
            load_helper<src_reg, 0, simd_len, 0, Vld1q_s8>(
                    src, nchw_src_ptr + 1 * iw * pack_iw_len, 0);
            cal_helper<0, 0, c_dim, Vdotq_s32_h, stride>(c, src, dot4_weight,
                                                         temp_c);

            weight_ptr += oc_step * filter_height * filter_width;
        }

        store_ocx_ow8_remain_static_dt<c_dim, remain_w, Op, dt_qint8*>(
                c, op, dst_ptr, ld_dst_oc);
    }
};

template <BiasMode bias_mode, typename Op, int remain_w, int oc_block>
struct KerNeonXXs2NchwNchw44<bias_mode, Op, remain_w, 3, oc_block, 1> {
    static void impl(const int8_t* src_ptr, const int8_t* weight_ptr,
                     const int32_t* bias_ptr, int8_t* dst_ptr, int ic, int ih,
                     int iw, int ld_dst_oc, const Op& op) {
        constexpr int stride = 1;
        constexpr int filter_height = 3;
        constexpr int filter_width = 4;
        constexpr int oc_step = 4;
        constexpr int loop_ic_step = 1;
        constexpr int simd_len = 16;
        constexpr int pack_iw_len = 16;
        constexpr int src_reg = 8;
        constexpr int weight_reg = 1;

        const int ic_stride = ih * iw * pack_iw_len;
        const int ld_weight_oc = oc_step * filter_height * filter_width * ic;
        constexpr int c_dim = OCHelper<oc_block>::val;
        int32x4_t c[c_dim][8];
        init_ocx_ow8<c_dim, bias_mode, 8>(c, bias_ptr, oc_step);

        for (int ic_idx = 0; ic_idx < ic; ic_idx += loop_ic_step) {
            const int8_t* nchw_src_ptr = src_ptr + ic_idx * ic_stride;
            int8x16_t src[src_reg];
            int8x16_t dot4_weight[c_dim][weight_reg];
            int16x8_t temp_c[4];
            load_helper<weight_reg, 0, simd_len, c_dim, Vld1q_s8>(
                    dot4_weight, weight_ptr, ld_weight_oc);

            load_helper<src_reg, 0, simd_len, 0, Vld1q_s8>(
                    src, nchw_src_ptr + 0 * iw * pack_iw_len, 0);
            cal_helper<0, 0, c_dim, Vdotq_s32_h, stride>(c, src, dot4_weight,
                                                         temp_c);
            load_helper<weight_reg, 0, simd_len, c_dim, Vld1q_s8>(
                    dot4_weight, weight_ptr + 1 * filter_width * oc_step,
                    ld_weight_oc);

            load_helper<src_reg, 0, simd_len, 0, Vld1q_s8>(
                    src, nchw_src_ptr + 1 * iw * pack_iw_len, 0);
            cal_helper<0, 0, c_dim, Vdotq_s32_h, stride>(c, src, dot4_weight,
                                                         temp_c);

            load_helper<weight_reg, 0, simd_len, c_dim, Vld1q_s8>(
                    dot4_weight, weight_ptr + 2 * filter_width * oc_step,
                    ld_weight_oc);
            load_helper<src_reg, 0, simd_len, 0, Vld1q_s8>(
                    src, nchw_src_ptr + 2 * iw * pack_iw_len, 0);
            cal_helper<0, 0, c_dim, Vdotq_s32_h, stride>(c, src, dot4_weight,
                                                         temp_c);

            weight_ptr += oc_step * filter_height * filter_width;
        }
        store_ocx_ow8_remain_static_dt<c_dim, remain_w, Op, dt_qint8*>(
                c, op, dst_ptr, ld_dst_oc);
    }
};

template <BiasMode bias_mode, typename Op, int remain_w, int oc_block>
struct KerNeonXXs2NchwNchw44<bias_mode, Op, remain_w, 5, oc_block, 1> {
    static void impl(const int8_t* src_ptr, const int8_t* weight_ptr,
                     const int32_t* bias_ptr, int8_t* dst_ptr, int ic, int ih,
                     int iw, int ld_dst_oc, const Op& op) {
        constexpr int stride = 1;
        constexpr int filter_height = 5;
        constexpr int filter_width = 8;
        constexpr int oc_step = 4;
        constexpr int loop_ic_step = 1;
        constexpr int simd_len = 16;
        constexpr int pack_iw_len = 16;
        constexpr int src_reg = 8;
        constexpr int weight_reg = 2;

        const int ic_stride = ih * iw * pack_iw_len;
        const int ld_weight_oc = oc_step * filter_height * filter_width * ic;
        constexpr int c_dim = OCHelper<oc_block>::val;
        int32x4_t c[c_dim][8];
        init_ocx_ow8<c_dim, bias_mode, 8>(c, bias_ptr, oc_step);

        for (int ic_idx = 0; ic_idx < ic; ic_idx += loop_ic_step) {
            const int8_t* nchw_src_ptr = src_ptr + ic_idx * ic_stride;
            int8x16_t src[src_reg];
            int8x16_t dot4_weight[c_dim][weight_reg];
            int16x8_t temp_c[4];
#define cb(step)                                                               \
    load_helper<weight_reg, 0, simd_len, c_dim, Vld1q_s8>(                     \
            dot4_weight, weight_ptr + step * filter_width * oc_step,           \
            ld_weight_oc);                                                     \
    load_helper<src_reg, 0, simd_len, 0, Vld1q_s8>(                            \
            src, nchw_src_ptr + step * iw * pack_iw_len, 0);                   \
    cal_helper<0, 0, c_dim, Vdotq_s32_h, stride>(c, src, dot4_weight, temp_c); \
    load_helper<4, 0, simd_len, 0, Vld1q_s8>(                                  \
            src,                                                               \
            nchw_src_ptr + step * iw * pack_iw_len + src_reg * pack_iw_len,    \
            0);                                                                \
    cal_helper<4, 1, c_dim, Vdotq_s32_h, stride>(c, src, dot4_weight, temp_c);
            UNROLL_CALL_RAW(5, cb);
#undef cb
            weight_ptr += oc_step * filter_height * filter_width;
        }
        store_ocx_ow8_remain_static_dt<c_dim, remain_w, Op, dt_qint8*>(
                c, op, dst_ptr, ld_dst_oc);
    }
};

template <BiasMode bias_mode, typename Op, int remain_w, int oc_block>
struct KerNeonXXs2NchwNchw44<bias_mode, Op, remain_w, 7, oc_block, 1> {
    static void impl(const int8_t* src_ptr, const int8_t* weight_ptr,
                     const int32_t* bias_ptr, int8_t* dst_ptr, int ic, int ih,
                     int iw, int ld_dst_oc, const Op& op) {
        constexpr int stride = 1;
        constexpr int filter_height = 7;
        constexpr int filter_width = 8;
        constexpr int oc_step = 4;
        constexpr int loop_ic_step = 1;
        constexpr int simd_len = 16;
        constexpr int pack_iw_len = 16;
        constexpr int src_reg = 8;
        constexpr int weight_reg = 2;

        const int ic_stride = ih * iw * pack_iw_len;
        const int ld_weight_oc = oc_step * filter_height * filter_width * ic;
        constexpr int c_dim = OCHelper<oc_block>::val;
        int32x4_t c[c_dim][8];
        init_ocx_ow8<c_dim, bias_mode, 8>(c, bias_ptr, oc_step);

        for (int ic_idx = 0; ic_idx < ic; ic_idx += loop_ic_step) {
            const int8_t* nchw_src_ptr = src_ptr + ic_idx * ic_stride;
            int8x16_t src[src_reg];
            int8x16_t dot4_weight[c_dim][weight_reg];
            int16x8_t temp_c[4];
#define cb(step)                                                               \
    load_helper<weight_reg, 0, simd_len, c_dim, Vld1q_s8>(                     \
            dot4_weight, weight_ptr + step * filter_width * oc_step,           \
            ld_weight_oc);                                                     \
    load_helper<src_reg, 0, simd_len, 0, Vld1q_s8>(                            \
            src, nchw_src_ptr + step * iw * pack_iw_len, 0);                   \
    cal_helper<0, 0, c_dim, Vdotq_s32_h, stride>(c, src, dot4_weight, temp_c); \
    load_helper<4, 0, simd_len, 0, Vld1q_s8>(                                  \
            src,                                                               \
            nchw_src_ptr + step * iw * pack_iw_len + src_reg * pack_iw_len,    \
            0);                                                                \
    cal_helper<4, 1, c_dim, Vdotq_s32_h, stride>(c, src, dot4_weight, temp_c);

            UNROLL_CALL_RAW(7, cb);
#undef cb
            weight_ptr += oc_step * filter_height * filter_width;
        }
        store_ocx_ow8_remain_static_dt<c_dim, remain_w, Op, dt_qint8*>(
                c, op, dst_ptr, ld_dst_oc);
    }
};

enum PACK_MODE { NO_PAD = 0, FIRST_PAD = 1, LAST_PAD = 2 };
template <PACK_MODE mode>
MEGDNN_ALWAYS_INLINE void pack_src_one_line(const int8_t* inptr, int8_t* outptr,
                                            int left_pad, int right_pad,
                                            const int iw) {
    const int8_t* src_row_0 = inptr;
    const int8_t* src_row_1 = inptr + iw;
    constexpr int combine_row = 2;
    constexpr int iw_step = 16;
    constexpr int src_expand = 4;
    constexpr int out_gap = iw_step * src_expand;
    const int iw_end = iw / iw_step * iw_step;

    memset(outptr, 0, combine_row * left_pad * src_expand * sizeof(int8_t));
    outptr += combine_row * left_pad * src_expand;

    for (int iw_idx = 0; iw_idx < iw_end; iw_idx += iw_step) {
        int8x16_t row0 = vld1q_s8(src_row_0 + iw_idx);
        int8x16_t row1 = vdupq_n_s8(0);
        if (mode == PACK_MODE::NO_PAD) {
            row1 = vld1q_s8(src_row_1 + iw_idx);
        } else if (mode == PACK_MODE::FIRST_PAD) {
            row1 = row0;
            row0 = vdupq_n_s8(0);
        }
        int8x16x2_t pack_rows = vzipq_s8(row0, row1);
#define STORE_8S8(step)                         \
    vst1_s8(outptr + step * 8,                  \
            vreinterpret_s8_s16(vdup_laneq_s16( \
                    vreinterpretq_s16_s8(pack_rows.val[0]), step)));

        UNROLL_CALL_RAW(8, STORE_8S8);
#undef STORE_8S8
#define STORE_8S8(step)                         \
    vst1_s8(outptr + out_gap + step * 8,        \
            vreinterpret_s8_s16(vdup_laneq_s16( \
                    vreinterpretq_s16_s8(pack_rows.val[1]), step)));

        UNROLL_CALL_RAW(8, STORE_8S8);
#undef STORE_8S8
        outptr += out_gap * combine_row;
    }
    for (int iw_idx = iw_end; iw_idx < iw; iw_idx++) {
        int8x8_t row0 = vld1_dup_s8(src_row_0 + iw_idx);
        int8x8_t row1 = vdup_n_s8(0);
        if (mode == PACK_MODE::NO_PAD) {
            row1 = vld1_dup_s8(src_row_1 + iw_idx);
        } else if (mode == PACK_MODE::FIRST_PAD) {
            row1 = row0;
            row0 = vdup_n_s8(0);
        }
        int8x8x2_t pack_rows = vzip_s8(row0, row1);
        vst1_s8(outptr, pack_rows.val[0]);
        outptr += src_expand * combine_row;
    }
    memset(outptr, 0, combine_row * right_pad * src_expand * sizeof(int8_t));
    outptr += combine_row * right_pad * src_expand;
}
template <int stride>
void pack_nchw_src_for_nchw44_conv(const int8_t* inptr, int8_t* outptr,
                                   const int ic, const int top_pad,
                                   const int bottom_pad, const int left_pad,
                                   const int right_pad, const int ih,
                                   const int iw, const int iw2, const int pw,
                                   int8_t* temp_ptr);
/**
 * pack (ic, h, w) to (ic, h / 2, 2 * w)
 * pack interleave two adjacent row in src and repeat 4 times, store to one row
 * */
template <>
void pack_nchw_src_for_nchw44_conv<2>(const int8_t* inptr, int8_t* outptr,
                                      const int ic, const int top_pad,
                                      const int bottom_pad, const int left_pad,
                                      const int right_pad, const int ih,
                                      const int iw, const int, const int,
                                      int8_t*) {
    constexpr int src_expand = 4;
    constexpr int oh_step = 2;
    const int oh = ih + top_pad + bottom_pad;
    const int oh_end = div_floor(ih + top_pad, oh_step) * oh_step;
    const int ow = (iw + left_pad + right_pad) * src_expand;

    for (int ic_idx = 0; ic_idx < ic; ++ic_idx) {
        int oh_idx = 0;
        for (; oh_idx < top_pad; oh_idx += oh_step) {
            if (top_pad - oh_idx >= oh_step) {
                memset(outptr, 0, oh_step * ow * sizeof(int8_t));
            } else {
                pack_src_one_line<PACK_MODE::FIRST_PAD>(inptr, outptr, left_pad,
                                                        right_pad, iw);
                inptr += iw;
            }
            outptr += oh_step * ow;
        }

        for (; oh_idx < oh_end; oh_idx += oh_step) {
            pack_src_one_line<PACK_MODE::NO_PAD>(inptr, outptr, left_pad,
                                                 right_pad, iw);
            inptr += oh_step * iw;
            outptr += oh_step * ow;
        }

        for (; oh_idx < oh; oh_idx += oh_step) {
            const int last_pad = oh_idx - ih - top_pad;
            if (last_pad >= 0) {
                memset(outptr, 0, oh_step * ow * sizeof(int8_t));
            } else {
                pack_src_one_line<PACK_MODE::LAST_PAD>(inptr, outptr, left_pad,
                                                       right_pad, iw);
                inptr += iw;
            }
            outptr += oh_step * ow;
        }
    }
}
/**
 * pack (ic, h, w) to (ic, h, w * 16)
 * pack interleave two adjacent row in src and repeat 4 times, store to one row
 * */
template <>
void pack_nchw_src_for_nchw44_conv<1>(const int8_t* sptr_origin,
                                      int8_t* sptr_base, const int ic,
                                      const int pad_top, const int pad_bottom,
                                      const int, const int, const int ih,
                                      const int iw, const int iw2, const int pw,
                                      int8_t* temp_ptr) {
    static uint8_t reorder_idx[16] = {0, 1, 0, 1, 0, 1, 0, 1,
                                      2, 3, 2, 3, 2, 3, 2, 3};
    uint8x16_t tbl_idx = vld1q_u8(&reorder_idx[0]);

    constexpr int iw_step = 4;
    constexpr int pack_iw_len = 16;
    const int ic_stride = ih * iw;
    const int iw_with_pad = iw + 2 * pw;
    const int iw_with_pad_end = iw_with_pad / iw_step * iw_step;
    rep(ic_idx, ic) {
        const int8_t* sptr = sptr_origin + ic_idx * ic_stride;
        memset(sptr_base, 0,
               sizeof(int8_t) * iw2 * (ih + pad_top + pad_bottom) *
                       pack_iw_len);
        sptr_base += iw2 * pad_top * pack_iw_len;
        rep(ih_idx, ih) {
            memset(temp_ptr, 0, iw_with_pad * sizeof(int8_t));
            memcpy(temp_ptr + pw, sptr, sizeof(int8_t) * iw);
            for (int iw_idx = 0; iw_idx < iw_with_pad_end; iw_idx += iw_step) {
                int8x16_t src[4];
                int8x16_t dst[4];
                src[0] = vld1q_s8(temp_ptr + iw_idx);
                src[1] = vld1q_s8(temp_ptr + iw_idx + 1);
                src[2] = vld1q_s8(temp_ptr + iw_idx + 2);
                src[3] = vld1q_s8(temp_ptr + iw_idx + 3);
                dst[0] = vqtbl1q_s8(src[0], tbl_idx);
                dst[1] = vqtbl1q_s8(src[1], tbl_idx);
                dst[2] = vqtbl1q_s8(src[2], tbl_idx);
                dst[3] = vqtbl1q_s8(src[3], tbl_idx);
                vst1q_s8(sptr_base + iw_idx * pack_iw_len + 0, dst[0]);
                vst1q_s8(sptr_base + iw_idx * pack_iw_len + 16, dst[1]);
                vst1q_s8(sptr_base + iw_idx * pack_iw_len + 32, dst[2]);
                vst1q_s8(sptr_base + iw_idx * pack_iw_len + 48, dst[3]);
            }
            for (int iw_idx = iw_with_pad_end; iw_idx < iw_with_pad; ++iw_idx) {
                int8x16_t src = vld1q_s8(temp_ptr + iw_idx);
                int8x16_t dst = vqtbl1q_s8(src, tbl_idx);
                vst1q_s8(sptr_base + iw_idx * pack_iw_len, dst);
            }
            sptr_base += iw2 * pack_iw_len;
            sptr += iw;
        }
        sptr_base += iw2 * pad_bottom * pack_iw_len;
    }
}

template <int stride>
void pack_nchw44_weight_for_nchw_conv(const int8_t* inptr, int8_t* outptr,
                                      const int ic, const int fh, const int fw,
                                      const int oc);
/**
 * pack {oc / 4, fh, fw, ic, 4(oc)} to {oc / 4, ic, fh * fw, 4(oc)}
 * pack interleave two adjacent row in filter to one row
 * */
template <>
void pack_nchw44_weight_for_nchw_conv<2>(const int8_t* inptr, int8_t* outptr,
                                         const int ic, const int fh,
                                         const int fw, const int oc) {
    constexpr int oc_step = 4;
    constexpr int ic_step = 2;
    constexpr int fh_step = 2;
    constexpr int fw_step = 2;
    const int ic_end = ic / ic_step * ic_step;
    const int ic_remain = ic - ic_end;
    const int fh_end = fh / fh_step * fh_step;
    const int fh_remain = fh - fh_end;
    const int fw_end = fw / fw_step * fw_step;
    const int fw_remain = fw - fw_end;
    const int filter_stride = ic * oc_step;
    static const uint8_t ic2_idx_h_buffer[16] = {0, 8,  1, 9,  2, 10, 3, 11,
                                                 4, 12, 5, 13, 6, 14, 7, 15};
    uint8x16_t ic2_idx_h = vld1q_u8(ic2_idx_h_buffer);
    for (int oc_idx = 0; oc_idx < oc; oc_idx += oc_step) {
        for (int ic_idx = 0; ic_idx < ic_end; ic_idx += ic_step) {
            const int ic_offset = ic_idx * oc_step;
            int8_t* output_ic0 = outptr + ic_idx * fh * fw * oc_step;
            int8_t* output_ic1 = output_ic0 + fh * fw * oc_step;
            for (int fh_idx = 0; fh_idx < fh_end; fh_idx += fh_step) {
                const int fh_offset = fh_idx * fw * filter_stride;
                for (int fw_idx = 0; fw_idx < fw; ++fw_idx) {
                    const int8_t* filter_ptr = inptr + fh_offset +
                                               fw_idx * filter_stride +
                                               ic_offset;
                    int8x8_t row_0 = vld1_s8(filter_ptr);
                    int8x8_t row_1 = vld1_s8(filter_ptr + fw * filter_stride);
                    int8x16_t combine_row = vcombine_s8(row_0, row_1);
                    combine_row = vqtbl1q_s8(combine_row, ic2_idx_h);
                    vst1_s8(output_ic0, vget_low_s8(combine_row));
                    vst1_s8(output_ic1, vget_high_s8(combine_row));
                    output_ic0 += 8;
                    output_ic1 += 8;
                }
            }
            if (fh_remain > 0) {
                const int fh_offset = fh_end * fw * filter_stride;
                for (int fw_idx = 0; fw_idx < fw_end; fw_idx += fw_step) {
                    const int8_t* filter_ptr = inptr + fh_offset +
                                               fw_idx * filter_stride +
                                               ic_offset;
                    int8x8_t row_0 = vld1_s8(filter_ptr);
                    int8x8_t row_1 = vld1_s8(filter_ptr + filter_stride);
                    int8x16_t combine_row = vcombine_s8(row_0, row_1);
                    combine_row = vqtbl1q_s8(combine_row, ic2_idx_h);
                    vst1_s8(output_ic0, vget_low_s8(combine_row));
                    vst1_s8(output_ic1, vget_high_s8(combine_row));
                    output_ic0 += 8;
                    output_ic1 += 8;
                }
                if (fw_remain > 0) {
                    const int8_t* filter_ptr = inptr + fh_offset +
                                               fw_end * filter_stride +
                                               ic_offset;
                    int8x8_t row_0 = vld1_s8(filter_ptr);
                    vst1_lane_s32((int32_t*)output_ic0,
                                  vreinterpret_s32_s8(row_0), 0);
                    vst1_lane_s32((int32_t*)output_ic1,
                                  vreinterpret_s32_s8(row_0), 1);
                    output_ic0 += 4;
                    output_ic1 += 4;
                }
            }
        }
        if (ic_remain > 0) {
            const int ic_offset = ic_end * oc_step;
            int8_t* output_ic0 = outptr + ic_end * fh * fw * oc_step;
            for (int fh_idx = 0; fh_idx < fh_end; fh_idx += fh_step) {
                const int fh_offset = fh_idx * fw * filter_stride;
                for (int fw_idx = 0; fw_idx < fw; ++fw_idx) {
                    const int8_t* filter_ptr = inptr + fh_offset +
                                               fw_idx * filter_stride +
                                               ic_offset;
                    int8x8_t row_0 = vreinterpret_s8_s32(
                            vld1_dup_s32((const int32_t*)(filter_ptr)));
                    int8x8_t row_1 = vreinterpret_s8_s32(vld1_dup_s32(
                            (const int32_t*)(filter_ptr + fw * filter_stride)));
                    int8x16_t combine_row = vcombine_s8(row_0, row_1);
                    combine_row = vqtbl1q_s8(combine_row, ic2_idx_h);
                    vst1_s8(output_ic0, vget_low_s8(combine_row));
                    output_ic0 += 8;
                }
            }
            if (fh_remain > 0) {
                const int fh_offset = fh_end * fw * filter_stride;
                for (int fw_idx = 0; fw_idx < fw_end; fw_idx += fw_step) {
                    const int8_t* filter_ptr = inptr + fh_offset +
                                               fw_idx * filter_stride +
                                               ic_offset;
                    int8x8_t row_0 = vreinterpret_s8_s32(
                            vld1_dup_s32((const int32_t*)(filter_ptr)));
                    int8x8_t row_1 = vreinterpret_s8_s32(vld1_dup_s32(
                            (const int32_t*)(filter_ptr + filter_stride)));
                    int8x16_t combine_row = vcombine_s8(row_0, row_1);
                    combine_row = vqtbl1q_s8(combine_row, ic2_idx_h);
                    vst1_s8(output_ic0, vget_low_s8(combine_row));
                    output_ic0 += 8;
                }
                if (fw_remain > 0) {
                    const int8_t* filter_ptr = inptr + fh_offset +
                                               fw_end * filter_stride +
                                               ic_offset;
                    *(int32_t*)(output_ic0) = *(const int32_t*)(filter_ptr);
                    output_ic0 += 4;
                }
            }
        }
        inptr += oc_step * fh * fw * ic;
        outptr += oc_step * fh * fw * ic;
    }
}
/**
 * pack {oc / 4, fh, fw, ic, 4(oc)} to {oc / 4, ic, fh ,fw/4, 4(oc)*4(fw)}
 * pack interleave two adjacent row in filter to one row
 * */
template <>
void pack_nchw44_weight_for_nchw_conv<1>(const int8_t* src_ptr, int8_t* dst_ptr,
                                         const int ic, const int fh,
                                         const int fw, const int oc) {
    constexpr int oc_step = 4;
    const int fw2 = round_up(fw, 4);
    const int fw_remain = fw2 - fw;
    const int dst_ic_stride = fh * fw2;
    const int oc_step_stride = fh * fw2 * ic * oc_step;
    static const uint8_t transpose_4x4_idx[16] = {0, 4,  1, 5,  2,  6,  3,  7,
                                                  8, 12, 9, 13, 10, 14, 11, 15};
    uint8x16_t tbl_transpose_4x4 = vld1q_u8(&transpose_4x4_idx[0]);
    rep_step(oc_idx, oc, oc_step) {
        int32_t* dst_temp_ptr =
                reinterpret_cast<int32_t*>(dst_ptr + oc_idx * ic * fh * fw2);
        const int32_t* src_temp_ptr = reinterpret_cast<const int32_t*>(
                src_ptr + oc_idx * ic * fh * fw);
        // transpose ic and pad
        rep(fh_idx, fh) {
            rep(fw_idx, fw) {
                rep(ic_idx, ic) {
                    *(dst_temp_ptr + ic_idx * dst_ic_stride) = *src_temp_ptr;
                    src_temp_ptr++;
                }
                dst_temp_ptr++;
            }
            rep(ic_idx, ic) {
                memset(dst_temp_ptr + ic_idx * dst_ic_stride, 0,
                       sizeof(int8_t) * oc_step * fw_remain);
            }
            dst_temp_ptr += fw_remain;
        }
        // transpose fw oc
        int8_t* trans_dst_temp_ptr =
                reinterpret_cast<int8_t*>(dst_ptr + oc_idx * ic * fh * fw2);

        rep_step(idx, oc_step_stride, 16) {
            int8x16_t temp = vld1q_s8(trans_dst_temp_ptr + idx);
            vst1q_s8(trans_dst_temp_ptr + idx,
                     vqtbl1q_s8(temp, tbl_transpose_4x4));
        }
    }
};
template <BiasMode bias_mode, typename Op, size_t filter_size, int stride>
struct ConvDiectStrideInt8NchwNchw44 {
    static void impl(const int8_t* src, const int8_t* filter,
                     const int32_t* bias, int32_t* temp, int8_t* dst,
                     const size_t oc, const size_t ic, const size_t ih,
                     const size_t iw, const size_t oh, const size_t ow,
                     const Op& op) {
        MEGDNN_MARK_USED_VAR(temp);
        constexpr size_t fh = filter_size;
        constexpr size_t fw =
                stride == 2 ? filter_size : (filter_size + 3) / 4 * 4;
        constexpr size_t ic_step = 1;
        constexpr size_t big_oc_step = 8;
        constexpr size_t oc_step = 4;
        constexpr size_t ih_step = stride == 2 ? 2 : 1;
        constexpr size_t oh_step = 1;
        constexpr size_t ow_step = stride == 2 ? 4 : 8;
        constexpr size_t stride_h = stride;
        constexpr size_t stride_w = stride;
        constexpr int pack_iw_len = 4;

        const size_t img_stride = oh * ow;
        const size_t ow_end = ow / ow_step * ow_step;
        const size_t ow_remain = ow - ow_end;
        const size_t oc_end = oc / big_oc_step * big_oc_step;
        const size_t oc_remain = oc - oc_end;
        const int ld_dst_oc = oc_step * img_stride;

        using remain_fun = std::function<void(
                const int8_t* src_ptr, const int8_t* weight_ptr,
                const int32_t* bias_ptr, int8_t* dst_ptr, int ic, int ih,
                int iw, int ld_dst_oc, const Op& op)>;
        remain_fun kern_big_oc_remain = nullptr;
        remain_fun kern_small_oc_remain = nullptr;
        switch (ow_remain) {
#define cb(step)                                                        \
    case step:                                                          \
        kern_big_oc_remain =                                            \
                KerNeonXXs2NchwNchw44<bias_mode, Op, step, filter_size, \
                                      big_oc_step, stride>::impl;       \
        kern_small_oc_remain =                                          \
                KerNeonXXs2NchwNchw44<bias_mode, Op, step, filter_size, \
                                      oc_step, stride>::impl;           \
        break;

            UNROLL_CALL_RAW(4, cb);
            default:
                megdnn_assert(0, "no remain %zu for kern", ow_remain);
        }

        for (size_t oc_idx = 0; oc_idx < oc_end; oc_idx += big_oc_step) {
            const size_t weight_offset = oc_idx * ic * fh * fw;
            for (size_t oh_idx = 0; oh_idx < oh; oh_idx += oh_step) {
                for (size_t ow_idx = 0; ow_idx < ow_end; ow_idx += ow_step) {
                    const size_t src_offset = (oh_idx * stride_h * iw +
                                               ow_idx * stride_w * ih_step) *
                                              ic_step * pack_iw_len;
                    const size_t dst_offset = oc_idx * img_stride +
                                              (oh_idx * ow + ow_idx) * oc_step;
                    KerNeonXXs2NchwNchw44<bias_mode, Op, 0, filter_size,
                                          big_oc_step,
                                          stride>::impl(src + src_offset,
                                                        filter + weight_offset,
                                                        bias + oc_idx,
                                                        dst + dst_offset, ic,
                                                        ih, iw, ld_dst_oc, op);
                }
                if (ow_remain > 0) {
                    const size_t src_offset = (oh_idx * stride_h * iw +
                                               ow_end * stride_w * ih_step) *
                                              ic_step * pack_iw_len;
                    const size_t dst_offset = oc_idx * img_stride +
                                              (oh_idx * ow + ow_end) * oc_step;
                    kern_big_oc_remain(src + src_offset, filter + weight_offset,
                                       bias + oc_idx, dst + dst_offset, ic, ih,
                                       iw, ld_dst_oc, op);
                }
            }
        }
        if (oc_remain > 0) {
            size_t oc_idx = oc_end;
            const size_t weight_offset = oc_idx * ic * fh * fw;
            for (size_t oh_idx = 0; oh_idx < oh; oh_idx += oh_step) {
                for (size_t ow_idx = 0; ow_idx < ow_end; ow_idx += ow_step) {
                    const size_t src_offset = (oh_idx * stride_h * iw +
                                               ow_idx * stride_w * ih_step) *
                                              ic_step * pack_iw_len;
                    const size_t dst_offset = oc_idx * img_stride +
                                              (oh_idx * ow + ow_idx) * oc_step;
                    KerNeonXXs2NchwNchw44<bias_mode, Op, 0, filter_size,
                                          oc_step,
                                          stride>::impl(src + src_offset,
                                                        filter + weight_offset,
                                                        bias + oc_idx,
                                                        dst + dst_offset, ic,
                                                        ih, iw, ld_dst_oc, op);
                }
                if (ow_remain > 0) {
                    const size_t src_offset = (oh_idx * stride_h * iw +
                                               ow_end * stride_w * ih_step) *
                                              ic_step * pack_iw_len;
                    const size_t dst_offset = oc_idx * img_stride +
                                              (oh_idx * ow + ow_end) * oc_step;
                    kern_small_oc_remain(src + src_offset,
                                         filter + weight_offset, bias + oc_idx,
                                         dst + dst_offset, ic, ih, iw,
                                         ld_dst_oc, op);
                }
            }
        }
    }
};

template <BiasMode bias_mode, typename Op, size_t filter_size>
struct ConvDiectStrideInt8NchwNchw44<bias_mode, Op, filter_size, 1> {
    static void impl(const int8_t* src, const int8_t* filter,
                     const int32_t* bias, int32_t* temp, int8_t* dst,
                     const size_t oc, const size_t ic, const size_t ih,
                     const size_t iw, const size_t oh, const size_t ow,
                     const Op& op) {
        MEGDNN_MARK_USED_VAR(temp);
        constexpr int stride = 1;
        constexpr size_t fh = filter_size;
        constexpr size_t fw = (filter_size + 3) / 4 * 4;
        constexpr size_t ic_step = 1;
        constexpr size_t big_oc_step = 8;
        constexpr size_t oc_step = 4;
        constexpr size_t ih_step = 1;
        constexpr size_t oh_step = 1;
        constexpr size_t ow_step = 8;
        constexpr size_t stride_h = stride;
        constexpr size_t stride_w = stride;
        constexpr int pack_iw_len = 16;

        const size_t img_stride = oh * ow;
        const size_t ow_end = ow / ow_step * ow_step;
        const size_t ow_remain = ow - ow_end;
        const size_t oc_end = oc / big_oc_step * big_oc_step;
        const size_t oc_remain = oc - oc_end;
        const int ld_dst_oc = oc_step * img_stride;

        using remain_fun = std::function<void(
                const int8_t* src_ptr, const int8_t* weight_ptr,
                const int32_t* bias_ptr, int8_t* dst_ptr, int ic, int ih,
                int iw, int ld_dst_oc, const Op& op)>;
        remain_fun kern_big_oc_remain = nullptr;
        remain_fun kern_small_oc_remain = nullptr;
        switch (ow_remain) {
#define cb(step)                                                        \
    case step:                                                          \
        kern_big_oc_remain =                                            \
                KerNeonXXs2NchwNchw44<bias_mode, Op, step, filter_size, \
                                      big_oc_step, stride>::impl;       \
        kern_small_oc_remain =                                          \
                KerNeonXXs2NchwNchw44<bias_mode, Op, step, filter_size, \
                                      oc_step, stride>::impl;           \
        break;

            UNROLL_CALL_RAW(8, cb);
            default:
                megdnn_assert(0, "no remain %zu for kern", ow_remain);
        }

        for (size_t oc_idx = 0; oc_idx < oc_end; oc_idx += big_oc_step) {
            const size_t weight_offset = oc_idx * ic * fh * fw;
            for (size_t oh_idx = 0; oh_idx < oh; oh_idx += oh_step) {
                for (size_t ow_idx = 0; ow_idx < ow_end; ow_idx += ow_step) {
                    const size_t src_offset = (oh_idx * stride_h * iw +
                                               ow_idx * stride_w * ih_step) *
                                              ic_step * pack_iw_len;
                    const size_t dst_offset = oc_idx * img_stride +
                                              (oh_idx * ow + ow_idx) * oc_step;

                    KerNeonXXs2NchwNchw44<bias_mode, Op, ow_step, filter_size,
                                          big_oc_step,
                                          stride>::impl(src + src_offset,
                                                        filter + weight_offset,
                                                        bias + oc_idx,
                                                        dst + dst_offset, ic,
                                                        ih, iw, ld_dst_oc, op);
                }
                if (ow_remain > 0) {
                    const size_t src_offset = (oh_idx * stride_h * iw +
                                               ow_end * stride_w * ih_step) *
                                              ic_step * pack_iw_len;
                    const size_t dst_offset = oc_idx * img_stride +
                                              (oh_idx * ow + ow_end) * oc_step;
                    kern_big_oc_remain(src + src_offset, filter + weight_offset,
                                       bias + oc_idx, dst + dst_offset, ic, ih,
                                       iw, ld_dst_oc, op);
                }
            }
        }

        if (oc_remain > 0) {
            size_t oc_idx = oc_end;
            const size_t weight_offset = oc_idx * ic * fh * fw;
            for (size_t oh_idx = 0; oh_idx < oh; oh_idx += oh_step) {
                for (size_t ow_idx = 0; ow_idx < ow_end; ow_idx += ow_step) {
                    const size_t src_offset = (oh_idx * stride_h * iw +
                                               ow_idx * stride_w * ih_step) *
                                              ic_step * pack_iw_len;
                    const size_t dst_offset = oc_idx * img_stride +
                                              (oh_idx * ow + ow_idx) * oc_step;
                    KerNeonXXs2NchwNchw44<bias_mode, Op, ow_step, filter_size,
                                          oc_step,
                                          stride>::impl(src + src_offset,
                                                        filter + weight_offset,
                                                        bias + oc_idx,
                                                        dst + dst_offset, ic,
                                                        ih, iw, ld_dst_oc, op);
                }
                if (ow_remain > 0) {
                    const size_t src_offset = (oh_idx * stride_h * iw +
                                               ow_end * stride_w * ih_step) *
                                              ic_step * pack_iw_len;
                    const size_t dst_offset = oc_idx * img_stride +
                                              (oh_idx * ow + ow_end) * oc_step;
                    kern_small_oc_remain(src + src_offset,
                                         filter + weight_offset, bias + oc_idx,
                                         dst + dst_offset, ic, ih, iw,
                                         ld_dst_oc, op);
                }
            }
        }
    }
};

template <BiasMode bias_mode, typename Op, size_t filter_size, int stride>
static void conv_direct_int8_nchw_nchw44(const int8_t* src,
                                         const int8_t* filter,
                                         const int32_t* bias, int32_t* temp,
                                         int8_t* dst, const size_t oc,
                                         const size_t ic, const size_t ih,
                                         const size_t iw, const size_t oh,
                                         const size_t ow, const Op& op) {
    ConvDiectStrideInt8NchwNchw44<bias_mode, Op, filter_size, stride>::impl(
            src, filter, bias, temp, dst, oc, ic, ih, iw, oh, ow, op);
}

}  // namespace
}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen