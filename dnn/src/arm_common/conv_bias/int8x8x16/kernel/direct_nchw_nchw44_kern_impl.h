/**
 * \file
 * dnn/src/arm_common/conv_bias/int8x8x16/kernel/direct_nchw_nchw44_kern_impl.h
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
#include "src/arm_common/conv_bias/int8x8x16/direct_nchw_nchw44_kern.h"
#include "src/arm_common/conv_bias/intrinsic_helper.h"
#include "src/arm_common/conv_bias/opr_impl.h"
#include "src/arm_common/elemwise_op.h"
#include "src/arm_common/simd_macro/marm_neon.h"
#include "src/common/unroll_macro.h"
#include "src/common/utils.h"
#include "src/fallback/conv_bias/common.h"

using namespace megdnn;
using namespace arm_common;
namespace {
/**
 * @brief kernel helper to do core computation
 *
 * @tparam src_idx src reg offset
 * @tparam weight_idx weight reg offset
 * @tparam c_dim first dim of c reg
 * @tparam ow_block output width
 * @tparam half_adv half calculation
 * @tparam stride
 * @tparam T
 * @tparam T2
 * @tparam T3
 * @tparam T4
 */
template <int src_idx, int weight_idx, int c_dim, int ow_block, bool half_adv,
          int stride, typename T, typename T2, typename T3, typename T4>
struct ShiftCalHelper {
    static MEGDNN_ALWAYS_INLINE void impl(T& c, T2& src, T3& weight);
};

template <int src_idx, int weight_idx, typename T, typename T2, typename T3,
          typename T4>
struct ShiftCalHelper<src_idx, weight_idx, 1, 8, false, 2, T, T2, T3, T4> {
    static MEGDNN_ALWAYS_INLINE void impl(T& c, T2& src, T3& weight) {
#define cb(step)                                                          \
    c[0][step] = vmlal_s8(c[0][step], vget_low_s8(weight[0][weight_idx]), \
                          vget_low_s8(src[step + src_idx]));              \
    c[0][step] = vmlal_high_s8(c[0][step], weight[0][weight_idx],         \
                               src[step + src_idx]);

        UNROLL_CALL_RAW(8, cb);

#undef cb
    }
};

template <int src_idx, int weight_idx, typename T, typename T2, typename T3,
          typename T4>
struct ShiftCalHelper<src_idx, weight_idx, 1, 8, true, 2, T, T2, T3, T4> {
    static MEGDNN_ALWAYS_INLINE void impl(T& c, T2& src, T3& weight) {
#define cb(step)                                                          \
    c[0][step] = vmlal_s8(c[0][step], vget_low_s8(weight[0][weight_idx]), \
                          vget_low_s8(src[step + src_idx]));

        UNROLL_CALL_RAW(8, cb);

#undef cb
    }
};

template <int src_idx, int weight_idx, typename T, typename T2, typename T3,
          typename T4>
struct ShiftCalHelper<src_idx, weight_idx, 1, 8, false, 1, T, T2, T3, T4> {
    static MEGDNN_ALWAYS_INLINE void impl(T& c, T2& src, T3& weight) {
        //! for compatible with stride2 kernel, step, weight_idx, src_idx should
        //! mul 2
#define cb(step)                                                             \
    c[0][2 * step] =                                                         \
            vmlal_s8(c[0][2 * step], vget_low_s8(weight[0][2 * weight_idx]), \
                     vget_low_s8(src[step + src_idx]));                      \
    c[0][2 * step + 1] =                                                     \
            vmlal_high_s8(c[0][2 * step + 1], weight[0][2 * weight_idx],     \
                          src[step + src_idx]);

        UNROLL_CALL_RAW(4, cb);

#undef cb
#define cb(step)                                                              \
    c[0][2 * step] =                                                          \
            vmlal_high_s8(c[0][2 * step], weight[0][2 * weight_idx + 1],      \
                          src[step + src_idx]);                               \
    c[0][2 * step + 1] = vmlal_s8(c[0][2 * step + 1],                         \
                                  vget_low_s8(weight[0][2 * weight_idx + 1]), \
                                  vget_low_s8(src[step + 1 + src_idx]));

        UNROLL_CALL_RAW(4, cb);

#undef cb
    }
};

template <int src_idx, int weight_idx, typename T, typename T2, typename T3,
          typename T4>
struct ShiftCalHelper<src_idx, weight_idx, 1, 8, true, 1, T, T2, T3, T4> {
    static MEGDNN_ALWAYS_INLINE void impl(T& c, T2& src, T3& weight) {
#define cb(step)                                                             \
    c[0][2 * step] =                                                         \
            vmlal_s8(c[0][2 * step], vget_low_s8(weight[0][2 * weight_idx]), \
                     vget_low_s8(src[step + src_idx]));                      \
    c[0][2 * step + 1] =                                                     \
            vmlal_high_s8(c[0][2 * step + 1], weight[0][2 * weight_idx],     \
                          src[step + src_idx]);

        UNROLL_CALL_RAW(4, cb);

#undef cb
    }
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

template <int src_idx, int weight_idx, int c_dim, int ow_block, bool half_adv,
          int stride, typename T, typename T2, typename T3>
MEGDNN_ALWAYS_INLINE void cal_helper(T& c, T2& src, T3& weight) {
    ShiftCalHelper<src_idx, weight_idx, c_dim, ow_block, half_adv, stride, T,
                   T2, T3, int>::impl(c, src, weight);
};
template <BiasMode bias_mode, typename Op, int filter_size, int oc_block,
          int stride, int ow_block>
struct KerNeonXXs2NchwNchw44I8I8I16 {
    static void impl(const int8_t* src_ptr_origin, const int8_t* weight_ptr,
                     const int16_t* bias_ptr, int16_t* dst_ptr, int ic, int ih,
                     int iw, int ld_dst_oc, const Op&, const int remain_ow);
};
template <BiasMode bias_mode, typename Op, int oc_block, int stride,
          int ow_block>
struct KerNeonXXs2NchwNchw44I8I8I16<bias_mode, Op, 2, oc_block, stride,
                                    ow_block> {
    static void impl(const int8_t* src_ptr_origin, const int8_t* weight_ptr,
                     const int16_t* bias_ptr, int16_t* dst_ptr, int ic, int ih,
                     int iw, int ld_dst_oc, const Op&, const int remain_ow) {
        constexpr int loop_ic_step = 1;
        constexpr int filter_size = 2;
        constexpr int iw_expand = 8;
        constexpr int simd_len = 16;
        constexpr int filter_pack_oc = stride == 1 ? 16 : 8;

        const int ld_src_ic = ih * iw * iw_expand;
        const int ld_src_iw = iw * iw_expand;

        constexpr int ld_weight_ic = filter_pack_oc * filter_size * filter_size;
        constexpr int c_dim = 1;
        constexpr int reg_pair = stride;
        constexpr int div_pad = stride - 1;
        constexpr int iw_reg =
                ow_block + (filter_size - stride + div_pad) / reg_pair;
        constexpr int filter_reg = (filter_size + div_pad) / reg_pair;
        int16x8_t c[c_dim][ow_block];
        init_ocx_ow8<c_dim, bias_mode, ow_block>(c, bias_ptr, 0);
        for (int ic_idx = 0; ic_idx < ic; ic_idx += loop_ic_step) {
            const int8_t* src_ptr = src_ptr_origin + ic_idx * ld_src_ic;

            int8x16_t src[iw_reg];
            int8x16_t weight[1][filter_reg];
#define cb(step)                                                          \
    load_helper<iw_reg, 0, simd_len, 0, Vld1q_s8>(                        \
            src, src_ptr + step * ld_src_iw, 0);                          \
    load_helper<filter_reg, 0, simd_len, c_dim, Vld1q_s8>(                \
            weight, weight_ptr + step * filter_size * filter_pack_oc, 0); \
    cal_helper<0, 0, c_dim, ow_block, false, stride>(c, src, weight);
            UNROLL_CALL_RAW(2, cb)
#undef cb

            src_ptr += ld_src_iw;
            weight_ptr += ld_weight_ic;
        }
        constexpr int output_c_group = OCHelper<oc_block>::val;
        store_oc4_ow8_remain_static<c_dim, ow_block, 2, output_c_group>(
                c, dst_ptr, ld_dst_oc, remain_ow);
    };
};

template <BiasMode bias_mode, typename Op, int oc_block, int stride,
          int ow_block>
struct KerNeonXXs2NchwNchw44I8I8I16<bias_mode, Op, 3, oc_block, stride,
                                    ow_block> {
    static void impl(const int8_t* src_ptr_origin, const int8_t* weight_ptr,
                     const int16_t* bias_ptr, int16_t* dst_ptr, int ic, int ih,
                     int iw, int ld_dst_oc, const Op&, const int remain_ow) {
        constexpr int loop_ic_step = 1;
        constexpr int filter_size = 3;
        constexpr int iw_expand = 8;
        constexpr int simd_len = 16;
        constexpr int filter_pack_oc = stride == 1 ? 16 : 8;
        constexpr int c_dim = 1;

        const int ld_src_ic = ih * iw * iw_expand;
        const int ld_src_iw = iw * iw_expand;

        constexpr int ld_weight_ic = filter_pack_oc * filter_size * filter_size;
        constexpr int reg_pair = stride;
        constexpr int div_pad = stride - 1;
        constexpr int iw_reg =
                ow_block + (filter_size - stride + div_pad) / reg_pair;
        constexpr int filter_reg = (filter_size + div_pad) / reg_pair;
        int16x8_t c[c_dim][ow_block];
        init_ocx_ow8<c_dim, bias_mode, ow_block>(c, bias_ptr, 0);
        for (int ic_idx = 0; ic_idx < ic; ic_idx += loop_ic_step) {
            const int8_t* src_ptr = src_ptr_origin + ic_idx * ld_src_ic;

            int8x16_t src[iw_reg];
            int8x16_t weight[1][filter_reg];
#define cb(step)                                                          \
    load_helper<iw_reg, 0, simd_len, 0, Vld1q_s8>(                        \
            src, src_ptr + step * ld_src_iw, 0);                          \
    load_helper<filter_reg, 0, simd_len, c_dim, Vld1q_s8>(                \
            weight, weight_ptr + step * filter_size * filter_pack_oc, 0); \
    cal_helper<0, 0, c_dim, ow_block, false, stride>(c, src, weight);     \
    cal_helper<1, 1, c_dim, ow_block, true, stride>(c, src, weight);
            UNROLL_CALL_RAW(3, cb)
#undef cb

            src_ptr += ld_src_iw;
            weight_ptr += ld_weight_ic;
        }
        constexpr int output_c_group = OCHelper<oc_block>::val;
        store_oc4_ow8_remain_static<c_dim, ow_block, 2, output_c_group>(
                c, dst_ptr, ld_dst_oc, remain_ow);
    };
};

template <BiasMode bias_mode, typename Op, int oc_block, int stride,
          int ow_block>
struct KerNeonXXs2NchwNchw44I8I8I16<bias_mode, Op, 5, oc_block, stride,
                                    ow_block> {
    static void impl(const int8_t* src_ptr_origin, const int8_t* weight_ptr,
                     const int16_t* bias_ptr, int16_t* dst_ptr, int ic, int ih,
                     int iw, int ld_dst_oc, const Op&, const int remain_ow) {
        constexpr int loop_ic_step = 1;
        constexpr int filter_size = 5;
        constexpr int iw_expand = 8;
        constexpr int simd_len = 16;
        constexpr int filter_pack_oc = stride == 1 ? 16 : 8;
        constexpr int c_dim = 1;

        const int ld_src_ic = ih * iw * iw_expand;
        const int ld_src_iw = iw * iw_expand;

        constexpr int ld_weight_ic = filter_pack_oc * filter_size * filter_size;
        constexpr int reg_pair = stride;
        constexpr int div_pad = stride - 1;
        constexpr int iw_reg =
                ow_block + (filter_size - stride + div_pad) / reg_pair;
        constexpr int filter_reg = (filter_size + div_pad) / reg_pair;
        int16x8_t c[c_dim][ow_block];
        init_ocx_ow8<c_dim, bias_mode, ow_block>(c, bias_ptr, 0);
        for (int ic_idx = 0; ic_idx < ic; ic_idx += loop_ic_step) {
            const int8_t* src_ptr = src_ptr_origin + ic_idx * ld_src_ic;

            int8x16_t src[iw_reg];
            int8x16_t weight[1][filter_reg];
#define cb(step)                                                          \
    load_helper<iw_reg, 0, simd_len, 0, Vld1q_s8>(                        \
            src, src_ptr + step * ld_src_iw, 0);                          \
    load_helper<filter_reg, 0, simd_len, c_dim, Vld1q_s8>(                \
            weight, weight_ptr + step * filter_size * filter_pack_oc, 0); \
    cal_helper<0, 0, c_dim, ow_block, false, stride>(c, src, weight);     \
    cal_helper<1, 1, c_dim, ow_block, false, stride>(c, src, weight);     \
    cal_helper<2, 2, c_dim, ow_block, true, stride>(c, src, weight);
            UNROLL_CALL_RAW(5, cb)
#undef cb

            src_ptr += ld_src_iw;
            weight_ptr += ld_weight_ic;
        }
        constexpr int output_c_group = OCHelper<oc_block>::val;
        store_oc4_ow8_remain_static<c_dim, ow_block, 2, output_c_group>(
                c, dst_ptr, ld_dst_oc, remain_ow);
    };
};

template <BiasMode bias_mode, typename Op, int oc_block, int stride,
          int ow_block>
struct KerNeonXXs2NchwNchw44I8I8I16<bias_mode, Op, 7, oc_block, stride,
                                    ow_block> {
    static void impl(const int8_t* src_ptr_origin, const int8_t* weight_ptr,
                     const int16_t* bias_ptr, int16_t* dst_ptr, int ic, int ih,
                     int iw, int ld_dst_oc, const Op&, const int remain_ow) {
        constexpr int loop_ic_step = 1;
        constexpr int filter_size = 7;
        constexpr int iw_expand = 8;
        constexpr int simd_len = 16;
        constexpr int filter_pack_oc = stride == 1 ? 16 : 8;
        constexpr int c_dim = 1;

        const int ld_src_ic = ih * iw * iw_expand;
        const int ld_src_iw = iw * iw_expand;

        constexpr int ld_weight_ic = filter_pack_oc * filter_size * filter_size;
        constexpr int reg_pair = stride;
        constexpr int div_pad = stride - 1;
        constexpr int iw_reg =
                ow_block + (filter_size - stride + div_pad) / reg_pair;
        constexpr int filter_reg = (filter_size + div_pad) / reg_pair;
        int16x8_t c[c_dim][ow_block];
        init_ocx_ow8<c_dim, bias_mode, ow_block>(c, bias_ptr, 0);
        for (int ic_idx = 0; ic_idx < ic; ic_idx += loop_ic_step) {
            const int8_t* src_ptr = src_ptr_origin + ic_idx * ld_src_ic;

            int8x16_t src[iw_reg];
            int8x16_t weight[1][filter_reg];
#define cb(step)                                                          \
    load_helper<iw_reg, 0, simd_len, 0, Vld1q_s8>(                        \
            src, src_ptr + step * ld_src_iw, 0);                          \
    load_helper<filter_reg, 0, simd_len, c_dim, Vld1q_s8>(                \
            weight, weight_ptr + step * filter_size * filter_pack_oc, 0); \
    cal_helper<0, 0, c_dim, ow_block, false, stride>(c, src, weight);     \
    cal_helper<1, 1, c_dim, ow_block, false, stride>(c, src, weight);     \
    cal_helper<2, 2, c_dim, ow_block, false, stride>(c, src, weight);     \
    cal_helper<3, 3, c_dim, ow_block, true, stride>(c, src, weight);
            UNROLL_CALL_RAW(7, cb)
#undef cb

            src_ptr += ld_src_iw;
            weight_ptr += ld_weight_ic;
        }
        constexpr int output_c_group = OCHelper<oc_block>::val;
        store_oc4_ow8_remain_static<c_dim, ow_block, 2, output_c_group>(
                c, dst_ptr, ld_dst_oc, remain_ow);
    };
};

}  // namespace

template <BiasMode bias_mode, typename Op, int filter_size, int stride>
void i8i8i16_direct_nchw_nchw44::conv_direct_i8i8i16_nchw_nchw44(
        const int8_t* src, const int8_t* filter, const int16_t* bias, int8_t*,
        int16_t* dst, const int oc, const int ic, const int ih, const int iw,
        const int oh, const int oh_block, const int ow, const Op& op, const int,
        const int) {
    constexpr int fh = filter_size;
    constexpr int fw = filter_size;
    constexpr int ic_step = 1;
    constexpr int big_oc_step = 8;
    constexpr int oc_step = 4;
    constexpr int ih_step = 1;
    constexpr int oh_step = 1;
    constexpr int ow_step = 8;
    constexpr int stride_h = stride;
    constexpr int stride_w = stride;
    constexpr int iw_expand = 8;
    constexpr int weight_expand = stride == 1 ? 2 : 1;

    const int img_stride = oh * ow;
    const int ow_end = ow / ow_step * ow_step;
    const int ow_remain = ow - ow_end;
    const int oc_end = oc / big_oc_step * big_oc_step;
    const int oc_remain = oc - oc_end;
    const int ld_dst_oc = oc_step * img_stride;

    for (int oc_idx = 0; oc_idx < oc_end; oc_idx += big_oc_step) {
        const int weight_offset = (oc_idx * ic * fh * fw) * weight_expand;
        for (int oh_idx = 0; oh_idx < oh_block; oh_idx += oh_step) {
            for (int ow_idx = 0; ow_idx < ow_end; ow_idx += ow_step) {
                const int src_offset =
                        (oh_idx * stride_h * iw + ow_idx * stride_w * ih_step) *
                        ic_step * iw_expand;
                const int dst_offset =
                        oc_idx * img_stride + (oh_idx * ow + ow_idx) * oc_step;
                KerNeonXXs2NchwNchw44I8I8I16<
                        bias_mode, Op, filter_size, big_oc_step, stride,
                        ow_step>::impl(src + src_offset, filter + weight_offset,
                                       bias + oc_idx, dst + dst_offset, ic, ih,
                                       iw, ld_dst_oc, op, ow_step);
            }
            if (ow_remain > 0) {
                const int src_offset =
                        (oh_idx * stride_h * iw + ow_end * stride_w * ih_step) *
                        ic_step * iw_expand;
                const int dst_offset =
                        oc_idx * img_stride + (oh_idx * ow + ow_end) * oc_step;
                KerNeonXXs2NchwNchw44I8I8I16<
                        bias_mode, Op, filter_size, big_oc_step, stride,
                        ow_step>::impl(src + src_offset, filter + weight_offset,
                                       bias + oc_idx, dst + dst_offset, ic, ih,
                                       iw, ld_dst_oc, op, ow_remain);
            }
        }
    }
    if (oc_remain > 0) {
        int oc_idx = oc_end;
        const int weight_offset = (oc_idx * ic * fh * fw) * weight_expand;
        for (int oh_idx = 0; oh_idx < oh_block; oh_idx += oh_step) {
            for (int ow_idx = 0; ow_idx < ow_end; ow_idx += ow_step) {
                const int src_offset =
                        (oh_idx * stride_h * iw + ow_idx * stride_w * ih_step) *
                        ic_step * iw_expand;
                const int dst_offset =
                        oc_idx * img_stride + (oh_idx * ow + ow_idx) * oc_step;
                KerNeonXXs2NchwNchw44I8I8I16<
                        bias_mode, Op, filter_size, oc_step, stride,
                        ow_step>::impl(src + src_offset, filter + weight_offset,
                                       bias + oc_idx, dst + dst_offset, ic, ih,
                                       iw, ld_dst_oc, op, ow_step);
            }
            if (ow_remain > 0) {
                const int src_offset =
                        (oh_idx * stride_h * iw + ow_end * stride_w * ih_step) *
                        ic_step * iw_expand;
                const int dst_offset =
                        oc_idx * img_stride + (oh_idx * ow + ow_end) * oc_step;
                KerNeonXXs2NchwNchw44I8I8I16<
                        bias_mode, Op, filter_size, oc_step, stride,
                        ow_step>::impl(src + src_offset, filter + weight_offset,
                                       bias + oc_idx, dst + dst_offset, ic, ih,
                                       iw, ld_dst_oc, op, ow_remain);
            }
        }
    }
}

#define INSTANTIATION(stride, filter_size, bias_mode, Op)                      \
    template void i8i8i16_direct_nchw_nchw44::conv_direct_i8i8i16_nchw_nchw44< \
            bias_mode, Op, filter_size, stride>(                               \
            const int8_t* src, const int8_t* filter, const int16_t* bias,      \
            int8_t*, int16_t* dst, const int oc, const int ic, const int ih,   \
            const int iw, const int oh, const int oh_block, const int ow,      \
            const Op& op, const int, const int);

#define FOR_OP(stride, filter, bias) \
    INSTANTIATION(stride, filter, bias, NoneOp<dt_int16>)

#define INSTANCE_CONV(filter, stride)         \
    FOR_OP(stride, filter, BiasMode::NO_BIAS) \
    FOR_OP(stride, filter, BiasMode::BROADCAST_CHANNEL_BIAS)
