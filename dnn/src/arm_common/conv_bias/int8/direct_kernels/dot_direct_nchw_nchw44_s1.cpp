/**
 * \file
 * dnn/src/arm_common/conv_bias/int8/direct_kernels/dot_direct_nchw_nchw44_s1.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#if __ARM_FEATURE_DOTPROD
#include "src/arm_common/conv_bias/int8/dot_direct_nchw_nchw44_kern.h"
namespace megdnn {
namespace arm_common {
namespace dot_direct_nchw_nchw44 {

template <int src_idx, int weight_idx, typename Func, typename T, typename T2,
          typename T3, typename T4>
struct ShiftCalHelper<src_idx, weight_idx, 2, Func, 8, 1, T, T2, T3, T4> {
    static void impl(T& c, T2& src, T3& weight) {
#define cb(step)                                                           \
    c[0][step] = Func::template impl<(src_idx + step) % 4>(                \
            c[0][step], weight[0][weight_idx], src[(src_idx + step) / 4]); \
    c[1][step] = Func::template impl<(src_idx + step) % 4>(                \
            c[1][step], weight[1][weight_idx], src[(src_idx + step) / 4]);

        UNROLL_CALL_RAW(8, cb);
#undef cb
    }
};

template <int src_idx, int weight_idx, typename Func, typename T, typename T2,
          typename T3, typename T4>
struct ShiftCalHelper<src_idx, weight_idx, 1, Func, 8, 1, T, T2, T3, T4> {
    static void impl(T& c, T2& src, T3& weight) {
#define cb(step)                                            \
    c[0][step] = Func::template impl<(src_idx + step) % 4>( \
            c[0][step], weight[0][weight_idx], src[(src_idx + step) / 4]);

        UNROLL_CALL_RAW(8, cb);
#undef cb
    }
};
////////////////////stride 1///////////////////
template <BiasMode bias_mode, typename Op, int remain_w, int oc_block,
          int ow_block>
struct KerNeonDotXXs2Nchw44Int8<bias_mode, Op, remain_w, 2, oc_block, ow_block,
                                1> {
    static void impl(const int8_t* src_ptr, const int8_t* weight_ptr,
                     const int32_t* bias_ptr, int8_t* dst_ptr, int ic, int ih,
                     int iw, int ld_dst_oc, const Op& op) {
        constexpr int stride = 1;
        constexpr int filter_hight = 2;
        constexpr int filter_width = 4;
        constexpr int weight_reg = 2;
        constexpr int src_reg = 2;

        constexpr int oc_step = 4;
        constexpr int ic_step = 1;
        constexpr int pack_iw_len = 4;
        constexpr int simd_len = 16;

        const int ld_bias = oc_step;
        const int ic_stride = ih * iw * pack_iw_len;
        const int ld_weight_oc = oc_step * filter_hight * filter_width * ic;
        constexpr int c_dim = OCHelper<oc_block>::val;

        int32x4_t c[c_dim][8];
        init_ocx_ow8<c_dim, bias_mode, remain_w>(c, bias_ptr, ld_bias);
        for (int ic_idx = 0; ic_idx < ic; ic_idx += ic_step) {
            int8x16_t src[src_reg];
            int8x16_t weight[c_dim][weight_reg];
            // row 0
            load_helper<src_reg, 0, simd_len, 0, Vld1q_s8>(
                    src, src_ptr + 0 * iw * pack_iw_len, 0);
            load_helper<weight_reg, 0, simd_len, c_dim, Vld1q_s8>(
                    weight, weight_ptr, ld_weight_oc);
            cal_helper<0, 0, c_dim, Vdotq_laneq_s32, ow_block, stride>(c, src,
                                                                       weight);
            // row 1
            load_helper<src_reg, 0, simd_len, 0, Vld1q_s8>(
                    src, src_ptr + 1 * iw * pack_iw_len, 0);
            cal_helper<0, 1, c_dim, Vdotq_laneq_s32, ow_block, stride>(c, src,
                                                                       weight);

            src_ptr += ic_stride;
            weight_ptr += filter_hight * filter_width * oc_step;
        }
        store_ocx_ow8_remain_static_dt<c_dim, remain_w, Op, dt_qint8*>(
                c, op, dst_ptr, ld_dst_oc);
    }
};
template <BiasMode bias_mode, typename Op, int remain_w, int oc_block,
          int ow_block>
struct KerNeonDotXXs2Nchw44Int8<bias_mode, Op, remain_w, 3, oc_block, ow_block,
                                1> {
    static void impl(const int8_t* src_ptr, const int8_t* weight_ptr,
                     const int32_t* bias_ptr, int8_t* dst_ptr, int ic, int ih,
                     int iw, int ld_dst_oc, const Op& op) {
        constexpr int stride = 1;
        constexpr int filter_hight = 3;
        constexpr int filter_width = 4;
        constexpr int weight_reg = 3;
        constexpr int src_reg = 2;

        constexpr int oc_step = 4;
        constexpr int ic_step = 1;
        constexpr int pack_iw_len = 4;
        constexpr int simd_len = 16;

        const int ld_bias = oc_step;
        const int ic_stride = ih * iw * pack_iw_len;
        const int ld_weight_oc = oc_step * filter_hight * filter_width * ic;
        constexpr int c_dim = OCHelper<oc_block>::val;

        int32x4_t c[c_dim][8];
        init_ocx_ow8<c_dim, bias_mode, remain_w>(c, bias_ptr, ld_bias);
        for (int ic_idx = 0; ic_idx < ic; ic_idx += ic_step) {
            int8x16_t src[src_reg];
            int8x16_t weight[c_dim][weight_reg];
            // row 0
            load_helper<src_reg, 0, simd_len, 0, Vld1q_s8>(
                    src, src_ptr + 0 * iw * pack_iw_len, 0);
            load_helper<weight_reg, 0, simd_len, c_dim, Vld1q_s8>(
                    weight, weight_ptr, ld_weight_oc);
            cal_helper<0, 0, c_dim, Vdotq_laneq_s32, ow_block, stride>(c, src,
                                                                       weight);
            // row 1
            load_helper<src_reg, 0, simd_len, 0, Vld1q_s8>(
                    src, src_ptr + 1 * iw * pack_iw_len, 0);
            cal_helper<0, 1, c_dim, Vdotq_laneq_s32, ow_block, stride>(c, src,
                                                                       weight);
            // row 2
            load_helper<src_reg, 0, simd_len, 0, Vld1q_s8>(
                    src, src_ptr + 2 * iw * pack_iw_len, 0);
            cal_helper<0, 2, c_dim, Vdotq_laneq_s32, ow_block, stride>(c, src,
                                                                       weight);

            src_ptr += ic_stride;
            weight_ptr += filter_hight * filter_width * oc_step;
        }
        store_ocx_ow8_remain_static_dt<c_dim, remain_w, Op, dt_qint8*>(
                c, op, dst_ptr, ld_dst_oc);
    }
};

template <BiasMode bias_mode, typename Op, int remain_w, int oc_block,
          int ow_block>
struct KerNeonDotXXs2Nchw44Int8<bias_mode, Op, remain_w, 5, oc_block, ow_block,
                                1> {
    static void impl(const int8_t* src_ptr, const int8_t* weight_ptr,
                     const int32_t* bias_ptr, int8_t* dst_ptr, int ic, int ih,
                     int iw, int ld_dst_oc, const Op& op) {
        constexpr int stride = 1;
        constexpr int filter_hight = 5;
        constexpr int filter_width = 8;
        constexpr int src_reg = 3;
        constexpr int weight_reg = 2;

        constexpr int oc_step = 4;
        constexpr int ic_step = 1;
        constexpr int pack_iw_len = 4;
        constexpr int simd_len = 16;

        const int ld_bias = oc_step;
        const int ic_stride = ih * iw * pack_iw_len;
        const int ld_weight_oc = oc_step * filter_hight * filter_width * ic;
        constexpr int c_dim = OCHelper<oc_block>::val;

        int32x4_t c[c_dim][8];
        init_ocx_ow8<c_dim, bias_mode, remain_w>(c, bias_ptr, ld_bias);

        for (int ic_idx = 0; ic_idx < ic; ic_idx += ic_step) {
            int8x16_t src[src_reg];
            int8x16_t weight[c_dim][weight_reg];

#define cb(step)                                                        \
    load_helper<src_reg, 0, simd_len, 0, Vld1q_s8>(                     \
            src, src_ptr + step * iw * pack_iw_len, 0);                 \
    load_helper<weight_reg, 0, simd_len, c_dim, Vld1q_s8>(              \
            weight, weight_ptr + step * 2 * simd_len, ld_weight_oc);    \
    cal_helper<0, 0, c_dim, Vdotq_laneq_s32, ow_block, stride>(c, src,  \
                                                               weight); \
    cal_helper<4, 1, c_dim, Vdotq_laneq_s32, ow_block, stride>(c, src, weight);

            UNROLL_CALL_RAW(5, cb);
#undef cb
            src_ptr += ic_stride;
            weight_ptr += filter_hight * filter_width * oc_step;
        }
        store_ocx_ow8_remain_static_dt<c_dim, remain_w, Op, dt_qint8*>(
                c, op, dst_ptr, ld_dst_oc);
    }
};

template <BiasMode bias_mode, typename Op, int remain_w, int oc_block,
          int ow_block>
struct KerNeonDotXXs2Nchw44Int8<bias_mode, Op, remain_w, 7, oc_block, ow_block,
                                1> {
    static void impl(const int8_t* src_ptr, const int8_t* weight_ptr,
                     const int32_t* bias_ptr, int8_t* dst_ptr, int ic, int ih,
                     int iw, int ld_dst_oc, const Op& op) {
        constexpr int stride = 1;
        constexpr int filter_hight = 7;
        constexpr int filter_width = 8;
        constexpr int src_reg = 3;
        constexpr int weight_reg = 2;

        constexpr int oc_step = 4;
        constexpr int ic_step = 1;
        constexpr int pack_iw_len = 4;
        constexpr int simd_len = 16;

        const int ld_bias = oc_step;
        const int ic_stride = ih * iw * pack_iw_len;
        const int ld_weight_oc = oc_step * filter_hight * filter_width * ic;
        constexpr int c_dim = OCHelper<oc_block>::val;

        int32x4_t c[c_dim][8];
        init_ocx_ow8<c_dim, bias_mode, remain_w>(c, bias_ptr, ld_bias);

        for (int ic_idx = 0; ic_idx < ic; ic_idx += ic_step) {
            int8x16_t src[src_reg];
            int8x16_t weight[c_dim][weight_reg];
#define cb(step)                                                        \
    load_helper<src_reg, 0, simd_len, 0, Vld1q_s8>(                     \
            src, src_ptr + step * iw * pack_iw_len, 0);                 \
    load_helper<weight_reg, 0, simd_len, c_dim, Vld1q_s8>(              \
            weight, weight_ptr + step * 2 * simd_len, ld_weight_oc);    \
    cal_helper<0, 0, c_dim, Vdotq_laneq_s32, ow_block, stride>(c, src,  \
                                                               weight); \
    cal_helper<4, 1, c_dim, Vdotq_laneq_s32, ow_block, stride>(c, src, weight);

            UNROLL_CALL_RAW(7, cb);
#undef cb
            src_ptr += ic_stride;
            weight_ptr += filter_hight * filter_width * oc_step;
        }
        store_ocx_ow8_remain_static_dt<c_dim, remain_w, Op, dt_qint8*>(
                c, op, dst_ptr, ld_dst_oc);
    }
};
template <>
void pack_src_int8_nchw_nchw44_dot<1>(int8_t* sptr_base,
                                      const int8_t* sptr_origin, const int,
                                      const int pw, const int, const int ih,
                                      const int iw, const int iw2,
                                      const int pad_top, const int pad_bottom,
                                      const int ic, const int ic_stride,
                                      int8_t* temp_ptr) {
    static uint8_t reorder_idx[16] = {0, 1, 2, 3, 1, 2, 3, 4,
                                      2, 3, 4, 5, 3, 4, 5, 6};
    uint8x16_t tbl_idx = vld1q_u8(&reorder_idx[0]);

    constexpr int iw_step = 16;
    constexpr int pack_iw_len = 4;
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
                src[1] = vld1q_s8(temp_ptr + iw_idx + 4);
                src[2] = vld1q_s8(temp_ptr + iw_idx + 8);
                src[3] = vld1q_s8(temp_ptr + iw_idx + 12);
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
                *(sptr_base + iw_idx * pack_iw_len + 0) =
                        *(temp_ptr + iw_idx + 0);
                *(sptr_base + iw_idx * pack_iw_len + 1) =
                        *(temp_ptr + iw_idx + 1);
                *(sptr_base + iw_idx * pack_iw_len + 2) =
                        *(temp_ptr + iw_idx + 2);
                *(sptr_base + iw_idx * pack_iw_len + 3) =
                        *(temp_ptr + iw_idx + 3);
            }
            sptr_base += iw2 * pack_iw_len;
            sptr += iw;
        }
        sptr_base += iw2 * pad_bottom * pack_iw_len;
    }
}

template <BiasMode bias_mode, typename Op, int filter_size, int stride>
void conv_direct_int8_nchw_nchw44_dot(const int8_t* src, const int8_t* filter,
                                      const int32_t* bias, int32_t* temp,
                                      int8_t* dst, const int oc, const int ic,
                                      const int ih, const int iw, const int oh,
                                      const int oh_block, const int ow,
                                      const Op& op) {
    MEGDNN_MARK_USED_VAR(temp);
    constexpr int fh = filter_size;
    constexpr int fw = (filter_size + 3) / 4 * 4;
#if MEGDNN_AARCH64
    constexpr int big_oc_step = 8;
#else
    constexpr int big_oc_step = 4;
#endif
    constexpr int oc_step = 4;
    constexpr int ih_step = 1;
    constexpr int oh_step = 1;
    constexpr int ow_step = 8;
    constexpr int stride_h = stride;
    constexpr int stride_w = stride;
    constexpr int pack_iw_len = stride == 2 ? 1 : 4;

    const int img_stride = oh * ow;
    const int ow_end = ow / ow_step * ow_step;
    const int ow_remain = ow - ow_end;
    const int oc_end = oc / big_oc_step * big_oc_step;
    const int oc_remain = oc - oc_end;
    const int ld_dst_oc = oc_step * img_stride;

    using remain_fun =
            std::function<void(const int8_t* src_ptr, const int8_t* weight_ptr,
                               const int32_t* bias_ptr, int8_t* dst_ptr, int ic,
                               int ih, int iw, int ld_dst_oc, const Op& op)>;
    remain_fun kern_big_oc_remain = nullptr;
    remain_fun kern_small_oc_remain = nullptr;
    switch (ow_remain) {
#define cb(step)                                                              \
    case step:                                                                \
        kern_big_oc_remain =                                                  \
                KerNeonDotXXs2Nchw44Int8<bias_mode, Op, step, filter_size,    \
                                         big_oc_step, ow_step, stride>::impl; \
        kern_small_oc_remain =                                                \
                KerNeonDotXXs2Nchw44Int8<bias_mode, Op, step, filter_size,    \
                                         oc_step, ow_step, stride>::impl;     \
        break;

        UNROLL_CALL_RAW(8, cb);
        default:
            megdnn_assert(0, "no remain %d for kern", ow_remain);
    }

    for (int oc_idx = 0; oc_idx < oc_end; oc_idx += big_oc_step) {
        const int weight_offset = oc_idx * ic * fh * fw;
        for (int oh_idx = 0; oh_idx < oh_block; oh_idx += oh_step) {
            for (int ow_idx = 0; ow_idx < ow_end; ow_idx += ow_step) {
                const int src_offset =
                        (oh_idx * stride_h * iw + ow_idx * stride_w * ih_step) *
                        pack_iw_len;
                const int dst_offset =
                        oc_idx * img_stride + (oh_idx * ow + ow_idx) * oc_step;
                KerNeonDotXXs2Nchw44Int8<bias_mode, Op, ow_step, filter_size,
                                         big_oc_step, ow_step,
                                         stride>::impl(src + src_offset,
                                                       filter + weight_offset,
                                                       bias + oc_idx,
                                                       dst + dst_offset, ic, ih,
                                                       iw, ld_dst_oc, op);
            }
            if (ow_remain > 0) {
                const int src_offset =
                        (oh_idx * stride_h * iw + ow_end * stride_w * ih_step) *
                        pack_iw_len;
                const int dst_offset =
                        oc_idx * img_stride + (oh_idx * ow + ow_end) * oc_step;
                kern_big_oc_remain(src + src_offset, filter + weight_offset,
                                   bias + oc_idx, dst + dst_offset, ic, ih, iw,
                                   ld_dst_oc, op);
            }
        }
    }
    if (oc_remain > 0) {
        int oc_idx = oc_end;
        const int weight_offset = oc_idx * ic * fh * fw;
        for (int oh_idx = 0; oh_idx < oh_block; oh_idx += oh_step) {
            for (int ow_idx = 0; ow_idx < ow_end; ow_idx += ow_step) {
                const int src_offset =
                        (oh_idx * stride_h * iw + ow_idx * stride_w * ih_step) *
                        pack_iw_len;
                const int dst_offset =
                        oc_idx * img_stride + (oh_idx * ow + ow_idx) * oc_step;
                KerNeonDotXXs2Nchw44Int8<bias_mode, Op, ow_step, filter_size,
                                         oc_step, ow_step,
                                         stride>::impl(src + src_offset,
                                                       filter + weight_offset,
                                                       bias + oc_idx,
                                                       dst + dst_offset, ic, ih,
                                                       iw, ld_dst_oc, op);
            }
            if (ow_remain > 0) {
                const int src_offset =
                        (oh_idx * stride_h * iw + ow_end * stride_w * ih_step) *
                        pack_iw_len;
                const int dst_offset =
                        oc_idx * img_stride + (oh_idx * ow + ow_end) * oc_step;
                kern_small_oc_remain(src + src_offset, filter + weight_offset,
                                     bias + oc_idx, dst + dst_offset, ic, ih,
                                     iw, ld_dst_oc, op);
            }
        }
    }
}
#define DO_CONV_KERN_FUN(stride, filter_size, bias_mode, Op)              \
    template void                                                         \
    conv_direct_int8_nchw_nchw44_dot<bias_mode, Op, filter_size, stride>( \
            const int8_t* src, const int8_t* filter, const int32_t* bias, \
            int32_t* temp, int8_t* dst, const int oc, const int ic,       \
            const int ih, const int iw, const int oh, const int oh_block, \
            const int ow, const Op& op);

#define GET_OP_PARAM(stride, filter, bias_mode)                  \
    DO_CONV_KERN_FUN(stride, filter, bias_mode,                  \
                     TypeCvtOp<dt_qint32 MEGDNN_COMMA dt_qint8>) \
    DO_CONV_KERN_FUN(stride, filter, bias_mode,                  \
                     ReluOp<dt_qint32 MEGDNN_COMMA dt_qint8>)    \
    DO_CONV_KERN_FUN(stride, filter, bias_mode,                  \
                     HSwishOp<dt_qint32 MEGDNN_COMMA dt_qint8>)

#define GET_BIAS_MODE_PARAM(stride, filter)         \
    GET_OP_PARAM(stride, filter, BiasMode::NO_BIAS) \
    GET_OP_PARAM(stride, filter, BiasMode::BROADCAST_CHANNEL_BIAS)

#define DISPATCH_CONV_KERN(stride) \
    GET_BIAS_MODE_PARAM(stride, 2) \
    GET_BIAS_MODE_PARAM(stride, 3) \
    GET_BIAS_MODE_PARAM(stride, 5) \
    GET_BIAS_MODE_PARAM(stride, 7)

DISPATCH_CONV_KERN(1);

}  // namespace dot_direct_nchw_nchw44
}  // namespace arm_common
}  // namespace megdnn
#endif
// vim: syntax=cpp.doxygen