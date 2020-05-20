/**
 * \file
 * dnn/src/arm_common/conv_bias/fp32/f32_direct_stride1_nchw44_kern.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/arm_common/conv_bias/fp32/f32_direct_stride1_nchw44_kern.h"
#include "src/arm_common/conv_bias/intrinsic_helper.h"
#include "src/arm_common/elemwise_op.h"
#include "src/arm_common/simd_macro/marm_neon.h"
#include "src/common/unroll_macro.h"
#include "src/common/utils.h"
#include "src/fallback/conv_bias/common.h"

using namespace megdnn;
using namespace arm_common;
namespace {

template <int src_idx, int weight_idx, int c_dim, typename Func, int ow_block,
          typename T, typename T2, typename T3, typename T4>
struct ShiftCalHelper {
    static void impl(T& c, T2& src, T3& weight);
};

template <int src_idx, int weight_idx, typename Func, typename T, typename T2,
          typename T3, typename T4>
struct ShiftCalHelper<src_idx, weight_idx, 2, Func, 8, T, T2, T3, T4> {
    static void impl(T& c, T2& src, T3& weight) {
#define cb(step, lane)                                                  \
    c[0][step] = Func::template impl<lane>(c[0][step], weight[0][lane], \
                                           src[(step + src_idx) % 8]);  \
    c[1][step] = Func::template impl<lane>(c[1][step], weight[1][lane], \
                                           src[(step + src_idx) % 8]);

        UNROLL_CALL_RAW(8, cb, 0);
        UNROLL_CALL_RAW(8, cb, 1);
        UNROLL_CALL_RAW(8, cb, 2);
        UNROLL_CALL_RAW(8, cb, 3);
#undef cb
    }
};
template <int src_idx, int weight_idx, typename Func, typename T, typename T2,
          typename T3, typename T4>
struct ShiftCalHelper<src_idx, weight_idx, 2, Func, 4, T, T2, T3, T4> {
    static void impl(T& c, T2& src, T3& weight) {
#define cb(step, lane)                                                  \
    c[0][step] = Func::template impl<lane>(c[0][step], weight[0][lane], \
                                           src[(step + src_idx) % 4]);  \
    c[1][step] = Func::template impl<lane>(c[1][step], weight[1][lane], \
                                           src[(step + src_idx) % 4]);

        UNROLL_CALL_RAW(4, cb, 0);
        UNROLL_CALL_RAW(4, cb, 1);
        UNROLL_CALL_RAW(4, cb, 2);
        UNROLL_CALL_RAW(4, cb, 3);
#undef cb
    }
};
template <int src_idx, int weight_idx, typename Func, typename T, typename T2,
          typename T3, typename T4>
struct ShiftCalHelper<src_idx, weight_idx, 1, Func, 8, T, T2, T3, T4> {
    static void impl(T& c, T2& src, T3& weight) {
#define cb(step, lane)                                                  \
    c[0][step] = Func::template impl<lane>(c[0][step], weight[0][lane], \
                                           src[(step + src_idx) % 8]);

        UNROLL_CALL_RAW(8, cb, 0);
        UNROLL_CALL_RAW(8, cb, 1);
        UNROLL_CALL_RAW(8, cb, 2);
        UNROLL_CALL_RAW(8, cb, 3);
#undef cb
    }
};
template <int src_idx, int weight_idx, typename Func, typename T, typename T2,
          typename T3, typename T4>
struct ShiftCalHelper<src_idx, weight_idx, 1, Func, 4, T, T2, T3, T4> {
    static void impl(T& c, T2& src, T3& weight) {
#define cb(step, lane)                                                  \
    c[0][step] = Func::template impl<lane>(c[0][step], weight[0][lane], \
                                           src[(step + src_idx) % 4]);

        UNROLL_CALL_RAW(4, cb, 0);
        UNROLL_CALL_RAW(4, cb, 1);
        UNROLL_CALL_RAW(4, cb, 2);
        UNROLL_CALL_RAW(4, cb, 3);
#undef cb
    }
};

template <int src_idx, int weight_idx, int c_dim, typename FUNC, int ow_block,
          typename T, typename T2, typename T3>
inline void cal_helper(T& c, T2& src, T3& weight) {
    ShiftCalHelper<src_idx, weight_idx, c_dim, FUNC, ow_block, T, T2, T3,
                   int>::impl(c, src, weight);
};
template <int oc>
struct OCHelper {
public:
    static const int val = -1;
};

template <>
struct OCHelper<4> {
public:
    static const int val = 1;
};
#if MEGDNN_AARCH64
template <>
struct OCHelper<8> {
public:
    static const int val = 2;
};
#endif

/**
 *  oc8_ow8(m = 8, n = 8) and oc4_ow8(m = 4, n = 8) gemm like kernel
 * */
template <BiasMode bias_mode, typename Op, int remain_w, int filter_size,
          int oc_block, int ow_block>
struct KerNeonXXs1Nchw44FP32 {
    static void impl(const float32_t* src_ptr, const float32_t* weight_ptr,
                     const float32_t* bias_ptr, float32_t* dst_ptr, int ic,
                     int ih, int iw, int ld_dst_oc, const Op& op);
};

template <BiasMode bias_mode, typename Op, int remain_w, int oc_block,
          int ow_block>
struct KerNeonXXs1Nchw44FP32<bias_mode, Op, remain_w, 2, oc_block, ow_block> {
    static void impl(const float32_t* src_ptr_origin,
                     const float32_t* weight_ptr, const float32_t* bias_ptr,
                     float32_t* dst_ptr, int ic, int ih, int iw, int ld_dst_oc,
                     const Op& op) {
        constexpr int ic_step = 4;
        constexpr int filter_size = 2;
        constexpr int oc_step = 4;
        constexpr int simd_len = 4;

        constexpr int ld_weight = oc_step * oc_step;
        const int ld_bias = bias_mode == BiasMode::BIAS ? ld_dst_oc : oc_step;
        const int ld_weight_oc = oc_step * filter_size * filter_size * ic;
        const int ld_weight_fh = oc_step * oc_step * filter_size;
        const int ld_src_ic = ih * iw;
        const int ld_src_iw = iw * oc_step;
        constexpr int c_dim = OCHelper<oc_block>::val;
        float32x4_t c[c_dim][ow_block];
        init_ocx_ow8<c_dim, bias_mode, ow_block>(c, bias_ptr, ld_bias);

        for (int ic_idx = 0; ic_idx < ic; ic_idx += ic_step) {
            const float* src_ptr = src_ptr_origin + ic_idx * ld_src_ic;
            for (int fh_idx = 0; fh_idx < filter_size; ++fh_idx) {
                float32x4_t src[ow_block];
                float32x4_t weight[c_dim][ic_step];
                load_helper<ow_block, 0, simd_len, 0, Vld1q_f32>(src, src_ptr,
                                                                 0);
                load_helper<ic_step, 0, oc_step, c_dim, Vld1q_f32>(
                        weight, weight_ptr, ld_weight_oc);
                cal_helper<0, 0, c_dim, Vfmaq_laneq_f32, ow_block>(c, src,
                                                                   weight);
                src[0] = vld1q_f32(src_ptr + (ow_block)*ic_step);
                load_helper<ic_step, 1 * ld_weight, oc_step, c_dim, Vld1q_f32>(
                        weight, weight_ptr, ld_weight_oc);
                cal_helper<1, 0, c_dim, Vfmaq_laneq_f32, ow_block>(c, src,
                                                                   weight);
                src_ptr += ld_src_iw;
                weight_ptr += ld_weight_fh;
            }
        }
        store_ocx_ow8_remain_static<c_dim, remain_w, Op>(c, op, dst_ptr,
                                                         ld_dst_oc);
    }
};

template <BiasMode bias_mode, typename Op, int remain_w, int oc_block,
          int ow_block>
struct KerNeonXXs1Nchw44FP32<bias_mode, Op, remain_w, 3, oc_block, ow_block> {
    static void impl(const float32_t* src_ptr_origin,
                     const float32_t* weight_ptr, const float32_t* bias_ptr,
                     float32_t* dst_ptr, int ic, int ih, int iw, int ld_dst_oc,
                     const Op& op) {
        constexpr int ic_step = 4;
        constexpr int filter_size = 3;
        constexpr int oc_step = 4;
        constexpr int simd_len = 4;

        constexpr int ld_weight = oc_step * oc_step;
        const int ld_bias = bias_mode == BiasMode::BIAS ? ld_dst_oc : oc_step;
        const int ld_weight_oc = oc_step * filter_size * filter_size * ic;
        const int ld_weight_fh = oc_step * oc_step * filter_size;
        const int ld_src_ic = ih * iw;
        const int ld_src_iw = iw * oc_step;
        constexpr int c_dim = OCHelper<oc_block>::val;
        float32x4_t c[c_dim][ow_block];
        init_ocx_ow8<c_dim, bias_mode, ow_block>(c, bias_ptr, ld_bias);

        for (int ic_idx = 0; ic_idx < ic; ic_idx += ic_step) {
            const float* src_ptr = src_ptr_origin + ic_idx * ld_src_ic;
            for (int fh_idx = 0; fh_idx < filter_size; ++fh_idx) {
                float32x4_t src[ow_block];
                float32x4_t weight[c_dim][ic_step];
                load_helper<ow_block, 0, simd_len, 0, Vld1q_f32>(src, src_ptr,
                                                                 0);
                load_helper<ic_step, 0, oc_step, c_dim, Vld1q_f32>(
                        weight, weight_ptr, ld_weight_oc);
                cal_helper<0, 0, c_dim, Vfmaq_laneq_f32, ow_block>(c, src,
                                                                   weight);
                src[0] = vld1q_f32(src_ptr + (ow_block)*ic_step);
                load_helper<ic_step, 1 * ld_weight, oc_step, c_dim, Vld1q_f32>(
                        weight, weight_ptr, ld_weight_oc);
                cal_helper<1, 0, c_dim, Vfmaq_laneq_f32, ow_block>(c, src,
                                                                   weight);
                src[1] = vld1q_f32(src_ptr + (ow_block + 1) * ic_step);
                load_helper<ic_step, 2 * ld_weight, oc_step, c_dim, Vld1q_f32>(
                        weight, weight_ptr, ld_weight_oc);
                cal_helper<2, 0, c_dim, Vfmaq_laneq_f32, ow_block>(c, src,
                                                                   weight);
                src_ptr += ld_src_iw;
                weight_ptr += ld_weight_fh;
            }
        }
        store_ocx_ow8_remain_static<c_dim, remain_w, Op>(c, op, dst_ptr,
                                                         ld_dst_oc);
    }
};
template <BiasMode bias_mode, typename Op, int remain_w, int oc_block,
          int ow_block>
struct KerNeonXXs1Nchw44FP32<bias_mode, Op, remain_w, 5, oc_block, ow_block> {
    static void impl(const float32_t* src_ptr_origin,
                     const float32_t* weight_ptr, const float32_t* bias_ptr,
                     float32_t* dst_ptr, int ic, int ih, int iw, int ld_dst_oc,
                     const Op& op) {
        constexpr int ic_step = 4;
        constexpr int filter_size = 5;
        constexpr int oc_step = 4;
        constexpr int simd_len = 4;

        constexpr int ld_weight = oc_step * oc_step;
        const int ld_bias = bias_mode == BiasMode::BIAS ? ld_dst_oc : oc_step;
        const int ld_weight_oc = oc_step * filter_size * filter_size * ic;
        const int ld_weight_fh = oc_step * oc_step * filter_size;
        const int ld_src_ic = ih * iw;
        const int ld_src_iw = iw * oc_step;
        constexpr int c_dim = OCHelper<oc_block>::val;
        float32x4_t c[c_dim][ow_block];
        init_ocx_ow8<c_dim, bias_mode, ow_block>(c, bias_ptr, ld_bias);

        for (int ic_idx = 0; ic_idx < ic; ic_idx += ic_step) {
            const float* src_ptr = src_ptr_origin + ic_idx * ld_src_ic;
            for (int fh_idx = 0; fh_idx < filter_size; ++fh_idx) {
                float32x4_t src[ow_block];
                float32x4_t weight[c_dim][ic_step];
                load_helper<ow_block, 0, simd_len, 0, Vld1q_f32>(src, src_ptr,
                                                                 0);
                load_helper<ic_step, 0, oc_step, c_dim, Vld1q_f32>(
                        weight, weight_ptr, ld_weight_oc);
                cal_helper<0, 0, c_dim, Vfmaq_laneq_f32, ow_block>(c, src,
                                                                   weight);

                src[0] = vld1q_f32(src_ptr + (ow_block)*ic_step);
                load_helper<ic_step, 1 * ld_weight, oc_step, c_dim, Vld1q_f32>(
                        weight, weight_ptr, ld_weight_oc);
                cal_helper<1, 0, c_dim, Vfmaq_laneq_f32, ow_block>(c, src,
                                                                   weight);

                src[1] = vld1q_f32(src_ptr + (ow_block + 1) * ic_step);
                load_helper<ic_step, 2 * ld_weight, oc_step, c_dim, Vld1q_f32>(
                        weight, weight_ptr, ld_weight_oc);
                cal_helper<2, 0, c_dim, Vfmaq_laneq_f32, ow_block>(c, src,
                                                                   weight);

                src[2] = vld1q_f32(src_ptr + (ow_block + 2) * ic_step);
                load_helper<ic_step, 3 * ld_weight, oc_step, c_dim, Vld1q_f32>(
                        weight, weight_ptr, ld_weight_oc);
                cal_helper<3, 0, c_dim, Vfmaq_laneq_f32, ow_block>(c, src,
                                                                   weight);

                src[3] = vld1q_f32(src_ptr + (ow_block + 3) * ic_step);
                load_helper<ic_step, 4 * ld_weight, oc_step, c_dim, Vld1q_f32>(
                        weight, weight_ptr, ld_weight_oc);
                cal_helper<4, 0, c_dim, Vfmaq_laneq_f32, ow_block>(c, src,
                                                                   weight);
                src_ptr += ld_src_iw;
                weight_ptr += ld_weight_fh;
            }
        }
        store_ocx_ow8_remain_static<c_dim, remain_w, Op>(c, op, dst_ptr,
                                                         ld_dst_oc);
    }
};

template <BiasMode bias_mode, typename Op, int remain_w, int oc_block,
          int ow_block>
struct KerNeonXXs1Nchw44FP32<bias_mode, Op, remain_w, 7, oc_block, ow_block> {
    static void impl(const float32_t* src_ptr_origin,
                     const float32_t* weight_ptr, const float32_t* bias_ptr,
                     float32_t* dst_ptr, int ic, int ih, int iw, int ld_dst_oc,
                     const Op& op) {
        constexpr int ic_step = 4;
        constexpr int filter_size = 7;
        constexpr int oc_step = 4;
        constexpr int simd_len = 4;

        constexpr int ld_weight = oc_step * oc_step;
        const int ld_bias = bias_mode == BiasMode::BIAS ? ld_dst_oc : oc_step;
        const int ld_weight_oc = oc_step * filter_size * filter_size * ic;
        const int ld_weight_fh = oc_step * oc_step * filter_size;
        const int ld_src_ic = ih * iw;
        const int ld_src_iw = iw * oc_step;
        constexpr int c_dim = OCHelper<oc_block>::val;
        float32x4_t c[c_dim][ow_block];
        init_ocx_ow8<c_dim, bias_mode, ow_block>(c, bias_ptr, ld_bias);

        for (int ic_idx = 0; ic_idx < ic; ic_idx += ic_step) {
            const float* src_ptr = src_ptr_origin + ic_idx * ld_src_ic;
            for (int fh_idx = 0; fh_idx < filter_size; ++fh_idx) {
                float32x4_t src[ow_block];
                float32x4_t weight[c_dim][ic_step];
                load_helper<ow_block, 0, simd_len, 0, Vld1q_f32>(src, src_ptr,
                                                                 0);
                load_helper<ic_step, 0, oc_step, c_dim, Vld1q_f32>(
                        weight, weight_ptr, ld_weight_oc);
                cal_helper<0, 0, c_dim, Vfmaq_laneq_f32, ow_block>(c, src,
                                                                   weight);

                src[0] = vld1q_f32(src_ptr + (ow_block)*ic_step);
                load_helper<ic_step, 1 * ld_weight, oc_step, c_dim, Vld1q_f32>(
                        weight, weight_ptr, ld_weight_oc);
                cal_helper<1, 0, c_dim, Vfmaq_laneq_f32, ow_block>(c, src,
                                                                   weight);

                src[1] = vld1q_f32(src_ptr + (ow_block + 1) * ic_step);
                load_helper<ic_step, 2 * ld_weight, oc_step, c_dim, Vld1q_f32>(
                        weight, weight_ptr, ld_weight_oc);
                cal_helper<2, 0, c_dim, Vfmaq_laneq_f32, ow_block>(c, src,
                                                                   weight);

                src[2] = vld1q_f32(src_ptr + (ow_block + 2) * ic_step);
                load_helper<ic_step, 3 * ld_weight, oc_step, c_dim, Vld1q_f32>(
                        weight, weight_ptr, ld_weight_oc);
                cal_helper<3, 0, c_dim, Vfmaq_laneq_f32, ow_block>(c, src,
                                                                   weight);

                src[3] = vld1q_f32(src_ptr + (ow_block + 3) * ic_step);
                load_helper<ic_step, 4 * ld_weight, oc_step, c_dim, Vld1q_f32>(
                        weight, weight_ptr, ld_weight_oc);
                cal_helper<4, 0, c_dim, Vfmaq_laneq_f32, ow_block>(c, src,
                                                                   weight);

                src[4] = vld1q_f32(src_ptr + (ow_block + 4) * ic_step);
                load_helper<ic_step, 5 * ld_weight, oc_step, c_dim, Vld1q_f32>(
                        weight, weight_ptr, ld_weight_oc);
                cal_helper<5, 0, c_dim, Vfmaq_laneq_f32, ow_block>(c, src,
                                                                   weight);

                src[5] = vld1q_f32(src_ptr + (ow_block + 5) * ic_step);
                load_helper<ic_step, 6 * ld_weight, oc_step, c_dim, Vld1q_f32>(
                        weight, weight_ptr, ld_weight_oc);
                cal_helper<6, 0, c_dim, Vfmaq_laneq_f32, ow_block>(c, src,
                                                                   weight);
                src_ptr += ld_src_iw;
                weight_ptr += ld_weight_fh;
            }
        }
        store_ocx_ow8_remain_static<c_dim, remain_w, Op>(c, op, dst_ptr,
                                                         ld_dst_oc);
    }
};

}  // namespace

void conv_bias::pack_src_fp32_nchw44_stride1(
        float* sptr_base, const float* sptr_origin, const int, const int pw,
        const int pad_right, const int ih, const int iw, const int iw2,
        const int pad_top, const int pad_bottom, const int ic,
        const int ic_stride) {
    constexpr int ic_step = 4;
    rep_step(ic_idx, ic, ic_step) {
        const float* sptr = sptr_origin + ic_idx * ic_stride;
        memset(sptr_base, 0, sizeof(float) * iw2 * pad_top * ic_step);
        sptr_base += iw2 * pad_top * ic_step;
        rep(ih_idx, ih) {
            memset(sptr_base, 0, sizeof(float) * pw * ic_step);
            sptr_base += pw * ic_step;
            memcpy(sptr_base, sptr, sizeof(float) * iw * ic_step);
            sptr_base += iw * ic_step;
            sptr += iw * ic_step;
            memset(sptr_base, 0, sizeof(float) * pad_right * ic_step);
            sptr_base += pad_right * ic_step;
        }
        memset(sptr_base, 0, sizeof(float) * iw2 * pad_bottom * ic_step);
        sptr_base += iw2 * pad_bottom * ic_step;
    }
}

template <BiasMode bias_mode, typename Op, int filter_size>
static void conv_direct_stride1_fp32_nchw44(
        const float32_t* src, const float32_t* filter, const float32_t* bias,
        float32_t*, float32_t* dst, const int oc, const int ic, const int ih,
        const int iw, const int oh, const int oh_block, const int ow,
        const Op& op, const int, const int) {
    constexpr int fh = filter_size;
    constexpr int fw = filter_size;
    constexpr int ic_step = 4;
#if MEGDNN_ARMV7
    constexpr int big_oc_step = 4;
#else
    constexpr int big_oc_step = 8;
#endif
    constexpr int oc_step = 4;
    constexpr int ih_step = 1;
    constexpr int oh_step = 1;
    constexpr int ow_step = 8;
    constexpr int stride_h = 1;
    constexpr int stride_w = 1;

    const int img_stride = oh * ow;
    const int ow_end = ow / ow_step * ow_step;
    const int ow_remain = ow - ow_end;
    const int oc_end = oc / big_oc_step * big_oc_step;
    const int oc_remain = oc - oc_end;
    const int ld_dst_oc = oc_step * img_stride;

    using remain_fun = std::function<void(
            const float32_t* src_ptr, const float32_t* weight_ptr,
            const float32_t* bias_ptr, float32_t* dst_ptr, int ic, int ih,
            int iw, int ld_dst_oc, const Op& op)>;
    remain_fun kern_big_oc_remain = nullptr;
    remain_fun kern_small_oc_remain = nullptr;

    switch (ow_remain) {
#define cb(step)                                                        \
    case step:                                                          \
        kern_big_oc_remain =                                            \
                KerNeonXXs1Nchw44FP32<bias_mode, Op, step, filter_size, \
                                      big_oc_step, ow_step>::impl;      \
        kern_small_oc_remain =                                          \
                KerNeonXXs1Nchw44FP32<bias_mode, Op, step, filter_size, \
                                      oc_step, ow_step>::impl;          \
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
                        ic_step;
                const int dst_offset =
                        oc_idx * img_stride + (oh_idx * ow + ow_idx) * oc_step;
                const int bias_offset =
                        bias_mode == BiasMode::BIAS ? dst_offset : oc_idx;
                KerNeonXXs1Nchw44FP32<bias_mode, Op, ow_step, filter_size,
                                      big_oc_step,
                                      ow_step>::impl(src + src_offset,
                                                     filter + weight_offset,
                                                     bias + bias_offset,
                                                     dst + dst_offset, ic, ih,
                                                     iw, ld_dst_oc, op);
            }
            if (ow_remain > 0) {
                const int src_offset =
                        (oh_idx * stride_h * iw + ow_end * stride_w * ih_step) *
                        ic_step;
                const int dst_offset =
                        oc_idx * img_stride + (oh_idx * ow + ow_end) * oc_step;
                const int bias_offset =
                        bias_mode == BiasMode::BIAS ? dst_offset : oc_idx;
                kern_big_oc_remain(src + src_offset, filter + weight_offset,
                                   bias + bias_offset, dst + dst_offset, ic, ih,
                                   iw, ld_dst_oc, op);
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
                        ic_step;
                const int dst_offset =
                        oc_idx * img_stride + (oh_idx * ow + ow_idx) * oc_step;
                const int bias_offset =
                        bias_mode == BiasMode::BIAS ? dst_offset : oc_idx;
                KerNeonXXs1Nchw44FP32<bias_mode, Op, ow_step, filter_size,
                                      oc_step,
                                      ow_step>::impl(src + src_offset,
                                                     filter + weight_offset,
                                                     bias + bias_offset,
                                                     dst + dst_offset, ic, ih,
                                                     iw, ld_dst_oc, op);
            }
            if (ow_remain > 0) {
                const int src_offset =
                        (oh_idx * stride_h * iw + ow_end * stride_w * ih_step) *
                        ic_step;
                const int dst_offset =
                        oc_idx * img_stride + (oh_idx * ow + ow_end) * oc_step;
                const int bias_offset =
                        bias_mode == BiasMode::BIAS ? dst_offset : oc_idx;
                kern_small_oc_remain(src + src_offset, filter + weight_offset,
                                     bias + bias_offset, dst + dst_offset, ic,
                                     ih, iw, ld_dst_oc, op);
            }
        }
    }
}

#define CONSTRUCT_FUNC(filter_size)                                          \
    template <BiasMode bias_mode, typename Op>                               \
    void conv_bias::                                                         \
            conv_direct_stride1_##filter_size##x##filter_size##_fp32_nchw44( \
                    const float32_t* src, const float32_t* filter,           \
                    const float32_t* bias, float32_t* temp, float32_t* dst,  \
                    const int oc, const int ic, const int ih, const int iw,  \
                    const int oh, const int oh_block, const int ow,          \
                    const Op& op, const int ph, const int pw) {              \
        conv_direct_stride1_fp32_nchw44<bias_mode, Op, filter_size>(         \
                src, filter, bias, temp, dst, oc, ic, ih, iw, oh, oh_block,  \
                ow, op, ph, pw);                                             \
    }
CONSTRUCT_FUNC(2);
CONSTRUCT_FUNC(3);
CONSTRUCT_FUNC(5);
CONSTRUCT_FUNC(7);
#undef CONSTRUCT_FUNC

#define INSTANTIATION(stride, i, bias, Op)                                     \
    template void conv_bias::conv_direct_##stride##_##i##x##i##_fp32_nchw44<   \
            bias, Op>(const float32_t*, const float32_t*, const float32_t*,    \
                      float32_t*, float32_t*, const int, const int, const int, \
                      const int, const int, const int, const int, const Op&,   \
                      const int, const int);

#define FOR_OP(stride, i, bias)                          \
    INSTANTIATION(stride, i, bias, NoneOp<dt_float32>)   \
    INSTANTIATION(stride, i, bias, ReluOp<dt_float32>)   \
    INSTANTIATION(stride, i, bias, HSwishOp<dt_float32>) \
    INSTANTIATION(stride, i, bias, SigmoidOp<dt_float32>)

#define FOR_BIAS(stride, i)                             \
    FOR_OP(stride, i, BiasMode::NO_BIAS)                \
    FOR_OP(stride, i, BiasMode::BROADCAST_CHANNEL_BIAS) \
    FOR_OP(stride, i, BiasMode::BIAS)

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
