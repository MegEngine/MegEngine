/**
 * \file dnn/src/arm_common/conv_bias/fp32/f32_direct_nchw_nchw44_kern.h
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
#include "src/arm_common/conv_bias/fp32/f32_direct_nchw_nchw44_kern.h"
#include "src/arm_common/conv_bias/intrinsic_helper.h"
#include "src/arm_common/conv_bias/opr_impl.h"
#include "src/arm_common/elemwise_op.h"
#include "src/arm_common/simd_macro/marm_neon.h"
#include "src/common/unroll_macro.h"
#include "src/common/utils.h"
#include "src/fallback/conv_bias/common.h"
#if MEGDNN_ARMV7
#include "src/armv7/matrix_mul/asm/common.h"
#endif

#if MGB_ENABLE_CPUINFO
#include "cpuinfo.h"
#endif

using namespace megdnn;
using namespace arm_common;

namespace {
/**
 *\brief ShiftCalHelper is core calculate code
 *\tparam src_idx is offset for src regs
 *\tparam weight_idx is offset for weight regs
 *\tparam T is type of output regs
 *\tparam T2 is type of src regs
 *\tparam T3 is type of weight regs
 */
template <int src_idx, int weight_idx, int c_dim, int stride, int remain_w,
          typename T, typename T2, typename T3>
struct ShiftCalHelper {
    static MEGDNN_ALWAYS_INLINE void impl(T& c, T2& src, T3& weight);
};

template <int src_idx, int weight_idx, int c_dim, int stride, typename T,
          typename T2, typename T3>
struct ShiftCalHelper<src_idx, weight_idx, c_dim, stride, 0, T, T2, T3> {
    static MEGDNN_ALWAYS_INLINE void impl(T&, T2&, T3&) {}
};

#define cb(step)                                                     \
    c[0][step] = vfmaq_laneq_f32(c[0][step], weight[0][weight_idx],  \
                                 src[(step * stride + src_idx) / 4], \
                                 (step * stride + src_idx) % 4);     \
    c[1][step] = vfmaq_laneq_f32(c[1][step], weight[1][weight_idx],  \
                                 src[(step * stride + src_idx) / 4], \
                                 (step * stride + src_idx) % 4);

#define cb2(step)                                                    \
    c[0][step] = vfmaq_laneq_f32(c[0][step], weight[0][weight_idx],  \
                                 src[(step * stride + src_idx) / 4], \
                                 (step * stride + src_idx) % 4);

#define SHIFT_CAL_HELPER(ow_remain)                                         \
    template <int src_idx, int weight_idx, int stride, typename T,          \
              typename T2, typename T3>                                     \
    struct ShiftCalHelper<src_idx, weight_idx, 2, stride, ow_remain, T, T2, \
                          T3> {                                             \
        static MEGDNN_ALWAYS_INLINE void impl(T& c, T2& src, T3& weight) {  \
            UNROLL_CALL_RAW(ow_remain, cb);                                 \
        }                                                                   \
    };                                                                      \
    template <int src_idx, int weight_idx, int stride, typename T,          \
              typename T2, typename T3>                                     \
    struct ShiftCalHelper<src_idx, weight_idx, 1, stride, ow_remain, T, T2, \
                          T3> {                                             \
        static MEGDNN_ALWAYS_INLINE void impl(T& c, T2& src, T3& weight) {  \
            UNROLL_CALL_RAW(ow_remain, cb2);                                \
        }                                                                   \
    };

SHIFT_CAL_HELPER(1)
SHIFT_CAL_HELPER(2)
SHIFT_CAL_HELPER(3)
SHIFT_CAL_HELPER(4)
SHIFT_CAL_HELPER(5)
SHIFT_CAL_HELPER(6)
SHIFT_CAL_HELPER(7)
SHIFT_CAL_HELPER(8)

#undef SHIFT_CAL_HELPER
#undef cb
#undef cb2

template <int src_idx, int weight_idx, int c_dim, int stride, int remain_w,
          typename T, typename T2, typename T3>
MEGDNN_ALWAYS_INLINE void cal_helper(T& c, T2& src, T3& weight) {
    ShiftCalHelper<src_idx, weight_idx, c_dim, stride, remain_w, T, T2,
                   T3>::impl(c, src, weight);
};
enum CpuTag {
    DEFAULT_CPU_TAG = 0,
    A7_TAG,
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

template <>
struct OCHelper<8> {
public:
    static const int val = 2;
};
/**
 *  oc8_ow8(m = 8, n = 8) and oc4_ow8(m = 4, n = 8) gemm like kernel
 **/
template <BiasMode bias_mode, typename Op, int remain_w, int filter_size,
          int oc_block, int stride, int ow_block,
          int tag = CpuTag::DEFAULT_CPU_TAG>
struct KerNeonXXs2NchwNchw44FP32 {
    static void impl(const float32_t* src_ptr, const float32_t* weight_ptr,
                     const float32_t* bias_ptr, float32_t* dst_ptr, int ic,
                     int ih, int iw, int ld_dst_oc, const Op& op);
};
template <BiasMode bias_mode, typename Op, int remain_w, int oc_block,
          int stride, int ow_block>
struct KerNeonXXs2NchwNchw44FP32<bias_mode, Op, remain_w, 7, oc_block, stride,
                                 ow_block> {
    static void impl(const float32_t* src_ptr, const float32_t* weight_ptr,
                     const float32_t* bias_ptr, float32_t* dst_ptr, int ic,
                     int ih, int iw, int ld_dst_oc, const Op& op) {
        constexpr int loop_ic_step = 1;
        constexpr int filter_size = 7;
        constexpr int oc_step = 4;
        constexpr int simd_len = 4;
        constexpr int src_reg_size =
                (ow_block * stride + filter_size - stride + simd_len - 1) /
                simd_len;

        constexpr int ld_weight_fw = oc_step * filter_size;
        const int ld_weight_oc = oc_step * filter_size * filter_size * ic;
        const int ld_weight_ic = oc_step * filter_size * filter_size;
        const int ld_src_ic = ih * iw;
        constexpr int c_dim = OCHelper<oc_block>::val;
        float32x4_t c[c_dim][8];
        init_ocx_ow8<c_dim, bias_mode, remain_w>(c, bias_ptr, oc_step);

        for (int ic_idx = 0; ic_idx < ic; ic_idx += loop_ic_step) {
            float32x4_t src[src_reg_size];
            float32x4_t weight[c_dim][filter_size];

#define KERNEL_CB(step)                                              \
    load_helper<src_reg_size, 0, simd_len, 0, Vld1q_f32>(            \
            src, src_ptr + step * iw, 0);                            \
    load_helper<filter_size, 0, oc_step, c_dim, Vld1q_f32>(          \
            weight, weight_ptr + step * ld_weight_fw, ld_weight_oc); \
    cal_helper<0, 0, c_dim, stride, remain_w>(c, src, weight);       \
    cal_helper<1, 1, c_dim, stride, remain_w>(c, src, weight);       \
    cal_helper<2, 2, c_dim, stride, remain_w>(c, src, weight);       \
    cal_helper<3, 3, c_dim, stride, remain_w>(c, src, weight);       \
    cal_helper<4, 4, c_dim, stride, remain_w>(c, src, weight);       \
    cal_helper<5, 5, c_dim, stride, remain_w>(c, src, weight);       \
    cal_helper<6, 6, c_dim, stride, remain_w>(c, src, weight);

            UNROLL_CALL_RAW(7, KERNEL_CB)
#undef KERNEL_CB

            src_ptr += ld_src_ic;
            weight_ptr += ld_weight_ic;
        }
        store_ocx_ow8_remain_static<c_dim, remain_w, Op>(c, op, dst_ptr,
                                                         ld_dst_oc);
    }
};

template <BiasMode bias_mode, typename Op, int remain_w, int oc_block,
          int stride, int ow_block>
struct KerNeonXXs2NchwNchw44FP32<bias_mode, Op, remain_w, 5, oc_block, stride,
                                 ow_block> {
    static void impl(const float32_t* src_ptr, const float32_t* weight_ptr,
                     const float32_t* bias_ptr, float32_t* dst_ptr, int ic,
                     int ih, int iw, int ld_dst_oc, const Op& op) {
        constexpr int loop_ic_step = 1;
        constexpr int filter_size = 5;
        constexpr int oc_step = 4;
        constexpr int simd_len = 4;
        constexpr int src_reg_size =
                (ow_block * stride + filter_size - stride + simd_len - 1) /
                simd_len;

        constexpr int ld_weight_fw = oc_step * filter_size;
        const int ld_weight_oc = oc_step * filter_size * filter_size * ic;
        const int ld_weight_ic = oc_step * filter_size * filter_size;
        const int ld_src_ic = ih * iw;
        constexpr int c_dim = OCHelper<oc_block>::val;
        float32x4_t c[c_dim][8];
        init_ocx_ow8<c_dim, bias_mode, remain_w>(c, bias_ptr, oc_step);

        for (int ic_idx = 0; ic_idx < ic; ic_idx += loop_ic_step) {
            float32x4_t src[src_reg_size];
            float32x4_t weight[c_dim][filter_size];

#define KERNEL_CB(step)                                              \
    load_helper<src_reg_size, 0, simd_len, 0, Vld1q_f32>(            \
            src, src_ptr + step * iw, 0);                            \
    load_helper<filter_size, 0, oc_step, c_dim, Vld1q_f32>(          \
            weight, weight_ptr + step * ld_weight_fw, ld_weight_oc); \
    cal_helper<0, 0, c_dim, stride, remain_w>(c, src, weight);       \
    cal_helper<1, 1, c_dim, stride, remain_w>(c, src, weight);       \
    cal_helper<2, 2, c_dim, stride, remain_w>(c, src, weight);       \
    cal_helper<3, 3, c_dim, stride, remain_w>(c, src, weight);       \
    cal_helper<4, 4, c_dim, stride, remain_w>(c, src, weight);
            UNROLL_CALL_RAW(5, KERNEL_CB)
#undef KERNEL_CB

            src_ptr += ld_src_ic;
            weight_ptr += ld_weight_ic;
        }
        store_ocx_ow8_remain_static<c_dim, remain_w, Op>(c, op, dst_ptr,
                                                         ld_dst_oc);
    }
};

template <BiasMode bias_mode, typename Op, int remain_w, int oc_block,
          int stride, int ow_block>
struct KerNeonXXs2NchwNchw44FP32<bias_mode, Op, remain_w, 3, oc_block, stride,
                                 ow_block> {
    static void impl(const float32_t* src_ptr, const float32_t* weight_ptr,
                     const float32_t* bias_ptr, float32_t* dst_ptr, int ic,
                     int ih, int iw, int ld_dst_oc, const Op& op) {
        constexpr int loop_ic_step = 1;
        constexpr int filter_size = 3;
        constexpr int oc_step = 4;
        constexpr int simd_len = 4;
        constexpr int src_reg_size =
                (ow_block * stride + filter_size - stride + simd_len - 1) /
                simd_len;

        constexpr int ld_weight_fw = oc_step * filter_size;
        const int ld_weight_oc = oc_step * filter_size * filter_size * ic;
        const int ld_weight_ic = oc_step * filter_size * filter_size;
        const int ld_src_ic = ih * iw;
        constexpr int c_dim = OCHelper<oc_block>::val;
        float32x4_t c[c_dim][8];
        init_ocx_ow8<c_dim, bias_mode, remain_w>(c, bias_ptr, oc_step);

        for (int ic_idx = 0; ic_idx < ic; ic_idx += loop_ic_step) {
            float32x4_t src[src_reg_size];
            float32x4_t weight[c_dim][filter_size];
            // row 0
            load_helper<src_reg_size, 0, simd_len, 0, Vld1q_f32>(src, src_ptr,
                                                                 0);
            load_helper<filter_size, 0, oc_step, c_dim, Vld1q_f32>(
                    weight, weight_ptr, ld_weight_oc);
            cal_helper<0, 0, c_dim, stride, remain_w>(c, src, weight);
            cal_helper<1, 1, c_dim, stride, remain_w>(c, src, weight);
            cal_helper<2, 2, c_dim, stride, remain_w>(c, src, weight);

            // row 1
            load_helper<src_reg_size, 0, simd_len, 0, Vld1q_f32>(
                    src, src_ptr + iw, 0);
            load_helper<filter_size, 0, oc_step, c_dim, Vld1q_f32>(
                    weight, weight_ptr + 1 * ld_weight_fw, ld_weight_oc);
            cal_helper<0, 0, c_dim, stride, remain_w>(c, src, weight);
            cal_helper<1, 1, c_dim, stride, remain_w>(c, src, weight);
            cal_helper<2, 2, c_dim, stride, remain_w>(c, src, weight);

            // row 2
            load_helper<src_reg_size, 0, simd_len, 0, Vld1q_f32>(
                    src, src_ptr + 2 * iw, 0);
            load_helper<filter_size, 0, oc_step, c_dim, Vld1q_f32>(
                    weight, weight_ptr + 2 * ld_weight_fw, ld_weight_oc);
            cal_helper<0, 0, c_dim, stride, remain_w>(c, src, weight);
            cal_helper<1, 1, c_dim, stride, remain_w>(c, src, weight);
            cal_helper<2, 2, c_dim, stride, remain_w>(c, src, weight);

            src_ptr += ld_src_ic;
            weight_ptr += ld_weight_ic;
        }
        store_ocx_ow8_remain_static<c_dim, remain_w, Op>(c, op, dst_ptr,
                                                         ld_dst_oc);
    }
};

#if MEGDNN_ARMV7

template <BiasMode bias_mode, typename Op>
struct KerNeonXXs2NchwNchw44FP32<bias_mode, Op, 8, 3, 4, 2, 8, CpuTag::A7_TAG> {
    static void impl(const float32_t* src_ptr, const float32_t* weight_ptr,
                     const float32_t* bias_ptr, float32_t* dst_ptr, int ic,
                     int ih, int iw, int ld_dst_oc, const Op& op) {
        constexpr int oc_block = 4;
        constexpr int stride = 2;
        constexpr int remain_w = 8;
        constexpr int ow_block = 8;
        constexpr int loop_ic_step = 1;
        constexpr int filter_size = 3;
        constexpr int oc_step = 4;
        constexpr int src_line_block = ow_block * stride + filter_size - stride;

        const int iw_skip_bytes =
                (iw - round_up(src_line_block, 2)) * sizeof(float);
        const int ld_src_ic_skip_bytes =
                iw * (ih - filter_size) * sizeof(float) + iw_skip_bytes;
        constexpr int c_dim = OCHelper<oc_block>::val;
        float32x4_t c[1][8];
        init_ocx_ow8<c_dim, bias_mode, 8>(c, bias_ptr, oc_step);
        const int img_stride = ih * iw;
        constexpr int filter_stride = filter_size * filter_size * oc_step;
        megdnn::armv7::prefetch_2x(src_ptr);
        megdnn::armv7::prefetch_2x(src_ptr + iw);
        megdnn::armv7::prefetch_2x(src_ptr + 2 * iw);
        megdnn::armv7::prefetch_2x(weight_ptr);

        /**
         * c q8-q15
         * src q0-q4
         * weight q5-q7
         * optimized for A7
         *
         */
        for (int ic_idx = 0; ic_idx < ic; ic_idx += loop_ic_step) {
            megdnn::armv7::prefetch_2x(src_ptr + img_stride);
            megdnn::armv7::prefetch_2x(src_ptr + img_stride + iw);
            megdnn::armv7::prefetch_2x(src_ptr + img_stride + 2 * iw);
            megdnn::armv7::prefetch_2x(weight_ptr + filter_stride);
            asm volatile(

                    "2:\n"
                    //! row 0
                    "vld1.32 {d10, d11}, [%[weight_ptr]]!\n"
                    "vld1.32 {d0, d1}, [%[src_ptr]]!\n"
                    "vld1.32 {d2, d3}, [%[src_ptr]]!\n"
                    "vld1.32 {d4, d5}, [%[src_ptr]]!\n"
                    "vld1.32 {d6, d7}, [%[src_ptr]]!\n"
                    "vld1.32 {d8}, [%[src_ptr]]!\n"
                    "vld1.32 {d12, d13}, [%[weight_ptr]]!\n"
                    "vld1.32 {d14, d15}, [%[weight_ptr]]!\n"
                    "add %[src_ptr], %[src_ptr], %[iw_skip_bytes]\n"

                    "vmla.f32 %q[c0], q5, d0[0]\n"
                    "vmla.f32 %q[c1], q5, d1[0]\n"
                    "vmla.f32 %q[c2], q5, d2[0]\n"
                    "vmla.f32 %q[c3], q5, d3[0]\n"
                    "vmla.f32 %q[c4], q5, d4[0]\n"
                    "vmla.f32 %q[c5], q5, d5[0]\n"
                    "vmla.f32 %q[c6], q5, d6[0]\n"
                    "vmla.f32 %q[c7], q5, d7[0]\n"

                    "vmla.f32 %q[c0], q6, d0[1]\n"
                    "vmla.f32 %q[c1], q6, d1[1]\n"
                    "vmla.f32 %q[c2], q6, d2[1]\n"
                    "vmla.f32 %q[c3], q6, d3[1]\n"
                    "vmla.f32 %q[c4], q6, d4[1]\n"
                    "vmla.f32 %q[c5], q6, d5[1]\n"
                    "vmla.f32 %q[c6], q6, d6[1]\n"
                    "vmla.f32 %q[c7], q6, d7[1]\n"

                    "vmla.f32 %q[c0], q7, d1[0]\n"
                    "vmla.f32 %q[c1], q7, d2[0]\n"
                    "vmla.f32 %q[c2], q7, d3[0]\n"
                    "vmla.f32 %q[c3], q7, d4[0]\n"
                    "vmla.f32 %q[c4], q7, d5[0]\n"
                    "vmla.f32 %q[c5], q7, d6[0]\n"
                    "vmla.f32 %q[c6], q7, d7[0]\n"
                    "vmla.f32 %q[c7], q7, d8[0]\n"

                    //! row 1
                    "vld1.32 {d10, d11}, [%[weight_ptr]]!\n"
                    "vld1.32 {d0, d1}, [%[src_ptr]]!\n"
                    "vld1.32 {d2, d3}, [%[src_ptr]]!\n"
                    "vld1.32 {d4, d5}, [%[src_ptr]]!\n"
                    "vld1.32 {d6, d7}, [%[src_ptr]]!\n"
                    "vld1.32 {d8}, [%[src_ptr]]!\n"
                    "vld1.32 {d12, d13}, [%[weight_ptr]]!\n"
                    "vld1.32 {d14, d15}, [%[weight_ptr]]!\n"
                    "add %[src_ptr], %[src_ptr], %[iw_skip_bytes]\n"

                    "vmla.f32 %q[c0], q5, d0[0]\n"
                    "vmla.f32 %q[c1], q5, d1[0]\n"
                    "vmla.f32 %q[c2], q5, d2[0]\n"
                    "vmla.f32 %q[c3], q5, d3[0]\n"
                    "vmla.f32 %q[c4], q5, d4[0]\n"
                    "vmla.f32 %q[c5], q5, d5[0]\n"
                    "vmla.f32 %q[c6], q5, d6[0]\n"
                    "vmla.f32 %q[c7], q5, d7[0]\n"

                    "vmla.f32 %q[c0], q6, d0[1]\n"
                    "vmla.f32 %q[c1], q6, d1[1]\n"
                    "vmla.f32 %q[c2], q6, d2[1]\n"
                    "vmla.f32 %q[c3], q6, d3[1]\n"
                    "vmla.f32 %q[c4], q6, d4[1]\n"
                    "vmla.f32 %q[c5], q6, d5[1]\n"
                    "vmla.f32 %q[c6], q6, d6[1]\n"
                    "vmla.f32 %q[c7], q6, d7[1]\n"

                    "vmla.f32 %q[c0], q7, d1[0]\n"
                    "vmla.f32 %q[c1], q7, d2[0]\n"
                    "vmla.f32 %q[c2], q7, d3[0]\n"
                    "vmla.f32 %q[c3], q7, d4[0]\n"
                    "vmla.f32 %q[c4], q7, d5[0]\n"
                    "vmla.f32 %q[c5], q7, d6[0]\n"
                    "vmla.f32 %q[c6], q7, d7[0]\n"
                    "vmla.f32 %q[c7], q7, d8[0]\n"

                    //! row 2
                    "vld1.32 {d10, d11}, [%[weight_ptr]]!\n"
                    "vld1.32 {d0, d1}, [%[src_ptr]]!\n"
                    "vld1.32 {d2, d3}, [%[src_ptr]]!\n"
                    "vld1.32 {d4, d5}, [%[src_ptr]]!\n"
                    "vld1.32 {d6, d7}, [%[src_ptr]]!\n"
                    "vld1.32 {d8}, [%[src_ptr]]!\n"
                    "vld1.32 {d12, d13}, [%[weight_ptr]]!\n"
                    "vld1.32 {d14, d15}, [%[weight_ptr]]!\n"
                    "add %[src_ptr], %[src_ptr], %[ld_src_ic_skip_bytes]\n"

                    "vmla.f32 %q[c0], q5, d0[0]\n"
                    "vmla.f32 %q[c1], q5, d1[0]\n"
                    "vmla.f32 %q[c2], q5, d2[0]\n"
                    "vmla.f32 %q[c3], q5, d3[0]\n"
                    "vmla.f32 %q[c4], q5, d4[0]\n"
                    "vmla.f32 %q[c5], q5, d5[0]\n"
                    "vmla.f32 %q[c6], q5, d6[0]\n"
                    "vmla.f32 %q[c7], q5, d7[0]\n"

                    "vmla.f32 %q[c0], q6, d0[1]\n"
                    "vmla.f32 %q[c1], q6, d1[1]\n"
                    "vmla.f32 %q[c2], q6, d2[1]\n"
                    "vmla.f32 %q[c3], q6, d3[1]\n"
                    "vmla.f32 %q[c4], q6, d4[1]\n"
                    "vmla.f32 %q[c5], q6, d5[1]\n"
                    "vmla.f32 %q[c6], q6, d6[1]\n"
                    "vmla.f32 %q[c7], q6, d7[1]\n"

                    "vmla.f32 %q[c0], q7, d1[0]\n"
                    "vmla.f32 %q[c1], q7, d2[0]\n"
                    "vmla.f32 %q[c2], q7, d3[0]\n"
                    "vmla.f32 %q[c3], q7, d4[0]\n"
                    "vmla.f32 %q[c4], q7, d5[0]\n"
                    "vmla.f32 %q[c5], q7, d6[0]\n"
                    "vmla.f32 %q[c6], q7, d7[0]\n"
                    "vmla.f32 %q[c7], q7, d8[0]\n"

                    "6:\n"
                    : [c0] "+w"(c[0][0]), [c1] "+w"(c[0][1]),
                      [c2] "+w"(c[0][2]), [c3] "+w"(c[0][3]),
                      [c4] "+w"(c[0][4]), [c5] "+w"(c[0][5]),
                      [c6] "+w"(c[0][6]), [c7] "+w"(c[0][7]),
                      [src_ptr] "+r"(src_ptr), [weight_ptr] "+r"(weight_ptr)

                    : [ld_src_ic_skip_bytes] "r"(ld_src_ic_skip_bytes),
                      [iw_skip_bytes] "r"(iw_skip_bytes)
                    : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8",
                      "d9", "d10", "d11", "d12", "d13", "d14", "d15", "r1",
                      "r2", "cc", "memory");
        }
        store_ocx_ow8_remain_static<c_dim, remain_w, Op>(c, op, dst_ptr,
                                                         ld_dst_oc);
    }
};

template <BiasMode bias_mode, typename Op>
struct KerNeonXXs2NchwNchw44FP32<bias_mode, Op, 8, 3, 4, 2, 8,
                                 CpuTag::DEFAULT_CPU_TAG> {
    static void impl(const float32_t* src_ptr, const float32_t* weight_ptr,
                     const float32_t* bias_ptr, float32_t* dst_ptr, int ic,
                     int ih, int iw, int ld_dst_oc, const Op& op) {
        constexpr int oc_block = 4;
        constexpr int stride = 2;
        constexpr int remain_w = 8;
        constexpr int ow_block = 8;
        constexpr int loop_ic_step = 1;
        constexpr int filter_size = 3;
        constexpr int oc_step = 4;
        constexpr int src_line_block = ow_block * stride + filter_size - stride;

        const int iw_skip_bytes =
                (iw - round_up(src_line_block, 2)) * sizeof(float);
        const int ld_src_ic_skip_bytes =
                iw * (ih - filter_size) * sizeof(float) + iw_skip_bytes;
        constexpr int c_dim = OCHelper<oc_block>::val;
        float32x4_t c[1][8];
        init_ocx_ow8<c_dim, bias_mode, 8>(c, bias_ptr, oc_step);
        /**
         * c q8-q15
         * src q0-q4
         * weight q5-q7
         * optimized for big core
         *
         */
        for (int ic_idx = 0; ic_idx < ic; ic_idx += loop_ic_step) {
            asm volatile(

                    "2:\n"
                    //! row 0
                    "vld1.32 {d10, d11}, [%[weight_ptr]]!\n"
                    "vld1.32 {d0, d1}, [%[src_ptr]]!\n"
                    "vld1.32 {d2, d3}, [%[src_ptr]]!\n"
                    "vld1.32 {d4, d5}, [%[src_ptr]]!\n"
                    "vmla.f32 %q[c0], q5, d0[0]\n"
                    "vld1.32 {d6, d7}, [%[src_ptr]]!\n"
                    "vmla.f32 %q[c1], q5, d1[0]\n"
                    "vmla.f32 %q[c2], q5, d2[0]\n"
                    "vld1.32 {d12, d13}, [%[weight_ptr]]!\n"
                    "vmla.f32 %q[c3], q5, d3[0]\n"
                    "vld1.32 {d14, d15}, [%[weight_ptr]]!\n"
                    "vmla.f32 %q[c4], q5, d4[0]\n"
                    "vld1.32 {d8}, [%[src_ptr]]!\n"
                    "vmla.f32 %q[c5], q5, d5[0]\n"
                    "add %[src_ptr], %[src_ptr], %[iw_skip_bytes]\n"
                    "vmla.f32 %q[c6], q5, d6[0]\n"
                    "vmla.f32 %q[c7], q5, d7[0]\n"
                    "vld1.32 {d10, d11}, [%[weight_ptr]]!\n"

                    "vmla.f32 %q[c0], q6, d0[1]\n"
                    "vmla.f32 %q[c1], q6, d1[1]\n"
                    "vmla.f32 %q[c2], q6, d2[1]\n"
                    "vmla.f32 %q[c3], q6, d3[1]\n"
                    "vmla.f32 %q[c4], q6, d4[1]\n"
                    "vmla.f32 %q[c5], q6, d5[1]\n"
                    "vmla.f32 %q[c6], q6, d6[1]\n"
                    "vmla.f32 %q[c7], q6, d7[1]\n"
                    "vld1.32 {d12, d13}, [%[weight_ptr]]!\n"

                    "vmla.f32 %q[c0], q7, d1[0]\n"
                    "vld1.32 {d0, d1}, [%[src_ptr]]!\n"
                    "vmla.f32 %q[c1], q7, d2[0]\n"
                    "vmla.f32 %q[c2], q7, d3[0]\n"
                    "vld1.32 {d2, d3}, [%[src_ptr]]!\n"
                    "vmla.f32 %q[c3], q7, d4[0]\n"
                    "vmla.f32 %q[c4], q7, d5[0]\n"
                    "vld1.32 {d4, d5}, [%[src_ptr]]!\n"
                    "vmla.f32 %q[c5], q7, d6[0]\n"
                    "vmla.f32 %q[c6], q7, d7[0]\n"
                    "vld1.32 {d6, d7}, [%[src_ptr]]!\n"
                    "vmla.f32 %q[c7], q7, d8[0]\n"
                    "vld1.32 {d14, d15}, [%[weight_ptr]]!\n"
                    //! row 1

                    "vmla.f32 %q[c0], q5, d0[0]\n"
                    "vld1.32 {d8}, [%[src_ptr]]!\n"
                    "vmla.f32 %q[c1], q5, d1[0]\n"
                    "add %[src_ptr], %[src_ptr], %[iw_skip_bytes]\n"
                    "vmla.f32 %q[c2], q5, d2[0]\n"
                    "vmla.f32 %q[c3], q5, d3[0]\n"
                    "vmla.f32 %q[c4], q5, d4[0]\n"
                    "vmla.f32 %q[c5], q5, d5[0]\n"
                    "vmla.f32 %q[c6], q5, d6[0]\n"
                    "vmla.f32 %q[c7], q5, d7[0]\n"
                    "vld1.32 {d10, d11}, [%[weight_ptr]]!\n"

                    "vmla.f32 %q[c0], q6, d0[1]\n"
                    "vmla.f32 %q[c1], q6, d1[1]\n"
                    "vmla.f32 %q[c2], q6, d2[1]\n"
                    "vmla.f32 %q[c3], q6, d3[1]\n"
                    "vmla.f32 %q[c4], q6, d4[1]\n"
                    "vmla.f32 %q[c5], q6, d5[1]\n"
                    "vmla.f32 %q[c6], q6, d6[1]\n"
                    "vmla.f32 %q[c7], q6, d7[1]\n"
                    "vld1.32 {d12, d13}, [%[weight_ptr]]!\n"

                    "vmla.f32 %q[c0], q7, d1[0]\n"
                    "vld1.32 {d0, d1}, [%[src_ptr]]!\n"
                    "vmla.f32 %q[c1], q7, d2[0]\n"
                    "vmla.f32 %q[c2], q7, d3[0]\n"
                    "vld1.32 {d2, d3}, [%[src_ptr]]!\n"
                    "vmla.f32 %q[c3], q7, d4[0]\n"
                    "vmla.f32 %q[c4], q7, d5[0]\n"
                    "vld1.32 {d4, d5}, [%[src_ptr]]!\n"
                    "vmla.f32 %q[c5], q7, d6[0]\n"
                    "vmla.f32 %q[c6], q7, d7[0]\n"
                    "vld1.32 {d6, d7}, [%[src_ptr]]!\n"
                    "vmla.f32 %q[c7], q7, d8[0]\n"
                    "vld1.32 {d14, d15}, [%[weight_ptr]]!\n"
                    //! row 2

                    "vmla.f32 %q[c0], q5, d0[0]\n"
                    "vld1.32 {d8}, [%[src_ptr]]!\n"
                    "vmla.f32 %q[c1], q5, d1[0]\n"
                    "add %[src_ptr], %[src_ptr], %[ld_src_ic_skip_bytes]\n"
                    "vmla.f32 %q[c2], q5, d2[0]\n"
                    "vmla.f32 %q[c3], q5, d3[0]\n"
                    "vmla.f32 %q[c4], q5, d4[0]\n"
                    "vmla.f32 %q[c5], q5, d5[0]\n"
                    "vmla.f32 %q[c6], q5, d6[0]\n"
                    "vmla.f32 %q[c7], q5, d7[0]\n"

                    "vmla.f32 %q[c0], q6, d0[1]\n"
                    "vmla.f32 %q[c1], q6, d1[1]\n"
                    "vmla.f32 %q[c2], q6, d2[1]\n"
                    "vmla.f32 %q[c3], q6, d3[1]\n"
                    "vmla.f32 %q[c4], q6, d4[1]\n"
                    "vmla.f32 %q[c5], q6, d5[1]\n"
                    "vmla.f32 %q[c6], q6, d6[1]\n"
                    "vmla.f32 %q[c7], q6, d7[1]\n"

                    "vmla.f32 %q[c0], q7, d1[0]\n"
                    "vmla.f32 %q[c1], q7, d2[0]\n"
                    "vmla.f32 %q[c2], q7, d3[0]\n"
                    "vmla.f32 %q[c3], q7, d4[0]\n"
                    "vmla.f32 %q[c4], q7, d5[0]\n"
                    "vmla.f32 %q[c5], q7, d6[0]\n"
                    "vmla.f32 %q[c6], q7, d7[0]\n"
                    "vmla.f32 %q[c7], q7, d8[0]\n"

                    "6:\n"
                    : [c0] "+w"(c[0][0]), [c1] "+w"(c[0][1]),
                      [c2] "+w"(c[0][2]), [c3] "+w"(c[0][3]),
                      [c4] "+w"(c[0][4]), [c5] "+w"(c[0][5]),
                      [c6] "+w"(c[0][6]), [c7] "+w"(c[0][7]),
                      [src_ptr] "+r"(src_ptr), [weight_ptr] "+r"(weight_ptr)

                    : [ld_src_ic_skip_bytes] "r"(ld_src_ic_skip_bytes),
                      [iw_skip_bytes] "r"(iw_skip_bytes)
                    : "d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d8",
                      "d9", "d10", "d11", "d12", "d13", "d14", "d15", "r1",
                      "r2", "cc", "memory");
        }
        store_ocx_ow8_remain_static<c_dim, remain_w, Op>(c, op, dst_ptr,
                                                         ld_dst_oc);
    }
};

#endif
template <BiasMode bias_mode, typename Op, int remain_w, int oc_block,
          int stride, int ow_block>
struct KerNeonXXs2NchwNchw44FP32<bias_mode, Op, remain_w, 2, oc_block, stride,
                                 ow_block> {
    static void impl(const float32_t* src_ptr, const float32_t* weight_ptr,
                     const float32_t* bias_ptr, float32_t* dst_ptr, int ic,
                     int ih, int iw, int ld_dst_oc, const Op& op) {
        constexpr int loop_ic_step = 1;
        constexpr int filter_size = 2;
        constexpr int oc_step = 4;
        constexpr int simd_len = 4;
        constexpr int src_reg_size =
                (ow_block * stride + filter_size - stride + simd_len - 1) /
                simd_len;

        constexpr int ld_weight_fw = oc_step * filter_size;
        const int ld_weight_oc = oc_step * filter_size * filter_size * ic;
        const int ld_weight_ic = oc_step * filter_size * filter_size;
        const int ld_src_ic = ih * iw;
        constexpr int c_dim = OCHelper<oc_block>::val;
        float32x4_t c[c_dim][8];
        init_ocx_ow8<c_dim, bias_mode, remain_w>(c, bias_ptr, oc_step);

        for (int ic_idx = 0; ic_idx < ic; ic_idx += loop_ic_step) {
            float32x4_t src[src_reg_size];
            float32x4_t weight[c_dim][filter_size];
            // row 0
            load_helper<src_reg_size, 0, simd_len, 0, Vld1q_f32>(src, src_ptr,
                                                                 0);
            load_helper<filter_size, 0, oc_step, c_dim, Vld1q_f32>(
                    weight, weight_ptr, ld_weight_oc);
            cal_helper<0, 0, c_dim, stride, remain_w>(c, src, weight);
            cal_helper<1, 1, c_dim, stride, remain_w>(c, src, weight);

            // row 1
            load_helper<src_reg_size, 0, simd_len, 0, Vld1q_f32>(
                    src, src_ptr + iw, 0);
            load_helper<filter_size, 0, oc_step, c_dim, Vld1q_f32>(
                    weight, weight_ptr + 1 * ld_weight_fw, ld_weight_oc);
            cal_helper<0, 0, c_dim, stride, remain_w>(c, src, weight);
            cal_helper<1, 1, c_dim, stride, remain_w>(c, src, weight);

            src_ptr += ld_src_ic;
            weight_ptr += ld_weight_ic;
        }
        store_ocx_ow8_remain_static<c_dim, remain_w, Op>(c, op, dst_ptr,
                                                         ld_dst_oc);
    }
};

template <BiasMode bias_mode, typename Op, int filter_size, int stride>
struct ConvDirectFp32NchwNchw44 {
    static MEGDNN_ALWAYS_INLINE void impl(
            const float32_t* src, const float32_t* filter,
            const float32_t* bias, float32_t*, float32_t* dst, const int oc,
            const int ic, const int ih, const int iw, const int oh,
            const int oh_block, const int ow, const Op& op) {
        constexpr int fh = filter_size;
        constexpr int fw = filter_size;
        constexpr int ic_step = 1;
#if MEGDNN_ARMV7
        constexpr int big_oc_step = 4;
#else
        constexpr int big_oc_step = 8;
#endif
        constexpr int oc_step = 4;
        constexpr int ih_step = 1;
        constexpr int oh_step = 1;
        constexpr int ow_step = 8;
        constexpr int stride_h = stride;
        constexpr int stride_w = stride;
        constexpr int pack_iw_len = 1;

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
#define cb(step)                                                               \
    case step:                                                                 \
        kern_big_oc_remain =                                                   \
                KerNeonXXs2NchwNchw44FP32<bias_mode, Op, step, filter_size,    \
                                          big_oc_step, stride, ow_step>::impl; \
        kern_small_oc_remain =                                                 \
                KerNeonXXs2NchwNchw44FP32<bias_mode, Op, step, filter_size,    \
                                          oc_step, stride, ow_step>::impl;     \
        break;

            UNROLL_CALL_RAW(8, cb);
            default:
                megdnn_assert(0, "no remain %d for kern", ow_remain);
        }
#undef cb
        for (int oc_idx = 0; oc_idx < oc_end; oc_idx += big_oc_step) {
            const int weight_offset = oc_idx * ic * fh * fw;
            for (int oh_idx = 0; oh_idx < oh_block; oh_idx += oh_step) {
                for (int ow_idx = 0; ow_idx < ow_end; ow_idx += ow_step) {
                    const int src_offset = (oh_idx * stride_h * iw +
                                            ow_idx * stride_w * ih_step) *
                                           ic_step * pack_iw_len;
                    const int dst_offset = oc_idx * img_stride +
                                           (oh_idx * ow + ow_idx) * oc_step;
                    KerNeonXXs2NchwNchw44FP32<
                            bias_mode, Op, ow_step, filter_size, big_oc_step,
                            stride, ow_step>::impl(src + src_offset,
                                                   filter + weight_offset,
                                                   bias + oc_idx,
                                                   dst + dst_offset, ic, ih, iw,
                                                   ld_dst_oc, op);
                }
                if (ow_remain > 0) {
                    const int src_offset = (oh_idx * stride_h * iw +
                                            ow_end * stride_w * ih_step) *
                                           ic_step * pack_iw_len;
                    const int dst_offset = oc_idx * img_stride +
                                           (oh_idx * ow + ow_end) * oc_step;
                    kern_big_oc_remain(src + src_offset, filter + weight_offset,
                                       bias + oc_idx, dst + dst_offset, ic, ih,
                                       iw, ld_dst_oc, op);
                }
            }
        }
        if (oc_remain > 0) {
            int oc_idx = oc_end;
            const int weight_offset = oc_idx * ic * fh * fw;
            for (int oh_idx = 0; oh_idx < oh_block; oh_idx += oh_step) {
                for (int ow_idx = 0; ow_idx < ow_end; ow_idx += ow_step) {
                    const int src_offset = (oh_idx * stride_h * iw +
                                            ow_idx * stride_w * ih_step) *
                                           ic_step * pack_iw_len;
                    const int dst_offset = oc_idx * img_stride +
                                           (oh_idx * ow + ow_idx) * oc_step;
                    KerNeonXXs2NchwNchw44FP32<
                            bias_mode, Op, ow_step, filter_size, oc_step,
                            stride, ow_step>::impl(src + src_offset,
                                                   filter + weight_offset,
                                                   bias + oc_idx,
                                                   dst + dst_offset, ic, ih, iw,
                                                   ld_dst_oc, op);
                }
                if (ow_remain > 0) {
                    const int src_offset = (oh_idx * stride_h * iw +
                                            ow_end * stride_w * ih_step) *
                                           ic_step * pack_iw_len;
                    const int dst_offset = oc_idx * img_stride +
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

#if MEGDNN_ARMV7
template <BiasMode bias_mode, typename Op>
struct ConvDirectFp32NchwNchw44<bias_mode, Op, 3, 2> {
    static MEGDNN_ALWAYS_INLINE void impl(
            const float32_t* src, const float32_t* filter,
            const float32_t* bias, float32_t*, float32_t* dst, const int oc,
            const int ic, const int ih, const int iw, const int oh,
            const int oh_block, const int ow, const Op& op) {
        constexpr int filter_size = 3;
        constexpr int stride = 2;
        constexpr int fh = filter_size;
        constexpr int fw = filter_size;
        constexpr int ic_step = 1;
        constexpr int oc_step = 4;
        constexpr int big_oc_step = oc_step;
        constexpr int ih_step = 1;
        constexpr int oh_step = 1;
        constexpr int ow_step = 8;
        constexpr int stride_h = stride;
        constexpr int stride_w = stride;
        constexpr int pack_iw_len = 1;

        const int img_stride = oh * ow;
        const int ow_end = ow / ow_step * ow_step;
        const int ow_remain = ow - ow_end;
        const int oc_end = oc / big_oc_step * big_oc_step;
        const int ld_dst_oc = oc_step * img_stride;

        using remain_fun = std::function<void(
                const float32_t* src_ptr, const float32_t* weight_ptr,
                const float32_t* bias_ptr, float32_t* dst_ptr, int ic, int ih,
                int iw, int ld_dst_oc, const Op& op)>;
        remain_fun kern_big_oc_remain = nullptr;

        switch (ow_remain) {
#define cb(step)                                                               \
    case step:                                                                 \
        kern_big_oc_remain =                                                   \
                KerNeonXXs2NchwNchw44FP32<bias_mode, Op, step, filter_size,    \
                                          big_oc_step, stride, ow_step>::impl; \
        break;

            UNROLL_CALL_RAW(8, cb);
            default:
                megdnn_assert(0, "no remain %d for kern", ow_remain);
        }
#undef cb
#if MGB_ENABLE_CPUINFO
        auto arch_tag =
                cpuinfo_get_current_core()->uarch == cpuinfo_uarch_cortex_a7
                        ? CpuTag::A7_TAG
                        : CpuTag::DEFAULT_CPU_TAG;
#else
        auto arch_tag = CpuTag::A7_TAG;
#endif
        if (arch_tag == CpuTag::A7_TAG) {
            for (int oc_idx = 0; oc_idx < oc_end; oc_idx += big_oc_step) {
                const int weight_offset = oc_idx * ic * fh * fw;
                for (int oh_idx = 0; oh_idx < oh_block; oh_idx += oh_step) {
                    for (int ow_idx = 0; ow_idx < ow_end; ow_idx += ow_step) {
                        const int src_offset = (oh_idx * stride_h * iw +
                                                ow_idx * stride_w * ih_step) *
                                               ic_step * pack_iw_len;
                        const int dst_offset = oc_idx * img_stride +
                                               (oh_idx * ow + ow_idx) * oc_step;
                        KerNeonXXs2NchwNchw44FP32<
                                bias_mode, Op, ow_step, filter_size,
                                big_oc_step, stride, ow_step,
                                CpuTag::A7_TAG>::impl(src + src_offset,
                                                      filter + weight_offset,
                                                      bias + oc_idx,
                                                      dst + dst_offset, ic, ih,
                                                      iw, ld_dst_oc, op);
                    }
                    if (ow_remain > 0) {
                        const int src_offset = (oh_idx * stride_h * iw +
                                                ow_end * stride_w * ih_step) *
                                               ic_step * pack_iw_len;
                        const int dst_offset = oc_idx * img_stride +
                                               (oh_idx * ow + ow_end) * oc_step;
                        kern_big_oc_remain(src + src_offset,
                                           filter + weight_offset,
                                           bias + oc_idx, dst + dst_offset, ic,
                                           ih, iw, ld_dst_oc, op);
                    }
                }
            }
        } else {
            for (int oc_idx = 0; oc_idx < oc_end; oc_idx += big_oc_step) {
                const int weight_offset = oc_idx * ic * fh * fw;
                for (int oh_idx = 0; oh_idx < oh_block; oh_idx += oh_step) {
                    for (int ow_idx = 0; ow_idx < ow_end; ow_idx += ow_step) {
                        const int src_offset = (oh_idx * stride_h * iw +
                                                ow_idx * stride_w * ih_step) *
                                               ic_step * pack_iw_len;
                        const int dst_offset = oc_idx * img_stride +
                                               (oh_idx * ow + ow_idx) * oc_step;
                        KerNeonXXs2NchwNchw44FP32<
                                bias_mode, Op, ow_step, filter_size,
                                big_oc_step, stride,
                                ow_step>::impl(src + src_offset,
                                               filter + weight_offset,
                                               bias + oc_idx, dst + dst_offset,
                                               ic, ih, iw, ld_dst_oc, op);
                    }
                    if (ow_remain > 0) {
                        const int src_offset = (oh_idx * stride_h * iw +
                                                ow_end * stride_w * ih_step) *
                                               ic_step * pack_iw_len;
                        const int dst_offset = oc_idx * img_stride +
                                               (oh_idx * ow + ow_end) * oc_step;
                        kern_big_oc_remain(src + src_offset,
                                           filter + weight_offset,
                                           bias + oc_idx, dst + dst_offset, ic,
                                           ih, iw, ld_dst_oc, op);
                    }
                }
            }
        }
    }
};

#endif

}  // namespace

template <BiasMode bias_mode, typename Op, int filter_size, int stride>
void fp32_direct_nchw_nchw44::conv_direct_fp32_nchw_nchw44(
        const float32_t* src, const float32_t* filter, const float32_t* bias,
        float32_t*, float32_t* dst, const int oc, const int ic, const int ih,
        const int iw, const int oh, const int oh_block, const int ow,
        const Op& op, const int, const int) {
    ConvDirectFp32NchwNchw44<bias_mode, Op, filter_size, stride>::impl(
            src, filter, bias, nullptr, dst, oc, ic, ih, iw, oh, oh_block, ow,
            op);
}

#define INSTANTIATION(stride, filter_size, bias_mode, Op)                    \
    template void fp32_direct_nchw_nchw44::conv_direct_fp32_nchw_nchw44<     \
            bias_mode, Op, filter_size, stride>(                             \
            const float32_t* src, const float32_t* filter,                   \
            const float32_t* bias, float32_t*, float32_t* dst, const int oc, \
            const int ic, const int ih, const int iw, const int oh,          \
            const int oh_block, const int ow, const Op& op, const int,       \
            const int);

#define FOR_OP(stride, filter, bias)                        \
    INSTANTIATION(stride, filter, bias, NoneOp<dt_float32>) \
    INSTANTIATION(stride, filter, bias, ReluOp<dt_float32>) \
    INSTANTIATION(stride, filter, bias, HSwishOp<dt_float32>)

#define INSTANCE_CONV(filter, stride)                        \
    FOR_OP(stride, filter, BiasMode::NO_BIAS)                \
    FOR_OP(stride, filter, BiasMode::BROADCAST_CHANNEL_BIAS) \
    FOR_OP(stride, filter, BiasMode::BIAS)

// vim: syntax=cpp.doxygen
