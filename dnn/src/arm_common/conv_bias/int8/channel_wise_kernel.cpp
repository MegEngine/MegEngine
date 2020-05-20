/**
 * \file dnn/src/arm_common/conv_bias/int8/channel_wise_kernel.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/arm_common/conv_bias/int8/channel_wise_kernel.h"
#include "src/arm_common/elemwise_op.h"
#include "src/arm_common/simd_macro/marm_neon.h"
#include "src/common/unroll_macro.h"
#include "src/common/utils.h"
#include "src/fallback/conv_bias/common.h"

using namespace megdnn;
using namespace arm_common;

static inline void accumulate_2_q_vector(int8x16_t& src0, int8x16_t& kern0,
                                         int8x16_t& src1, int8x16_t& kern1,
                                         int32x4_t* sum) {
    int16x8_t tmp_sum0 = vmull_s8(vget_low_s8(src0), vget_low_s8(kern0));
    int16x8_t tmp_sum1 = vmull_high_s8(src0, kern0);
    tmp_sum0 = vmlal_s8(tmp_sum0, vget_low_s8(src1), vget_low_s8(kern1));
    tmp_sum1 = vmlal_high_s8(tmp_sum1, src1, kern1);
    sum[0] = vaddw_s16(sum[0], vget_low_s16(tmp_sum0));
    sum[1] = vaddw_s16(sum[1], vget_high_s16(tmp_sum0));
    sum[2] = vaddw_s16(sum[2], vget_low_s16(tmp_sum1));
    sum[3] = vaddw_s16(sum[3], vget_high_s16(tmp_sum1));
}

static inline void accumulate_1_q_vector(int8x16_t& src0, int8x16_t& kern0,
                                         int32x4_t* sum) {
    int16x8_t tmp_sum0 = vmull_s8(vget_low_s8(src0), vget_low_s8(kern0));
    int16x8_t tmp_sum1 = vmull_high_s8(src0, kern0);
    sum[0] = vaddw_s16(sum[0], vget_low_s16(tmp_sum0));
    sum[1] = vaddw_s16(sum[1], vget_high_s16(tmp_sum0));
    sum[2] = vaddw_s16(sum[2], vget_low_s16(tmp_sum1));
    sum[3] = vaddw_s16(sum[3], vget_high_s16(tmp_sum1));
}

static inline void accumulate_2_d_vector(int8x16_t& src0, int8x8_t& kern0,
                                         int8x16_t& src1, int8x8_t& kern1,
                                         int32x4_t& sum0, int32x4_t& sum1) {
    int16x8_t tmp_sum0 = vmull_s8(vget_low_s8(src0), kern0);
    int16x8_t tmp_sum1 = vmull_s8(vget_high_s8(src0), kern0);
    tmp_sum0 = vmlal_s8(tmp_sum0, vget_low_s8(src1), kern1);
    tmp_sum1 = vmlal_s8(tmp_sum1, vget_high_s8(src1), kern1);
    sum0 = vaddw_s16(sum0, vget_low_s16(tmp_sum0));
    sum1 = vaddw_s16(sum1, vget_low_s16(tmp_sum1));
    sum0 = vaddw_s16(sum0, vget_high_s16(tmp_sum0));
    sum1 = vaddw_s16(sum1, vget_high_s16(tmp_sum1));
}

static inline void accumulate_1_line_horizon(const int8x8_t& src0,
                                             const int8x8_t& kern0,
                                             const int8x8_t& src1,
                                             const int8x8_t& kern1,
                                             int32x4_t& sum) {
    int16x8_t tmp_sum = vmull_s8(src0, kern0);
    tmp_sum = vmlal_s8(tmp_sum, src1, kern1);
    sum = vaddw_s16(sum, vget_low_s16(tmp_sum));
    sum = vaddw_s16(sum, vget_high_s16(tmp_sum));
}

static inline void accumulate_1_d_vector(const int8x8_t& src0,
                                         const int8x8_t& kern0,
                                         int32x4_t& sum) {
    int16x8_t tmp_sum = vmull_s8(src0, kern0);
    sum = vaddw_s16(sum, vget_low_s16(tmp_sum));
    sum = vaddw_s16(sum, vget_high_s16(tmp_sum));
}

#define ACC_S16_S32(sum, tmp_sum)                \
    sum = vaddw_s16(sum, vget_low_s16(tmp_sum)); \
    sum = vaddw_s16(sum, vget_high_s16(tmp_sum));

#define STORE_1_LINE(dst, oh, ow, OW, sum)                               \
    if (quantized) {                                                     \
        dt_qint8* dptr =                                                 \
                reinterpret_cast<dt_qint8*>(dst) + oh * OW * 4 + ow * 4; \
        op({{sum[0], sum[1]}}, dptr);                                    \
        op({{sum[2], sum[3]}}, dptr + 8);                                \
    } else {                                                             \
        dt_int32* dptr =                                                 \
                reinterpret_cast<dt_int32*>(dst) + oh * OW * 4 + ow * 4; \
        vst1q_s32(dptr, sum[0]);                                         \
        vst1q_s32(dptr + 4, sum[1]);                                     \
        vst1q_s32(dptr + 8, sum[2]);                                     \
        vst1q_s32(dptr + 12, sum[3]);                                    \
    }

#define STORE_1_LINE_REMAIN(dst, oh, ow, OW, sum, remain)                \
    if (quantized) {                                                     \
        dt_qint8* dptr =                                                 \
                reinterpret_cast<dt_qint8*>(dst) + oh * OW * 4 + ow * 4; \
        if (remain == 1) {                                               \
            op(sum[0], dptr);                                            \
        } else if (remain == 2) {                                        \
            op({{sum[0], sum[1]}}, dptr);                                \
        } else if (remain == 3) {                                        \
            op({{sum[0], sum[1]}}, dptr);                                \
            op(sum[2], dptr + 8);                                        \
        }                                                                \
    } else {                                                             \
        dt_int32* dptr =                                                 \
                reinterpret_cast<dt_int32*>(dst) + oh * OW * 4 + ow * 4; \
        if (remain == 1) {                                               \
            vst1q_s32(dptr, sum[0]);                                     \
        } else if (remain == 2) {                                        \
            vst1q_s32(dptr, sum[0]);                                     \
            vst1q_s32(dptr + 4, sum[1]);                                 \
        } else if (remain == 3) {                                        \
            vst1q_s32(dptr, sum[0]);                                     \
            vst1q_s32(dptr + 4, sum[1]);                                 \
            vst1q_s32(dptr + 8, sum[2]);                                 \
        }                                                                \
    }

template <bool quantized, BiasMode bias_mode, typename Op>
void channel_wise_nchw44::direct_stride1_2x2_int8(
        const int8_t* src, const int8_t* filter, const int32_t* bias, void* dst,
        const size_t IH, const size_t IW, const size_t OH, const size_t OW,
        const Op& op) {
    MEGDNN_MARK_USED_VAR(IH);
    int8x8_t kern01 = vld1_s8(filter);
    int8x8_t kern23 = vld1_s8(filter + 8);
    size_t oh = 0_z;
    for (; oh + 2 <= OH; oh += 2) {
        size_t ih = oh;
        size_t ow = 0_z;
        for (; ow + 4 <= OW; ow += 4) {
            size_t iw = ow;
            const int8_t* __restrict sptr0 = src + ih * IW * 4 + iw * 4;
            const int8_t* __restrict sptr1 = src + (ih + 1) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr2 = src + (ih + 2) * IW * 4 + iw * 4;
            const int32_t* __restrict bptr = bias;
            int32x4_t sum00;
            if (bias_mode == BiasMode::BROADCAST_CHANNEL_BIAS) {
                sum00 = vld1q_s32(bptr);
            } else {
                sum00 = vdupq_n_s32(0);
            }
            int32x4_t sum01 = sum00, sum02 = sum00, sum03 = sum00,
                      sum10 = sum00, sum11 = sum00, sum12 = sum00,
                      sum13 = sum00;
            int8x16_t src0 = vld1q_s8(sptr0);
            int8x8_t src03 = vld1_s8(sptr0 + 3 * 4), src00 = vget_low_s8(src0),
                     src02 = vget_high_s8(src0);
            int8x8_t src01 = vext_s8(src00, src02, 4);

            int8x16_t src1 = vld1q_s8(sptr1);
            int8x8_t src13 = vld1_s8(sptr1 + 3 * 4), src10 = vget_low_s8(src1),
                     src12 = vget_high_s8(src1);
            int8x8_t src11 = vext_s8(src10, src12, 4);

            int8x16_t src2 = vld1q_s8(sptr2);
            int8x8_t src23 = vld1_s8(sptr2 + 3 * 4), src20 = vget_low_s8(src2),
                     src22 = vget_high_s8(src2);
            int8x8_t src21 = vext_s8(src20, src22, 4);
            //! first line
            int16x8_t tmp_sum00 = vmull_s8(src00, kern01);
            tmp_sum00 = vmlal_s8(tmp_sum00, src10, kern23);
            ACC_S16_S32(sum00, tmp_sum00);

            int16x8_t tmp_sum01 = vmull_s8(src01, kern01);
            tmp_sum01 = vmlal_s8(tmp_sum01, src11, kern23);
            ACC_S16_S32(sum01, tmp_sum01);

            int16x8_t tmp_sum02 = vmull_s8(src02, kern01);
            tmp_sum02 = vmlal_s8(tmp_sum02, src12, kern23);
            ACC_S16_S32(sum02, tmp_sum02);

            int16x8_t tmp_sum03 = vmull_s8(src03, kern01);
            tmp_sum03 = vmlal_s8(tmp_sum03, src13, kern23);
            ACC_S16_S32(sum03, tmp_sum03);
            //! second line
            int16x8_t tmp_sum10 = vmull_s8(src10, kern01);
            tmp_sum10 = vmlal_s8(tmp_sum10, src20, kern23);
            ACC_S16_S32(sum10, tmp_sum10);

            int16x8_t tmp_sum11 = vmull_s8(src11, kern01);
            tmp_sum11 = vmlal_s8(tmp_sum11, src21, kern23);
            ACC_S16_S32(sum11, tmp_sum11);

            int16x8_t tmp_sum12 = vmull_s8(src12, kern01);
            tmp_sum12 = vmlal_s8(tmp_sum12, src22, kern23);
            ACC_S16_S32(sum12, tmp_sum12);

            int16x8_t tmp_sum13 = vmull_s8(src13, kern01);
            tmp_sum13 = vmlal_s8(tmp_sum13, src23, kern23);
            ACC_S16_S32(sum13, tmp_sum13);
            if (quantized) {
                dt_qint8* dptr =
                        reinterpret_cast<dt_qint8*>(dst) + oh * OW * 4 + ow * 4;
                op({{sum00, sum01}}, dptr);
                op({{sum02, sum03}}, dptr + 8);
                op({{sum10, sum11}}, dptr + OW * 4);
                op({{sum12, sum13}}, dptr + OW * 4 + 8);
            } else {
                dt_int32* dptr =
                        reinterpret_cast<dt_int32*>(dst) + oh * OW * 4 + ow * 4;
                vst1q_s32(dptr, sum00);
                vst1q_s32(dptr + 4, sum01);
                vst1q_s32(dptr + 8, sum02);
                vst1q_s32(dptr + 12, sum03);
                vst1q_s32(dptr + OW * 4, sum10);
                vst1q_s32(dptr + OW * 4 + 4, sum11);
                vst1q_s32(dptr + OW * 4 + 8, sum12);
                vst1q_s32(dptr + OW * 4 + 12, sum13);
            }
        }
        for (; ow < OW; ow++) {
            size_t iw = ow;
            const int8_t* __restrict sptr0 = src + ih * IW * 4 + iw * 4;
            const int8_t* __restrict sptr1 = src + (ih + 1) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr2 = src + (ih + 2) * IW * 4 + iw * 4;
            const int32_t* __restrict bptr = bias;
            int32x4_t sum00;
            if (bias_mode == BiasMode::BROADCAST_CHANNEL_BIAS) {
                sum00 = vld1q_s32(bptr);
            } else {
                sum00 = vdupq_n_s32(0);
            }
            int32x4_t sum10 = sum00;
            int8x8_t src00 = vld1_s8(sptr0);
            int8x8_t src10 = vld1_s8(sptr1);
            int8x8_t src20 = vld1_s8(sptr2);

            int16x8_t tmp_sum00 = vmull_s8(src00, kern01);
            tmp_sum00 = vmlal_s8(tmp_sum00, src10, kern23);
            ACC_S16_S32(sum00, tmp_sum00);

            int16x8_t tmp_sum10 = vmull_s8(src10, kern01);
            tmp_sum10 = vmlal_s8(tmp_sum10, src20, kern23);
            ACC_S16_S32(sum10, tmp_sum10);

            if (quantized) {
                dt_qint8* dptr =
                        reinterpret_cast<dt_qint8*>(dst) + oh * OW * 4 + ow * 4;
                op(sum00, dptr);
                op(sum10, dptr + OW * 4);
            } else {
                dt_int32* dptr =
                        reinterpret_cast<dt_int32*>(dst) + oh * OW * 4 + ow * 4;
                vst1q_s32(dptr, sum00);
                vst1q_s32(dptr + OW * 4, sum10);
            }
        }
    }
    for (; oh < OH; oh++) {
        size_t ih = oh;
        size_t ow = 0_z;
        for (; ow + 4 <= OW; ow += 4) {
            size_t iw = ow;
            const int8_t* __restrict sptr0 = src + ih * IW * 4 + iw * 4;
            const int8_t* __restrict sptr1 = src + (ih + 1) * IW * 4 + iw * 4;
            const int32_t* __restrict bptr = bias;
            int32x4_t sum00;
            if (bias_mode == BiasMode::BROADCAST_CHANNEL_BIAS) {
                sum00 = vld1q_s32(bptr);
            } else {
                sum00 = vdupq_n_s32(0);
            }
            int32x4_t sum01 = sum00, sum02 = sum00, sum03 = sum00;

            int8x16_t src0 = vld1q_s8(sptr0);
            int8x8_t src03 = vld1_s8(sptr0 + 3 * 4), src00 = vget_low_s8(src0),
                     src02 = vget_high_s8(src0);
            int8x8_t src01 = vext_s8(src00, src02, 4);

            int8x16_t src1 = vld1q_s8(sptr1);
            int8x8_t src13 = vld1_s8(sptr1 + 3 * 4), src10 = vget_low_s8(src1),
                     src12 = vget_high_s8(src1);
            int8x8_t src11 = vext_s8(src10, src12, 4);

            int16x8_t tmp_sum00 = vmull_s8(src00, kern01);
            tmp_sum00 = vmlal_s8(tmp_sum00, src10, kern23);
            ACC_S16_S32(sum00, tmp_sum00);

            int16x8_t tmp_sum01 = vmull_s8(src01, kern01);
            tmp_sum01 = vmlal_s8(tmp_sum01, src11, kern23);
            ACC_S16_S32(sum01, tmp_sum01);

            int16x8_t tmp_sum02 = vmull_s8(src02, kern01);
            tmp_sum02 = vmlal_s8(tmp_sum02, src12, kern23);
            ACC_S16_S32(sum02, tmp_sum02);

            int16x8_t tmp_sum03 = vmull_s8(src03, kern01);
            tmp_sum03 = vmlal_s8(tmp_sum03, src13, kern23);
            ACC_S16_S32(sum03, tmp_sum03);

            if (quantized) {
                dt_qint8* dptr =
                        reinterpret_cast<dt_qint8*>(dst) + oh * OW * 4 + ow * 4;
                op({{sum00, sum01}}, dptr);
                op({{sum02, sum03}}, dptr + 8);
            } else {
                dt_int32* dptr =
                        reinterpret_cast<dt_int32*>(dst) + oh * OW * 4 + ow * 4;
                vst1q_s32(dptr, sum00);
                vst1q_s32(dptr + 4, sum01);
                vst1q_s32(dptr + 8, sum02);
                vst1q_s32(dptr + 12, sum03);
            }
        }
        for (; ow < OW; ow++) {
            size_t iw = ow;
            const int8_t* __restrict sptr0 = src + ih * IW * 4 + iw * 4;
            const int8_t* __restrict sptr1 = src + (ih + 1) * IW * 4 + iw * 4;
            const int32_t* __restrict bptr = bias;
            int32x4_t sum00;
            if (bias_mode == BiasMode::BROADCAST_CHANNEL_BIAS) {
                sum00 = vld1q_s32(bptr);
            } else {
                sum00 = vdupq_n_s32(0);
            }
            int8x8_t src00 = vld1_s8(sptr0);
            int8x8_t src10 = vld1_s8(sptr1);
            int16x8_t tmp_sum00 = vmull_s8(src00, kern01);
            tmp_sum00 = vmlal_s8(tmp_sum00, src10, kern23);
            ACC_S16_S32(sum00, tmp_sum00);
            if (quantized) {
                dt_qint8* dptr =
                        reinterpret_cast<dt_qint8*>(dst) + oh * OW * 4 + ow * 4;
                op(sum00, dptr);
            } else {
                dt_int32* dptr =
                        reinterpret_cast<dt_int32*>(dst) + oh * OW * 4 + ow * 4;
                vst1q_s32(dptr, sum00);
            }
        }
    }
}
#undef ACC_S16_S32

template <bool quantized, BiasMode bias_mode, typename Op>
void channel_wise_nchw44::direct_stride1_3x3_int8(
        const int8_t* sptr, const int8_t* fptr, const int32_t* bias, void* dst,
        const size_t IH, const size_t IW, const size_t OH, const size_t OW,
        const Op& op) {
    MEGDNN_MARK_USED_VAR(IH);
    const int32_t* __restrict bptr = bias;
    int32x4_t init_v;
    if (bias_mode == BiasMode::BROADCAST_CHANNEL_BIAS) {
        init_v = vld1q_s32(bptr);
    } else {
        init_v = vdupq_n_s32(0);
    }
    const int* filter = reinterpret_cast<const int*>(fptr);
    int8x16_t kern[9];
#define cb(i) kern[i] = (int8x16_t)vld1q_dup_s32(filter + i);
    UNROLL_CALL_NOWRAPPER(9, cb);
#undef cb

#define LOAD_2_LINE_SRC(sptr0, sptr1)              \
    src[0][0] = vld1q_s8(sptr0);                   \
    src[0][2] = vld1q_s8(sptr0 + 16);              \
    src[1][0] = vld1q_s8(sptr1);                   \
    src[1][2] = vld1q_s8(sptr1 + 16);              \
    src[0][1] = vextq_s8(src[0][0], src[0][2], 4); \
    src[1][1] = vextq_s8(src[1][0], src[1][2], 4); \
    src[0][2] = vextq_s8(src[0][0], src[0][2], 8); \
    src[1][2] = vextq_s8(src[1][0], src[1][2], 8);

#define LOAD_1_LINE_SRC(sptr0, src)       \
    src[0] = vld1q_s8(sptr0);             \
    src[2] = vld1q_s8(sptr0 + 16);        \
    src[1] = vextq_s8(src[0], src[2], 4); \
    src[2] = vextq_s8(src[0], src[2], 8);

#define ACC_1_LINE(src, kern0, kern1, kern2, sum)             \
    accumulate_2_q_vector(src[0], kern0, src[1], kern1, sum); \
    accumulate_1_q_vector(src[2], kern2, sum);

    size_t oh = 0_z;
    for (; oh + 3 <= OH; oh += 3) {
        size_t ih = oh;
        size_t ow = 0_z;
        for (; ow + 4 <= OW; ow += 4) {
            size_t iw = ow;
            const int8_t* __restrict sptr0 = sptr + ih * IW * 4 + iw * 4;
            const int8_t* __restrict sptr1 = sptr + (ih + 1) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr2 = sptr + (ih + 2) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr3 = sptr + (ih + 3) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr4 = sptr + (ih + 4) * IW * 4 + iw * 4;
            int32x4_t sum0[4], sum1[4], sum2[4];
#define cb(j)         \
    sum0[j] = init_v; \
    sum1[j] = init_v; \
    sum2[j] = init_v;
            UNROLL_CALL_NOWRAPPER(4, cb);
#undef cb
//! gcc will report error of "more than 30 operands in 'asm'"
#if MEGDNN_AARCH64 && defined(__clang__)
            asm volatile(
                    //! load src 0,1
                    "ldr q21, [%[sptr0]]\n"
                    "ldr q24, [%[sptr1]]\n"

                    //! sum0 line<0,1>
                    "smull  v27.8h, v21.8b, %[k0].8b\n"
                    "ldr q23, [%[sptr0], #16]\n"
                    "smull2 v28.8h, v21.16b, %[k0].16b\n"
                    "ldr q26, [%[sptr1], #16]\n"
                    "smlal  v27.8h, v24.8b, %[k3].8b\n"
                    "ext v22.16b, v21.16b, v23.16b, #4\n"
                    "smlal2 v28.8h, v24.16b, %[k3].16b\n"
                    "ext v23.16b, v21.16b, v23.16b, #8\n"
                    "saddw   %[sum00].4s, %[sum00].4s, v27.4h\n"
                    "ext v25.16b, v24.16b, v26.16b, #4\n"
                    "saddw2  %[sum01].4s, %[sum01].4s, v27.8h\n"
                    "saddw   %[sum02].4s, %[sum02].4s, v28.4h\n"
                    "ext v26.16b, v24.16b, v26.16b, #8\n"
                    "saddw2  %[sum03].4s, %[sum03].4s, v28.8h\n"

                    "ldr q21, [%[sptr2]]\n"
                    "smull  v29.8h, v22.8b, %[k1].8b\n"
                    "smull2 v30.8h, v22.16b, %[k1].16b\n"
                    "ldr q31, [%[sptr2], #16]\n"
                    "smull  v27.8h, v23.8b, %[k2].8b\n"
                    "ext v22.16b, v21.16b, v31.16b, #4\n"
                    "smull2 v28.8h, v23.16b, %[k2].16b\n"
                    "ext v23.16b, v21.16b, v31.16b, #8\n"
                    "smlal  v29.8h, v25.8b, %[k4].8b\n"
                    "smlal2 v30.8h, v25.16b, %[k4].16b\n"
                    "saddw   %[sum00].4s, %[sum00].4s, v29.4h\n"
                    "smlal  v27.8h, v26.8b, %[k5].8b\n"
                    "saddw2  %[sum01].4s, %[sum01].4s, v29.8h\n"
                    "smlal2 v28.8h, v26.16b, %[k5].16b\n"
                    "saddw   %[sum02].4s, %[sum02].4s, v30.4h\n"
                    "saddw2  %[sum03].4s, %[sum03].4s, v30.8h\n"
                    //! load src 2

                    //! sum0 line<2>
                    "smull  v29.8h, v21.8b, %[k6].8b\n"
                    "saddw   %[sum00].4s, %[sum00].4s, v27.4h\n"
                    "smull2 v30.8h, v21.16b, %[k6].16b\n"
                    "saddw2  %[sum01].4s, %[sum01].4s, v27.8h\n"
                    "smull  v27.8h, v23.8b, %[k8].8b\n"
                    "saddw   %[sum02].4s, %[sum02].4s, v28.4h\n"
                    "smlal  v29.8h, v22.8b, %[k7].8b\n"
                    "saddw2  %[sum03].4s, %[sum03].4s, v28.8h\n"
                    "smlal2 v30.8h, v22.16b, %[k7].16b\n"
                    "saddw   %[sum00].4s, %[sum00].4s, v29.4h\n"
                    "smull2 v28.8h, v23.16b, %[k8].16b\n"
                    "saddw2  %[sum01].4s, %[sum01].4s, v29.8h\n"
                    "saddw   %[sum02].4s, %[sum02].4s, v30.4h\n"
                    "saddw2  %[sum03].4s, %[sum03].4s, v30.8h\n"

                    //! sum1 line<0,1>
                    "saddw2  %[sum03].4s, %[sum03].4s, v28.8h\n"
                    "smull  v29.8h, v24.8b, %[k0].8b\n"
                    "saddw   %[sum00].4s, %[sum00].4s, v27.4h\n"
                    "smull2 v30.8h, v24.16b, %[k0].16b\n"
                    "saddw2  %[sum01].4s, %[sum01].4s, v27.8h\n"
                    "smull  v27.8h, v25.8b, %[k1].8b\n"
                    "saddw   %[sum02].4s, %[sum02].4s, v28.4h\n"
                    "smull2 v28.8h, v25.16b, %[k1].16b\n"
                    "smlal  v29.8h, v21.8b, %[k3].8b\n"
                    "smlal2 v30.8h, v21.16b, %[k3].16b\n"
                    "saddw   %[sum10].4s, %[sum10].4s, v29.4h\n"
                    "smlal  v27.8h, v22.8b, %[k4].8b\n"
                    "saddw2  %[sum11].4s, %[sum11].4s, v29.8h\n"
                    "smlal2 v28.8h, v22.16b, %[k4].16b\n"
                    "saddw   %[sum12].4s, %[sum12].4s, v30.4h\n"
                    "saddw2  %[sum13].4s, %[sum13].4s, v30.8h\n"

                    "ldr q24, [%[sptr3]]\n"
                    "smull  v29.8h, v26.8b, %[k2].8b\n"
                    "saddw   %[sum10].4s, %[sum10].4s, v27.4h\n"
                    "smull2 v30.8h, v26.16b, %[k2].16b\n"
                    "saddw2  %[sum11].4s, %[sum11].4s, v27.8h\n"
                    "smlal  v29.8h, v23.8b, %[k5].8b\n"
                    "saddw   %[sum12].4s, %[sum12].4s, v28.4h\n"
                    "smlal2 v30.8h, v23.16b, %[k5].16b\n"
                    "saddw2  %[sum13].4s, %[sum13].4s, v28.8h\n"
                    "ldr q26, [%[sptr3], #16]\n"
                    "saddw   %[sum10].4s, %[sum10].4s, v29.4h\n"
                    "ext v25.16b, v24.16b, v26.16b, #4\n"
                    "saddw2  %[sum11].4s, %[sum11].4s, v29.8h\n"
                    "ext v26.16b, v24.16b, v26.16b, #8\n"
                    "saddw   %[sum12].4s, %[sum12].4s, v30.4h\n"
                    //! src line 3

                    //! sum1 line<2>
                    "smull  v27.8h, v24.8b, %[k6].8b\n"
                    "saddw2  %[sum13].4s, %[sum13].4s, v30.8h\n"
                    "smull2 v28.8h, v24.16b, %[k6].16b\n"
                    "smlal  v27.8h, v25.8b, %[k7].8b\n"
                    "smlal2 v28.8h, v25.16b, %[k7].16b\n"
                    "saddw   %[sum10].4s, %[sum10].4s, v27.4h\n"
                    "saddw2  %[sum11].4s, %[sum11].4s, v27.8h\n"

                    "smull  v29.8h, v26.8b, %[k8].8b\n"
                    "saddw   %[sum12].4s, %[sum12].4s, v28.4h\n"
                    "smull2 v30.8h, v26.16b, %[k8].16b\n"
                    "saddw2  %[sum13].4s, %[sum13].4s, v28.8h\n"

                    //! sum2 line<0,1>
                    "smull  v27.8h, v21.8b, %[k0].8b\n"
                    "saddw   %[sum10].4s, %[sum10].4s, v29.4h\n"
                    "smull2 v28.8h, v21.16b, %[k0].16b\n"
                    "saddw2  %[sum11].4s, %[sum11].4s, v29.8h\n"
                    "smull  v29.8h, v22.8b, %[k1].8b\n"
                    "saddw   %[sum12].4s, %[sum12].4s, v30.4h\n"
                    "smlal  v27.8h, v24.8b, %[k3].8b\n"
                    "saddw2  %[sum13].4s, %[sum13].4s, v30.8h\n"
                    "smull2 v30.8h, v22.16b, %[k1].16b\n"
                    "ldr q21, [%[sptr4]]\n"
                    "saddw   %[sum20].4s, %[sum20].4s, v27.4h\n"
                    "smlal2 v28.8h, v24.16b, %[k3].16b\n"
                    "saddw2  %[sum21].4s, %[sum21].4s, v27.8h\n"
                    "smlal  v29.8h, v25.8b, %[k4].8b\n"
                    "saddw   %[sum22].4s, %[sum22].4s, v28.4h\n"
                    "smlal2 v30.8h, v25.16b, %[k4].16b\n"
                    "saddw2  %[sum23].4s, %[sum23].4s, v28.8h\n"

                    "smull  v27.8h, v23.8b, %[k2].8b\n"
                    "saddw   %[sum20].4s, %[sum20].4s, v29.4h\n"
                    "smull2 v28.8h, v23.16b, %[k2].16b\n"
                    "saddw2  %[sum21].4s, %[sum21].4s, v29.8h\n"
                    "ldr q23, [%[sptr4], #16]\n"
                    "smlal  v27.8h, v26.8b, %[k5].8b\n"
                    "saddw   %[sum22].4s, %[sum22].4s, v30.4h\n"
                    "smlal2 v28.8h, v26.16b, %[k5].16b\n"
                    "saddw2  %[sum23].4s, %[sum23].4s, v30.8h\n"
                    "ext v22.16b, v21.16b, v23.16b, #4\n"
                    "saddw   %[sum20].4s, %[sum20].4s, v27.4h\n"
                    "ext v23.16b, v21.16b, v23.16b, #8\n"
                    "saddw2  %[sum21].4s, %[sum21].4s, v27.8h\n"
                    //! src line 3

                    //! sum2 line<2>
                    "smull  v29.8h, v21.8b, %[k6].8b\n"
                    "saddw   %[sum22].4s, %[sum22].4s, v28.4h\n"
                    "smull2 v30.8h, v21.16b, %[k6].16b\n"
                    "saddw2  %[sum23].4s, %[sum23].4s, v28.8h\n"
                    "smull  v27.8h, v23.8b, %[k8].8b\n"
                    "smull2 v28.8h, v23.16b, %[k8].16b\n"
                    "smlal  v29.8h, v22.8b, %[k7].8b\n"
                    "smlal2 v30.8h, v22.16b, %[k7].16b\n"
                    "saddw   %[sum20].4s, %[sum20].4s, v29.4h\n"
                    "saddw2  %[sum21].4s, %[sum21].4s, v29.8h\n"
                    "saddw   %[sum22].4s, %[sum22].4s, v30.4h\n"
                    "saddw2  %[sum23].4s, %[sum23].4s, v30.8h\n"
                    "saddw   %[sum20].4s, %[sum20].4s, v27.4h\n"
                    "saddw2  %[sum21].4s, %[sum21].4s, v27.8h\n"
                    "saddw   %[sum22].4s, %[sum22].4s, v28.4h\n"
                    "saddw2  %[sum23].4s, %[sum23].4s, v28.8h\n"
                    : [k0] "+w"(kern[0]), [k1] "+w"(kern[1]),
                      [k2] "+w"(kern[2]), [k3] "+w"(kern[3]),
                      [k4] "+w"(kern[4]), [k5] "+w"(kern[5]),
                      [k6] "+w"(kern[6]), [k7] "+w"(kern[7]),
                      [k8] "+w"(kern[8]), [sum00] "+w"(sum0[0]),
                      [sum01] "+w"(sum0[1]), [sum02] "+w"(sum0[2]),
                      [sum03] "+w"(sum0[3]), [sum10] "+w"(sum1[0]),
                      [sum11] "+w"(sum1[1]), [sum12] "+w"(sum1[2]),
                      [sum13] "+w"(sum1[3]), [sum20] "+w"(sum2[0]),
                      [sum21] "+w"(sum2[1]), [sum22] "+w"(sum2[2]),
                      [sum23] "+w"(sum2[3]), [sptr0] "+r"(sptr0),
                      [sptr1] "+r"(sptr1), [sptr2] "+r"(sptr2),
                      [sptr3] "+r"(sptr3), [sptr4] "+r"(sptr4)
                    :
                    : "v21", "v22", "v23", "v24", "v25", "v26", "v27", "v28",
                      "v29", "v30", "v31", "cc", "memory");

            STORE_1_LINE(dst, (oh), ow, OW, sum0);
            STORE_1_LINE(dst, (oh + 1), ow, OW, sum1);
            STORE_1_LINE(dst, (oh + 2), ow, OW, sum2);
#else
            int8x16_t src[2][3];
            LOAD_2_LINE_SRC(sptr0, sptr1);

            accumulate_2_q_vector(src[0][0], kern[0], src[1][0], kern[3], sum0);
            accumulate_2_q_vector(src[0][1], kern[1], src[1][1], kern[4], sum0);
            accumulate_2_q_vector(src[0][2], kern[2], src[1][2], kern[5], sum0);

            LOAD_1_LINE_SRC(sptr2, src[0]);

            ACC_1_LINE(src[0], kern[6], kern[7], kern[8], sum0);

            accumulate_2_q_vector(src[1][0], kern[0], src[0][0], kern[3], sum1);
            accumulate_2_q_vector(src[1][1], kern[1], src[0][1], kern[4], sum1);
            accumulate_2_q_vector(src[1][2], kern[2], src[0][2], kern[5], sum1);

            STORE_1_LINE(dst, oh, ow, OW, sum0);

            LOAD_1_LINE_SRC(sptr3, src[1]);
            ACC_1_LINE(src[1], kern[6], kern[7], kern[8], sum1);

            accumulate_2_q_vector(src[0][0], kern[0], src[1][0], kern[3], sum2);
            accumulate_2_q_vector(src[0][1], kern[1], src[1][1], kern[4], sum2);
            accumulate_2_q_vector(src[0][2], kern[2], src[1][2], kern[5], sum2);

            STORE_1_LINE(dst, (oh + 1), ow, OW, sum1);
            LOAD_1_LINE_SRC(sptr4, src[0]);
            ACC_1_LINE(src[0], kern[6], kern[7], kern[8], sum2);

            STORE_1_LINE(dst, (oh + 2), ow, OW, sum2);
#endif
        }
        if (ow < OW) {
            size_t iw = ow;
            size_t remain = OW - ow;
            const int8_t* __restrict sptr0 = sptr + ih * IW * 4 + iw * 4;
            const int8_t* __restrict sptr1 = sptr + (ih + 1) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr2 = sptr + (ih + 2) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr3 = sptr + (ih + 3) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr4 = sptr + (ih + 4) * IW * 4 + iw * 4;
            int32x4_t sum0[4], sum1[4], sum2[4];
            int8x16_t src[2][3];
#define cb(j)         \
    sum0[j] = init_v; \
    sum1[j] = init_v; \
    sum2[j] = init_v;
            UNROLL_CALL_NOWRAPPER(4, cb);
#undef cb
            LOAD_2_LINE_SRC(sptr0, sptr1);

            accumulate_2_q_vector(src[0][0], kern[0], src[1][0], kern[3], sum0);
            accumulate_2_q_vector(src[0][1], kern[1], src[1][1], kern[4], sum0);
            accumulate_2_q_vector(src[0][2], kern[2], src[1][2], kern[5], sum0);

            LOAD_1_LINE_SRC(sptr2, src[0]);
            ACC_1_LINE(src[0], kern[6], kern[7], kern[8], sum0);

            accumulate_2_q_vector(src[1][0], kern[0], src[0][0], kern[3], sum1);
            accumulate_2_q_vector(src[1][1], kern[1], src[0][1], kern[4], sum1);
            accumulate_2_q_vector(src[1][2], kern[2], src[0][2], kern[5], sum1);

            STORE_1_LINE_REMAIN(dst, oh, ow, OW, sum0, remain);

            LOAD_1_LINE_SRC(sptr3, src[1]);
            ACC_1_LINE(src[1], kern[6], kern[7], kern[8], sum1);

            accumulate_2_q_vector(src[0][0], kern[0], src[1][0], kern[3], sum2);
            accumulate_2_q_vector(src[0][1], kern[1], src[1][1], kern[4], sum2);
            accumulate_2_q_vector(src[0][2], kern[2], src[1][2], kern[5], sum2);

            STORE_1_LINE_REMAIN(dst, (oh + 1), ow, OW, sum1, remain);
            LOAD_1_LINE_SRC(sptr4, src[0]);
            ACC_1_LINE(src[0], kern[6], kern[7], kern[8], sum2);

            STORE_1_LINE_REMAIN(dst, (oh + 2), ow, OW, sum2, remain);
        }
    }
    for (; oh < OH; oh++) {
        size_t ih = oh;
        size_t ow = 0_z;
        for (; ow + 4 <= OW; ow += 4) {
            size_t iw = ow;
            const int8_t* __restrict sptr0 = sptr + ih * IW * 4 + iw * 4;
            const int8_t* __restrict sptr1 = sptr + (ih + 1) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr2 = sptr + (ih + 2) * IW * 4 + iw * 4;
            int32x4_t sum0[4];
            int8x16_t src[2][3];
#define cb(i) sum0[i] = init_v;
            UNROLL_CALL_NOWRAPPER(4, cb);
#undef cb
            LOAD_2_LINE_SRC(sptr0, sptr1);
            accumulate_2_q_vector(src[0][0], kern[0], src[1][0], kern[3], sum0);
            accumulate_2_q_vector(src[0][1], kern[1], src[1][1], kern[4], sum0);
            accumulate_2_q_vector(src[0][2], kern[2], src[1][2], kern[5], sum0);
            LOAD_1_LINE_SRC(sptr2, src[0]);
            ACC_1_LINE(src[0], kern[6], kern[7], kern[8], sum0);
            STORE_1_LINE(dst, oh, ow, OW, sum0);
        }
        if (ow < OW) {
            size_t iw = ow;
            const int8_t* __restrict sptr0 = sptr + ih * IW * 4 + iw * 4;
            const int8_t* __restrict sptr1 = sptr + (ih + 1) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr2 = sptr + (ih + 2) * IW * 4 + iw * 4;
            int32x4_t sum0[4];
            int8x16_t src[2][3];
#define cb(i) sum0[i] = init_v;
            UNROLL_CALL_NOWRAPPER(4, cb);
#undef cb
            LOAD_2_LINE_SRC(sptr0, sptr1);
            accumulate_2_q_vector(src[0][0], kern[0], src[1][0], kern[3], sum0);
            accumulate_2_q_vector(src[0][1], kern[1], src[1][1], kern[4], sum0);
            accumulate_2_q_vector(src[0][2], kern[2], src[1][2], kern[5], sum0);
            LOAD_1_LINE_SRC(sptr2, src[0]);
            ACC_1_LINE(src[0], kern[6], kern[7], kern[8], sum0);
            STORE_1_LINE_REMAIN(dst, oh, ow, OW, sum0, (OW - ow));
        }
    }
#undef LOAD_1_LINE_SRC
#undef LOAD_2_LINE_SRC
#undef ACC_1_LINE
}

template <bool quantized, BiasMode bias_mode, typename Op>
void channel_wise_nchw44::direct_stride1_5x5_int8(
        const int8_t* sptr, const int8_t* fptr, const int32_t* bias, void* dst,
        const size_t IH, const size_t IW, const size_t OH, const size_t OW,
        const Op& op) {
    MEGDNN_MARK_USED_VAR(IH);
    const int32_t* __restrict bptr = bias;
    int32x4_t init_v;
    if (bias_mode == BiasMode::BROADCAST_CHANNEL_BIAS) {
        init_v = vld1q_s32(bptr);
    } else {
        init_v = vdupq_n_s32(0);
    }
    const int* filter = reinterpret_cast<const int*>(fptr);

#define LOAD_1_LINE_SRC(sptr, src)        \
    src[0] = vld1q_s8(sptr);              \
    src[4] = vld1q_s8(sptr + 16);         \
    src[1] = vextq_s8(src[0], src[4], 4); \
    src[2] = vextq_s8(src[0], src[4], 8); \
    src[3] = vextq_s8(src[0], src[4], 12);

#define ACC_1_LINE(src, kern, sum)                                \
    accumulate_2_q_vector(src[0], kern[0], src[1], kern[1], sum); \
    accumulate_2_q_vector(src[2], kern[2], src[3], kern[3], sum); \
    accumulate_1_q_vector(src[4], kern[4], sum);

    size_t oh = 0_z;
    for (; oh + 2 <= OH; oh += 2) {
        size_t ih = oh;
        size_t ow = 0_z;
        for (; ow + 4 <= OW; ow += 4) {
            size_t iw = ow;
            const int8_t* __restrict sptr0 = sptr + ih * IW * 4 + iw * 4;
            const int8_t* __restrict sptr1 = sptr0 + IW * 4;
            const int8_t* __restrict sptr2 = sptr1 + IW * 4;
            const int8_t* __restrict sptr3 = sptr2 + IW * 4;
            const int8_t* __restrict sptr4 = sptr3 + IW * 4;
            const int8_t* __restrict sptr5 = sptr4 + IW * 4;
            int32x4_t sum0[4], sum1[4];
            int8x16_t src[2][5];
            int8x16_t kern[2][5];
#define cb(j)         \
    sum0[j] = init_v; \
    sum1[j] = init_v;
            UNROLL_CALL_NOWRAPPER(4, cb);
#undef cb

            //! first two line in filter
#define cb(i, kern, filter) kern[i] = (int8x16_t)vld1q_dup_s32((filter) + i);
            UNROLL_CALL(5, cb, kern[0], filter);
            UNROLL_CALL(5, cb, kern[1], (filter + 5));
#undef cb
            LOAD_1_LINE_SRC(sptr0, src[0]);
            LOAD_1_LINE_SRC(sptr1, src[1]);
#define cb(i, sum) \
    accumulate_2_q_vector(src[0][i], kern[0][i], src[1][i], kern[1][i], sum);
            UNROLL_CALL(5, cb, sum0);
#undef cb

            LOAD_1_LINE_SRC(sptr2, src[0]);

#define cb(i, sum) \
    accumulate_2_q_vector(src[1][i], kern[0][i], src[0][i], kern[1][i], sum);
            UNROLL_CALL(5, cb, sum1);
#undef cb
            //! second two line in filter
            LOAD_1_LINE_SRC(sptr3, src[1]);

#define cb(i, kern, filter) kern[i] = (int8x16_t)vld1q_dup_s32((filter) + i);
            UNROLL_CALL(5, cb, kern[0], filter + 10);
            UNROLL_CALL(5, cb, kern[1], (filter + 15));
#undef cb
#define cb(i, sum) \
    accumulate_2_q_vector(src[0][i], kern[0][i], src[1][i], kern[1][i], sum);
            UNROLL_CALL(5, cb, sum0);
#undef cb
            LOAD_1_LINE_SRC(sptr4, src[0]);

#define cb(i, sum) \
    accumulate_2_q_vector(src[1][i], kern[0][i], src[0][i], kern[1][i], sum);
            UNROLL_CALL(5, cb, sum1);
#undef cb
            //! last line in filter
#define cb(i, kern, filter) kern[i] = (int8x16_t)vld1q_dup_s32((filter) + i);
            UNROLL_CALL(5, cb, kern[0], filter + 20);
#undef cb

            ACC_1_LINE(src[0], kern[0], sum0);

            LOAD_1_LINE_SRC(sptr5, src[1]);

            ACC_1_LINE(src[1], kern[0], sum1);

            STORE_1_LINE(dst, oh, ow, OW, sum0);
            STORE_1_LINE(dst, (oh + 1), ow, OW, sum1);
        }
        if (ow < OW) {
            size_t remain = OW - ow;
            size_t iw = ow;
            const int8_t* __restrict sptr0 = sptr + ih * IW * 4 + iw * 4;
            const int8_t* __restrict sptr1 = sptr0 + IW * 4;
            const int8_t* __restrict sptr2 = sptr1 + IW * 4;
            const int8_t* __restrict sptr3 = sptr2 + IW * 4;
            const int8_t* __restrict sptr4 = sptr3 + IW * 4;
            const int8_t* __restrict sptr5 = sptr4 + IW * 4;
            int32x4_t sum0[4], sum1[4];
            int8x16_t src[2][5];
            int8x16_t kern[2][5];
#define cb(j)         \
    sum0[j] = init_v; \
    sum1[j] = init_v;
            UNROLL_CALL_NOWRAPPER(4, cb);
#undef cb

            //! first two line in filter
#define cb(i, kern, filter) kern[i] = (int8x16_t)vld1q_dup_s32((filter) + i);
            UNROLL_CALL(5, cb, kern[0], filter);
            UNROLL_CALL(5, cb, kern[1], (filter + 5));
#undef cb
            LOAD_1_LINE_SRC(sptr0, src[0]);
            LOAD_1_LINE_SRC(sptr1, src[1]);
#define cb(i, sum) \
    accumulate_2_q_vector(src[0][i], kern[0][i], src[1][i], kern[1][i], sum);
            UNROLL_CALL(5, cb, sum0);
#undef cb

            LOAD_1_LINE_SRC(sptr2, src[0]);
#define cb(i, sum) \
    accumulate_2_q_vector(src[1][i], kern[0][i], src[0][i], kern[1][i], sum);
            UNROLL_CALL(5, cb, sum1);
#undef cb
            //! second two line in filter
            LOAD_1_LINE_SRC(sptr3, src[1]);

#define cb(i, kern, filter) kern[i] = (int8x16_t)vld1q_dup_s32((filter) + i);
            UNROLL_CALL(5, cb, kern[0], filter + 10);
            UNROLL_CALL(5, cb, kern[1], (filter + 15));
#undef cb
#define cb(i, sum) \
    accumulate_2_q_vector(src[0][i], kern[0][i], src[1][i], kern[1][i], sum);
            UNROLL_CALL(5, cb, sum0);
#undef cb
            LOAD_1_LINE_SRC(sptr4, src[0]);

#define cb(i, sum) \
    accumulate_2_q_vector(src[1][i], kern[0][i], src[0][i], kern[1][i], sum);
            UNROLL_CALL(5, cb, sum1);
#undef cb
            //! last line in filter
#define cb(i, kern, filter) kern[i] = (int8x16_t)vld1q_dup_s32((filter) + i);
            UNROLL_CALL(5, cb, kern[0], filter + 20);
#undef cb

            ACC_1_LINE(src[0], kern[0], sum0);

            LOAD_1_LINE_SRC(sptr5, src[1]);

            ACC_1_LINE(src[1], kern[0], sum1);

            STORE_1_LINE_REMAIN(dst, oh, ow, OW, sum0, remain);
            STORE_1_LINE_REMAIN(dst, (oh + 1), ow, OW, sum1, remain);
        }
    }
    for (; oh < OH; oh++) {
        size_t ih = oh;
        size_t ow = 0_z;
        for (; ow + 4 <= OW; ow += 4) {
            size_t iw = ow;
            const int8_t* __restrict sptr0 = sptr + ih * IW * 4 + iw * 4;
            const int8_t* __restrict sptr1 = sptr0 + IW * 4;
            const int8_t* __restrict sptr2 = sptr1 + IW * 4;
            const int8_t* __restrict sptr3 = sptr2 + IW * 4;
            const int8_t* __restrict sptr4 = sptr3 + IW * 4;
            int32x4_t sum0[4];
            int8x16_t src[2][5];
            int8x16_t kern[2][5];
#define cb(j) sum0[j] = init_v;
            UNROLL_CALL_NOWRAPPER(4, cb);
#undef cb
            //! first two line in filter
#define cb(i, kern, filter) kern[i] = (int8x16_t)vld1q_dup_s32((filter) + i);
            UNROLL_CALL(5, cb, kern[0], filter);
            UNROLL_CALL(5, cb, kern[1], (filter + 5));
#undef cb
            LOAD_1_LINE_SRC(sptr0, src[0]);
            LOAD_1_LINE_SRC(sptr1, src[1]);
#define cb(i, sum) \
    accumulate_2_q_vector(src[0][i], kern[0][i], src[1][i], kern[1][i], sum);
            UNROLL_CALL(5, cb, sum0);
#undef cb
            //! second two line in filter
            LOAD_1_LINE_SRC(sptr2, src[0]);
            LOAD_1_LINE_SRC(sptr3, src[1]);
#define cb(i, kern, filter) kern[i] = (int8x16_t)vld1q_dup_s32((filter) + i);
            UNROLL_CALL(5, cb, kern[0], filter + 10);
            UNROLL_CALL(5, cb, kern[1], (filter + 15));
#undef cb
#define cb(i, sum) \
    accumulate_2_q_vector(src[0][i], kern[0][i], src[1][i], kern[1][i], sum);
            UNROLL_CALL(5, cb, sum0);
#undef cb
            //! last line in filter
            LOAD_1_LINE_SRC(sptr4, src[0]);
#define cb(i, kern, filter) kern[i] = (int8x16_t)vld1q_dup_s32((filter) + i);
            UNROLL_CALL(5, cb, kern[0], filter + 20);
#undef cb
            ACC_1_LINE(src[0], kern[0], sum0);
            STORE_1_LINE(dst, oh, ow, OW, sum0);
        }
        if (ow < OW) {
            size_t remain = OW - ow;
            size_t iw = ow;
            const int8_t* __restrict sptr0 = sptr + ih * IW * 4 + iw * 4;
            const int8_t* __restrict sptr1 = sptr0 + IW * 4;
            const int8_t* __restrict sptr2 = sptr1 + IW * 4;
            const int8_t* __restrict sptr3 = sptr2 + IW * 4;
            const int8_t* __restrict sptr4 = sptr3 + IW * 4;
            int32x4_t sum0[4];
            int8x16_t src[2][5];
            int8x16_t kern[2][5];
#define cb(j) sum0[j] = init_v;
            UNROLL_CALL_NOWRAPPER(4, cb);
#undef cb
            //! first two line in filter
#define cb(i, kern, filter) kern[i] = (int8x16_t)vld1q_dup_s32((filter) + i);
            UNROLL_CALL(5, cb, kern[0], filter);
            UNROLL_CALL(5, cb, kern[1], (filter + 5));
#undef cb
            LOAD_1_LINE_SRC(sptr0, src[0]);
            LOAD_1_LINE_SRC(sptr1, src[1]);
#define cb(i, sum) \
    accumulate_2_q_vector(src[0][i], kern[0][i], src[1][i], kern[1][i], sum);
            UNROLL_CALL(5, cb, sum0);
#undef cb
            //! second two line in filter
            LOAD_1_LINE_SRC(sptr2, src[0]);
            LOAD_1_LINE_SRC(sptr3, src[1]);
#define cb(i, kern, filter) kern[i] = (int8x16_t)vld1q_dup_s32((filter) + i);
            UNROLL_CALL(5, cb, kern[0], filter + 10);
            UNROLL_CALL(5, cb, kern[1], (filter + 15));
#undef cb
#define cb(i, sum) \
    accumulate_2_q_vector(src[0][i], kern[0][i], src[1][i], kern[1][i], sum);
            UNROLL_CALL(5, cb, sum0);
#undef cb
            //! last line in filter
            LOAD_1_LINE_SRC(sptr4, src[0]);
#define cb(i, kern, filter) kern[i] = (int8x16_t)vld1q_dup_s32((filter) + i);
            UNROLL_CALL(5, cb, kern[0], filter + 20);
#undef cb
            ACC_1_LINE(src[0], kern[0], sum0);
            STORE_1_LINE_REMAIN(dst, oh, ow, OW, sum0, remain);
        }
    }
#undef LOAD_1_LINE_SRC
#undef LOAD_2_LINE_SRC
#undef ACC_1_LINE
}

template <bool quantized, BiasMode bias_mode, typename Op>
void channel_wise_nchw44::direct_stride2_2x2_int8(
        const int8_t* src, const int8_t* filter, const int32_t* bias, void* dst,
        const size_t IH, const size_t IW, const size_t OH, const size_t OW,
        const Op& op) {
    MEGDNN_MARK_USED_VAR(IH);
    int32x4_t init_v;
    if (bias_mode == BiasMode::BROADCAST_CHANNEL_BIAS) {
        init_v = vld1q_s32(bias);
    } else {
        init_v = vdupq_n_s32(0);
    }
    int8x8_t kern01 = vld1_s8(filter);
    int8x8_t kern23 = vld1_s8(filter + 8);
    size_t oh = 0_z;
    for (; oh + 2 <= OH; oh += 2) {
        size_t ih = oh * 2;
        size_t ow = 0_z;
        for (; ow + 4 <= OW; ow += 4) {
            size_t iw = ow * 2;
            const int8_t* __restrict sptr0 = src + ih * IW * 4 + iw * 4;
            const int8_t* __restrict sptr1 = src + (ih + 1) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr2 = src + (ih + 2) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr3 = src + (ih + 3) * IW * 4 + iw * 4;

            int32x4_t sum[2][4];
#define cb(i)           \
    sum[0][i] = init_v; \
    sum[1][i] = init_v;
            UNROLL_CALL_NOWRAPPER(4, cb);
#undef cb
            int8x16_t src00 = vld1q_s8(sptr0);
            int8x16_t src01 = vld1q_s8(sptr0 + 16);

            int8x16_t src10 = vld1q_s8(sptr1);
            int8x16_t src11 = vld1q_s8(sptr1 + 16);

            accumulate_2_d_vector(src00, kern01, src10, kern23, sum[0][0],
                                  sum[0][1]);
            accumulate_2_d_vector(src01, kern01, src11, kern23, sum[0][2],
                                  sum[0][3]);

            int8x16_t src20 = vld1q_s8(sptr2);
            int8x16_t src21 = vld1q_s8(sptr2 + 16);

            int8x16_t src30 = vld1q_s8(sptr3);
            int8x16_t src31 = vld1q_s8(sptr3 + 16);

            accumulate_2_d_vector(src20, kern01, src30, kern23, sum[1][0],
                                  sum[1][1]);
            accumulate_2_d_vector(src21, kern01, src31, kern23, sum[1][2],
                                  sum[1][3]);

            STORE_1_LINE(dst, oh, ow, OW, sum[0]);
            STORE_1_LINE(dst, (oh + 1), ow, OW, sum[1]);
        }
        if (ow < OW) {
            size_t iw = ow * 2;
            size_t remain = OW - ow;
            const int8_t* __restrict sptr0 = src + ih * IW * 4 + iw * 4;
            const int8_t* __restrict sptr1 = src + (ih + 1) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr2 = src + (ih + 2) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr3 = src + (ih + 3) * IW * 4 + iw * 4;

            int32x4_t sum[2][4];
#define cb(i)           \
    sum[0][i] = init_v; \
    sum[1][i] = init_v;
            UNROLL_CALL_NOWRAPPER(4, cb);
#undef cb

            int8x16_t src00 = vld1q_s8(sptr0);
            int8x16_t src01 = vld1q_s8(sptr0 + 16);

            int8x16_t src10 = vld1q_s8(sptr1);
            int8x16_t src11 = vld1q_s8(sptr1 + 16);

            accumulate_2_d_vector(src00, kern01, src10, kern23, sum[0][0],
                                  sum[0][1]);
            accumulate_2_d_vector(src01, kern01, src11, kern23, sum[0][2],
                                  sum[0][3]);

            int8x16_t src20 = vld1q_s8(sptr2);
            int8x16_t src21 = vld1q_s8(sptr2 + 16);

            int8x16_t src30 = vld1q_s8(sptr3);
            int8x16_t src31 = vld1q_s8(sptr3 + 16);

            accumulate_2_d_vector(src20, kern01, src30, kern23, sum[1][0],
                                  sum[1][1]);
            accumulate_2_d_vector(src21, kern01, src31, kern23, sum[1][2],
                                  sum[1][3]);

            STORE_1_LINE_REMAIN(dst, oh, ow, OW, sum[0], remain);
            STORE_1_LINE_REMAIN(dst, (oh + 1), ow, OW, sum[1], remain);
        }
    }
    for (; oh < OH; oh++) {
        size_t ih = oh * 2;
        size_t ow = 0_z;
        for (; ow + 4 <= OW; ow += 4) {
            size_t iw = ow * 2;
            const int8_t* __restrict sptr0 = src + ih * IW * 4 + iw * 4;
            const int8_t* __restrict sptr1 = src + (ih + 1) * IW * 4 + iw * 4;
            int32x4_t sum0[4];
#define cb(i) sum0[i] = init_v;
            UNROLL_CALL_NOWRAPPER(4, cb);
#undef cb
            //! first  two line
            int8x16_t src00 = vld1q_s8(sptr0);
            int8x16_t src01 = vld1q_s8(sptr0 + 16);

            int8x16_t src10 = vld1q_s8(sptr1);
            int8x16_t src11 = vld1q_s8(sptr1 + 16);

            accumulate_2_d_vector(src00, kern01, src10, kern23, sum0[0],
                                  sum0[1]);
            accumulate_2_d_vector(src01, kern01, src11, kern23, sum0[2],
                                  sum0[3]);

            STORE_1_LINE(dst, oh, ow, OW, sum0);
        }
        if (OW > ow) {
            size_t iw = ow * 2;
            size_t remain = OW - ow;
            const int8_t* __restrict sptr0 = src + ih * IW * 4 + iw * 4;
            const int8_t* __restrict sptr1 = src + (ih + 1) * IW * 4 + iw * 4;
            int32x4_t sum0[4];
#define cb(i) sum0[i] = init_v;
            UNROLL_CALL_NOWRAPPER(4, cb);
#undef cb
            //! first  two line
            int8x16_t src00 = vld1q_s8(sptr0);
            int8x16_t src01 = vld1q_s8(sptr0 + 16);

            int8x16_t src10 = vld1q_s8(sptr1);
            int8x16_t src11 = vld1q_s8(sptr1 + 16);

            accumulate_2_d_vector(src00, kern01, src10, kern23, sum0[0],
                                  sum0[1]);
            accumulate_2_d_vector(src01, kern01, src11, kern23, sum0[2],
                                  sum0[3]);

            STORE_1_LINE_REMAIN(dst, oh, ow, OW, sum0, remain);
        }
    }
}

template <bool quantized, BiasMode bias_mode, typename Op>
void channel_wise_nchw44::direct_stride2_3x3_int8(
        const int8_t* src, const int8_t* filter, const int32_t* bias, void* dst,
        const size_t IH, const size_t IW, const size_t OH, const size_t OW,
        const Op& op) {
    MEGDNN_MARK_USED_VAR(IH);
    int32x4_t init_v;
    if (bias_mode == BiasMode::BROADCAST_CHANNEL_BIAS) {
        init_v = vld1q_s32(bias);
    } else {
        init_v = vdupq_n_s32(0);
    }
    int32x2_t zero = vdup_n_s32(0);
    int8x8_t kern01 = vld1_s8(filter);
    int8x8_t kern20 = vreinterpret_s8_s32(
            vzip_s32(vreinterpret_s32_s8(vld1_s8(filter + 8)), zero).val[0]);
    int8x8_t kern34 = vld1_s8(filter + 12);
    int8x8_t kern50 = vreinterpret_s8_s32(
            vzip_s32(vreinterpret_s32_s8(vld1_s8(filter + 20)), zero).val[0]);
    int8x8_t kern67 = vld1_s8(filter + 24);
    //! in case of illegal read
    int8x8_t kern80 = vreinterpret_s8_s32(
            vzip_s32(vreinterpret_s32_s8(vld1_s8(filter + 28)), zero).val[1]);

#define COMPUTE_ONE_LINE(src00, src01, src02, kern01, kern20, sum)             \
    accumulate_1_line_horizon(vget_low_s8(src00), kern01, vget_high_s8(src00), \
                              kern20, sum[0]);                                 \
    accumulate_1_line_horizon(vget_high_s8(src00), kern01, vget_low_s8(src01), \
                              kern20, sum[1]);                                 \
    accumulate_1_line_horizon(vget_low_s8(src01), kern01, vget_high_s8(src01), \
                              kern20, sum[2]);                                 \
    accumulate_1_line_horizon(vget_high_s8(src01), kern01, src02, kern20,      \
                              sum[3]);

    size_t oh = 0_z;
    for (; oh + 2 <= OH; oh += 2) {
        size_t ih = oh * 2;
        size_t ow = 0_z;
        for (; ow + 4 <= OW; ow += 4) {
            size_t iw = ow * 2;
            const int8_t* __restrict sptr0 = src + ih * IW * 4 + iw * 4;
            const int8_t* __restrict sptr1 = src + (ih + 1) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr2 = src + (ih + 2) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr3 = src + (ih + 3) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr4 = src + (ih + 4) * IW * 4 + iw * 4;
            int32x4_t sum[2][4];
#define cb(i)           \
    sum[0][i] = init_v; \
    sum[1][i] = init_v;
            UNROLL_CALL_NOWRAPPER(4, cb);
#undef cb
            //! line 0
            int8x16_t src00 = vld1q_s8(sptr0);
            int8x16_t src01 = vld1q_s8(sptr0 + 16);
            int8x8_t src02 = vld1_s8(sptr0 + 32);
            COMPUTE_ONE_LINE(src00, src01, src02, kern01, kern20, sum[0]);

            //! line 1
            int8x16_t src10 = vld1q_s8(sptr1);
            int8x16_t src11 = vld1q_s8(sptr1 + 16);
            int8x8_t src12 = vld1_s8(sptr1 + 32);
            COMPUTE_ONE_LINE(src10, src11, src12, kern34, kern50, sum[0]);

            //! line 2
            int8x16_t src20 = vld1q_s8(sptr2);
            int8x16_t src21 = vld1q_s8(sptr2 + 16);
            int8x8_t src22 = vld1_s8(sptr2 + 32);
            COMPUTE_ONE_LINE(src20, src21, src22, kern67, kern80, sum[0]);
            //! sum1
            COMPUTE_ONE_LINE(src20, src21, src22, kern01, kern20, sum[1]);

            //! line 3
            int8x16_t src30 = vld1q_s8(sptr3);
            int8x16_t src31 = vld1q_s8(sptr3 + 16);
            int8x8_t src32 = vld1_s8(sptr3 + 32);
            COMPUTE_ONE_LINE(src30, src31, src32, kern34, kern50, sum[1]);

            //! line 4
            int8x16_t src40 = vld1q_s8(sptr4);
            int8x16_t src41 = vld1q_s8(sptr4 + 16);
            int8x8_t src42 = vld1_s8(sptr4 + 32);
            COMPUTE_ONE_LINE(src40, src41, src42, kern67, kern80, sum[1]);

            STORE_1_LINE(dst, oh, ow, OW, sum[0]);
            STORE_1_LINE(dst, (oh + 1), ow, OW, sum[1]);
        }
        if (ow < OW) {
            size_t iw = ow * 2;
            size_t remain = OW - ow;
            const int8_t* __restrict sptr0 = src + ih * IW * 4 + iw * 4;
            const int8_t* __restrict sptr1 = src + (ih + 1) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr2 = src + (ih + 2) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr3 = src + (ih + 3) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr4 = src + (ih + 4) * IW * 4 + iw * 4;

            int32x4_t sum[2][4];
#define cb(i)           \
    sum[0][i] = init_v; \
    sum[1][i] = init_v;
            UNROLL_CALL_NOWRAPPER(4, cb);
#undef cb
            //! line 0
            int8x16_t src00 = vld1q_s8(sptr0);
            int8x16_t src01 = vld1q_s8(sptr0 + 16);
            int8x8_t src02 = vld1_s8(sptr0 + 32);
            COMPUTE_ONE_LINE(src00, src01, src02, kern01, kern20, sum[0]);

            //! line 1
            int8x16_t src10 = vld1q_s8(sptr1);
            int8x16_t src11 = vld1q_s8(sptr1 + 16);
            int8x8_t src12 = vld1_s8(sptr1 + 32);
            COMPUTE_ONE_LINE(src10, src11, src12, kern34, kern50, sum[0]);

            //! line 2
            int8x16_t src20 = vld1q_s8(sptr2);
            int8x16_t src21 = vld1q_s8(sptr2 + 16);
            int8x8_t src22 = vld1_s8(sptr2 + 32);
            COMPUTE_ONE_LINE(src20, src21, src22, kern67, kern80, sum[0]);
            //! sum1
            COMPUTE_ONE_LINE(src20, src21, src22, kern01, kern20, sum[1]);

            //! line 3
            int8x16_t src30 = vld1q_s8(sptr3);
            int8x16_t src31 = vld1q_s8(sptr3 + 16);
            int8x8_t src32 = vld1_s8(sptr3 + 32);
            COMPUTE_ONE_LINE(src30, src31, src32, kern34, kern50, sum[1]);

            //! line 4
            int8x16_t src40 = vld1q_s8(sptr4);
            int8x16_t src41 = vld1q_s8(sptr4 + 16);
            int8x8_t src42 = vld1_s8(sptr4 + 32);
            COMPUTE_ONE_LINE(src40, src41, src42, kern67, kern80, sum[1]);

            STORE_1_LINE_REMAIN(dst, oh, ow, OW, sum[0], remain);
            STORE_1_LINE_REMAIN(dst, (oh + 1), ow, OW, sum[1], remain);
        }
    }
    for (; oh < OH; oh++) {
        size_t ih = oh * 2;
        size_t ow = 0_z;
        for (; ow + 4 <= OW; ow += 4) {
            size_t iw = ow * 2;
            const int8_t* __restrict sptr0 = src + ih * IW * 4 + iw * 4;
            const int8_t* __restrict sptr1 = src + (ih + 1) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr2 = src + (ih + 2) * IW * 4 + iw * 4;
            int32x4_t sum[4];
#define cb(i) sum[i] = init_v;
            UNROLL_CALL_NOWRAPPER(4, cb);
#undef cb
            //! line 0
            int8x16_t src00 = vld1q_s8(sptr0);
            int8x16_t src01 = vld1q_s8(sptr0 + 16);
            int8x8_t src02 = vld1_s8(sptr0 + 32);
            COMPUTE_ONE_LINE(src00, src01, src02, kern01, kern20, sum);

            //! line 1
            int8x16_t src10 = vld1q_s8(sptr1);
            int8x16_t src11 = vld1q_s8(sptr1 + 16);
            int8x8_t src12 = vld1_s8(sptr1 + 32);
            COMPUTE_ONE_LINE(src10, src11, src12, kern34, kern50, sum);

            //! line 2
            int8x16_t src20 = vld1q_s8(sptr2);
            int8x16_t src21 = vld1q_s8(sptr2 + 16);
            int8x8_t src22 = vld1_s8(sptr2 + 32);
            COMPUTE_ONE_LINE(src20, src21, src22, kern67, kern80, sum);

            STORE_1_LINE(dst, oh, ow, OW, sum);
        }
        if (OW > ow) {
            size_t iw = ow * 2;
            size_t remain = OW - ow;
            const int8_t* __restrict sptr0 = src + ih * IW * 4 + iw * 4;
            const int8_t* __restrict sptr1 = src + (ih + 1) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr2 = src + (ih + 2) * IW * 4 + iw * 4;
            int32x4_t sum[4];
#define cb(i) sum[i] = init_v;
            UNROLL_CALL_NOWRAPPER(4, cb);
#undef cb
            //! line 0
            int8x16_t src00 = vld1q_s8(sptr0);
            int8x16_t src01 = vld1q_s8(sptr0 + 16);
            int8x8_t src02 = vld1_s8(sptr0 + 32);
            COMPUTE_ONE_LINE(src00, src01, src02, kern01, kern20, sum);

            //! line 1
            int8x16_t src10 = vld1q_s8(sptr1);
            int8x16_t src11 = vld1q_s8(sptr1 + 16);
            int8x8_t src12 = vld1_s8(sptr1 + 32);
            COMPUTE_ONE_LINE(src10, src11, src12, kern34, kern50, sum);

            //! line 2
            int8x16_t src20 = vld1q_s8(sptr2);
            int8x16_t src21 = vld1q_s8(sptr2 + 16);
            int8x8_t src22 = vld1_s8(sptr2 + 32);
            COMPUTE_ONE_LINE(src20, src21, src22, kern67, kern80, sum);

            STORE_1_LINE_REMAIN(dst, oh, ow, OW, sum, remain);
        }
    }
#undef COMPUTE_ONE_LINE
}

template <bool quantized, BiasMode bias_mode, typename Op>
void channel_wise_nchw44::direct_stride2_5x5_int8(
        const int8_t* src, const int8_t* filter, const int32_t* bias, void* dst,
        const size_t IH, const size_t IW, const size_t OH, const size_t OW,
        const Op& op) {
    MEGDNN_MARK_USED_VAR(IH);
    int32x4_t init_v;
    if (bias_mode == BiasMode::BROADCAST_CHANNEL_BIAS) {
        init_v = vld1q_s32(bias);
    } else {
        init_v = vdupq_n_s32(0);
    }
    int8x8_t kern0[3], kern1[3], kern2[3], kern3[3], kern4[3];
    int32x2_t zero = vdup_n_s32(0);
    kern0[0] = vld1_s8(filter);
    kern0[1] = vld1_s8(filter + 8);
    kern0[2] = vreinterpret_s8_s32(
            vzip_s32(vreinterpret_s32_s8(vld1_s8(filter + 16)), zero).val[0]);
    kern1[0] = vld1_s8(filter + 20);
    kern1[1] = vld1_s8(filter + 28);
    kern1[2] = vreinterpret_s8_s32(
            vzip_s32(vreinterpret_s32_s8(vld1_s8(filter + 36)), zero).val[0]);
    kern2[0] = vld1_s8(filter + 40);
    kern2[1] = vld1_s8(filter + 48);
    kern2[2] = vreinterpret_s8_s32(
            vzip_s32(vreinterpret_s32_s8(vld1_s8(filter + 56)), zero).val[0]);
    kern3[0] = vld1_s8(filter + 60);
    kern3[1] = vld1_s8(filter + 68);
    kern3[2] = vreinterpret_s8_s32(
            vzip_s32(vreinterpret_s32_s8(vld1_s8(filter + 76)), zero).val[0]);
    kern4[0] = vld1_s8(filter + 80);
    kern4[1] = vld1_s8(filter + 88);
    //! in case of illegal read
    kern4[2] = vreinterpret_s8_s32(
            vzip_s32(vreinterpret_s32_s8(vld1_s8(filter + 92)), zero).val[1]);

#define COMPUTE_ONE_VECTOR(src00, src01, src02, src10, src11, src12, kern0, \
                           kern1, sum)                                      \
    accumulate_1_line_horizon(src00, kern0[0], src10, kern1[0], sum);       \
    accumulate_1_line_horizon(src01, kern0[1], src11, kern1[1], sum);       \
    accumulate_1_line_horizon(src02, kern0[2], src12, kern1[2], sum);

#define COMPUTE_TWO_LINE(src0, src1, kern0, kern1, sum)                    \
    COMPUTE_ONE_VECTOR(vget_low_s8(src0[0]), vget_high_s8(src0[0]),        \
                       vget_low_s8(src0[1]), vget_low_s8(src1[0]),         \
                       vget_high_s8(src1[0]), vget_low_s8(src1[1]), kern0, \
                       kern1, sum[0])                                      \
    COMPUTE_ONE_VECTOR(vget_high_s8(src0[0]), vget_low_s8(src0[1]),        \
                       vget_high_s8(src0[1]), vget_high_s8(src1[0]),       \
                       vget_low_s8(src1[1]), vget_high_s8(src1[1]), kern0, \
                       kern1, sum[1])                                      \
    COMPUTE_ONE_VECTOR(vget_low_s8(src0[1]), vget_high_s8(src0[1]),        \
                       vget_low_s8(src0[2]), vget_low_s8(src1[1]),         \
                       vget_high_s8(src1[1]), vget_low_s8(src1[2]), kern0, \
                       kern1, sum[2])                                      \
    COMPUTE_ONE_VECTOR(vget_high_s8(src0[1]), vget_low_s8(src0[2]),        \
                       vget_high_s8(src0[2]), vget_high_s8(src1[1]),       \
                       vget_low_s8(src1[2]), vget_high_s8(src1[2]), kern0, \
                       kern1, sum[3])

#define COMPUTE_ONE_LINE(src, kern, sum)                              \
    accumulate_1_line_horizon(vget_low_s8(src[0]), kern[0],           \
                              vget_high_s8(src[0]), kern[1], sum[0]); \
    accumulate_1_line_horizon(vget_high_s8(src[0]), kern[0],          \
                              vget_low_s8(src[1]), kern[1], sum[1]);  \
    accumulate_1_line_horizon(vget_low_s8(src[1]), kern[0],           \
                              vget_high_s8(src[1]), kern[1], sum[2]); \
    accumulate_1_line_horizon(vget_high_s8(src[1]), kern[0],          \
                              vget_low_s8(src[2]), kern[1], sum[3]);  \
    accumulate_1_d_vector(vget_low_s8(src[1]), kern[2], sum[0]);      \
    accumulate_1_d_vector(vget_high_s8(src[1]), kern[2], sum[1]);     \
    accumulate_1_d_vector(vget_low_s8(src[2]), kern[2], sum[2]);      \
    accumulate_1_d_vector(vget_high_s8(src[2]), kern[2], sum[3])

    size_t oh = 0_z;
    for (; oh + 2 <= OH; oh += 2) {
        size_t ih = oh * 2;
        size_t ow = 0_z;
        for (; ow + 4 <= OW; ow += 4) {
            size_t iw = ow * 2;
            const int8_t* __restrict sptr0 = src + ih * IW * 4 + iw * 4;
            const int8_t* __restrict sptr1 = src + (ih + 1) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr2 = src + (ih + 2) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr3 = src + (ih + 3) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr4 = src + (ih + 4) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr5 = src + (ih + 5) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr6 = src + (ih + 6) * IW * 4 + iw * 4;
            int32x4_t sum[2][4];
#define cb(i)           \
    sum[0][i] = init_v; \
    sum[1][i] = init_v;
            UNROLL_CALL_NOWRAPPER(4, cb);
#undef cb
            int8x16_t src0[3], src1[3];
            //! line 0, 1
            src0[0] = vld1q_s8(sptr0);
            src0[1] = vld1q_s8(sptr0 + 16);
            src0[2] = vld1q_s8(sptr0 + 32);

            src1[0] = vld1q_s8(sptr1);
            src1[1] = vld1q_s8(sptr1 + 16);
            src1[2] = vld1q_s8(sptr1 + 32);

            COMPUTE_TWO_LINE(src0, src1, kern0, kern1, sum[0]);

            //! line 2,3
            src0[0] = vld1q_s8(sptr2);
            src0[1] = vld1q_s8(sptr2 + 16);
            src0[2] = vld1q_s8(sptr2 + 32);

            src1[0] = vld1q_s8(sptr3);
            src1[1] = vld1q_s8(sptr3 + 16);
            src1[2] = vld1q_s8(sptr3 + 32);

            COMPUTE_TWO_LINE(src0, src1, kern2, kern3, sum[0]);
            COMPUTE_TWO_LINE(src0, src1, kern0, kern1, sum[1]);

            //! line 4,5
            src0[0] = vld1q_s8(sptr4);
            src0[1] = vld1q_s8(sptr4 + 16);
            src0[2] = vld1q_s8(sptr4 + 32);

            src1[0] = vld1q_s8(sptr5);
            src1[1] = vld1q_s8(sptr5 + 16);
            src1[2] = vld1q_s8(sptr5 + 32);
            COMPUTE_ONE_LINE(src0, kern4, sum[0]);
            COMPUTE_TWO_LINE(src0, src1, kern2, kern3, sum[1]);

            //! line 6
            src0[0] = vld1q_s8(sptr6);
            src0[1] = vld1q_s8(sptr6 + 16);
            src0[2] = vld1q_s8(sptr6 + 32);

            COMPUTE_ONE_LINE(src0, kern4, sum[1]);

            STORE_1_LINE(dst, oh, ow, OW, sum[0]);
            STORE_1_LINE(dst, (oh + 1), ow, OW, sum[1]);
        }
        if (ow < OW) {
            size_t iw = ow * 2;
            size_t remain = OW - ow;
            const int8_t* __restrict sptr0 = src + ih * IW * 4 + iw * 4;
            const int8_t* __restrict sptr1 = src + (ih + 1) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr2 = src + (ih + 2) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr3 = src + (ih + 3) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr4 = src + (ih + 4) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr5 = src + (ih + 5) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr6 = src + (ih + 6) * IW * 4 + iw * 4;
            int32x4_t sum[2][4];
#define cb(i)           \
    sum[0][i] = init_v; \
    sum[1][i] = init_v;
            UNROLL_CALL_NOWRAPPER(4, cb);
#undef cb
            int8x16_t src0[3], src1[3];
            //! line 0, 1
            src0[0] = vld1q_s8(sptr0);
            src0[1] = vld1q_s8(sptr0 + 16);
            src0[2] = vld1q_s8(sptr0 + 32);

            src1[0] = vld1q_s8(sptr1);
            src1[1] = vld1q_s8(sptr1 + 16);
            src1[2] = vld1q_s8(sptr1 + 32);

            COMPUTE_TWO_LINE(src0, src1, kern0, kern1, sum[0]);

            //! line 2,3
            src0[0] = vld1q_s8(sptr2);
            src0[1] = vld1q_s8(sptr2 + 16);
            src0[2] = vld1q_s8(sptr2 + 32);

            src1[0] = vld1q_s8(sptr3);
            src1[1] = vld1q_s8(sptr3 + 16);
            src1[2] = vld1q_s8(sptr3 + 32);

            COMPUTE_TWO_LINE(src0, src1, kern2, kern3, sum[0]);
            COMPUTE_TWO_LINE(src0, src1, kern0, kern1, sum[1]);

            //! line 4,5
            src0[0] = vld1q_s8(sptr4);
            src0[1] = vld1q_s8(sptr4 + 16);
            src0[2] = vld1q_s8(sptr4 + 32);

            src1[0] = vld1q_s8(sptr5);
            src1[1] = vld1q_s8(sptr5 + 16);
            src1[2] = vld1q_s8(sptr5 + 32);
            COMPUTE_ONE_LINE(src0, kern4, sum[0]);
            COMPUTE_TWO_LINE(src0, src1, kern2, kern3, sum[1]);

            //! line 6
            src0[0] = vld1q_s8(sptr6);
            src0[1] = vld1q_s8(sptr6 + 16);
            src0[2] = vld1q_s8(sptr6 + 32);

            COMPUTE_ONE_LINE(src0, kern4, sum[1]);

            STORE_1_LINE_REMAIN(dst, oh, ow, OW, sum[0], remain);
            STORE_1_LINE_REMAIN(dst, (oh + 1), ow, OW, sum[1], remain);
        }
    }
    for (; oh < OH; oh++) {
        size_t ih = oh * 2;
        size_t ow = 0_z;
        for (; ow + 4 <= OW; ow += 4) {
            size_t iw = ow * 2;
            const int8_t* __restrict sptr0 = src + ih * IW * 4 + iw * 4;
            const int8_t* __restrict sptr1 = src + (ih + 1) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr2 = src + (ih + 2) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr3 = src + (ih + 3) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr4 = src + (ih + 4) * IW * 4 + iw * 4;
            int32x4_t sum[4];
#define cb(i) sum[i] = init_v;
            UNROLL_CALL_NOWRAPPER(4, cb);
#undef cb
            int8x16_t src0[3], src1[3];
            //! line 0, 1
            src0[0] = vld1q_s8(sptr0);
            src0[1] = vld1q_s8(sptr0 + 16);
            src0[2] = vld1q_s8(sptr0 + 32);

            src1[0] = vld1q_s8(sptr1);
            src1[1] = vld1q_s8(sptr1 + 16);
            src1[2] = vld1q_s8(sptr1 + 32);

            COMPUTE_TWO_LINE(src0, src1, kern0, kern1, sum);

            //! line 2,3
            src0[0] = vld1q_s8(sptr2);
            src0[1] = vld1q_s8(sptr2 + 16);
            src0[2] = vld1q_s8(sptr2 + 32);

            src1[0] = vld1q_s8(sptr3);
            src1[1] = vld1q_s8(sptr3 + 16);
            src1[2] = vld1q_s8(sptr3 + 32);

            COMPUTE_TWO_LINE(src0, src1, kern2, kern3, sum);

            //! line 4,5
            src0[0] = vld1q_s8(sptr4);
            src0[1] = vld1q_s8(sptr4 + 16);
            src0[2] = vld1q_s8(sptr4 + 32);

            COMPUTE_ONE_LINE(src0, kern4, sum);

            STORE_1_LINE(dst, oh, ow, OW, sum);
        }
        if (OW > ow) {
            size_t iw = ow * 2;
            size_t remain = OW - ow;
            const int8_t* __restrict sptr0 = src + ih * IW * 4 + iw * 4;
            const int8_t* __restrict sptr1 = src + (ih + 1) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr2 = src + (ih + 2) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr3 = src + (ih + 3) * IW * 4 + iw * 4;
            const int8_t* __restrict sptr4 = src + (ih + 4) * IW * 4 + iw * 4;
            int32x4_t sum[4];
#define cb(i) sum[i] = init_v;
            UNROLL_CALL_NOWRAPPER(4, cb);
#undef cb
            int8x16_t src0[3], src1[3];
            //! line 0, 1
            src0[0] = vld1q_s8(sptr0);
            src0[1] = vld1q_s8(sptr0 + 16);
            src0[2] = vld1q_s8(sptr0 + 32);

            src1[0] = vld1q_s8(sptr1);
            src1[1] = vld1q_s8(sptr1 + 16);
            src1[2] = vld1q_s8(sptr1 + 32);

            COMPUTE_TWO_LINE(src0, src1, kern0, kern1, sum);

            //! line 2,3
            src0[0] = vld1q_s8(sptr2);
            src0[1] = vld1q_s8(sptr2 + 16);
            src0[2] = vld1q_s8(sptr2 + 32);

            src1[0] = vld1q_s8(sptr3);
            src1[1] = vld1q_s8(sptr3 + 16);
            src1[2] = vld1q_s8(sptr3 + 32);

            COMPUTE_TWO_LINE(src0, src1, kern2, kern3, sum);

            //! line 4,5
            src0[0] = vld1q_s8(sptr4);
            src0[1] = vld1q_s8(sptr4 + 16);
            src0[2] = vld1q_s8(sptr4 + 32);

            COMPUTE_ONE_LINE(src0, kern4, sum);

            STORE_1_LINE_REMAIN(dst, oh, ow, OW, sum, remain);
        }
    }
#undef COMPUTE_ONE_VECTOR
#undef COMPUTE_ONE_LINE
#undef COMPUTE_TWO_LINE
}

#undef STORE_1_LINE
#undef STORE_1_LINE_REMAIN

#define INSTANTIATION(quantized, stride, i, bias, Op)                          \
    template void channel_wise_nchw44::direct_##stride##_##i##x##i##_int8<     \
            quantized, bias, Op>(const int8_t*, const int8_t*, const int32_t*, \
                                 void*, const size_t, const size_t,            \
                                 const size_t, const size_t, const Op&);

#define FOR_OP(stride, i, bias)                               \
    INSTANTIATION(true, stride, i, bias,                      \
                  TypeCvtOp<dt_qint32 MEGDNN_COMMA dt_qint8>) \
    INSTANTIATION(true, stride, i, bias,                      \
                  ReluOp<dt_qint32 MEGDNN_COMMA dt_qint8>)    \
    INSTANTIATION(true, stride, i, bias,                      \
                  HSwishOp<dt_qint32 MEGDNN_COMMA dt_qint8>)  \
    INSTANTIATION(false, stride, i, bias,                     \
                  NoneOp<dt_qint32 MEGDNN_COMMA dt_qint8>)

#define FOR_BIAS(stride, i)              \
    FOR_OP(stride, i, BiasMode::NO_BIAS) \
    FOR_OP(stride, i, BiasMode::BROADCAST_CHANNEL_BIAS)

#define FOR_FILTER(stride) \
    FOR_BIAS(stride, 2)    \
    FOR_BIAS(stride, 3)    \
    FOR_BIAS(stride, 5)

#define FOR_STRIDE      \
    FOR_FILTER(stride1) \
    FOR_FILTER(stride2)

FOR_STRIDE

#undef FOR_STRIDE
#undef FOR_FILTER
#undef FOR_BIAS
#undef FOR_OP
#undef INSTANTIATION

// vim: syntax=cpp.doxygen
