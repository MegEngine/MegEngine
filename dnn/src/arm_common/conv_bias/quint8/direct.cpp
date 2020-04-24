/**
 * \file dnn/src/arm_common/conv_bias/quint8/direct.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/arm_common/conv_bias/quint8/direct.h"
#include "src/arm_common/elemwise_op.h"
#include "src/arm_common/simd_macro/marm_neon.h"
#include "src/common/utils.h"
#include "src/fallback/conv_bias/common.h"

using namespace megdnn;
using namespace arm_common;

#include "midout.h"
MIDOUT_DECL(conv_direct_stride)

#define ACC_S16_S32(dst0, dst1, src)           \
    dst0 = vaddw_s16(dst0, vget_low_s16(src)); \
    dst1 = vaddw_s16(dst1, vget_high_s16(src));

#define SUB128(n) static_cast<int8_t>(static_cast<int32_t>(n) - 128)

#define SUB128VECTOR(src) \
    vqmovn_s16(vsubq_s16(vreinterpretq_s16_u16(vmovl_u8(src)), v128))

#define MLSFZP(s, f) vmlsl_s8(vmull_s8(s, f), s, fzp)

#define POSTPROCESS(dst0, dst1, tptr, dptr)                     \
    if (last_ic) {                                              \
        op({{dst0, dst1}}, reinterpret_cast<dt_quint8*>(dptr)); \
    } else {                                                    \
        vst1q_s32(tptr, dst0);                                  \
        vst1q_s32(tptr + 4, dst1);                              \
    }

template <bool first_ic, bool last_ic, BiasMode bias_mode, typename Op>
void conv_bias::conv_direct_stride1_2x2_quint8(
        const uint8_t* src, const uint8_t* filter, const int32_t* bias,
        int32_t* temp, uint8_t* dst, const size_t IH, const size_t IW,
        const size_t OH, const size_t OW, const int8_t src_zp,
        const int8_t filter_zp, const int32_t src_filter_zp, const Op& op) {
    MEGDNN_MARK_USED_VAR(IH);
    MIDOUT_BEGIN(conv_direct_stride, 0, 0) {
        int16x8_t v128 = vdupq_n_s16(128);
        int32x4_t vsrc_filter_zp = vdupq_n_s32(src_filter_zp);
        int8x8_t f00 = vdup_n_s8(SUB128(filter[0]));
        int8x8_t f01 = vdup_n_s8(SUB128(filter[1]));
        int8x8_t f10 = vdup_n_s8(SUB128(filter[2]));
        int8x8_t f11 = vdup_n_s8(SUB128(filter[3]));

        int8x8_t fzp = vdup_n_s8(filter_zp);

        // get filter * src_zp for one IC
        int32_t fxszp = 0;
        for (size_t i = 0; i < 4; ++i)
            fxszp += static_cast<int32_t>(filter[i]) - 128;
        int32x4_t vfxszp = vdupq_n_s32(fxszp * static_cast<int32_t>(src_zp));

        // 4x8 block
        size_t oh = 0;
        for (; oh + 4 <= OH; oh += 4) {
            size_t ih = oh;
            size_t ow = 0;
            for (; ow < OW; ow += 8) {
                size_t iw = ow;
                int32_t* __restrict tptr = temp + oh * OW + ow;
                uint8_t* __restrict dptr = dst + oh * OW + ow;
                const uint8_t* __restrict sptr = src + ih * IW + iw;
                const int32_t* __restrict bptr = bias;
                int32x4_t sum00, sum01, sum10, sum11, sum20, sum21, sum30, sum31;

                if (!first_ic) {
                    sum00 = vld1q_s32(tptr + 0 * OW);
                    sum01 = vld1q_s32(tptr + 0 * OW + 4);
                    sum10 = vld1q_s32(tptr + 1 * OW);
                    sum11 = vld1q_s32(tptr + 1 * OW + 4);
                    sum20 = vld1q_s32(tptr + 2 * OW);
                    sum21 = vld1q_s32(tptr + 2 * OW + 4);
                    sum30 = vld1q_s32(tptr + 3 * OW);
                    sum31 = vld1q_s32(tptr + 3 * OW + 4);
                } else {
                    if (bias_mode == BiasMode::BROADCAST_CHANNEL_BIAS) {
                        sum00 = vdupq_n_s32(bptr[0]);
                        sum01 = sum00;
                        sum10 = sum00;
                        sum11 = sum00;
                        sum20 = sum00;
                        sum21 = sum00;
                        sum30 = sum00;
                        sum31 = sum00;
                    } else {
                        sum00 = vdupq_n_s32(0);
                        sum01 = vdupq_n_s32(0);
                        sum10 = vdupq_n_s32(0);
                        sum11 = vdupq_n_s32(0);
                        sum20 = vdupq_n_s32(0);
                        sum21 = vdupq_n_s32(0);
                        sum30 = vdupq_n_s32(0);
                        sum31 = vdupq_n_s32(0);
                    }

                    // src_zp * filter_zp for one OC
                    sum00 += vsrc_filter_zp;
                    sum01 += vsrc_filter_zp;
                    sum10 += vsrc_filter_zp;
                    sum11 += vsrc_filter_zp;
                    sum20 += vsrc_filter_zp;
                    sum21 += vsrc_filter_zp;
                    sum30 += vsrc_filter_zp;
                    sum31 += vsrc_filter_zp;
                }

                int8x8_t s = SUB128VECTOR(vld1_u8(sptr + 0 * IW));
                ACC_S16_S32(sum00, sum01, MLSFZP(s, f00));

                s = SUB128VECTOR(vld1_u8(sptr + 1 * IW));
                ACC_S16_S32(sum00, sum01, MLSFZP(s, f10));
                ACC_S16_S32(sum10, sum11, MLSFZP(s, f00));

                s = SUB128VECTOR(vld1_u8(sptr + 2 * IW));
                ACC_S16_S32(sum10, sum11, MLSFZP(s, f10));
                ACC_S16_S32(sum20, sum21, MLSFZP(s, f00));

                s = SUB128VECTOR(vld1_u8(sptr + 3 * IW));
                ACC_S16_S32(sum20, sum21, MLSFZP(s, f10));
                ACC_S16_S32(sum30, sum31, MLSFZP(s, f00));

                s = SUB128VECTOR(vld1_u8(sptr + 4 * IW));
                ACC_S16_S32(sum30, sum31, MLSFZP(s, f10));

                ++sptr;

                s = SUB128VECTOR(vld1_u8(sptr + 0 * IW));
                ACC_S16_S32(sum00, sum01, MLSFZP(s, f01));

                s = SUB128VECTOR(vld1_u8(sptr + 1 * IW));
                ACC_S16_S32(sum00, sum01, MLSFZP(s, f11));
                sum00 = vsubq_s32(sum00, vfxszp);
                sum01 = vsubq_s32(sum01, vfxszp);
                POSTPROCESS(sum00, sum01, tptr + 0 * OW, dptr + 0 * OW);
                ACC_S16_S32(sum10, sum11, MLSFZP(s, f01));

                s = SUB128VECTOR(vld1_u8(sptr + 2 * IW));
                ACC_S16_S32(sum10, sum11, MLSFZP(s, f11));
                sum10 = vsubq_s32(sum10, vfxszp);
                sum11 = vsubq_s32(sum11, vfxszp);
                POSTPROCESS(sum10, sum11, tptr + 1 * OW, dptr + 1 * OW);
                ACC_S16_S32(sum20, sum21, MLSFZP(s, f01));

                s = SUB128VECTOR(vld1_u8(sptr + 3 * IW));
                ACC_S16_S32(sum20, sum21, MLSFZP(s, f11));
                sum20 = vsubq_s32(sum20, vfxszp);
                sum21 = vsubq_s32(sum21, vfxszp);
                POSTPROCESS(sum20, sum21, tptr + 2 * OW, dptr + 2 * OW);
                ACC_S16_S32(sum30, sum31, MLSFZP(s, f01));

                s = SUB128VECTOR(vld1_u8(sptr + 4 * IW));
                ACC_S16_S32(sum30, sum31, MLSFZP(s, f11));
                sum30 = vsubq_s32(sum30, vfxszp);
                sum31 = vsubq_s32(sum31, vfxszp);

                POSTPROCESS(sum30, sum31, tptr + 3 * OW, dptr + 3 * OW);
            }
        }

        if (oh + 3 == OH) {
            size_t ih = oh;
            size_t ow = 0;
            for (; ow < OW; ow += 8) {
                size_t iw = ow;
                int32_t* __restrict tptr = temp + oh * OW + ow;
                uint8_t* __restrict dptr = dst + oh * OW + ow;
                const uint8_t* __restrict sptr = src + ih * IW + iw;
                const int32_t* __restrict bptr = bias;
                int32x4_t sum00, sum01, sum10, sum11, sum20, sum21;

                if (!first_ic) {
                    sum00 = vld1q_s32(tptr + 0 * OW);
                    sum01 = vld1q_s32(tptr + 0 * OW + 4);
                    sum10 = vld1q_s32(tptr + 1 * OW);
                    sum11 = vld1q_s32(tptr + 1 * OW + 4);
                    sum20 = vld1q_s32(tptr + 2 * OW);
                    sum21 = vld1q_s32(tptr + 2 * OW + 4);
                } else {
                    if (bias_mode == BiasMode::BROADCAST_CHANNEL_BIAS) {
                        sum00 = vdupq_n_s32(bptr[0]);
                        sum01 = sum00;
                        sum10 = sum00;
                        sum11 = sum00;
                        sum20 = sum00;
                        sum21 = sum00;
                    } else {
                        sum00 = vdupq_n_s32(0);
                        sum01 = vdupq_n_s32(0);
                        sum10 = vdupq_n_s32(0);
                        sum11 = vdupq_n_s32(0);
                        sum20 = vdupq_n_s32(0);
                        sum21 = vdupq_n_s32(0);
                    }
                    sum00 += vsrc_filter_zp;
                    sum01 += vsrc_filter_zp;
                    sum10 += vsrc_filter_zp;
                    sum11 += vsrc_filter_zp;
                    sum20 += vsrc_filter_zp;
                    sum21 += vsrc_filter_zp;
                }

                int8x8_t s = SUB128VECTOR(vld1_u8(sptr + 0 * IW));
                ACC_S16_S32(sum00, sum01, MLSFZP(s, f00));

                s = SUB128VECTOR(vld1_u8(sptr + 1 * IW));
                ACC_S16_S32(sum00, sum01, MLSFZP(s, f10));
                ACC_S16_S32(sum10, sum11, MLSFZP(s, f00));

                s = SUB128VECTOR(vld1_u8(sptr + 2 * IW));
                ACC_S16_S32(sum10, sum11, MLSFZP(s, f10));
                ACC_S16_S32(sum20, sum21, MLSFZP(s, f00));

                s = SUB128VECTOR(vld1_u8(sptr + 3 * IW));
                ACC_S16_S32(sum20, sum21, MLSFZP(s, f10));

                ++sptr;

                s = SUB128VECTOR(vld1_u8(sptr + 0 * IW));
                ACC_S16_S32(sum00, sum01, MLSFZP(s, f01));

                s = SUB128VECTOR(vld1_u8(sptr + 1 * IW));
                ACC_S16_S32(sum00, sum01, MLSFZP(s, f11));
                sum00 = vsubq_s32(sum00, vfxszp);
                sum01 = vsubq_s32(sum01, vfxszp);
                POSTPROCESS(sum00, sum01, tptr + 0 * OW, dptr + 0 * OW);
                ACC_S16_S32(sum10, sum11, MLSFZP(s, f01));

                s = SUB128VECTOR(vld1_u8(sptr + 2 * IW));
                ACC_S16_S32(sum10, sum11, MLSFZP(s, f11));
                sum10 = vsubq_s32(sum10, vfxszp);
                sum11 = vsubq_s32(sum11, vfxszp);
                POSTPROCESS(sum10, sum11, tptr + 1 * OW, dptr + 1 * OW);
                ACC_S16_S32(sum20, sum21, MLSFZP(s, f01));

                s = SUB128VECTOR(vld1_u8(sptr + 3 * IW));
                ACC_S16_S32(sum20, sum21, MLSFZP(s, f11));
                sum20 = vsubq_s32(sum20, vfxszp);
                sum21 = vsubq_s32(sum21, vfxszp);
                POSTPROCESS(sum20, sum21, tptr + 2 * OW, dptr + 2 * OW);
            }
        } else if (oh + 2 == OH) {
            size_t ih = oh;
            size_t ow = 0;
            for (; ow < OW; ow += 8) {
                size_t iw = ow;
                int32_t* __restrict tptr = temp + oh * OW + ow;
                uint8_t* __restrict dptr = dst + oh * OW + ow;
                const uint8_t* __restrict sptr = src + ih * IW + iw;
                const int32_t* __restrict bptr = bias;
                int32x4_t sum00, sum01, sum10, sum11;

                if (!first_ic) {
                    sum00 = vld1q_s32(tptr + 0 * OW);
                    sum01 = vld1q_s32(tptr + 0 * OW + 4);
                    sum10 = vld1q_s32(tptr + 1 * OW);
                    sum11 = vld1q_s32(tptr + 1 * OW + 4);
                } else {
                    if (bias_mode == BiasMode::BROADCAST_CHANNEL_BIAS) {
                        sum00 = vdupq_n_s32(bptr[0]);
                        sum01 = sum00;
                        sum10 = sum00;
                        sum11 = sum00;
                    } else {
                        sum00 = vdupq_n_s32(0);
                        sum01 = vdupq_n_s32(0);
                        sum10 = vdupq_n_s32(0);
                        sum11 = vdupq_n_s32(0);
                    }
                    sum00 += vsrc_filter_zp;
                    sum01 += vsrc_filter_zp;
                    sum10 += vsrc_filter_zp;
                    sum11 += vsrc_filter_zp;
                }

                int8x8_t s = SUB128VECTOR(vld1_u8(sptr + 0 * IW));
                ACC_S16_S32(sum00, sum01, MLSFZP(s, f00));

                s = SUB128VECTOR(vld1_u8(sptr + 1 * IW));
                ACC_S16_S32(sum00, sum01, MLSFZP(s, f10));
                ACC_S16_S32(sum10, sum11, MLSFZP(s, f00));

                s = SUB128VECTOR(vld1_u8(sptr + 2 * IW));
                ACC_S16_S32(sum10, sum11, MLSFZP(s, f10));

                ++sptr;

                s = SUB128VECTOR(vld1_u8(sptr + 0 * IW));
                ACC_S16_S32(sum00, sum01, MLSFZP(s, f01));

                s = SUB128VECTOR(vld1_u8(sptr + 1 * IW));
                ACC_S16_S32(sum00, sum01, MLSFZP(s, f11));
                sum00 = vsubq_s32(sum00, vfxszp);
                sum01 = vsubq_s32(sum01, vfxszp);
                POSTPROCESS(sum00, sum01, tptr + 0 * OW, dptr + 0 * OW);
                ACC_S16_S32(sum10, sum11, MLSFZP(s, f01));

                s = SUB128VECTOR(vld1_u8(sptr + 2 * IW));
                ACC_S16_S32(sum10, sum11, MLSFZP(s, f11));

                sum10 = vsubq_s32(sum10, vfxszp);
                sum11 = vsubq_s32(sum11, vfxszp);
                POSTPROCESS(sum10, sum11, tptr + 1 * OW, dptr + 1 * OW);
            }
        } else if (oh + 1 == OH) {
            size_t ih = oh;
            size_t ow = 0;
            for (; ow < OW; ow += 8) {
                size_t iw = ow;
                int32_t* __restrict tptr = temp + oh * OW + ow;
                uint8_t* __restrict dptr = dst + oh * OW + ow;
                const uint8_t* __restrict sptr = src + ih * IW + iw;
                const int32_t* __restrict bptr = bias;
                int32x4_t sum00, sum01;

                if (!first_ic) {
                    sum00 = vld1q_s32(tptr + 0 * OW);
                    sum01 = vld1q_s32(tptr + 0 * OW + 4);
                } else {
                    if (bias_mode == BiasMode::BROADCAST_CHANNEL_BIAS) {
                        sum00 = vdupq_n_s32(bptr[0]);
                        sum01 = sum00;
                    } else {
                        sum00 = vdupq_n_s32(0);
                        sum01 = vdupq_n_s32(0);
                    }

                    sum00 += vsrc_filter_zp;
                    sum01 += vsrc_filter_zp;
                }

                int8x8_t s = SUB128VECTOR(vld1_u8(sptr + 0 * IW));
                ACC_S16_S32(sum00, sum01, MLSFZP(s, f00));

                s = SUB128VECTOR(vld1_u8(sptr + 1 * IW));
                ACC_S16_S32(sum00, sum01, MLSFZP(s, f10));

                ++sptr;

                s = SUB128VECTOR(vld1_u8(sptr + 0 * IW));
                ACC_S16_S32(sum00, sum01, MLSFZP(s, f01));

                s = SUB128VECTOR(vld1_u8(sptr + 1 * IW));
                ACC_S16_S32(sum00, sum01, MLSFZP(s, f11));
                sum00 = vsubq_s32(sum00, vfxszp);
                sum01 = vsubq_s32(sum01, vfxszp);
                POSTPROCESS(sum00, sum01, tptr + 0 * OW, dptr + 0 * OW);
            }
        }
    }
    MIDOUT_END();
}

template <bool first_ic, bool last_ic, BiasMode bias_mode, typename Op>
void conv_bias::conv_direct_stride1_3x3_quint8(
        const uint8_t* src, const uint8_t* filter, const int32_t* bias,
        int32_t* temp, uint8_t* dst, const size_t IH, const size_t IW,
        const size_t OH, const size_t OW, const int8_t src_zp,
        const int8_t filter_zp, const int32_t src_filter_zp, const Op& op) {
    MEGDNN_MARK_USED_VAR(IH);
    MIDOUT_BEGIN(conv_direct_stride, 0, 1) {
        int16x8_t v128 = vdupq_n_s16(128);
        int32x4_t vsrc_filter_zp = vdupq_n_s32(src_filter_zp);
        int8x8_t fzp = vdup_n_s8(filter_zp);

        int8x8_t f00 = vdup_n_s8(SUB128(filter[0]));
        int8x8_t f01 = vdup_n_s8(SUB128(filter[1]));
        int8x8_t f02 = vdup_n_s8(SUB128(filter[2]));
        int8x8_t f10 = vdup_n_s8(SUB128(filter[3]));
        int8x8_t f11 = vdup_n_s8(SUB128(filter[4]));
        int8x8_t f12 = vdup_n_s8(SUB128(filter[5]));
        int8x8_t f20 = vdup_n_s8(SUB128(filter[6]));
        int8x8_t f21 = vdup_n_s8(SUB128(filter[7]));
        int8x8_t f22 = vdup_n_s8(SUB128(filter[8]));

        int32_t fxszp = 0;
        for (size_t i = 0; i < 9; ++i)
            fxszp += static_cast<int32_t>(filter[i]) - 128;
        int32x4_t vfxszp = vdupq_n_s32(fxszp * static_cast<int32_t>(src_zp));

        // block 2x8
        size_t oh = 0;
        for (; oh + 1 < OH; oh += 2) {
            size_t ih = oh;
            for (size_t ow = 0; ow < OW; ow += 8) {
                size_t iw = ow;
                int32_t* __restrict tptr = temp + oh * OW + ow;
                uint8_t* __restrict dptr = dst + oh * OW + ow;
                const uint8_t* __restrict sptr = src + ih * IW + iw;
                const int32_t* __restrict bptr = bias;
                int32x4_t sum00, sum01, sum10, sum11;

                if (!first_ic) {
                    sum00 = vld1q_s32(tptr + 0 * OW);
                    sum01 = vld1q_s32(tptr + 0 * OW + 4);
                    sum10 = vld1q_s32(tptr + 1 * OW);
                    sum11 = vld1q_s32(tptr + 1 * OW + 4);
                } else {
                    if (bias_mode == BiasMode::BROADCAST_CHANNEL_BIAS) {
                        sum00 = vdupq_n_s32(bptr[0]);
                        sum01 = sum00;
                        sum10 = sum00;
                        sum11 = sum00;
                    } else {
                        sum00 = vdupq_n_s32(0);
                        sum01 = vdupq_n_s32(0);
                        sum10 = vdupq_n_s32(0);
                        sum11 = vdupq_n_s32(0);
                    }
                    sum00 += vsrc_filter_zp;
                    sum01 += vsrc_filter_zp;
                    sum10 += vsrc_filter_zp;
                    sum11 += vsrc_filter_zp;
                }

                int8x8_t _r00 = SUB128VECTOR(vld1_u8(sptr + 0 * IW));
                int8x8_t _r0n = SUB128VECTOR(vld1_u8(sptr + 0 * IW + 8));
                int8x8_t _r01 = vext_s8(_r00, _r0n, 1);
                int8x8_t _r02 = vext_s8(_r00, _r0n, 2);
                ACC_S16_S32(sum00, sum01, MLSFZP(_r00, f00));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r01, f01));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r02, f02));

                int8x8_t _r10 = SUB128VECTOR(vld1_u8(sptr + 1 * IW));
                int8x8_t _r1n = SUB128VECTOR(vld1_u8(sptr + 1 * IW + 8));
                int8x8_t _r11 = vext_s8(_r10, _r1n, 1);
                int8x8_t _r12 = vext_s8(_r10, _r1n, 2);
                ACC_S16_S32(sum00, sum01, MLSFZP(_r10, f10));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r11, f11));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r12, f12));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r10, f00));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r11, f01));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r12, f02));

                int8x8_t _r20 = SUB128VECTOR(vld1_u8(sptr + 2 * IW));
                int8x8_t _r2n = SUB128VECTOR(vld1_u8(sptr + 2 * IW + 8));
                int8x8_t _r21 = vext_s8(_r20, _r2n, 1);
                int8x8_t _r22 = vext_s8(_r20, _r2n, 2);
                ACC_S16_S32(sum00, sum01, MLSFZP(_r20, f20));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r21, f21));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r22, f22));
                sum00 = vsubq_s32(sum00, vfxszp);
                sum01 = vsubq_s32(sum01, vfxszp);
                POSTPROCESS(sum00, sum01, tptr + 0 * OW, dptr + 0 * OW);
                ACC_S16_S32(sum10, sum11, MLSFZP(_r20, f10));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r21, f11));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r22, f12));

                int8x8_t _r30 = SUB128VECTOR(vld1_u8(sptr + 3 * IW));
                int8x8_t _r3n = SUB128VECTOR(vld1_u8(sptr + 3 * IW + 8));
                int8x8_t _r31 = vext_s8(_r30, _r3n, 1);
                int8x8_t _r32 = vext_s8(_r30, _r3n, 2);
                ACC_S16_S32(sum10, sum11, MLSFZP(_r30, f20));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r31, f21));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r32, f22));
                sum10 = vsubq_s32(sum10, vfxszp);
                sum11 = vsubq_s32(sum11, vfxszp);
                POSTPROCESS(sum10, sum11, tptr + 1 * OW, dptr + 1 * OW);
            }
        }

        if (oh < OH) {
            size_t ih = oh;
            for (size_t ow = 0; ow < OW; ow += 8) {
                size_t iw = ow;
                int32_t* __restrict tptr = temp + oh * OW + ow;
                uint8_t* __restrict dptr = dst + oh * OW + ow;
                const uint8_t* __restrict sptr = src + ih * IW + iw;
                const int32_t* __restrict bptr = bias;
                int32x4_t sum00, sum01;

                if (!first_ic) {
                    sum00 = vld1q_s32(tptr + 0 * OW);
                    sum01 = vld1q_s32(tptr + 0 * OW + 4);
                } else {
                    if (bias_mode == BiasMode::BROADCAST_CHANNEL_BIAS) {
                        sum00 = vdupq_n_s32(bptr[0]);
                        sum01 = sum00;
                    } else {
                        sum00 = vdupq_n_s32(0);
                        sum01 = vdupq_n_s32(0);
                    }
                    sum00 += vsrc_filter_zp;
                    sum01 += vsrc_filter_zp;
                }

                int8x8_t _r00 = SUB128VECTOR(vld1_u8(sptr + 0 * IW));
                int8x8_t _r0n = SUB128VECTOR(vld1_u8(sptr + 0 * IW + 8));
                int8x8_t _r01 = vext_s8(_r00, _r0n, 1);
                int8x8_t _r02 = vext_s8(_r00, _r0n, 2);
                ACC_S16_S32(sum00, sum01, MLSFZP(_r00, f00));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r01, f01));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r02, f02));

                int8x8_t _r10 = SUB128VECTOR(vld1_u8(sptr + 1 * IW));
                int8x8_t _r1n = SUB128VECTOR(vld1_u8(sptr + 1 * IW + 8));
                int8x8_t _r11 = vext_s8(_r10, _r1n, 1);
                int8x8_t _r12 = vext_s8(_r10, _r1n, 2);
                ACC_S16_S32(sum00, sum01, MLSFZP(_r10, f10));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r11, f11));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r12, f12));

                int8x8_t _r20 = SUB128VECTOR(vld1_u8(sptr + 2 * IW));
                int8x8_t _r2n = SUB128VECTOR(vld1_u8(sptr + 2 * IW + 8));
                int8x8_t _r21 = vext_s8(_r20, _r2n, 1);
                int8x8_t _r22 = vext_s8(_r20, _r2n, 2);
                ACC_S16_S32(sum00, sum01, MLSFZP(_r20, f20));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r21, f21));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r22, f22));
                sum00 = vsubq_s32(sum00, vfxszp);
                sum01 = vsubq_s32(sum01, vfxszp);
                POSTPROCESS(sum00, sum01, tptr + 0 * OW, dptr + 0 * OW);
            }
        }
    }
    MIDOUT_END();
}

template <bool first_ic, bool last_ic, BiasMode bias_mode, typename Op>
void conv_bias::conv_direct_stride1_5x5_quint8(
        const uint8_t* src, const uint8_t* filter, const int32_t* bias,
        int32_t* temp, uint8_t* dst, const size_t IH, const size_t IW,
        const size_t OH, const size_t OW, const int8_t src_zp,
        const int8_t filter_zp, const int32_t src_filter_zp, const Op& op) {
    MEGDNN_MARK_USED_VAR(IH);
    MIDOUT_BEGIN(conv_direct_stride, 0, 2) {
        int16x8_t v128 = vdupq_n_s16(128);
        int32x4_t vsrc_filter_zp = vdupq_n_s32(src_filter_zp);
        int8x8_t fzp = vdup_n_s8(filter_zp);

        int8x8_t f00 = vdup_n_s8(SUB128(filter[0]));
        int8x8_t f01 = vdup_n_s8(SUB128(filter[1]));
        int8x8_t f02 = vdup_n_s8(SUB128(filter[2]));
        int8x8_t f03 = vdup_n_s8(SUB128(filter[3]));
        int8x8_t f04 = vdup_n_s8(SUB128(filter[4]));
        int8x8_t f10 = vdup_n_s8(SUB128(filter[5]));
        int8x8_t f11 = vdup_n_s8(SUB128(filter[6]));
        int8x8_t f12 = vdup_n_s8(SUB128(filter[7]));
        int8x8_t f13 = vdup_n_s8(SUB128(filter[8]));
        int8x8_t f14 = vdup_n_s8(SUB128(filter[9]));
        int8x8_t f20 = vdup_n_s8(SUB128(filter[10]));
        int8x8_t f21 = vdup_n_s8(SUB128(filter[11]));
        int8x8_t f22 = vdup_n_s8(SUB128(filter[12]));
        int8x8_t f23 = vdup_n_s8(SUB128(filter[13]));
        int8x8_t f24 = vdup_n_s8(SUB128(filter[14]));
        int8x8_t f30 = vdup_n_s8(SUB128(filter[15]));
        int8x8_t f31 = vdup_n_s8(SUB128(filter[16]));
        int8x8_t f32 = vdup_n_s8(SUB128(filter[17]));
        int8x8_t f33 = vdup_n_s8(SUB128(filter[18]));
        int8x8_t f34 = vdup_n_s8(SUB128(filter[19]));
        int8x8_t f40 = vdup_n_s8(SUB128(filter[20]));
        int8x8_t f41 = vdup_n_s8(SUB128(filter[21]));
        int8x8_t f42 = vdup_n_s8(SUB128(filter[22]));
        int8x8_t f43 = vdup_n_s8(SUB128(filter[23]));
        int8x8_t f44 = vdup_n_s8(SUB128(filter[24]));

        // get filter * src_zp for one IC
        int32_t fxszp = 0;
        for (size_t i = 0; i < 25; ++i)
            fxszp += static_cast<int32_t>(filter[i]) - 128;
        int32x4_t vfxszp = vdupq_n_s32(fxszp * static_cast<int32_t>(src_zp));

        // block 2x8
        size_t oh = 0;
        for (; oh + 1 < OH; oh += 2) {
            size_t ih = oh;
            for (size_t ow = 0; ow < OW; ow += 8) {
                size_t iw = ow;
                int32_t* __restrict tptr = temp + oh * OW + ow;
                uint8_t* __restrict dptr = dst + oh * OW + ow;
                const uint8_t* __restrict sptr = src + ih * IW + iw;
                const int32_t* __restrict bptr = bias;
                int32x4_t sum00, sum01, sum10, sum11;

                if (!first_ic) {
                    sum00 = vld1q_s32(tptr + 0 * OW);
                    sum01 = vld1q_s32(tptr + 0 * OW + 4);
                    sum10 = vld1q_s32(tptr + 1 * OW);
                    sum11 = vld1q_s32(tptr + 1 * OW + 4);
                } else {
                    if (bias_mode == BiasMode::BROADCAST_CHANNEL_BIAS) {
                        sum00 = vdupq_n_s32(bptr[0]);
                        sum01 = sum00;
                        sum10 = sum00;
                        sum11 = sum00;
                    } else {
                        sum00 = vdupq_n_s32(0);
                        sum01 = vdupq_n_s32(0);
                        sum10 = vdupq_n_s32(0);
                        sum11 = vdupq_n_s32(0);
                    }
                    sum00 += vsrc_filter_zp;
                    sum01 += vsrc_filter_zp;
                    sum10 += vsrc_filter_zp;
                    sum11 += vsrc_filter_zp;
                }

                int8x8_t _r00 = SUB128VECTOR(vld1_u8(sptr + 0 * IW));
                int8x8_t _r0n = SUB128VECTOR(vld1_u8(sptr + 0 * IW + 8));
                int8x8_t _r01 = vext_s8(_r00, _r0n, 1);
                int8x8_t _r02 = vext_s8(_r00, _r0n, 2);
                int8x8_t _r03 = vext_s8(_r00, _r0n, 3);
                int8x8_t _r04 = vext_s8(_r00, _r0n, 4);
                ACC_S16_S32(sum00, sum01, MLSFZP(_r00, f00));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r01, f01));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r02, f02));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r03, f03));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r04, f04));

                int8x8_t _r10 = SUB128VECTOR(vld1_u8(sptr + 1 * IW));
                int8x8_t _r1n = SUB128VECTOR(vld1_u8(sptr + 1 * IW + 8));
                int8x8_t _r11 = vext_s8(_r10, _r1n, 1);
                int8x8_t _r12 = vext_s8(_r10, _r1n, 2);
                int8x8_t _r13 = vext_s8(_r10, _r1n, 3);
                int8x8_t _r14 = vext_s8(_r10, _r1n, 4);
                ACC_S16_S32(sum00, sum01, MLSFZP(_r10, f10));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r11, f11));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r12, f12));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r13, f13));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r14, f14));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r10, f00));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r11, f01));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r12, f02));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r13, f03));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r14, f04))

                int8x8_t _r20 = SUB128VECTOR(vld1_u8(sptr + 2 * IW));
                int8x8_t _r2n = SUB128VECTOR(vld1_u8(sptr + 2 * IW + 8));
                int8x8_t _r21 = vext_s8(_r20, _r2n, 1);
                int8x8_t _r22 = vext_s8(_r20, _r2n, 2);
                int8x8_t _r23 = vext_s8(_r20, _r2n, 3);
                int8x8_t _r24 = vext_s8(_r20, _r2n, 4);
                ACC_S16_S32(sum00, sum01, MLSFZP(_r20, f20));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r21, f21));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r22, f22));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r23, f23));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r24, f24));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r20, f10));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r21, f11));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r22, f12));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r23, f13));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r24, f14))

                int8x8_t _r30 = SUB128VECTOR(vld1_u8(sptr + 3 * IW));
                int8x8_t _r3n = SUB128VECTOR(vld1_u8(sptr + 3 * IW + 8));
                int8x8_t _r31 = vext_s8(_r30, _r3n, 1);
                int8x8_t _r32 = vext_s8(_r30, _r3n, 2);
                int8x8_t _r33 = vext_s8(_r30, _r3n, 3);
                int8x8_t _r34 = vext_s8(_r30, _r3n, 4);
                ACC_S16_S32(sum00, sum01, MLSFZP(_r30, f30));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r31, f31));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r32, f32));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r33, f33));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r34, f34));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r30, f20));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r31, f21));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r32, f22));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r33, f23));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r34, f24));

                int8x8_t _r40 = SUB128VECTOR(vld1_u8(sptr + 4 * IW));
                int8x8_t _r4n = SUB128VECTOR(vld1_u8(sptr + 4 * IW + 8));
                int8x8_t _r41 = vext_s8(_r40, _r4n, 1);
                int8x8_t _r42 = vext_s8(_r40, _r4n, 2);
                int8x8_t _r43 = vext_s8(_r40, _r4n, 3);
                int8x8_t _r44 = vext_s8(_r40, _r4n, 4);
                ACC_S16_S32(sum00, sum01, MLSFZP(_r40, f40));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r41, f41));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r42, f42));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r43, f43));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r44, f44));
                sum00 = vsubq_s32(sum00, vfxszp);
                sum01 = vsubq_s32(sum01, vfxszp);
                POSTPROCESS(sum00, sum01, tptr + 0 * OW, dptr + 0 * OW);
                ACC_S16_S32(sum10, sum11, MLSFZP(_r40, f30));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r41, f31));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r42, f32));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r43, f33));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r44, f34));

                int8x8_t _r50 = SUB128VECTOR(vld1_u8(sptr + 5 * IW));
                int8x8_t _r5n = SUB128VECTOR(vld1_u8(sptr + 5 * IW + 8));
                int8x8_t _r51 = vext_s8(_r50, _r5n, 1);
                int8x8_t _r52 = vext_s8(_r50, _r5n, 2);
                int8x8_t _r53 = vext_s8(_r50, _r5n, 3);
                int8x8_t _r54 = vext_s8(_r50, _r5n, 4);
                ACC_S16_S32(sum10, sum11, MLSFZP(_r50, f40));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r51, f41));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r52, f42));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r53, f43));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r54, f44));
                sum10 = vsubq_s32(sum10, vfxszp);
                sum11 = vsubq_s32(sum11, vfxszp);
                POSTPROCESS(sum10, sum11, tptr + 1 * OW, dptr + 1 * OW);
            }
        }

        if (oh < OH) {
            size_t ih = oh;
            for (size_t ow = 0; ow < OW; ow += 8) {
                size_t iw = ow;
                int32_t* __restrict tptr = temp + oh * OW + ow;
                uint8_t* __restrict dptr = dst + oh * OW + ow;
                const uint8_t* __restrict sptr = src + ih * IW + iw;
                const int32_t* __restrict bptr = bias;
                int32x4_t sum00, sum01;

                if (!first_ic) {
                    sum00 = vld1q_s32(tptr + 0 * OW);
                    sum01 = vld1q_s32(tptr + 0 * OW + 4);
                } else {
                    if (bias_mode == BiasMode::BROADCAST_CHANNEL_BIAS) {
                        sum00 = vdupq_n_s32(bptr[0]);
                        sum01 = sum00;
                    } else {
                        sum00 = vdupq_n_s32(0);
                        sum01 = vdupq_n_s32(0);
                    }
                    sum00 += vsrc_filter_zp;
                    sum01 += vsrc_filter_zp;
                }

                int8x8_t _r00 = SUB128VECTOR(vld1_u8(sptr + 0 * IW));
                int8x8_t _r0n = SUB128VECTOR(vld1_u8(sptr + 0 * IW + 8));
                int8x8_t _r01 = vext_s8(_r00, _r0n, 1);
                int8x8_t _r02 = vext_s8(_r00, _r0n, 2);
                int8x8_t _r03 = vext_s8(_r00, _r0n, 3);
                int8x8_t _r04 = vext_s8(_r00, _r0n, 4);
                ACC_S16_S32(sum00, sum01, MLSFZP(_r00, f00));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r01, f01));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r02, f02));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r03, f03));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r04, f04));

                int8x8_t _r10 = SUB128VECTOR(vld1_u8(sptr + 1 * IW));
                int8x8_t _r1n = SUB128VECTOR(vld1_u8(sptr + 1 * IW + 8));
                int8x8_t _r11 = vext_s8(_r10, _r1n, 1);
                int8x8_t _r12 = vext_s8(_r10, _r1n, 2);
                int8x8_t _r13 = vext_s8(_r10, _r1n, 3);
                int8x8_t _r14 = vext_s8(_r10, _r1n, 4);
                ACC_S16_S32(sum00, sum01, MLSFZP(_r10, f10));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r11, f11));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r12, f12));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r13, f13));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r14, f14));

                int8x8_t _r20 = SUB128VECTOR(vld1_u8(sptr + 2 * IW));
                int8x8_t _r2n = SUB128VECTOR(vld1_u8(sptr + 2 * IW + 8));
                int8x8_t _r21 = vext_s8(_r20, _r2n, 1);
                int8x8_t _r22 = vext_s8(_r20, _r2n, 2);
                int8x8_t _r23 = vext_s8(_r20, _r2n, 3);
                int8x8_t _r24 = vext_s8(_r20, _r2n, 4);
                ACC_S16_S32(sum00, sum01, MLSFZP(_r20, f20));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r21, f21));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r22, f22));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r23, f23));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r24, f24));

                int8x8_t _r30 = SUB128VECTOR(vld1_u8(sptr + 3 * IW));
                int8x8_t _r3n = SUB128VECTOR(vld1_u8(sptr + 3 * IW + 8));
                int8x8_t _r31 = vext_s8(_r30, _r3n, 1);
                int8x8_t _r32 = vext_s8(_r30, _r3n, 2);
                int8x8_t _r33 = vext_s8(_r30, _r3n, 3);
                int8x8_t _r34 = vext_s8(_r30, _r3n, 4);
                ACC_S16_S32(sum00, sum01, MLSFZP(_r30, f30));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r31, f31));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r32, f32));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r33, f33));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r34, f34));

                int8x8_t _r40 = SUB128VECTOR(vld1_u8(sptr + 4 * IW));
                int8x8_t _r4n = SUB128VECTOR(vld1_u8(sptr + 4 * IW + 8));
                int8x8_t _r41 = vext_s8(_r40, _r4n, 1);
                int8x8_t _r42 = vext_s8(_r40, _r4n, 2);
                int8x8_t _r43 = vext_s8(_r40, _r4n, 3);
                int8x8_t _r44 = vext_s8(_r40, _r4n, 4);
                ACC_S16_S32(sum00, sum01, MLSFZP(_r40, f40));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r41, f41));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r42, f42));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r43, f43));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r44, f44));
                sum00 = vsubq_s32(sum00, vfxszp);
                sum01 = vsubq_s32(sum01, vfxszp);
                POSTPROCESS(sum00, sum01, tptr + 0 * OW, dptr + 0 * OW);
            }
        }
    }
    MIDOUT_END();
}

template <bool first_ic, bool last_ic, BiasMode bias_mode, typename Op>
void conv_bias::conv_direct_stride1_7x7_quint8(
        const uint8_t* src, const uint8_t* filter, const int32_t* bias,
        int32_t* temp, uint8_t* dst, const size_t IH, const size_t IW,
        const size_t OH, const size_t OW, const int8_t src_zp,
        const int8_t filter_zp, const int32_t src_filter_zp, const Op& op) {
    MEGDNN_MARK_USED_VAR(IH);
    MIDOUT_BEGIN(conv_direct_stride, 0, 3) {
        int16x8_t v128 = vdupq_n_s16(128);
        int32x4_t vsrc_filter_zp = vdupq_n_s32(src_filter_zp);
        int8x8_t fzp = vdup_n_s8(filter_zp);

        int8x8_t f00 = vdup_n_s8(SUB128(filter[0]));
        int8x8_t f01 = vdup_n_s8(SUB128(filter[1]));
        int8x8_t f02 = vdup_n_s8(SUB128(filter[2]));
        int8x8_t f03 = vdup_n_s8(SUB128(filter[3]));
        int8x8_t f04 = vdup_n_s8(SUB128(filter[4]));
        int8x8_t f05 = vdup_n_s8(SUB128(filter[5]));
        int8x8_t f06 = vdup_n_s8(SUB128(filter[6]));

        int8x8_t f10 = vdup_n_s8(SUB128(filter[7]));
        int8x8_t f11 = vdup_n_s8(SUB128(filter[8]));
        int8x8_t f12 = vdup_n_s8(SUB128(filter[9]));
        int8x8_t f13 = vdup_n_s8(SUB128(filter[10]));
        int8x8_t f14 = vdup_n_s8(SUB128(filter[11]));
        int8x8_t f15 = vdup_n_s8(SUB128(filter[12]));
        int8x8_t f16 = vdup_n_s8(SUB128(filter[13]));

        int8x8_t f20 = vdup_n_s8(SUB128(filter[14]));
        int8x8_t f21 = vdup_n_s8(SUB128(filter[15]));
        int8x8_t f22 = vdup_n_s8(SUB128(filter[16]));
        int8x8_t f23 = vdup_n_s8(SUB128(filter[17]));
        int8x8_t f24 = vdup_n_s8(SUB128(filter[18]));
        int8x8_t f25 = vdup_n_s8(SUB128(filter[19]));
        int8x8_t f26 = vdup_n_s8(SUB128(filter[20]));

        int8x8_t f30 = vdup_n_s8(SUB128(filter[21]));
        int8x8_t f31 = vdup_n_s8(SUB128(filter[22]));
        int8x8_t f32 = vdup_n_s8(SUB128(filter[23]));
        int8x8_t f33 = vdup_n_s8(SUB128(filter[24]));
        int8x8_t f34 = vdup_n_s8(SUB128(filter[25]));
        int8x8_t f35 = vdup_n_s8(SUB128(filter[26]));
        int8x8_t f36 = vdup_n_s8(SUB128(filter[27]));

        int8x8_t f40 = vdup_n_s8(SUB128(filter[28]));
        int8x8_t f41 = vdup_n_s8(SUB128(filter[29]));
        int8x8_t f42 = vdup_n_s8(SUB128(filter[30]));
        int8x8_t f43 = vdup_n_s8(SUB128(filter[31]));
        int8x8_t f44 = vdup_n_s8(SUB128(filter[32]));
        int8x8_t f45 = vdup_n_s8(SUB128(filter[33]));
        int8x8_t f46 = vdup_n_s8(SUB128(filter[34]));

        int8x8_t f50 = vdup_n_s8(SUB128(filter[35]));
        int8x8_t f51 = vdup_n_s8(SUB128(filter[36]));
        int8x8_t f52 = vdup_n_s8(SUB128(filter[37]));
        int8x8_t f53 = vdup_n_s8(SUB128(filter[38]));
        int8x8_t f54 = vdup_n_s8(SUB128(filter[39]));
        int8x8_t f55 = vdup_n_s8(SUB128(filter[40]));
        int8x8_t f56 = vdup_n_s8(SUB128(filter[41]));

        int8x8_t f60 = vdup_n_s8(SUB128(filter[42]));
        int8x8_t f61 = vdup_n_s8(SUB128(filter[43]));
        int8x8_t f62 = vdup_n_s8(SUB128(filter[44]));
        int8x8_t f63 = vdup_n_s8(SUB128(filter[45]));
        int8x8_t f64 = vdup_n_s8(SUB128(filter[46]));
        int8x8_t f65 = vdup_n_s8(SUB128(filter[47]));
        int8x8_t f66 = vdup_n_s8(SUB128(filter[48]));

        // get filter * src_zp for one IC
        int32_t fxszp = 0;
        for (size_t i = 0; i < 49; ++i)
            fxszp += static_cast<int32_t>(filter[i]) - 128;
        int32x4_t vfxszp = vdupq_n_s32(fxszp * static_cast<int32_t>(src_zp));

        // block 2x8
        size_t oh = 0;
        for (; oh + 1 < OH; oh += 2) {
            size_t ih = oh;
            for (size_t ow = 0; ow < OW; ow += 8) {
                size_t iw = ow;
                int32_t* __restrict tptr = temp + oh * OW + ow;
                uint8_t* __restrict dptr = dst + oh * OW + ow;
                const uint8_t* __restrict sptr = src + ih * IW + iw;
                const int32_t* __restrict bptr = bias;
                int32x4_t sum00, sum01, sum10, sum11;

                if (!first_ic) {
                    sum00 = vld1q_s32(tptr + 0 * OW);
                    sum01 = vld1q_s32(tptr + 0 * OW + 4);
                    sum10 = vld1q_s32(tptr + 1 * OW);
                    sum11 = vld1q_s32(tptr + 1 * OW + 4);
                } else {
                    if (bias_mode == BiasMode::BROADCAST_CHANNEL_BIAS) {
                        sum00 = vdupq_n_s32(bptr[0]);
                        sum01 = sum00;
                        sum10 = sum00;
                        sum11 = sum00;
                    } else {
                        sum00 = vdupq_n_s32(0);
                        sum01 = vdupq_n_s32(0);
                        sum10 = vdupq_n_s32(0);
                        sum11 = vdupq_n_s32(0);
                    }
                    sum00 += vsrc_filter_zp;
                    sum01 += vsrc_filter_zp;
                    sum10 += vsrc_filter_zp;
                    sum11 += vsrc_filter_zp;
                }

                int8x8_t _r00 = SUB128VECTOR(vld1_u8(sptr + 0 * IW));
                int8x8_t _r0n = SUB128VECTOR(vld1_u8(sptr + 0 * IW + 8));
                int8x8_t _r01 = vext_s8(_r00, _r0n, 1);
                int8x8_t _r02 = vext_s8(_r00, _r0n, 2);
                int8x8_t _r03 = vext_s8(_r00, _r0n, 3);
                int8x8_t _r04 = vext_s8(_r00, _r0n, 4);
                int8x8_t _r05 = vext_s8(_r00, _r0n, 5);
                int8x8_t _r06 = vext_s8(_r00, _r0n, 6);
                ACC_S16_S32(sum00, sum01, MLSFZP(_r00, f00));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r01, f01));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r02, f02));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r03, f03));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r04, f04));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r05, f05));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r06, f06));

                int8x8_t _r10 = SUB128VECTOR(vld1_u8(sptr + 1 * IW));
                int8x8_t _r1n = SUB128VECTOR(vld1_u8(sptr + 1 * IW + 8));
                int8x8_t _r11 = vext_s8(_r10, _r1n, 1);
                int8x8_t _r12 = vext_s8(_r10, _r1n, 2);
                int8x8_t _r13 = vext_s8(_r10, _r1n, 3);
                int8x8_t _r14 = vext_s8(_r10, _r1n, 4);
                int8x8_t _r15 = vext_s8(_r10, _r1n, 5);
                int8x8_t _r16 = vext_s8(_r10, _r1n, 6);
                ACC_S16_S32(sum00, sum01, MLSFZP(_r10, f10));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r11, f11));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r12, f12));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r13, f13));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r14, f14));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r15, f15));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r16, f16));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r10, f00));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r11, f01));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r12, f02));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r13, f03));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r14, f04));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r15, f05));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r16, f06));

                int8x8_t _r20 = SUB128VECTOR(vld1_u8(sptr + 2 * IW));
                int8x8_t _r2n = SUB128VECTOR(vld1_u8(sptr + 2 * IW + 8));
                int8x8_t _r21 = vext_s8(_r20, _r2n, 1);
                int8x8_t _r22 = vext_s8(_r20, _r2n, 2);
                int8x8_t _r23 = vext_s8(_r20, _r2n, 3);
                int8x8_t _r24 = vext_s8(_r20, _r2n, 4);
                int8x8_t _r25 = vext_s8(_r20, _r2n, 5);
                int8x8_t _r26 = vext_s8(_r20, _r2n, 6);
                ACC_S16_S32(sum00, sum01, MLSFZP(_r20, f20));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r21, f21));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r22, f22));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r23, f23));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r24, f24));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r25, f25));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r26, f26));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r20, f10));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r21, f11));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r22, f12));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r23, f13));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r24, f14));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r25, f15));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r26, f16));

                int8x8_t _r30 = SUB128VECTOR(vld1_u8(sptr + 3 * IW));
                int8x8_t _r3n = SUB128VECTOR(vld1_u8(sptr + 3 * IW + 8));
                int8x8_t _r31 = vext_s8(_r30, _r3n, 1);
                int8x8_t _r32 = vext_s8(_r30, _r3n, 2);
                int8x8_t _r33 = vext_s8(_r30, _r3n, 3);
                int8x8_t _r34 = vext_s8(_r30, _r3n, 4);
                int8x8_t _r35 = vext_s8(_r30, _r3n, 5);
                int8x8_t _r36 = vext_s8(_r30, _r3n, 6);
                ACC_S16_S32(sum00, sum01, MLSFZP(_r30, f30));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r31, f31));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r32, f32));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r33, f33));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r34, f34));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r35, f35));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r36, f36));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r30, f20));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r31, f21));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r32, f22));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r33, f23));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r34, f24));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r35, f25));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r36, f26));

                int8x8_t _r40 = SUB128VECTOR(vld1_u8(sptr + 4 * IW));
                int8x8_t _r4n = SUB128VECTOR(vld1_u8(sptr + 4 * IW + 8));
                int8x8_t _r41 = vext_s8(_r40, _r4n, 1);
                int8x8_t _r42 = vext_s8(_r40, _r4n, 2);
                int8x8_t _r43 = vext_s8(_r40, _r4n, 3);
                int8x8_t _r44 = vext_s8(_r40, _r4n, 4);
                int8x8_t _r45 = vext_s8(_r40, _r4n, 5);
                int8x8_t _r46 = vext_s8(_r40, _r4n, 6);
                ACC_S16_S32(sum00, sum01, MLSFZP(_r40, f40));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r41, f41));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r42, f42));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r43, f43));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r44, f44));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r45, f45));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r46, f46));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r40, f30));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r41, f31));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r42, f32));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r43, f33));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r44, f34));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r45, f35));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r46, f36));

                int8x8_t _r50 = SUB128VECTOR(vld1_u8(sptr + 5 * IW));
                int8x8_t _r5n = SUB128VECTOR(vld1_u8(sptr + 5 * IW + 8));
                int8x8_t _r51 = vext_s8(_r50, _r5n, 1);
                int8x8_t _r52 = vext_s8(_r50, _r5n, 2);
                int8x8_t _r53 = vext_s8(_r50, _r5n, 3);
                int8x8_t _r54 = vext_s8(_r50, _r5n, 4);
                int8x8_t _r55 = vext_s8(_r50, _r5n, 5);
                int8x8_t _r56 = vext_s8(_r50, _r5n, 6);
                ACC_S16_S32(sum00, sum01, MLSFZP(_r50, f50));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r51, f51));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r52, f52));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r53, f53));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r54, f54));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r55, f55));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r56, f56));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r50, f40));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r51, f41));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r52, f42));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r53, f43));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r54, f44));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r55, f45));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r56, f46));

                int8x8_t _r60 = SUB128VECTOR(vld1_u8(sptr + 6 * IW));
                int8x8_t _r6n = SUB128VECTOR(vld1_u8(sptr + 6 * IW + 8));
                int8x8_t _r61 = vext_s8(_r60, _r6n, 1);
                int8x8_t _r62 = vext_s8(_r60, _r6n, 2);
                int8x8_t _r63 = vext_s8(_r60, _r6n, 3);
                int8x8_t _r64 = vext_s8(_r60, _r6n, 4);
                int8x8_t _r65 = vext_s8(_r60, _r6n, 5);
                int8x8_t _r66 = vext_s8(_r60, _r6n, 6);
                ACC_S16_S32(sum00, sum01, MLSFZP(_r60, f60));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r61, f61));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r62, f62));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r63, f63));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r64, f64));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r65, f65));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r66, f66));
                sum00 = vsubq_s32(sum00, vfxszp);
                sum01 = vsubq_s32(sum01, vfxszp);
                POSTPROCESS(sum00, sum01, tptr + 0 * OW, dptr + 0 * OW);
                ACC_S16_S32(sum10, sum11, MLSFZP(_r60, f50));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r61, f51));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r62, f52));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r63, f53));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r64, f54));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r65, f55));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r66, f56));

                int8x8_t _r70 = SUB128VECTOR(vld1_u8(sptr + 7 * IW));
                int8x8_t _r7n = SUB128VECTOR(vld1_u8(sptr + 7 * IW + 8));
                int8x8_t _r71 = vext_s8(_r70, _r7n, 1);
                int8x8_t _r72 = vext_s8(_r70, _r7n, 2);
                int8x8_t _r73 = vext_s8(_r70, _r7n, 3);
                int8x8_t _r74 = vext_s8(_r70, _r7n, 4);
                int8x8_t _r75 = vext_s8(_r70, _r7n, 5);
                int8x8_t _r76 = vext_s8(_r70, _r7n, 6);
                ACC_S16_S32(sum10, sum11, MLSFZP(_r70, f60));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r71, f61));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r72, f62));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r73, f63));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r74, f64));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r75, f65));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r76, f66));
                sum10 = vsubq_s32(sum10, vfxszp);
                sum11 = vsubq_s32(sum11, vfxszp);
                POSTPROCESS(sum10, sum11, tptr + 1 * OW, dptr + 1 * OW);
            }
        }

        if (oh < OH) {
            size_t ih = oh;
            for (size_t ow = 0; ow < OW; ow += 8) {
                size_t iw = ow;
                int32_t* __restrict tptr = temp + oh * OW + ow;
                uint8_t* __restrict dptr = dst + oh * OW + ow;
                const uint8_t* __restrict sptr = src + ih * IW + iw;
                const int32_t* __restrict bptr = bias;
                int32x4_t sum00, sum01;

                if (!first_ic) {
                    sum00 = vld1q_s32(tptr + 0 * OW);
                    sum01 = vld1q_s32(tptr + 0 * OW + 4);
                } else {
                    if (bias_mode == BiasMode::BROADCAST_CHANNEL_BIAS) {
                        sum00 = vdupq_n_s32(bptr[0]);
                        sum01 = sum00;
                    } else {
                        sum00 = vdupq_n_s32(0);
                        sum01 = vdupq_n_s32(0);
                    }
                    sum00 += vsrc_filter_zp;
                    sum01 += vsrc_filter_zp;
                }

                int8x8_t _r00 = SUB128VECTOR(vld1_u8(sptr + 0 * IW));
                int8x8_t _r0n = SUB128VECTOR(vld1_u8(sptr + 0 * IW + 8));
                int8x8_t _r01 = vext_s8(_r00, _r0n, 1);
                int8x8_t _r02 = vext_s8(_r00, _r0n, 2);
                int8x8_t _r03 = vext_s8(_r00, _r0n, 3);
                int8x8_t _r04 = vext_s8(_r00, _r0n, 4);
                int8x8_t _r05 = vext_s8(_r00, _r0n, 5);
                int8x8_t _r06 = vext_s8(_r00, _r0n, 6);
                ACC_S16_S32(sum00, sum01, MLSFZP(_r00, f00));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r01, f01));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r02, f02));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r03, f03));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r04, f04));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r05, f05));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r06, f06));

                int8x8_t _r10 = SUB128VECTOR(vld1_u8(sptr + 1 * IW));
                int8x8_t _r1n = SUB128VECTOR(vld1_u8(sptr + 1 * IW + 8));
                int8x8_t _r11 = vext_s8(_r10, _r1n, 1);
                int8x8_t _r12 = vext_s8(_r10, _r1n, 2);
                int8x8_t _r13 = vext_s8(_r10, _r1n, 3);
                int8x8_t _r14 = vext_s8(_r10, _r1n, 4);
                int8x8_t _r15 = vext_s8(_r10, _r1n, 5);
                int8x8_t _r16 = vext_s8(_r10, _r1n, 6);
                ACC_S16_S32(sum00, sum01, MLSFZP(_r10, f10));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r11, f11));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r12, f12));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r13, f13));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r14, f14));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r15, f15));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r16, f16));

                int8x8_t _r20 = SUB128VECTOR(vld1_u8(sptr + 2 * IW));
                int8x8_t _r2n = SUB128VECTOR(vld1_u8(sptr + 2 * IW + 8));
                int8x8_t _r21 = vext_s8(_r20, _r2n, 1);
                int8x8_t _r22 = vext_s8(_r20, _r2n, 2);
                int8x8_t _r23 = vext_s8(_r20, _r2n, 3);
                int8x8_t _r24 = vext_s8(_r20, _r2n, 4);
                int8x8_t _r25 = vext_s8(_r20, _r2n, 5);
                int8x8_t _r26 = vext_s8(_r20, _r2n, 6);
                ACC_S16_S32(sum00, sum01, MLSFZP(_r20, f20));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r21, f21));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r22, f22));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r23, f23));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r24, f24));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r25, f25));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r26, f26));

                int8x8_t _r30 = SUB128VECTOR(vld1_u8(sptr + 3 * IW));
                int8x8_t _r3n = SUB128VECTOR(vld1_u8(sptr + 3 * IW + 8));
                int8x8_t _r31 = vext_s8(_r30, _r3n, 1);
                int8x8_t _r32 = vext_s8(_r30, _r3n, 2);
                int8x8_t _r33 = vext_s8(_r30, _r3n, 3);
                int8x8_t _r34 = vext_s8(_r30, _r3n, 4);
                int8x8_t _r35 = vext_s8(_r30, _r3n, 5);
                int8x8_t _r36 = vext_s8(_r30, _r3n, 6);
                ACC_S16_S32(sum00, sum01, MLSFZP(_r30, f30));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r31, f31));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r32, f32));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r33, f33));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r34, f34));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r35, f35));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r36, f36));

                int8x8_t _r40 = SUB128VECTOR(vld1_u8(sptr + 4 * IW));
                int8x8_t _r4n = SUB128VECTOR(vld1_u8(sptr + 4 * IW + 8));
                int8x8_t _r41 = vext_s8(_r40, _r4n, 1);
                int8x8_t _r42 = vext_s8(_r40, _r4n, 2);
                int8x8_t _r43 = vext_s8(_r40, _r4n, 3);
                int8x8_t _r44 = vext_s8(_r40, _r4n, 4);
                int8x8_t _r45 = vext_s8(_r40, _r4n, 5);
                int8x8_t _r46 = vext_s8(_r40, _r4n, 6);
                ACC_S16_S32(sum00, sum01, MLSFZP(_r40, f40));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r41, f41));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r42, f42));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r43, f43));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r44, f44));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r45, f45));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r46, f46));

                int8x8_t _r50 = SUB128VECTOR(vld1_u8(sptr + 5 * IW));
                int8x8_t _r5n = SUB128VECTOR(vld1_u8(sptr + 5 * IW + 8));
                int8x8_t _r51 = vext_s8(_r50, _r5n, 1);
                int8x8_t _r52 = vext_s8(_r50, _r5n, 2);
                int8x8_t _r53 = vext_s8(_r50, _r5n, 3);
                int8x8_t _r54 = vext_s8(_r50, _r5n, 4);
                int8x8_t _r55 = vext_s8(_r50, _r5n, 5);
                int8x8_t _r56 = vext_s8(_r50, _r5n, 6);
                ACC_S16_S32(sum00, sum01, MLSFZP(_r50, f50));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r51, f51));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r52, f52));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r53, f53));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r54, f54));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r55, f55));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r56, f56));

                int8x8_t _r60 = SUB128VECTOR(vld1_u8(sptr + 6 * IW));
                int8x8_t _r6n = SUB128VECTOR(vld1_u8(sptr + 6 * IW + 8));
                int8x8_t _r61 = vext_s8(_r60, _r6n, 1);
                int8x8_t _r62 = vext_s8(_r60, _r6n, 2);
                int8x8_t _r63 = vext_s8(_r60, _r6n, 3);
                int8x8_t _r64 = vext_s8(_r60, _r6n, 4);
                int8x8_t _r65 = vext_s8(_r60, _r6n, 5);
                int8x8_t _r66 = vext_s8(_r60, _r6n, 6);
                ACC_S16_S32(sum00, sum01, MLSFZP(_r60, f60));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r61, f61));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r62, f62));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r63, f63));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r64, f64));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r65, f65));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r66, f66));

                sum00 = vsubq_s32(sum00, vfxszp);
                sum01 = vsubq_s32(sum01, vfxszp);
                POSTPROCESS(sum00, sum01, tptr + 0 * OW, dptr + 0 * OW);
            }
        }
    }
    MIDOUT_END();
}

template <bool first_ic, bool last_ic, BiasMode bias_mode, typename Op>
void conv_bias::conv_direct_stride2_2x2_quint8(
        const uint8_t* src, const uint8_t* filter, const int32_t* bias,
        int32_t* temp, uint8_t* dst, const size_t IH, const size_t IW,
        const size_t OH, const size_t OW, const int8_t src_zp,
        const int8_t filter_zp, const int32_t src_filter_zp, const Op& op) {
        MEGDNN_MARK_USED_VAR(IH);
#define GET_R2(sptr)                                                      \
    _r00 = SUB128VECTOR(vld1_u8(sptr));                                   \
    _r00 = vtbl1_s8(_r00, _idx);                                          \
    _r01 = SUB128VECTOR(vld1_u8(sptr + 8));                               \
    _r01 = vtbl1_s8(_r01, _idx);                                          \
    _rn = vzip_s32(vreinterpret_s32_s8(_r00), vreinterpret_s32_s8(_r01)); \
    _r00 = vreinterpret_s8_s32(_rn.val[0]);                               \
    _r01 = vreinterpret_s8_s32(_rn.val[1]);

    MIDOUT_BEGIN(conv_direct_stride, 0, 4) {
        int16x8_t v128 = vdupq_n_s16(128);
        int32x4_t vsrc_filter_zp = vdupq_n_s32(src_filter_zp);
        int8x8_t fzp = vdup_n_s8(filter_zp);

        int8x8_t f00 = vdup_n_s8(SUB128(filter[0]));
        int8x8_t f01 = vdup_n_s8(SUB128(filter[1]));
        int8x8_t f10 = vdup_n_s8(SUB128(filter[2]));
        int8x8_t f11 = vdup_n_s8(SUB128(filter[3]));

        // get filter * src_zp for one IC
        int32_t fxszp = 0;
        for (size_t i = 0; i < 4; ++i)
            fxszp += static_cast<int32_t>(filter[i]) - 128;
        int32x4_t vfxszp = vdupq_n_s32(fxszp * static_cast<int32_t>(src_zp));

        int8x8_t _idx = {0, 2, 4, 6, 1, 3, 5, 7};
        size_t oh = 0;
        for (; oh < OH; ++oh) {
            size_t ih = oh * 2;
            size_t ow = 0;
            for (; ow < OW; ow += 8) {
                size_t iw = ow * 2;
                int32_t* __restrict tptr = temp + oh * OW + ow;
                uint8_t* __restrict dptr = dst + oh * OW + ow;
                const uint8_t* __restrict sptr = src + ih * IW + iw;
                const int32_t* __restrict bptr = bias;
                int32x4_t sum00, sum01;
                int32x2x2_t _rn;
                int8x8_t _r00, _r01;

                if (!first_ic) {
                    sum00 = vld1q_s32(tptr + 0 * OW);
                    sum01 = vld1q_s32(tptr + 0 * OW + 4);
                } else {
                    if (bias_mode == BiasMode::BROADCAST_CHANNEL_BIAS) {
                        sum00 = vdupq_n_s32(bptr[0]);
                        sum01 = sum00;
                    } else {
                        sum00 = vdupq_n_s32(0);
                        sum01 = vdupq_n_s32(0);
                    }
                    sum00 += vsrc_filter_zp;
                    sum01 += vsrc_filter_zp;
                }

                GET_R2(sptr);
                ACC_S16_S32(sum00, sum01, MLSFZP(_r00, f00));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r01, f01));

                GET_R2(sptr + IW);
                ACC_S16_S32(sum00, sum01, MLSFZP(_r00, f10));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r01, f11));

                sum00 = vsubq_s32(sum00, vfxszp);
                sum01 = vsubq_s32(sum01, vfxszp);
                POSTPROCESS(sum00, sum01, tptr + 0 * OW, dptr + 0 * OW);
            }
        }
    }
    MIDOUT_END();

#undef GET_R2
}

template <bool first_ic, bool last_ic, BiasMode bias_mode, typename Op>
void conv_bias::conv_direct_stride2_3x3_quint8(
        const uint8_t* src, const uint8_t* filter, const int32_t* bias,
        int32_t* temp, uint8_t* dst, const size_t IH, const size_t IW,
        const size_t OH, const size_t OW, const int8_t src_zp,
        const int8_t filter_zp, const int32_t src_filter_zp, const Op& op) {
    MEGDNN_MARK_USED_VAR(IH);
#define GET_R3(sptr)                                                      \
    _r00 = SUB128VECTOR(vld1_u8(sptr));                                   \
    _r00 = vtbl1_s8(_r00, _idx);                                          \
    _r01 = SUB128VECTOR(vld1_u8(sptr + 8));                               \
    _r01 = vtbl1_s8(_r01, _idx);                                          \
    _rn = vzip_s32(vreinterpret_s32_s8(_r00), vreinterpret_s32_s8(_r01)); \
    _r00 = vreinterpret_s8_s32(_rn.val[0]);                               \
    _r01 = vreinterpret_s8_s32(_rn.val[1]);                               \
    _r02 = SUB128VECTOR(vld1_u8(sptr + 16));                              \
    _r02 = vext_s8(_r00, _r02, 1);

    MIDOUT_BEGIN(conv_direct_stride, 0, 5) {
        int16x8_t v128 = vdupq_n_s16(128);
        int32x4_t vsrc_filter_zp = vdupq_n_s32(src_filter_zp);
        int8x8_t fzp = vdup_n_s8(filter_zp);

        int8x8_t f00 = vdup_n_s8(SUB128(filter[0]));
        int8x8_t f01 = vdup_n_s8(SUB128(filter[1]));
        int8x8_t f02 = vdup_n_s8(SUB128(filter[2]));
        int8x8_t f10 = vdup_n_s8(SUB128(filter[3]));
        int8x8_t f11 = vdup_n_s8(SUB128(filter[4]));
        int8x8_t f12 = vdup_n_s8(SUB128(filter[5]));
        int8x8_t f20 = vdup_n_s8(SUB128(filter[6]));
        int8x8_t f21 = vdup_n_s8(SUB128(filter[7]));
        int8x8_t f22 = vdup_n_s8(SUB128(filter[8]));

        // get filter * src_zp for one IC
        int32_t fxszp = 0;
        for (size_t i = 0; i < 9; ++i)
            fxszp += static_cast<int32_t>(filter[i]) - 128;
        int32x4_t vfxszp = vdupq_n_s32(fxszp * static_cast<int32_t>(src_zp));

        int8x8_t _idx = {0, 2, 4, 6, 1, 3, 5, 7};

        // 2x8 block
        size_t oh = 0;
        for (; oh + 1 < OH; oh += 2) {
            size_t ih = oh * 2;
            size_t ow = 0;
            for (; ow < OW; ow += 8) {
                size_t iw = ow * 2;
                int32_t* __restrict tptr = temp + oh * OW + ow;
                uint8_t* __restrict dptr = dst + oh * OW + ow;
                const uint8_t* __restrict sptr = src + ih * IW + iw;
                const int32_t* __restrict bptr = bias;
                int32x4_t sum00, sum01, sum10, sum11;
                int32x2x2_t _rn;
                int8x8_t _r00, _r01, _r02;

                if (!first_ic) {
                    sum00 = vld1q_s32(tptr + 0 * OW);
                    sum01 = vld1q_s32(tptr + 0 * OW + 4);
                    sum10 = vld1q_s32(tptr + 1 * OW);
                    sum11 = vld1q_s32(tptr + 1 * OW + 4);
                } else {
                    if (bias_mode == BiasMode::BROADCAST_CHANNEL_BIAS) {
                        sum00 = vdupq_n_s32(bptr[0]);
                        sum01 = sum00;
                        sum10 = sum00;
                        sum11 = sum00;
                    } else {
                        sum00 = vdupq_n_s32(0);
                        sum01 = vdupq_n_s32(0);
                        sum10 = vdupq_n_s32(0);
                        sum11 = vdupq_n_s32(0);
                    }
                    sum00 += vsrc_filter_zp;
                    sum01 += vsrc_filter_zp;
                    sum10 += vsrc_filter_zp;
                    sum11 += vsrc_filter_zp;
                }
                GET_R3(sptr);
                ACC_S16_S32(sum00, sum01, MLSFZP(_r00, f00));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r01, f01));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r02, f02));

                GET_R3(sptr + IW);
                ACC_S16_S32(sum00, sum01, MLSFZP(_r00, f10));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r01, f11));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r02, f12));

                GET_R3(sptr + 2 * IW);
                ACC_S16_S32(sum00, sum01, MLSFZP(_r00, f20));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r01, f21));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r02, f22));
                sum00 = vsubq_s32(sum00, vfxszp);
                sum01 = vsubq_s32(sum01, vfxszp);
                POSTPROCESS(sum00, sum01, tptr + 0 * OW, dptr + 0 * OW);
                ACC_S16_S32(sum10, sum11, MLSFZP(_r00, f00));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r01, f01));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r02, f02));

                GET_R3(sptr + 3 * IW);
                ACC_S16_S32(sum10, sum11, MLSFZP(_r00, f10));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r01, f11));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r02, f12));

                GET_R3(sptr + 4 * IW);
                ACC_S16_S32(sum10, sum11, MLSFZP(_r00, f20));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r01, f21));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r02, f22));
                sum10 = vsubq_s32(sum10, vfxszp);
                sum11 = vsubq_s32(sum11, vfxszp);
                POSTPROCESS(sum10, sum11, tptr + 1 * OW, dptr + 1 * OW);
            }
        }
        if (oh < OH) {
            size_t ih = oh * 2;
            size_t ow = 0;
            for (; ow < OW; ow += 8) {
                size_t iw = ow * 2;
                int32_t* __restrict tptr = temp + oh * OW + ow;
                uint8_t* __restrict dptr = dst + oh * OW + ow;
                const uint8_t* __restrict sptr = src + ih * IW + iw;
                const int32_t* __restrict bptr = bias;
                int32x4_t sum00, sum01;
                int32x2x2_t _rn;
                int8x8_t _r00, _r01, _r02;

                if (!first_ic) {
                    sum00 = vld1q_s32(tptr + 0 * OW);
                    sum01 = vld1q_s32(tptr + 0 * OW + 4);
                } else {
                    if (bias_mode == BiasMode::BROADCAST_CHANNEL_BIAS) {
                        sum00 = vdupq_n_s32(bptr[0]);
                        sum01 = sum00;
                    } else {
                        sum00 = vdupq_n_s32(0);
                        sum01 = vdupq_n_s32(0);
                    }
                    sum00 += vsrc_filter_zp;
                    sum01 += vsrc_filter_zp;
                }

                GET_R3(sptr);
                ACC_S16_S32(sum00, sum01, MLSFZP(_r00, f00));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r01, f01));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r02, f02));

                GET_R3(sptr + IW);
                ACC_S16_S32(sum00, sum01, MLSFZP(_r00, f10));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r01, f11));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r02, f12));

                GET_R3(sptr + 2 * IW);
                ACC_S16_S32(sum00, sum01, MLSFZP(_r00, f20));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r01, f21));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r02, f22));
                sum00 = vsubq_s32(sum00, vfxszp);
                sum01 = vsubq_s32(sum01, vfxszp);
                POSTPROCESS(sum00, sum01, tptr + 0 * OW, dptr + 0 * OW);
            }
        }
    }
    MIDOUT_END();
#undef GET_R3
}

template <bool first_ic, bool last_ic, BiasMode bias_mode, typename Op>
void conv_bias::conv_direct_stride2_5x5_quint8(
        const uint8_t* src, const uint8_t* filter, const int32_t* bias,
        int32_t* temp, uint8_t* dst, const size_t IH, const size_t IW,
        const size_t OH, const size_t OW, const int8_t src_zp,
        const int8_t filter_zp, const int32_t src_filter_zp, const Op& op) {
    MEGDNN_MARK_USED_VAR(IH);
#define GET_R5(sptr)                                                      \
    _r00 = SUB128VECTOR(vld1_u8(sptr));                                   \
    _r00 = vtbl1_s8(_r00, _idx);                                          \
    _r01 = SUB128VECTOR(vld1_u8(sptr + 8));                               \
    _r01 = vtbl1_s8(_r01, _idx);                                          \
    _rn = vzip_s32(vreinterpret_s32_s8(_r00), vreinterpret_s32_s8(_r01)); \
    _r00 = vreinterpret_s8_s32(_rn.val[0]);                               \
    _r01 = vreinterpret_s8_s32(_rn.val[1]);                               \
    _r03 = SUB128VECTOR(vld1_u8(sptr + 16));                              \
    _r03 = vtbl1_s8(_r03, _idx);                                          \
    _r02 = vext_s8(_r00, _r03, 1);                                        \
    _r04 = vext_s8(_r00, _r03, 2);                                        \
    _r03 = vtbl1_s8(_r03, _idxn);                                         \
    _r03 = vext_s8(_r01, _r03, 1);

    MIDOUT_BEGIN(conv_direct_stride, 0, 6) {
        int16x8_t v128 = vdupq_n_s16(128);
        int32x4_t vsrc_filter_zp = vdupq_n_s32(src_filter_zp);
        int8x8_t fzp = vdup_n_s8(filter_zp);

        int8x8_t f00 = vdup_n_s8(SUB128(filter[0]));
        int8x8_t f01 = vdup_n_s8(SUB128(filter[1]));
        int8x8_t f02 = vdup_n_s8(SUB128(filter[2]));
        int8x8_t f03 = vdup_n_s8(SUB128(filter[3]));
        int8x8_t f04 = vdup_n_s8(SUB128(filter[4]));
        int8x8_t f10 = vdup_n_s8(SUB128(filter[5]));
        int8x8_t f11 = vdup_n_s8(SUB128(filter[6]));
        int8x8_t f12 = vdup_n_s8(SUB128(filter[7]));
        int8x8_t f13 = vdup_n_s8(SUB128(filter[8]));
        int8x8_t f14 = vdup_n_s8(SUB128(filter[9]));
        int8x8_t f20 = vdup_n_s8(SUB128(filter[10]));
        int8x8_t f21 = vdup_n_s8(SUB128(filter[11]));
        int8x8_t f22 = vdup_n_s8(SUB128(filter[12]));
        int8x8_t f23 = vdup_n_s8(SUB128(filter[13]));
        int8x8_t f24 = vdup_n_s8(SUB128(filter[14]));
        int8x8_t f30 = vdup_n_s8(SUB128(filter[15]));
        int8x8_t f31 = vdup_n_s8(SUB128(filter[16]));
        int8x8_t f32 = vdup_n_s8(SUB128(filter[17]));
        int8x8_t f33 = vdup_n_s8(SUB128(filter[18]));
        int8x8_t f34 = vdup_n_s8(SUB128(filter[19]));
        int8x8_t f40 = vdup_n_s8(SUB128(filter[20]));
        int8x8_t f41 = vdup_n_s8(SUB128(filter[21]));
        int8x8_t f42 = vdup_n_s8(SUB128(filter[22]));
        int8x8_t f43 = vdup_n_s8(SUB128(filter[23]));
        int8x8_t f44 = vdup_n_s8(SUB128(filter[24]));

        // get filter * src_zp for one IC
        int32_t fxszp = 0;
        for (size_t i = 0; i < 25; ++i)
            fxszp += static_cast<int32_t>(filter[i]) - 128;
        int32x4_t vfxszp = vdupq_n_s32(fxszp * static_cast<int32_t>(src_zp));

        int8x8_t _idx = {0, 2, 4, 6, 1, 3, 5, 7};
        int8x8_t _idxn = {4, 5, 6, 7, 0, 1, 2, 3};
        // 2x8 block
        size_t oh = 0;
        for (; oh + 1 < OH; oh += 2) {
            size_t ih = oh * 2;
            size_t ow = 0;
            for (; ow < OW; ow += 8) {
                size_t iw = ow * 2;
                int32_t* __restrict tptr = temp + oh * OW + ow;
                uint8_t* __restrict dptr = dst + oh * OW + ow;
                const uint8_t* __restrict sptr = src + ih * IW + iw;
                const int32_t* __restrict bptr = bias;
                int32x4_t sum00, sum01, sum10, sum11;
                int32x2x2_t _rn;
                int8x8_t _r00, _r01, _r02, _r03, _r04;

                if (!first_ic) {
                    sum00 = vld1q_s32(tptr + 0 * OW);
                    sum01 = vld1q_s32(tptr + 0 * OW + 4);
                    sum10 = vld1q_s32(tptr + 1 * OW);
                    sum11 = vld1q_s32(tptr + 1 * OW + 4);
                } else {
                    if (bias_mode == BiasMode::BROADCAST_CHANNEL_BIAS) {
                        sum00 = vdupq_n_s32(bptr[0]);
                        sum01 = sum00;
                        sum10 = sum00;
                        sum11 = sum00;
                    } else {
                        sum00 = vdupq_n_s32(0);
                        sum01 = vdupq_n_s32(0);
                        sum10 = vdupq_n_s32(0);
                        sum11 = vdupq_n_s32(0);
                    }
                    sum00 += vsrc_filter_zp;
                    sum01 += vsrc_filter_zp;
                    sum10 += vsrc_filter_zp;
                    sum11 += vsrc_filter_zp;
                }

                GET_R5(sptr);
                ACC_S16_S32(sum00, sum01, MLSFZP(_r00, f00));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r01, f01));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r02, f02));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r03, f03));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r04, f04));

                GET_R5(sptr + IW);
                ACC_S16_S32(sum00, sum01, MLSFZP(_r00, f10));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r01, f11));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r02, f12));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r03, f13));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r04, f14));

                GET_R5(sptr + 2 * IW);
                ACC_S16_S32(sum00, sum01, MLSFZP(_r00, f20));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r01, f21));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r02, f22));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r03, f23));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r04, f24));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r00, f00));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r01, f01));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r02, f02));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r03, f03));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r04, f04));

                GET_R5(sptr + 3 * IW);
                ACC_S16_S32(sum00, sum01, MLSFZP(_r00, f30));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r01, f31));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r02, f32));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r03, f33));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r04, f34));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r00, f10));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r01, f11));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r02, f12));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r03, f13));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r04, f14));

                GET_R5(sptr + 4 * IW);
                ACC_S16_S32(sum00, sum01, MLSFZP(_r00, f40));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r01, f41));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r02, f42));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r03, f43));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r04, f44));
                sum00 = vsubq_s32(sum00, vfxszp);
                sum01 = vsubq_s32(sum01, vfxszp);
                POSTPROCESS(sum00, sum01, tptr + 0 * OW, dptr + 0 * OW);
                ACC_S16_S32(sum10, sum11, MLSFZP(_r00, f20));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r01, f21));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r02, f22));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r03, f23));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r04, f24));

                GET_R5(sptr + 5 * IW);
                ACC_S16_S32(sum10, sum11, MLSFZP(_r00, f30));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r01, f31));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r02, f32));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r03, f33));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r04, f34));

                GET_R5(sptr + 6 * IW);
                ACC_S16_S32(sum10, sum11, MLSFZP(_r00, f40));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r01, f41));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r02, f42));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r03, f43));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r04, f44));
                sum10 = vsubq_s32(sum10, vfxszp);
                sum11 = vsubq_s32(sum11, vfxszp);
                POSTPROCESS(sum10, sum11, tptr + 1 * OW, dptr + 1 * OW);
            }
        }
        if (oh < OH) {
            size_t ih = oh * 2;
            size_t ow = 0;
            for (; ow < OW; ow += 8) {
                size_t iw = ow * 2;
                int32_t* __restrict tptr = temp + oh * OW + ow;
                uint8_t* __restrict dptr = dst + oh * OW + ow;
                const uint8_t* __restrict sptr = src + ih * IW + iw;
                const int32_t* __restrict bptr = bias;
                int32x4_t sum00, sum01;
                int32x2x2_t _rn;
                int8x8_t _r00, _r01, _r02, _r03, _r04;

                if (!first_ic) {
                    sum00 = vld1q_s32(tptr + 0 * OW);
                    sum01 = vld1q_s32(tptr + 0 * OW + 4);
                } else {
                    if (bias_mode == BiasMode::BROADCAST_CHANNEL_BIAS) {
                        sum00 = vdupq_n_s32(bptr[0]);
                        sum01 = sum00;
                    } else {
                        sum00 = vdupq_n_s32(0);
                        sum01 = vdupq_n_s32(0);
                    }
                    sum00 += vsrc_filter_zp;
                    sum01 += vsrc_filter_zp;
                }

                GET_R5(sptr);
                ACC_S16_S32(sum00, sum01, MLSFZP(_r00, f00));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r01, f01));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r02, f02));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r03, f03));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r04, f04));

                GET_R5(sptr + IW);
                ACC_S16_S32(sum00, sum01, MLSFZP(_r00, f10));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r01, f11));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r02, f12));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r03, f13));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r04, f14));

                GET_R5(sptr + 2 * IW);
                ACC_S16_S32(sum00, sum01, MLSFZP(_r00, f20));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r01, f21));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r02, f22));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r03, f23));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r04, f24));

                GET_R5(sptr + 3 * IW);
                ACC_S16_S32(sum00, sum01, MLSFZP(_r00, f30));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r01, f31));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r02, f32));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r03, f33));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r04, f34));

                GET_R5(sptr + 4 * IW);
                ACC_S16_S32(sum00, sum01, MLSFZP(_r00, f40));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r01, f41));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r02, f42));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r03, f43));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r04, f44));
                sum00 = vsubq_s32(sum00, vfxszp);
                sum01 = vsubq_s32(sum01, vfxszp);
                POSTPROCESS(sum00, sum01, tptr + 0 * OW, dptr + 0 * OW);
            }
        }
    }
    MIDOUT_END();
#undef GET_R5
}

template <bool first_ic, bool last_ic, BiasMode bias_mode, typename Op>
void conv_bias::conv_direct_stride2_7x7_quint8(
        const uint8_t* src, const uint8_t* filter, const int32_t* bias,
        int32_t* temp, uint8_t* dst, const size_t IH, const size_t IW,
        const size_t OH, const size_t OW, const int8_t src_zp,
        const int8_t filter_zp, const int32_t src_filter_zp, const Op& op) {
    MEGDNN_MARK_USED_VAR(IH);
#define GET_R7(sptr)                                                      \
    _r00 = SUB128VECTOR(vld1_u8(sptr));                                   \
    _r00 = vtbl1_s8(_r00, _idx);                                          \
    _r01 = SUB128VECTOR(vld1_u8(sptr + 8));                               \
    _r01 = vtbl1_s8(_r01, _idx);                                          \
    _rn = vzip_s32(vreinterpret_s32_s8(_r00), vreinterpret_s32_s8(_r01)); \
    _r00 = vreinterpret_s8_s32(_rn.val[0]);                               \
    _r01 = vreinterpret_s8_s32(_rn.val[1]);                               \
    _r05 = SUB128VECTOR(vld1_u8(sptr + 16));                              \
    _r05 = vtbl1_s8(_r05, _idx);                                          \
    _r02 = vext_s8(_r00, _r05, 1);                                        \
    _r04 = vext_s8(_r00, _r05, 2);                                        \
    _r06 = vext_s8(_r00, _r05, 3);                                        \
    _r05 = vtbl1_s8(_r05, _idxn);                                         \
    _r03 = vext_s8(_r01, _r05, 1);                                        \
    _r05 = vext_s8(_r01, _r05, 2);

    MIDOUT_BEGIN(conv_direct_stride, 0, 7) {
        int16x8_t v128 = vdupq_n_s16(128);
        int32x4_t vsrc_filter_zp = vdupq_n_s32(src_filter_zp);
        int8x8_t fzp = vdup_n_s8(filter_zp);

        int8x8_t f00 = vdup_n_s8(SUB128(filter[0]));
        int8x8_t f01 = vdup_n_s8(SUB128(filter[1]));
        int8x8_t f02 = vdup_n_s8(SUB128(filter[2]));
        int8x8_t f03 = vdup_n_s8(SUB128(filter[3]));
        int8x8_t f04 = vdup_n_s8(SUB128(filter[4]));
        int8x8_t f05 = vdup_n_s8(SUB128(filter[5]));
        int8x8_t f06 = vdup_n_s8(SUB128(filter[6]));

        int8x8_t f10 = vdup_n_s8(SUB128(filter[7]));
        int8x8_t f11 = vdup_n_s8(SUB128(filter[8]));
        int8x8_t f12 = vdup_n_s8(SUB128(filter[9]));
        int8x8_t f13 = vdup_n_s8(SUB128(filter[10]));
        int8x8_t f14 = vdup_n_s8(SUB128(filter[11]));
        int8x8_t f15 = vdup_n_s8(SUB128(filter[12]));
        int8x8_t f16 = vdup_n_s8(SUB128(filter[13]));

        int8x8_t f20 = vdup_n_s8(SUB128(filter[14]));
        int8x8_t f21 = vdup_n_s8(SUB128(filter[15]));
        int8x8_t f22 = vdup_n_s8(SUB128(filter[16]));
        int8x8_t f23 = vdup_n_s8(SUB128(filter[17]));
        int8x8_t f24 = vdup_n_s8(SUB128(filter[18]));
        int8x8_t f25 = vdup_n_s8(SUB128(filter[19]));
        int8x8_t f26 = vdup_n_s8(SUB128(filter[20]));

        int8x8_t f30 = vdup_n_s8(SUB128(filter[21]));
        int8x8_t f31 = vdup_n_s8(SUB128(filter[22]));
        int8x8_t f32 = vdup_n_s8(SUB128(filter[23]));
        int8x8_t f33 = vdup_n_s8(SUB128(filter[24]));
        int8x8_t f34 = vdup_n_s8(SUB128(filter[25]));
        int8x8_t f35 = vdup_n_s8(SUB128(filter[26]));
        int8x8_t f36 = vdup_n_s8(SUB128(filter[27]));

        int8x8_t f40 = vdup_n_s8(SUB128(filter[28]));
        int8x8_t f41 = vdup_n_s8(SUB128(filter[29]));
        int8x8_t f42 = vdup_n_s8(SUB128(filter[30]));
        int8x8_t f43 = vdup_n_s8(SUB128(filter[31]));
        int8x8_t f44 = vdup_n_s8(SUB128(filter[32]));
        int8x8_t f45 = vdup_n_s8(SUB128(filter[33]));
        int8x8_t f46 = vdup_n_s8(SUB128(filter[34]));

        int8x8_t f50 = vdup_n_s8(SUB128(filter[35]));
        int8x8_t f51 = vdup_n_s8(SUB128(filter[36]));
        int8x8_t f52 = vdup_n_s8(SUB128(filter[37]));
        int8x8_t f53 = vdup_n_s8(SUB128(filter[38]));
        int8x8_t f54 = vdup_n_s8(SUB128(filter[39]));
        int8x8_t f55 = vdup_n_s8(SUB128(filter[40]));
        int8x8_t f56 = vdup_n_s8(SUB128(filter[41]));

        int8x8_t f60 = vdup_n_s8(SUB128(filter[42]));
        int8x8_t f61 = vdup_n_s8(SUB128(filter[43]));
        int8x8_t f62 = vdup_n_s8(SUB128(filter[44]));
        int8x8_t f63 = vdup_n_s8(SUB128(filter[45]));
        int8x8_t f64 = vdup_n_s8(SUB128(filter[46]));
        int8x8_t f65 = vdup_n_s8(SUB128(filter[47]));
        int8x8_t f66 = vdup_n_s8(SUB128(filter[48]));

        // get filter * src_zp for one IC
        int32_t fxszp = 0;
        for (size_t i = 0; i < 49; ++i)
            fxszp += static_cast<int32_t>(filter[i]) - 128;
        int32x4_t vfxszp = vdupq_n_s32(fxszp * static_cast<int32_t>(src_zp));

        int8x8_t _idx = {0, 2, 4, 6, 1, 3, 5, 7};
        int8x8_t _idxn = {4, 5, 6, 7, 0, 1, 2, 3};

        // 2x8 block
        size_t oh = 0;
        for (; oh + 1 < OH; oh += 2) {
            size_t ih = oh * 2;
            size_t ow = 0;
            for (; ow < OW; ow += 8) {
                size_t iw = ow * 2;
                int32_t* __restrict tptr = temp + oh * OW + ow;
                uint8_t* __restrict dptr = dst + oh * OW + ow;
                const uint8_t* __restrict sptr = src + ih * IW + iw;
                const int32_t* __restrict bptr = bias;
                int32x4_t sum00, sum01, sum10, sum11;
                int32x2x2_t _rn;
                int8x8_t _r00, _r01, _r02, _r03, _r04, _r05, _r06;

                if (!first_ic) {
                    sum00 = vld1q_s32(tptr + 0 * OW);
                    sum01 = vld1q_s32(tptr + 0 * OW + 4);
                    sum10 = vld1q_s32(tptr + 1 * OW);
                    sum11 = vld1q_s32(tptr + 1 * OW + 4);
                } else {
                    if (bias_mode == BiasMode::BROADCAST_CHANNEL_BIAS) {
                        sum00 = vdupq_n_s32(bptr[0]);
                        sum01 = sum00;
                        sum10 = sum00;
                        sum11 = sum00;
                    } else {
                        sum00 = vdupq_n_s32(0);
                        sum01 = vdupq_n_s32(0);
                        sum10 = vdupq_n_s32(0);
                        sum11 = vdupq_n_s32(0);
                    }
                    sum00 += vsrc_filter_zp;
                    sum01 += vsrc_filter_zp;
                    sum10 += vsrc_filter_zp;
                    sum11 += vsrc_filter_zp;
                }

                GET_R7(sptr);
                ACC_S16_S32(sum00, sum01, MLSFZP(_r00, f00));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r01, f01));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r02, f02));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r03, f03));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r04, f04));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r05, f05));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r06, f06));

                GET_R7(sptr + IW);
                ACC_S16_S32(sum00, sum01, MLSFZP(_r00, f10));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r01, f11));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r02, f12));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r03, f13));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r04, f14));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r05, f15));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r06, f16));

                GET_R7(sptr + 2 * IW);
                ACC_S16_S32(sum00, sum01, MLSFZP(_r00, f20));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r01, f21));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r02, f22));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r03, f23));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r04, f24));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r05, f25));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r06, f26));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r00, f00));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r01, f01));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r02, f02));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r03, f03));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r04, f04));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r05, f05));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r06, f06));

                GET_R7(sptr + 3 * IW);
                ACC_S16_S32(sum00, sum01, MLSFZP(_r00, f30));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r01, f31));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r02, f32));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r03, f33));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r04, f34));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r05, f35));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r06, f36));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r00, f10));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r01, f11));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r02, f12));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r03, f13));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r04, f14));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r05, f15));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r06, f16));

                GET_R7(sptr + 4 * IW);
                ACC_S16_S32(sum00, sum01, MLSFZP(_r00, f40));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r01, f41));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r02, f42));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r03, f43));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r04, f44));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r05, f45));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r06, f46));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r00, f20));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r01, f21));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r02, f22));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r03, f23));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r04, f24));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r05, f25));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r06, f26));

                GET_R7(sptr + 5 * IW);
                ACC_S16_S32(sum00, sum01, MLSFZP(_r00, f50));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r01, f51));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r02, f52));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r03, f53));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r04, f54));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r05, f55));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r06, f56));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r00, f30));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r01, f31));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r02, f32));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r03, f33));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r04, f34));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r05, f35));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r06, f36));

                GET_R7(sptr + 6 * IW);
                ACC_S16_S32(sum00, sum01, MLSFZP(_r00, f60));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r01, f61));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r02, f62));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r03, f63));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r04, f64));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r05, f65));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r06, f66));
                sum00 = vsubq_s32(sum00, vfxszp);
                sum01 = vsubq_s32(sum01, vfxszp);
                POSTPROCESS(sum00, sum01, tptr + 0 * OW, dptr + 0 * OW);
                ACC_S16_S32(sum10, sum11, MLSFZP(_r00, f40));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r01, f41));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r02, f42));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r03, f43));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r04, f44));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r05, f45));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r06, f46));

                GET_R7(sptr + 7 * IW);
                ACC_S16_S32(sum10, sum11, MLSFZP(_r00, f50));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r01, f51));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r02, f52));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r03, f53));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r04, f54));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r05, f55));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r06, f56));

                GET_R7(sptr + 8 * IW);
                ACC_S16_S32(sum10, sum11, MLSFZP(_r00, f60));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r01, f61));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r02, f62));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r03, f63));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r04, f64));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r05, f65));
                ACC_S16_S32(sum10, sum11, MLSFZP(_r06, f66));
                sum10 = vsubq_s32(sum10, vfxszp);
                sum11 = vsubq_s32(sum11, vfxszp);
                POSTPROCESS(sum10, sum11, tptr + 1 * OW, dptr + 1 * OW);
            }
        }
        if (oh < OH) {
            size_t ih = oh * 2;
            size_t ow = 0;
            for (; ow < OW; ow += 8) {
                size_t iw = ow * 2;
                int32_t* __restrict tptr = temp + oh * OW + ow;
                uint8_t* __restrict dptr = dst + oh * OW + ow;
                const uint8_t* __restrict sptr = src + ih * IW + iw;
                const int32_t* __restrict bptr = bias;
                int32x4_t sum00, sum01;
                int32x2x2_t _rn;
                int8x8_t _r00, _r01, _r02, _r03, _r04, _r05, _r06;

                if (!first_ic) {
                    sum00 = vld1q_s32(tptr + 0 * OW);
                    sum01 = vld1q_s32(tptr + 0 * OW + 4);
                } else {
                    if (bias_mode == BiasMode::BROADCAST_CHANNEL_BIAS) {
                        sum00 = vdupq_n_s32(bptr[0]);
                        sum01 = sum00;
                    } else {
                        sum00 = vdupq_n_s32(0);
                        sum01 = vdupq_n_s32(0);
                    }
                    sum00 += vsrc_filter_zp;
                    sum01 += vsrc_filter_zp;
                }

                GET_R7(sptr);
                ACC_S16_S32(sum00, sum01, MLSFZP(_r00, f00));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r01, f01));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r02, f02));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r03, f03));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r04, f04));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r05, f05));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r06, f06));

                GET_R7(sptr + IW);
                ACC_S16_S32(sum00, sum01, MLSFZP(_r00, f10));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r01, f11));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r02, f12));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r03, f13));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r04, f14));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r05, f15));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r06, f16));

                GET_R7(sptr + 2 * IW);
                ACC_S16_S32(sum00, sum01, MLSFZP(_r00, f20));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r01, f21));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r02, f22));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r03, f23));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r04, f24));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r05, f25));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r06, f26));

                GET_R7(sptr + 3 * IW);
                ACC_S16_S32(sum00, sum01, MLSFZP(_r00, f30));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r01, f31));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r02, f32));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r03, f33));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r04, f34));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r05, f35));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r06, f36));

                GET_R7(sptr + 4 * IW);
                ACC_S16_S32(sum00, sum01, MLSFZP(_r00, f40));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r01, f41));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r02, f42));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r03, f43));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r04, f44));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r05, f45));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r06, f46));

                GET_R7(sptr + 5 * IW);
                ACC_S16_S32(sum00, sum01, MLSFZP(_r00, f50));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r01, f51));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r02, f52));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r03, f53));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r04, f54));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r05, f55));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r06, f56));

                GET_R7(sptr + 6 * IW);
                ACC_S16_S32(sum00, sum01, MLSFZP(_r00, f60));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r01, f61));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r02, f62));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r03, f63));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r04, f64));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r05, f65));
                ACC_S16_S32(sum00, sum01, MLSFZP(_r06, f66));
                sum00 = vsubq_s32(sum00, vfxszp);
                sum01 = vsubq_s32(sum01, vfxszp);
                POSTPROCESS(sum00, sum01, tptr + 0 * OW, dptr + 0 * OW);
            }
        }
    }
    MIDOUT_END();
#undef GET_R7
}

#undef MLSFZP
#undef SUB128
#undef SUB128VECTOR
#undef POSTPROCESS
#undef ACC_S16_S32

#define INSTANTIATION(stride, i, first_ic, last_ic, bias, Op)                 \
    template void conv_bias::conv_direct_##stride##_##i##x##i##_quint8<       \
            first_ic, last_ic, bias, Op>(                                     \
            const uint8_t*, const uint8_t*, const int32_t*, int32_t*,         \
            uint8_t*, const size_t, const size_t, const size_t, const size_t, \
            const int8_t, const int8_t, const int32_t, const Op&);

#define FOR_NONLINEAR(stride, i, first_ic, last_ic, bias)      \
    INSTANTIATION(stride, i, first_ic, last_ic, bias,          \
                  TypeCvtOp<dt_qint32 MEGDNN_COMMA dt_quint8>) \
    INSTANTIATION(stride, i, first_ic, last_ic, bias,          \
                  ReluOp<dt_qint32 MEGDNN_COMMA dt_quint8>)    \
    INSTANTIATION(stride, i, first_ic, last_ic, bias,          \
                  HSwishOp<dt_qint32 MEGDNN_COMMA dt_quint8>)

#define FOR_BIAS(stride, i, first_ic, last_ic)                     \
    FOR_NONLINEAR(stride, i, first_ic, last_ic, BiasMode::NO_BIAS) \
    FOR_NONLINEAR(stride, i, first_ic, last_ic,                    \
                  BiasMode::BROADCAST_CHANNEL_BIAS)

#define FOR_IC(stride, i)             \
    FOR_BIAS(stride, i, true, true)   \
    FOR_BIAS(stride, i, true, false)  \
    FOR_BIAS(stride, i, false, false) \
    FOR_BIAS(stride, i, false, true)

#define FOR_FILTER(stride) \
    FOR_IC(stride, 2)      \
    FOR_IC(stride, 3)      \
    FOR_IC(stride, 5)      \
    FOR_IC(stride, 7)

#define FOR_STRIDE      \
    FOR_FILTER(stride1) \
    FOR_FILTER(stride2)

FOR_STRIDE

#undef FOR_STRIDE
#undef FOR_FILTER
#undef FOR_IC
#undef FOR_BIAS
#undef FOR_NONLINEAR
#undef INSTANTIATION

// vim: syntax=cpp.doxygen
