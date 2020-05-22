/**
 * \file dnn/src/arm_common/conv_bias/int8/direct.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/arm_common/conv_bias/int8/direct.h"
#include "src/arm_common/elemwise_op.h"
#include "src/arm_common/simd_macro/marm_neon.h"
#include "src/common/utils.h"
#include "src/fallback/conv_bias/common.h"

using namespace megdnn;
using namespace arm_common;

#define ACC_S16_S32(dst0, dst1, src)           \
    dst0 = vaddw_s16(dst0, vget_low_s16(src)); \
    dst1 = vaddw_s16(dst1, vget_high_s16(src));

#define POSTPROCESS(dst0, dst1, tptr, dptr)                    \
    if (last_ic) {                                             \
        op({{dst0, dst1}}, reinterpret_cast<dt_qint8*>(dptr)); \
    } else {                                                   \
        vst1q_s32(tptr, dst0);                                 \
        vst1q_s32(tptr + 4, dst1);                             \
    }

template <bool first_ic, bool last_ic, BiasMode bias_mode, typename Op>
void conv_bias::conv_direct_stride1_2x2_int8_nchw(const int8_t* src,
                                             const int8_t* filter,
                                             const int32_t* bias, int32_t* temp,
                                             int8_t* dst, const size_t IH,
                                             const size_t IW, const size_t OH,
                                             const size_t OW, const Op& op) {
    MEGDNN_MARK_USED_VAR(IH);
    int8x8_t k00 = vdup_n_s8(filter[0]);
    int8x8_t k01 = vdup_n_s8(filter[1]);
    int8x8_t k10 = vdup_n_s8(filter[2]);
    int8x8_t k11 = vdup_n_s8(filter[3]);

    // 4x8 block
    size_t oh = 0;
    for (; oh + 4 <= OH; oh += 4) {
        size_t ih = oh;
        for (size_t ow = 0; ow < OW; ow += 8) {
            size_t iw = ow;
            int32_t* __restrict tptr = temp + oh * OW + ow;
            int8_t* __restrict dptr = dst + oh * OW + ow;
            const int8_t* __restrict sptr = src + ih * IW + iw;
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
            }

            int8x8_t s = vld1_s8(sptr + 0 * IW);
            int16x8_t d0 = vmull_s8(k00, s);

            s = vld1_s8(sptr + 1 * IW);
            d0 = vmlal_s8(d0, k10, s);
            ACC_S16_S32(sum00, sum01, d0);
            int16x8_t d1 = vmull_s8(k00, s);

            s = vld1_s8(sptr + 2 * IW);
            d1 = vmlal_s8(d1, k10, s);
            ACC_S16_S32(sum10, sum11, d1);
            int16x8_t d2 = vmull_s8(k00, s);

            s = vld1_s8(sptr + 3 * IW);
            d2 = vmlal_s8(d2, k10, s);
            ACC_S16_S32(sum20, sum21, d2);
            int16x8_t d3 = vmull_s8(k00, s);

            s = vld1_s8(sptr + 4 * IW);
            d3 = vmlal_s8(d3, k10, s);
            ACC_S16_S32(sum30, sum31, d3);

            ++sptr;

            s = vld1_s8(sptr + 0 * IW);
            d0 = vmull_s8(k01, s);

            s = vld1_s8(sptr + 1 * IW);
            d0 = vmlal_s8(d0, k11, s);
            ACC_S16_S32(sum00, sum01, d0);
            POSTPROCESS(sum00, sum01, tptr + 0 * OW, dptr + 0 * OW);
            d1 = vmull_s8(k01, s);

            s = vld1_s8(sptr + 2 * IW);
            d1 = vmlal_s8(d1, k11, s);
            ACC_S16_S32(sum10, sum11, d1);
            POSTPROCESS(sum10, sum11, tptr + 1 * OW, dptr + 1 * OW);
            d2 = vmull_s8(k01, s);

            s = vld1_s8(sptr + 3 * IW);
            d2 = vmlal_s8(d2, k11, s);
            ACC_S16_S32(sum20, sum21, d2);
            POSTPROCESS(sum20, sum21, tptr + 2 * OW, dptr + 2 * OW);
            d3 = vmull_s8(k01, s);

            s = vld1_s8(sptr + 4 * IW);
            d3 = vmlal_s8(d3, k11, s);

            ACC_S16_S32(sum30, sum31, d3);
            POSTPROCESS(sum30, sum31, tptr + 3 * OW, dptr + 3 * OW);
        }
    }
    if (oh + 3 == OH) {
        size_t ih = oh;
        for (size_t ow = 0; ow < OW; ow += 8) {
            size_t iw = ow;
            int32_t* __restrict tptr = temp + oh * OW + ow;
            int8_t* __restrict dptr = dst + oh * OW + ow;
            const int8_t* __restrict sptr = src + ih * IW + iw;
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
            }

            int8x8_t s = vld1_s8(sptr + 0 * IW);
            int16x8_t d0 = vmull_s8(k00, s);

            s = vld1_s8(sptr + 1 * IW);
            d0 = vmlal_s8(d0, k10, s);
            ACC_S16_S32(sum00, sum01, d0);
            int16x8_t d1 = vmull_s8(k00, s);

            s = vld1_s8(sptr + 2 * IW);
            d1 = vmlal_s8(d1, k10, s);
            ACC_S16_S32(sum10, sum11, d1);
            int16x8_t d2 = vmull_s8(k00, s);

            s = vld1_s8(sptr + 3 * IW);
            d2 = vmlal_s8(d2, k10, s);
            ACC_S16_S32(sum20, sum21, d2);

            ++sptr;

            s = vld1_s8(sptr + 0 * IW);
            d0 = vmull_s8(k01, s);

            s = vld1_s8(sptr + 1 * IW);
            d0 = vmlal_s8(d0, k11, s);
            ACC_S16_S32(sum00, sum01, d0);
            POSTPROCESS(sum00, sum01, tptr + 0 * OW, dptr + 0 * OW);
            ;
            d1 = vmull_s8(k01, s);

            s = vld1_s8(sptr + 2 * IW);
            d1 = vmlal_s8(d1, k11, s);
            ACC_S16_S32(sum10, sum11, d1);
            POSTPROCESS(sum10, sum11, tptr + 1 * OW, dptr + 1 * OW);
            d2 = vmull_s8(k01, s);

            s = vld1_s8(sptr + 3 * IW);
            d2 = vmlal_s8(d2, k11, s);
            ACC_S16_S32(sum20, sum21, d2);
            POSTPROCESS(sum20, sum21, tptr + 2 * OW, dptr + 2 * OW);
        }
    } else if (oh + 2 == OH) {
        size_t ih = oh;
        for (size_t ow = 0; ow < OW; ow += 8) {
            size_t iw = ow;
            int32_t* __restrict tptr = temp + oh * OW + ow;
            int8_t* __restrict dptr = dst + oh * OW + ow;
            const int8_t* __restrict sptr = src + ih * IW + iw;
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
            }

            int8x8_t s = vld1_s8(sptr + 0 * IW);
            int16x8_t d0 = vmull_s8(k00, s);

            s = vld1_s8(sptr + 1 * IW);
            d0 = vmlal_s8(d0, k10, s);
            int16x8_t d1 = vmull_s8(k00, s);

            s = vld1_s8(sptr + 2 * IW);
            d1 = vmlal_s8(d1, k10, s);

            ACC_S16_S32(sum00, sum01, d0);
            ACC_S16_S32(sum10, sum11, d1);

            ++sptr;

            s = vld1_s8(sptr + 0 * IW);
            d0 = vmull_s8(k01, s);

            s = vld1_s8(sptr + 1 * IW);
            d0 = vmlal_s8(d0, k11, s);
            ACC_S16_S32(sum00, sum01, d0);
            POSTPROCESS(sum00, sum01, tptr + 0 * OW, dptr + 0 * OW);
            d1 = vmull_s8(k01, s);

            s = vld1_s8(sptr + 2 * IW);
            d1 = vmlal_s8(d1, k11, s);
            ACC_S16_S32(sum10, sum11, d1);
            POSTPROCESS(sum10, sum11, tptr + 1 * OW, dptr + 1 * OW);
        }
    } else if (oh + 1 == OH) {
        size_t ih = oh;
        for (size_t ow = 0; ow < OW; ow += 8) {
            size_t iw = ow;
            int32_t* __restrict tptr = temp + oh * OW + ow;
            int8_t* __restrict dptr = dst + oh * OW + ow;
            const int8_t* __restrict sptr = src + ih * IW + iw;
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
            }

            int8x8_t s = vld1_s8(sptr + 0 * IW);
            int16x8_t d0 = vmull_s8(k00, s);

            s = vld1_s8(sptr + 1 * IW);
            d0 = vmlal_s8(d0, k10, s);
            ACC_S16_S32(sum00, sum01, d0);

            ++sptr;

            s = vld1_s8(sptr + 0 * IW);
            d0 = vmull_s8(k01, s);

            s = vld1_s8(sptr + 1 * IW);
            d0 = vmlal_s8(d0, k11, s);
            ACC_S16_S32(sum00, sum01, d0);
            POSTPROCESS(sum00, sum01, tptr + 0 * OW, dptr + 0 * OW);
        }
    }
}

template <bool first_ic, bool last_ic, BiasMode bias_mode, typename Op>
void conv_bias::conv_direct_stride1_3x3_int8_nchw(const int8_t* src,
                                             const int8_t* filter,
                                             const int32_t* bias, int32_t* temp,
                                             int8_t* dst, const size_t IH,
                                             const size_t IW, const size_t OH,
                                             const size_t OW, const Op& op) {
    MEGDNN_MARK_USED_VAR(IH);
    int8x8_t k00 = vdup_n_s8(filter[0]);
    int8x8_t k01 = vdup_n_s8(filter[1]);
    int8x8_t k02 = vdup_n_s8(filter[2]);
    int8x8_t k10 = vdup_n_s8(filter[3]);
    int8x8_t k11 = vdup_n_s8(filter[4]);
    int8x8_t k12 = vdup_n_s8(filter[5]);
    int8x8_t k20 = vdup_n_s8(filter[6]);
    int8x8_t k21 = vdup_n_s8(filter[7]);
    int8x8_t k22 = vdup_n_s8(filter[8]);

    // block 2x8
    size_t oh = 0;
    for (; oh + 1 < OH; oh += 2) {
        size_t ih = oh;
        for (size_t ow = 0; ow < OW; ow += 8) {
            size_t iw = ow;
            int32_t* __restrict tptr = temp + oh * OW + ow;
            int8_t* __restrict dptr = dst + oh * OW + ow;
            const int8_t* __restrict sptr = src + ih * IW + iw;
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
            }

            int8x8_t _r00 = vld1_s8(sptr + 0 * IW);
            int8x8_t _r0n = vld1_s8(sptr + 0 * IW + 8);
            int8x8_t _r01 = vext_s8(_r00, _r0n, 1);
            int8x8_t _r02 = vext_s8(_r00, _r0n, 2);

            int16x8_t d0 = vmull_s8(_r00, k00);
            d0 = vmlal_s8(d0, _r01, k01);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r02, k02);

            int8x8_t _r10 = vld1_s8(sptr + 1 * IW);
            int8x8_t _r1n = vld1_s8(sptr + 1 * IW + 8);
            int8x8_t _r11 = vext_s8(_r10, _r1n, 1);
            int8x8_t _r12 = vext_s8(_r10, _r1n, 2);
            d0 = vmlal_s8(d0, _r10, k10);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r11, k11);
            d0 = vmlal_s8(d0, _r12, k12);
            ACC_S16_S32(sum00, sum01, d0);
            int16x8_t d1 = vmull_s8(_r10, k00);
            d1 = vmlal_s8(d1, _r11, k01);
            ACC_S16_S32(sum10, sum11, d1);
            d1 = vmull_s8(_r12, k02);

            int8x8_t _r20 = vld1_s8(sptr + 2 * IW);
            int8x8_t _r2n = vld1_s8(sptr + 2 * IW + 8);
            int8x8_t _r21 = vext_s8(_r20, _r2n, 1);
            int8x8_t _r22 = vext_s8(_r20, _r2n, 2);
            d0 = vmull_s8(_r20, k20);
            d0 = vmlal_s8(d0, _r21, k21);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r22, k22);
            ACC_S16_S32(sum00, sum01, d0);
            POSTPROCESS(sum00, sum01, tptr + 0 * OW, dptr + 0 * OW);
            d1 = vmlal_s8(d1, _r20, k10);
            ACC_S16_S32(sum10, sum11, d1);
            d1 = vmull_s8(_r21, k11);
            d1 = vmlal_s8(d1, _r22, k12);
            ACC_S16_S32(sum10, sum11, d1);

            int8x8_t _r30 = vld1_s8(sptr + 3 * IW);
            int8x8_t _r3n = vld1_s8(sptr + 3 * IW + 8);
            int8x8_t _r31 = vext_s8(_r30, _r3n, 1);
            int8x8_t _r32 = vext_s8(_r30, _r3n, 2);
            d1 = vmull_s8(_r30, k20);
            d1 = vmlal_s8(d1, _r31, k21);
            ACC_S16_S32(sum10, sum11, d1);
            d1 = vmull_s8(_r32, k22);
            ACC_S16_S32(sum10, sum11, d1);
            POSTPROCESS(sum10, sum11, tptr + 1 * OW, dptr + 1 * OW);
        }
    }

    if (oh < OH) {
        size_t ih = oh;
        for (size_t ow = 0; ow < OW; ow += 8) {
            size_t iw = ow;
            int32_t* __restrict tptr = temp + oh * OW + ow;
            int8_t* __restrict dptr = dst + oh * OW + ow;
            const int8_t* __restrict sptr = src + ih * IW + iw;
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
            }
            int8x8_t _r00 = vld1_s8(sptr + 0 * IW);
            int8x8_t _r0n = vld1_s8(sptr + 0 * IW + 8);
            int8x8_t _r01 = vext_s8(_r00, _r0n, 1);
            int8x8_t _r02 = vext_s8(_r00, _r0n, 2);

            int16x8_t d0 = vmull_s8(_r00, k00);
            d0 = vmlal_s8(d0, _r01, k01);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r02, k02);

            int8x8_t _r10 = vld1_s8(sptr + 1 * IW);
            int8x8_t _r1n = vld1_s8(sptr + 1 * IW + 8);
            int8x8_t _r11 = vext_s8(_r10, _r1n, 1);
            int8x8_t _r12 = vext_s8(_r10, _r1n, 2);
            d0 = vmlal_s8(d0, _r10, k10);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r11, k11);
            d0 = vmlal_s8(d0, _r12, k12);
            ACC_S16_S32(sum00, sum01, d0);

            int8x8_t _r20 = vld1_s8(sptr + 2 * IW);
            int8x8_t _r2n = vld1_s8(sptr + 2 * IW + 8);
            int8x8_t _r21 = vext_s8(_r20, _r2n, 1);
            int8x8_t _r22 = vext_s8(_r20, _r2n, 2);
            d0 = vmull_s8(_r20, k20);
            d0 = vmlal_s8(d0, _r21, k21);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r22, k22);
            ACC_S16_S32(sum00, sum01, d0);

            POSTPROCESS(sum00, sum01, tptr + 0 * OW, dptr + 0 * OW);
        }
    }
}

template <bool first_ic, bool last_ic, BiasMode bias_mode, typename Op>
void conv_bias::conv_direct_stride1_5x5_int8_nchw(const int8_t* src,
                                             const int8_t* filter,
                                             const int32_t* bias, int32_t* temp,
                                             int8_t* dst, const size_t IH,
                                             const size_t IW, const size_t OH,
                                             const size_t OW, const Op& op) {
    MEGDNN_MARK_USED_VAR(IH);
    int8x8_t k00 = vdup_n_s8(filter[0]);
    int8x8_t k01 = vdup_n_s8(filter[1]);
    int8x8_t k02 = vdup_n_s8(filter[2]);
    int8x8_t k03 = vdup_n_s8(filter[3]);
    int8x8_t k04 = vdup_n_s8(filter[4]);
    int8x8_t k10 = vdup_n_s8(filter[5]);
    int8x8_t k11 = vdup_n_s8(filter[6]);
    int8x8_t k12 = vdup_n_s8(filter[7]);
    int8x8_t k13 = vdup_n_s8(filter[8]);
    int8x8_t k14 = vdup_n_s8(filter[9]);
    int8x8_t k20 = vdup_n_s8(filter[10]);
    int8x8_t k21 = vdup_n_s8(filter[11]);
    int8x8_t k22 = vdup_n_s8(filter[12]);
    int8x8_t k23 = vdup_n_s8(filter[13]);
    int8x8_t k24 = vdup_n_s8(filter[14]);
    int8x8_t k30 = vdup_n_s8(filter[15]);
    int8x8_t k31 = vdup_n_s8(filter[16]);
    int8x8_t k32 = vdup_n_s8(filter[17]);
    int8x8_t k33 = vdup_n_s8(filter[18]);
    int8x8_t k34 = vdup_n_s8(filter[19]);
    int8x8_t k40 = vdup_n_s8(filter[20]);
    int8x8_t k41 = vdup_n_s8(filter[21]);
    int8x8_t k42 = vdup_n_s8(filter[22]);
    int8x8_t k43 = vdup_n_s8(filter[23]);
    int8x8_t k44 = vdup_n_s8(filter[24]);

    // block 2x8
    size_t oh = 0;
    for (; oh + 1 < OH; oh += 2) {
        size_t ih = oh;
        for (size_t ow = 0; ow < OW; ow += 8) {
            size_t iw = ow;
            int32_t* __restrict tptr = temp + oh * OW + ow;
            int8_t* __restrict dptr = dst + oh * OW + ow;
            const int8_t* __restrict sptr = src + ih * IW + iw;
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
            }

            int8x8_t _r00 = vld1_s8(sptr + 0 * IW);
            int8x8_t _r0n = vld1_s8(sptr + 0 * IW + 8);
            int8x8_t _r01 = vext_s8(_r00, _r0n, 1);
            int8x8_t _r02 = vext_s8(_r00, _r0n, 2);
            int8x8_t _r03 = vext_s8(_r00, _r0n, 3);
            int8x8_t _r04 = vext_s8(_r00, _r0n, 4);
            int16x8_t d0 = vmull_s8(_r00, k00);
            d0 = vmlal_s8(d0, _r01, k01);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r02, k02);
            d0 = vmlal_s8(d0, _r03, k03);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r04, k04);

            int8x8_t _r10 = vld1_s8(sptr + 1 * IW);
            int8x8_t _r1n = vld1_s8(sptr + 1 * IW + 8);
            int8x8_t _r11 = vext_s8(_r10, _r1n, 1);
            int8x8_t _r12 = vext_s8(_r10, _r1n, 2);
            int8x8_t _r13 = vext_s8(_r10, _r1n, 3);
            int8x8_t _r14 = vext_s8(_r10, _r1n, 4);
            d0 = vmlal_s8(d0, _r10, k10);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r11, k11);
            d0 = vmlal_s8(d0, _r12, k12);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r13, k13);
            d0 = vmlal_s8(d0, _r14, k14);
            ACC_S16_S32(sum00, sum01, d0);
            int16x8_t d1 = vmull_s8(_r10, k00);
            d1 = vmlal_s8(d1, _r11, k01);
            ACC_S16_S32(sum10, sum11, d1);
            d1 = vmull_s8(_r12, k02);
            d1 = vmlal_s8(d1, _r13, k03);
            ACC_S16_S32(sum10, sum11, d1);
            d1 = vmull_s8(_r14, k04);

            int8x8_t _r20 = vld1_s8(sptr + 2 * IW);
            int8x8_t _r2n = vld1_s8(sptr + 2 * IW + 8);
            int8x8_t _r21 = vext_s8(_r20, _r2n, 1);
            int8x8_t _r22 = vext_s8(_r20, _r2n, 2);
            int8x8_t _r23 = vext_s8(_r20, _r2n, 3);
            int8x8_t _r24 = vext_s8(_r20, _r2n, 4);
            d0 = vmull_s8(_r20, k20);
            d0 = vmlal_s8(d0, _r21, k21);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r22, k22);
            d0 = vmlal_s8(d0, _r23, k23);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r24, k24);
            d1 = vmlal_s8(d1, _r20, k10);
            ACC_S16_S32(sum10, sum11, d1);
            d1 = vmull_s8(_r21, k11);
            d1 = vmlal_s8(d1, _r22, k12);
            ACC_S16_S32(sum10, sum11, d1);
            d1 = vmull_s8(_r23, k13);
            d1 = vmlal_s8(d1, _r24, k14);
            ACC_S16_S32(sum10, sum11, d1);

            int8x8_t _r30 = vld1_s8(sptr + 3 * IW);
            int8x8_t _r3n = vld1_s8(sptr + 3 * IW + 8);
            int8x8_t _r31 = vext_s8(_r30, _r3n, 1);
            int8x8_t _r32 = vext_s8(_r30, _r3n, 2);
            int8x8_t _r33 = vext_s8(_r30, _r3n, 3);
            int8x8_t _r34 = vext_s8(_r30, _r3n, 4);
            d0 = vmlal_s8(d0, _r30, k30);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r31, k31);
            d0 = vmlal_s8(d0, _r32, k32);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r33, k33);
            d0 = vmlal_s8(d0, _r34, k34);
            ACC_S16_S32(sum00, sum01, d0);
            d1 = vmull_s8(_r30, k20);
            d1 = vmlal_s8(d1, _r31, k21);
            ACC_S16_S32(sum10, sum11, d1);
            d1 = vmull_s8(_r32, k22);
            d1 = vmlal_s8(d1, _r33, k23);
            ACC_S16_S32(sum10, sum11, d1);
            d1 = vmull_s8(_r34, k24);

            int8x8_t _r40 = vld1_s8(sptr + 4 * IW);
            int8x8_t _r4n = vld1_s8(sptr + 4 * IW + 8);
            int8x8_t _r41 = vext_s8(_r40, _r4n, 1);
            int8x8_t _r42 = vext_s8(_r40, _r4n, 2);
            int8x8_t _r43 = vext_s8(_r40, _r4n, 3);
            int8x8_t _r44 = vext_s8(_r40, _r4n, 4);
            d0 = vmull_s8(_r40, k40);
            d0 = vmlal_s8(d0, _r41, k41);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r42, k42);
            d0 = vmlal_s8(d0, _r43, k43);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r44, k44);
            ACC_S16_S32(sum00, sum01, d0);
            POSTPROCESS(sum00, sum01, tptr + 0 * OW, dptr + 0 * OW);
            d1 = vmlal_s8(d1, _r40, k30);
            ACC_S16_S32(sum10, sum11, d1);
            d1 = vmull_s8(_r41, k31);
            d1 = vmlal_s8(d1, _r42, k32);
            ACC_S16_S32(sum10, sum11, d1);
            d1 = vmull_s8(_r43, k33);
            d1 = vmlal_s8(d1, _r44, k34);
            ACC_S16_S32(sum10, sum11, d1);

            int8x8_t _r50 = vld1_s8(sptr + 5 * IW);
            int8x8_t _r5n = vld1_s8(sptr + 5 * IW + 8);
            int8x8_t _r51 = vext_s8(_r50, _r5n, 1);
            int8x8_t _r52 = vext_s8(_r50, _r5n, 2);
            int8x8_t _r53 = vext_s8(_r50, _r5n, 3);
            int8x8_t _r54 = vext_s8(_r50, _r5n, 4);
            d1 = vmull_s8(_r50, k40);
            d1 = vmlal_s8(d1, _r51, k41);
            ACC_S16_S32(sum10, sum11, d1);
            d1 = vmull_s8(_r52, k42);
            d1 = vmlal_s8(d1, _r53, k43);
            ACC_S16_S32(sum10, sum11, d1);
            d1 = vmull_s8(_r54, k44);
            ACC_S16_S32(sum10, sum11, d1);
            POSTPROCESS(sum10, sum11, tptr + 1 * OW, dptr + 1 * OW);
        }
    }

    if (oh < OH) {
        size_t ih = oh;
        for (size_t ow = 0; ow < OW; ow += 8) {
            size_t iw = ow;
            int32_t* __restrict tptr = temp + oh * OW + ow;
            int8_t* __restrict dptr = dst + oh * OW + ow;
            const int8_t* __restrict sptr = src + ih * IW + iw;
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
            }

            int8x8_t _r00 = vld1_s8(sptr + 0 * IW);
            int8x8_t _r0n = vld1_s8(sptr + 0 * IW + 8);
            int8x8_t _r01 = vext_s8(_r00, _r0n, 1);
            int8x8_t _r02 = vext_s8(_r00, _r0n, 2);
            int8x8_t _r03 = vext_s8(_r00, _r0n, 3);
            int8x8_t _r04 = vext_s8(_r00, _r0n, 4);
            int16x8_t d0 = vmull_s8(_r00, k00);
            d0 = vmlal_s8(d0, _r01, k01);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r02, k02);
            d0 = vmlal_s8(d0, _r03, k03);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r04, k04);

            int8x8_t _r10 = vld1_s8(sptr + 1 * IW);
            int8x8_t _r1n = vld1_s8(sptr + 1 * IW + 8);
            int8x8_t _r11 = vext_s8(_r10, _r1n, 1);
            int8x8_t _r12 = vext_s8(_r10, _r1n, 2);
            int8x8_t _r13 = vext_s8(_r10, _r1n, 3);
            int8x8_t _r14 = vext_s8(_r10, _r1n, 4);
            d0 = vmlal_s8(d0, _r10, k10);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r11, k11);
            d0 = vmlal_s8(d0, _r12, k12);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r13, k13);
            d0 = vmlal_s8(d0, _r14, k14);
            ACC_S16_S32(sum00, sum01, d0);

            int8x8_t _r20 = vld1_s8(sptr + 2 * IW);
            int8x8_t _r2n = vld1_s8(sptr + 2 * IW + 8);
            int8x8_t _r21 = vext_s8(_r20, _r2n, 1);
            int8x8_t _r22 = vext_s8(_r20, _r2n, 2);
            int8x8_t _r23 = vext_s8(_r20, _r2n, 3);
            int8x8_t _r24 = vext_s8(_r20, _r2n, 4);
            d0 = vmull_s8(_r20, k20);
            d0 = vmlal_s8(d0, _r21, k21);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r22, k22);
            d0 = vmlal_s8(d0, _r23, k23);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r24, k24);

            int8x8_t _r30 = vld1_s8(sptr + 3 * IW);
            int8x8_t _r3n = vld1_s8(sptr + 3 * IW + 8);
            int8x8_t _r31 = vext_s8(_r30, _r3n, 1);
            int8x8_t _r32 = vext_s8(_r30, _r3n, 2);
            int8x8_t _r33 = vext_s8(_r30, _r3n, 3);
            int8x8_t _r34 = vext_s8(_r30, _r3n, 4);
            d0 = vmlal_s8(d0, _r30, k30);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r31, k31);
            d0 = vmlal_s8(d0, _r32, k32);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r33, k33);
            d0 = vmlal_s8(d0, _r34, k34);
            ACC_S16_S32(sum00, sum01, d0);

            int8x8_t _r40 = vld1_s8(sptr + 4 * IW);
            int8x8_t _r4n = vld1_s8(sptr + 4 * IW + 8);
            int8x8_t _r41 = vext_s8(_r40, _r4n, 1);
            int8x8_t _r42 = vext_s8(_r40, _r4n, 2);
            int8x8_t _r43 = vext_s8(_r40, _r4n, 3);
            int8x8_t _r44 = vext_s8(_r40, _r4n, 4);
            d0 = vmull_s8(_r40, k40);
            d0 = vmlal_s8(d0, _r41, k41);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r42, k42);
            d0 = vmlal_s8(d0, _r43, k43);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r44, k44);
            ACC_S16_S32(sum00, sum01, d0);
            POSTPROCESS(sum00, sum01, tptr + 0 * OW, dptr + 0 * OW);
        }
    }
}

template <bool first_ic, bool last_ic, BiasMode bias_mode, typename Op>
void conv_bias::conv_direct_stride1_7x7_int8_nchw(const int8_t* src,
                                             const int8_t* filter,
                                             const int32_t* bias, int32_t* temp,
                                             int8_t* dst, const size_t IH,
                                             const size_t IW, const size_t OH,
                                             const size_t OW, const Op& op) {
    MEGDNN_MARK_USED_VAR(IH);
    int8x8_t k00 = vdup_n_s8(filter[0]);
    int8x8_t k01 = vdup_n_s8(filter[1]);
    int8x8_t k02 = vdup_n_s8(filter[2]);
    int8x8_t k03 = vdup_n_s8(filter[3]);
    int8x8_t k04 = vdup_n_s8(filter[4]);
    int8x8_t k05 = vdup_n_s8(filter[5]);
    int8x8_t k06 = vdup_n_s8(filter[6]);

    int8x8_t k10 = vdup_n_s8(filter[7]);
    int8x8_t k11 = vdup_n_s8(filter[8]);
    int8x8_t k12 = vdup_n_s8(filter[9]);
    int8x8_t k13 = vdup_n_s8(filter[10]);
    int8x8_t k14 = vdup_n_s8(filter[11]);
    int8x8_t k15 = vdup_n_s8(filter[12]);
    int8x8_t k16 = vdup_n_s8(filter[13]);

    int8x8_t k20 = vdup_n_s8(filter[14]);
    int8x8_t k21 = vdup_n_s8(filter[15]);
    int8x8_t k22 = vdup_n_s8(filter[16]);
    int8x8_t k23 = vdup_n_s8(filter[17]);
    int8x8_t k24 = vdup_n_s8(filter[18]);
    int8x8_t k25 = vdup_n_s8(filter[19]);
    int8x8_t k26 = vdup_n_s8(filter[20]);

    int8x8_t k30 = vdup_n_s8(filter[21]);
    int8x8_t k31 = vdup_n_s8(filter[22]);
    int8x8_t k32 = vdup_n_s8(filter[23]);
    int8x8_t k33 = vdup_n_s8(filter[24]);
    int8x8_t k34 = vdup_n_s8(filter[25]);
    int8x8_t k35 = vdup_n_s8(filter[26]);
    int8x8_t k36 = vdup_n_s8(filter[27]);

    int8x8_t k40 = vdup_n_s8(filter[28]);
    int8x8_t k41 = vdup_n_s8(filter[29]);
    int8x8_t k42 = vdup_n_s8(filter[30]);
    int8x8_t k43 = vdup_n_s8(filter[31]);
    int8x8_t k44 = vdup_n_s8(filter[32]);
    int8x8_t k45 = vdup_n_s8(filter[33]);
    int8x8_t k46 = vdup_n_s8(filter[34]);

    int8x8_t k50 = vdup_n_s8(filter[35]);
    int8x8_t k51 = vdup_n_s8(filter[36]);
    int8x8_t k52 = vdup_n_s8(filter[37]);
    int8x8_t k53 = vdup_n_s8(filter[38]);
    int8x8_t k54 = vdup_n_s8(filter[39]);
    int8x8_t k55 = vdup_n_s8(filter[40]);
    int8x8_t k56 = vdup_n_s8(filter[41]);

    int8x8_t k60 = vdup_n_s8(filter[42]);
    int8x8_t k61 = vdup_n_s8(filter[43]);
    int8x8_t k62 = vdup_n_s8(filter[44]);
    int8x8_t k63 = vdup_n_s8(filter[45]);
    int8x8_t k64 = vdup_n_s8(filter[46]);
    int8x8_t k65 = vdup_n_s8(filter[47]);
    int8x8_t k66 = vdup_n_s8(filter[48]);

    // block 2x8
    size_t oh = 0;
    for (; oh + 1 < OH; oh += 2) {
        size_t ih = oh;
        for (size_t ow = 0; ow < OW; ow += 8) {
            size_t iw = ow;
            int32_t* __restrict tptr = temp + oh * OW + ow;
            int8_t* __restrict dptr = dst + oh * OW + ow;
            const int8_t* __restrict sptr = src + ih * IW + iw;
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
            }

            int8x8_t _r00 = vld1_s8(sptr + 0 * IW);
            int8x8_t _r0n = vld1_s8(sptr + 0 * IW + 8);
            int8x8_t _r01 = vext_s8(_r00, _r0n, 1);
            int8x8_t _r02 = vext_s8(_r00, _r0n, 2);
            int8x8_t _r03 = vext_s8(_r00, _r0n, 3);
            int8x8_t _r04 = vext_s8(_r00, _r0n, 4);
            int8x8_t _r05 = vext_s8(_r00, _r0n, 5);
            int8x8_t _r06 = vext_s8(_r00, _r0n, 6);
            int16x8_t d0 = vmull_s8(_r00, k00);
            d0 = vmlal_s8(d0, _r01, k01);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r02, k02);
            d0 = vmlal_s8(d0, _r03, k03);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r04, k04);
            d0 = vmlal_s8(d0, _r05, k05);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r06, k06);

            int8x8_t _r10 = vld1_s8(sptr + 1 * IW);
            int8x8_t _r1n = vld1_s8(sptr + 1 * IW + 8);
            int8x8_t _r11 = vext_s8(_r10, _r1n, 1);
            int8x8_t _r12 = vext_s8(_r10, _r1n, 2);
            int8x8_t _r13 = vext_s8(_r10, _r1n, 3);
            int8x8_t _r14 = vext_s8(_r10, _r1n, 4);
            int8x8_t _r15 = vext_s8(_r10, _r1n, 5);
            int8x8_t _r16 = vext_s8(_r10, _r1n, 6);
            d0 = vmlal_s8(d0, _r10, k10);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r11, k11);
            d0 = vmlal_s8(d0, _r12, k12);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r13, k13);
            d0 = vmlal_s8(d0, _r14, k14);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r15, k15);
            d0 = vmlal_s8(d0, _r16, k16);
            ACC_S16_S32(sum00, sum01, d0);
            int16x8_t d1 = vmull_s8(_r10, k00);
            d1 = vmlal_s8(d1, _r11, k01);
            ACC_S16_S32(sum10, sum11, d1);
            d1 = vmull_s8(_r12, k02);
            d1 = vmlal_s8(d1, _r13, k03);
            ACC_S16_S32(sum10, sum11, d1);
            d1 = vmull_s8(_r14, k04);
            d1 = vmlal_s8(d1, _r15, k05);
            ACC_S16_S32(sum10, sum11, d1);
            d1 = vmull_s8(_r16, k06);

            int8x8_t _r20 = vld1_s8(sptr + 2 * IW);
            int8x8_t _r2n = vld1_s8(sptr + 2 * IW + 8);
            int8x8_t _r21 = vext_s8(_r20, _r2n, 1);
            int8x8_t _r22 = vext_s8(_r20, _r2n, 2);
            int8x8_t _r23 = vext_s8(_r20, _r2n, 3);
            int8x8_t _r24 = vext_s8(_r20, _r2n, 4);
            int8x8_t _r25 = vext_s8(_r20, _r2n, 5);
            int8x8_t _r26 = vext_s8(_r20, _r2n, 6);
            d0 = vmull_s8(_r20, k20);
            d0 = vmlal_s8(d0, _r21, k21);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r22, k22);
            d0 = vmlal_s8(d0, _r23, k23);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r24, k24);
            d0 = vmlal_s8(d0, _r25, k25);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r26, k26);
            d1 = vmlal_s8(d1, _r20, k10);
            ACC_S16_S32(sum10, sum11, d1);
            d1 = vmull_s8(_r21, k11);
            d1 = vmlal_s8(d1, _r22, k12);
            ACC_S16_S32(sum10, sum11, d1);
            d1 = vmull_s8(_r23, k13);
            d1 = vmlal_s8(d1, _r24, k14);
            ACC_S16_S32(sum10, sum11, d1);
            d1 = vmull_s8(_r25, k15);
            d1 = vmlal_s8(d1, _r26, k16);
            ACC_S16_S32(sum10, sum11, d1);

            int8x8_t _r30 = vld1_s8(sptr + 3 * IW);
            int8x8_t _r3n = vld1_s8(sptr + 3 * IW + 8);
            int8x8_t _r31 = vext_s8(_r30, _r3n, 1);
            int8x8_t _r32 = vext_s8(_r30, _r3n, 2);
            int8x8_t _r33 = vext_s8(_r30, _r3n, 3);
            int8x8_t _r34 = vext_s8(_r30, _r3n, 4);
            int8x8_t _r35 = vext_s8(_r30, _r3n, 5);
            int8x8_t _r36 = vext_s8(_r30, _r3n, 6);
            d0 = vmlal_s8(d0, _r30, k30);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r31, k31);
            d0 = vmlal_s8(d0, _r32, k32);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r33, k33);
            d0 = vmlal_s8(d0, _r34, k34);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r35, k35);
            d0 = vmlal_s8(d0, _r36, k36);
            ACC_S16_S32(sum00, sum01, d0);
            d1 = vmull_s8(_r30, k20);
            d1 = vmlal_s8(d1, _r31, k21);
            ACC_S16_S32(sum10, sum11, d1);
            d1 = vmull_s8(_r32, k22);
            d1 = vmlal_s8(d1, _r33, k23);
            ACC_S16_S32(sum10, sum11, d1);
            d1 = vmull_s8(_r34, k24);
            d1 = vmlal_s8(d1, _r35, k25);
            ACC_S16_S32(sum10, sum11, d1);
            d1 = vmull_s8(_r36, k26);

            int8x8_t _r40 = vld1_s8(sptr + 4 * IW);
            int8x8_t _r4n = vld1_s8(sptr + 4 * IW + 8);
            int8x8_t _r41 = vext_s8(_r40, _r4n, 1);
            int8x8_t _r42 = vext_s8(_r40, _r4n, 2);
            int8x8_t _r43 = vext_s8(_r40, _r4n, 3);
            int8x8_t _r44 = vext_s8(_r40, _r4n, 4);
            int8x8_t _r45 = vext_s8(_r40, _r4n, 5);
            int8x8_t _r46 = vext_s8(_r40, _r4n, 6);
            d0 = vmull_s8(_r40, k40);
            d0 = vmlal_s8(d0, _r41, k41);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r42, k42);
            d0 = vmlal_s8(d0, _r43, k43);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r44, k44);
            d0 = vmlal_s8(d0, _r45, k45);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r46, k46);
            d1 = vmlal_s8(d1, _r40, k30);
            ACC_S16_S32(sum10, sum11, d1);
            d1 = vmull_s8(_r41, k31);
            d1 = vmlal_s8(d1, _r42, k32);
            ACC_S16_S32(sum10, sum11, d1);
            d1 = vmull_s8(_r43, k33);
            d1 = vmlal_s8(d1, _r44, k34);
            ACC_S16_S32(sum10, sum11, d1);
            d1 = vmull_s8(_r45, k35);
            d1 = vmlal_s8(d1, _r46, k36);
            ACC_S16_S32(sum10, sum11, d1);

            int8x8_t _r50 = vld1_s8(sptr + 5 * IW);
            int8x8_t _r5n = vld1_s8(sptr + 5 * IW + 8);
            int8x8_t _r51 = vext_s8(_r50, _r5n, 1);
            int8x8_t _r52 = vext_s8(_r50, _r5n, 2);
            int8x8_t _r53 = vext_s8(_r50, _r5n, 3);
            int8x8_t _r54 = vext_s8(_r50, _r5n, 4);
            int8x8_t _r55 = vext_s8(_r50, _r5n, 5);
            int8x8_t _r56 = vext_s8(_r50, _r5n, 6);
            d0 = vmlal_s8(d0, _r50, k50);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r51, k51);
            d0 = vmlal_s8(d0, _r52, k52);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r53, k53);
            d0 = vmlal_s8(d0, _r54, k54);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r55, k55);
            d0 = vmlal_s8(d0, _r56, k56);
            ACC_S16_S32(sum00, sum01, d0);
            d1 = vmull_s8(_r50, k40);
            d1 = vmlal_s8(d1, _r51, k41);
            ACC_S16_S32(sum10, sum11, d1);
            d1 = vmull_s8(_r52, k42);
            d1 = vmlal_s8(d1, _r53, k43);
            ACC_S16_S32(sum10, sum11, d1);
            d1 = vmull_s8(_r54, k44);
            d1 = vmlal_s8(d1, _r55, k45);
            ACC_S16_S32(sum10, sum11, d1);
            d1 = vmull_s8(_r56, k46);

            int8x8_t _r60 = vld1_s8(sptr + 6 * IW);
            int8x8_t _r6n = vld1_s8(sptr + 6 * IW + 8);
            int8x8_t _r61 = vext_s8(_r60, _r6n, 1);
            int8x8_t _r62 = vext_s8(_r60, _r6n, 2);
            int8x8_t _r63 = vext_s8(_r60, _r6n, 3);
            int8x8_t _r64 = vext_s8(_r60, _r6n, 4);
            int8x8_t _r65 = vext_s8(_r60, _r6n, 5);
            int8x8_t _r66 = vext_s8(_r60, _r6n, 6);
            d0 = vmull_s8(_r60, k60);
            d0 = vmlal_s8(d0, _r61, k61);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r62, k62);
            d0 = vmlal_s8(d0, _r63, k63);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r64, k64);
            d0 = vmlal_s8(d0, _r65, k65);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r66, k66);
            ACC_S16_S32(sum00, sum01, d0);
            POSTPROCESS(sum00, sum01, tptr + 0 * OW, dptr + 0 * OW);
            d1 = vmlal_s8(d1, _r60, k50);
            ACC_S16_S32(sum10, sum11, d1);
            d1 = vmull_s8(_r61, k51);
            d1 = vmlal_s8(d1, _r62, k52);
            ACC_S16_S32(sum10, sum11, d1);
            d1 = vmull_s8(_r63, k53);
            d1 = vmlal_s8(d1, _r64, k54);
            ACC_S16_S32(sum10, sum11, d1);
            d1 = vmull_s8(_r65, k55);
            d1 = vmlal_s8(d1, _r66, k56);
            ACC_S16_S32(sum10, sum11, d1);

            int8x8_t _r70 = vld1_s8(sptr + 7 * IW);
            int8x8_t _r7n = vld1_s8(sptr + 7 * IW + 8);
            int8x8_t _r71 = vext_s8(_r70, _r7n, 1);
            int8x8_t _r72 = vext_s8(_r70, _r7n, 2);
            int8x8_t _r73 = vext_s8(_r70, _r7n, 3);
            int8x8_t _r74 = vext_s8(_r70, _r7n, 4);
            int8x8_t _r75 = vext_s8(_r70, _r7n, 5);
            int8x8_t _r76 = vext_s8(_r70, _r7n, 6);
            d1 = vmull_s8(_r70, k60);
            d1 = vmlal_s8(d1, _r71, k61);
            ACC_S16_S32(sum10, sum11, d1);
            d1 = vmull_s8(_r72, k62);
            d1 = vmlal_s8(d1, _r73, k63);
            ACC_S16_S32(sum10, sum11, d1);
            d1 = vmull_s8(_r74, k64);
            d1 = vmlal_s8(d1, _r75, k65);
            ACC_S16_S32(sum10, sum11, d1);
            d1 = vmull_s8(_r76, k66);
            ACC_S16_S32(sum10, sum11, d1);
            POSTPROCESS(sum10, sum11, tptr + 1 * OW, dptr + 1 * OW);
        }
    }

    if (oh < OH) {
        size_t ih = oh;
        for (size_t ow = 0; ow < OW; ow += 8) {
            size_t iw = ow;
            int32_t* __restrict tptr = temp + oh * OW + ow;
            int8_t* __restrict dptr = dst + oh * OW + ow;
            const int8_t* __restrict sptr = src + ih * IW + iw;
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
            }

            int8x8_t _r00 = vld1_s8(sptr + 0 * IW);
            int8x8_t _r0n = vld1_s8(sptr + 0 * IW + 8);
            int8x8_t _r01 = vext_s8(_r00, _r0n, 1);
            int8x8_t _r02 = vext_s8(_r00, _r0n, 2);
            int8x8_t _r03 = vext_s8(_r00, _r0n, 3);
            int8x8_t _r04 = vext_s8(_r00, _r0n, 4);
            int8x8_t _r05 = vext_s8(_r00, _r0n, 5);
            int8x8_t _r06 = vext_s8(_r00, _r0n, 6);
            int16x8_t d0 = vmull_s8(_r00, k00);
            d0 = vmlal_s8(d0, _r01, k01);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r02, k02);
            d0 = vmlal_s8(d0, _r03, k03);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r04, k04);
            d0 = vmlal_s8(d0, _r05, k05);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r06, k06);

            int8x8_t _r10 = vld1_s8(sptr + 1 * IW);
            int8x8_t _r1n = vld1_s8(sptr + 1 * IW + 8);
            int8x8_t _r11 = vext_s8(_r10, _r1n, 1);
            int8x8_t _r12 = vext_s8(_r10, _r1n, 2);
            int8x8_t _r13 = vext_s8(_r10, _r1n, 3);
            int8x8_t _r14 = vext_s8(_r10, _r1n, 4);
            int8x8_t _r15 = vext_s8(_r10, _r1n, 5);
            int8x8_t _r16 = vext_s8(_r10, _r1n, 6);
            d0 = vmlal_s8(d0, _r10, k10);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r11, k11);
            d0 = vmlal_s8(d0, _r12, k12);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r13, k13);
            d0 = vmlal_s8(d0, _r14, k14);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r15, k15);
            d0 = vmlal_s8(d0, _r16, k16);
            ACC_S16_S32(sum00, sum01, d0);

            int8x8_t _r20 = vld1_s8(sptr + 2 * IW);
            int8x8_t _r2n = vld1_s8(sptr + 2 * IW + 8);
            int8x8_t _r21 = vext_s8(_r20, _r2n, 1);
            int8x8_t _r22 = vext_s8(_r20, _r2n, 2);
            int8x8_t _r23 = vext_s8(_r20, _r2n, 3);
            int8x8_t _r24 = vext_s8(_r20, _r2n, 4);
            int8x8_t _r25 = vext_s8(_r20, _r2n, 5);
            int8x8_t _r26 = vext_s8(_r20, _r2n, 6);
            d0 = vmull_s8(_r20, k20);
            d0 = vmlal_s8(d0, _r21, k21);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r22, k22);
            d0 = vmlal_s8(d0, _r23, k23);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r24, k24);
            d0 = vmlal_s8(d0, _r25, k25);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r26, k26);

            int8x8_t _r30 = vld1_s8(sptr + 3 * IW);
            int8x8_t _r3n = vld1_s8(sptr + 3 * IW + 8);
            int8x8_t _r31 = vext_s8(_r30, _r3n, 1);
            int8x8_t _r32 = vext_s8(_r30, _r3n, 2);
            int8x8_t _r33 = vext_s8(_r30, _r3n, 3);
            int8x8_t _r34 = vext_s8(_r30, _r3n, 4);
            int8x8_t _r35 = vext_s8(_r30, _r3n, 5);
            int8x8_t _r36 = vext_s8(_r30, _r3n, 6);
            d0 = vmlal_s8(d0, _r30, k30);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r31, k31);
            d0 = vmlal_s8(d0, _r32, k32);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r33, k33);
            d0 = vmlal_s8(d0, _r34, k34);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r35, k35);
            d0 = vmlal_s8(d0, _r36, k36);
            ACC_S16_S32(sum00, sum01, d0);

            int8x8_t _r40 = vld1_s8(sptr + 4 * IW);
            int8x8_t _r4n = vld1_s8(sptr + 4 * IW + 8);
            int8x8_t _r41 = vext_s8(_r40, _r4n, 1);
            int8x8_t _r42 = vext_s8(_r40, _r4n, 2);
            int8x8_t _r43 = vext_s8(_r40, _r4n, 3);
            int8x8_t _r44 = vext_s8(_r40, _r4n, 4);
            int8x8_t _r45 = vext_s8(_r40, _r4n, 5);
            int8x8_t _r46 = vext_s8(_r40, _r4n, 6);
            d0 = vmull_s8(_r40, k40);
            d0 = vmlal_s8(d0, _r41, k41);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r42, k42);
            d0 = vmlal_s8(d0, _r43, k43);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r44, k44);
            d0 = vmlal_s8(d0, _r45, k45);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r46, k46);

            int8x8_t _r50 = vld1_s8(sptr + 5 * IW);
            int8x8_t _r5n = vld1_s8(sptr + 5 * IW + 8);
            int8x8_t _r51 = vext_s8(_r50, _r5n, 1);
            int8x8_t _r52 = vext_s8(_r50, _r5n, 2);
            int8x8_t _r53 = vext_s8(_r50, _r5n, 3);
            int8x8_t _r54 = vext_s8(_r50, _r5n, 4);
            int8x8_t _r55 = vext_s8(_r50, _r5n, 5);
            int8x8_t _r56 = vext_s8(_r50, _r5n, 6);
            d0 = vmlal_s8(d0, _r50, k50);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r51, k51);
            d0 = vmlal_s8(d0, _r52, k52);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r53, k53);
            d0 = vmlal_s8(d0, _r54, k54);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r55, k55);
            d0 = vmlal_s8(d0, _r56, k56);
            ACC_S16_S32(sum00, sum01, d0);

            int8x8_t _r60 = vld1_s8(sptr + 6 * IW);
            int8x8_t _r6n = vld1_s8(sptr + 6 * IW + 8);
            int8x8_t _r61 = vext_s8(_r60, _r6n, 1);
            int8x8_t _r62 = vext_s8(_r60, _r6n, 2);
            int8x8_t _r63 = vext_s8(_r60, _r6n, 3);
            int8x8_t _r64 = vext_s8(_r60, _r6n, 4);
            int8x8_t _r65 = vext_s8(_r60, _r6n, 5);
            int8x8_t _r66 = vext_s8(_r60, _r6n, 6);
            d0 = vmull_s8(_r60, k60);
            d0 = vmlal_s8(d0, _r61, k61);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r62, k62);
            d0 = vmlal_s8(d0, _r63, k63);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r64, k64);
            d0 = vmlal_s8(d0, _r65, k65);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r66, k66);
            ACC_S16_S32(sum00, sum01, d0);
            POSTPROCESS(sum00, sum01, tptr + 0 * OW, dptr + 0 * OW);
        }
    }
}

template <bool first_ic, bool last_ic, BiasMode bias_mode, typename Op>
void conv_bias::conv_direct_stride2_2x2_int8_nchw(const int8_t* src,
                                             const int8_t* filter,
                                             const int32_t* bias, int32_t* temp,
                                             int8_t* dst, const size_t IH,
                                             const size_t IW, const size_t OH,
                                             const size_t OW, const Op& op) {
    MEGDNN_MARK_USED_VAR(IH);
#define GET_R2(sptr)                                                      \
    _r00 = vld1_s8(sptr);                                                 \
    _r00 = vtbl1_s8(_r00, _idx);                                          \
    _r01 = vld1_s8(sptr + 8);                                             \
    _r01 = vtbl1_s8(_r01, _idx);                                          \
    _rn = vzip_s32(vreinterpret_s32_s8(_r00), vreinterpret_s32_s8(_r01)); \
    _r00 = vreinterpret_s8_s32(_rn.val[0]);                               \
    _r01 = vreinterpret_s8_s32(_rn.val[1]);

    int8x8_t k00 = vdup_n_s8(filter[0]);
    int8x8_t k01 = vdup_n_s8(filter[1]);
    int8x8_t k10 = vdup_n_s8(filter[2]);
    int8x8_t k11 = vdup_n_s8(filter[3]);

    int8x8_t _idx = {0, 2, 4, 6, 1, 3, 5, 7};
    size_t oh = 0;
    for (; oh < OH; ++oh) {
        size_t ih = oh * 2;
        for (size_t ow = 0; ow < OW; ow += 8) {
            size_t iw = ow * 2;
            int32_t* __restrict tptr = temp + oh * OW + ow;
            int8_t* __restrict dptr = dst + oh * OW + ow;
            const int8_t* __restrict sptr = src + ih * IW + iw;
            const int32_t* __restrict bptr = bias;
            int32x4_t sum00, sum01;
            int16x8_t d0;
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
            }

            GET_R2(sptr);
            d0 = vmull_s8(_r00, k00);
            d0 = vmlal_s8(d0, _r01, k01);
            ACC_S16_S32(sum00, sum01, d0);

            GET_R2(sptr + IW);
            d0 = vmull_s8(_r00, k10);
            d0 = vmlal_s8(d0, _r01, k11);
            ACC_S16_S32(sum00, sum01, d0);

            POSTPROCESS(sum00, sum01, tptr + 0 * OW, dptr + 0 * OW);
        }
    }
#undef GET_R2
}

template <bool first_ic, bool last_ic, BiasMode bias_mode, typename Op>
void conv_bias::conv_direct_stride2_3x3_int8_nchw(const int8_t* src,
                                             const int8_t* filter,
                                             const int32_t* bias, int32_t* temp,
                                             int8_t* dst, const size_t IH,
                                             const size_t IW, const size_t OH,
                                             const size_t OW, const Op& op) {
    MEGDNN_MARK_USED_VAR(IH);
#define GET_R3(sptr)                                                      \
    _r00 = vld1_s8(sptr);                                                 \
    _r00 = vtbl1_s8(_r00, _idx);                                          \
    _r01 = vld1_s8(sptr + 8);                                             \
    _r01 = vtbl1_s8(_r01, _idx);                                          \
    _rn = vzip_s32(vreinterpret_s32_s8(_r00), vreinterpret_s32_s8(_r01)); \
    _r00 = vreinterpret_s8_s32(_rn.val[0]);                               \
    _r01 = vreinterpret_s8_s32(_rn.val[1]);                               \
    _r02 = vld1_s8(sptr + 16);                                            \
    _r02 = vext_s8(_r00, _r02, 1);

    int8x8_t k00 = vdup_n_s8(filter[0]);
    int8x8_t k01 = vdup_n_s8(filter[1]);
    int8x8_t k02 = vdup_n_s8(filter[2]);
    int8x8_t k10 = vdup_n_s8(filter[3]);
    int8x8_t k11 = vdup_n_s8(filter[4]);
    int8x8_t k12 = vdup_n_s8(filter[5]);
    int8x8_t k20 = vdup_n_s8(filter[6]);
    int8x8_t k21 = vdup_n_s8(filter[7]);
    int8x8_t k22 = vdup_n_s8(filter[8]);

    int8x8_t _idx = {0, 2, 4, 6, 1, 3, 5, 7};

    // block 2x8
    size_t oh = 0;
    for (; oh + 1 < OH; oh += 2) {
        size_t ih = oh * 2;
        for (size_t ow = 0; ow < OW; ow += 8) {
            size_t iw = ow * 2;
            int32_t* __restrict tptr = temp + oh * OW + ow;
            int8_t* __restrict dptr = dst + oh * OW + ow;
            const int8_t* __restrict sptr = src + ih * IW + iw;
            const int32_t* __restrict bptr = bias;
            int32x4_t sum00, sum01, sum10, sum11;
            int16x8_t d0, d1;
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
            }
            GET_R3(sptr);
            d0 = vmull_s8(_r00, k00);
            d0 = vmlal_s8(d0, _r01, k01);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r02, k02);

            GET_R3(sptr + IW);
            d0 = vmlal_s8(d0, _r00, k10);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r01, k11);
            d0 = vmlal_s8(d0, _r02, k12);
            ACC_S16_S32(sum00, sum01, d0);

            GET_R3(sptr + 2 * IW);
            d0 = vmull_s8(_r00, k20);
            d0 = vmlal_s8(d0, _r01, k21);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r02, k22);
            ACC_S16_S32(sum00, sum01, d0);
            POSTPROCESS(sum00, sum01, tptr + 0 * OW, dptr + 0 * OW);
            d1 = vmull_s8(_r00, k00);
            d1 = vmlal_s8(d1, _r01, k01);
            ACC_S16_S32(sum10, sum11, d1);
            d1 = vmull_s8(_r02, k02);

            GET_R3(sptr + 3 * IW);
            d1 = vmlal_s8(d1, _r00, k10);
            ACC_S16_S32(sum10, sum11, d1);
            d1 = vmull_s8(_r01, k11);
            d1 = vmlal_s8(d1, _r02, k12);
            ACC_S16_S32(sum10, sum11, d1);

            GET_R3(sptr + 4 * IW);
            d1 = vmull_s8(_r00, k20);
            d1 = vmlal_s8(d1, _r01, k21);
            ACC_S16_S32(sum10, sum11, d1);
            d1 = vmull_s8(_r02, k22);
            ACC_S16_S32(sum10, sum11, d1);
            POSTPROCESS(sum10, sum11, tptr + 1 * OW, dptr + 1 * OW);
        }
    }

    if (oh < OH) {
        size_t ih = oh * 2;
        for (size_t ow = 0; ow < OW; ow += 8) {
            size_t iw = ow * 2;
            int32_t* __restrict tptr = temp + oh * OW + ow;
            int8_t* __restrict dptr = dst + oh * OW + ow;
            const int8_t* __restrict sptr = src + ih * IW + iw;
            const int32_t* __restrict bptr = bias;
            int32x4_t sum00, sum01;
            int16x8_t d0;
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
            }
            GET_R3(sptr);
            d0 = vmull_s8(_r00, k00);
            d0 = vmlal_s8(d0, _r01, k01);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r02, k02);

            GET_R3(sptr + IW);
            d0 = vmlal_s8(d0, _r00, k10);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r01, k11);
            d0 = vmlal_s8(d0, _r02, k12);
            ACC_S16_S32(sum00, sum01, d0);

            GET_R3(sptr + 2 * IW);
            d0 = vmull_s8(_r00, k20);
            d0 = vmlal_s8(d0, _r01, k21);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r02, k22);
            ACC_S16_S32(sum00, sum01, d0);

            POSTPROCESS(sum00, sum01, tptr + 0 * OW, dptr + 0 * OW);
        }
    }
#undef GET_R3
}

template <bool first_ic, bool last_ic, BiasMode bias_mode, typename Op>
void conv_bias::conv_direct_stride2_5x5_int8_nchw(const int8_t* src,
                                             const int8_t* filter,
                                             const int32_t* bias, int32_t* temp,
                                             int8_t* dst, const size_t IH,
                                             const size_t IW, const size_t OH,
                                             const size_t OW, const Op& op) {
    MEGDNN_MARK_USED_VAR(IH);
#define GET_R5(sptr)                                                      \
    _r00 = vld1_s8(sptr);                                                 \
    _r00 = vtbl1_s8(_r00, _idx);                                          \
    _r01 = vld1_s8(sptr + 8);                                             \
    _r01 = vtbl1_s8(_r01, _idx);                                          \
    _rn = vzip_s32(vreinterpret_s32_s8(_r00), vreinterpret_s32_s8(_r01)); \
    _r00 = vreinterpret_s8_s32(_rn.val[0]);                               \
    _r01 = vreinterpret_s8_s32(_rn.val[1]);                               \
    _r03 = vld1_s8(sptr + 16);                                            \
    _r03 = vtbl1_s8(_r03, _idx);                                          \
    _r02 = vext_s8(_r00, _r03, 1);                                        \
    _r04 = vext_s8(_r00, _r03, 2);                                        \
    _r03 = vtbl1_s8(_r03, _idxn);                                         \
    _r03 = vext_s8(_r01, _r03, 1);

    int8x8_t k00 = vdup_n_s8(filter[0]);
    int8x8_t k01 = vdup_n_s8(filter[1]);
    int8x8_t k02 = vdup_n_s8(filter[2]);
    int8x8_t k03 = vdup_n_s8(filter[3]);
    int8x8_t k04 = vdup_n_s8(filter[4]);
    int8x8_t k10 = vdup_n_s8(filter[5]);
    int8x8_t k11 = vdup_n_s8(filter[6]);
    int8x8_t k12 = vdup_n_s8(filter[7]);
    int8x8_t k13 = vdup_n_s8(filter[8]);
    int8x8_t k14 = vdup_n_s8(filter[9]);
    int8x8_t k20 = vdup_n_s8(filter[10]);
    int8x8_t k21 = vdup_n_s8(filter[11]);
    int8x8_t k22 = vdup_n_s8(filter[12]);
    int8x8_t k23 = vdup_n_s8(filter[13]);
    int8x8_t k24 = vdup_n_s8(filter[14]);
    int8x8_t k30 = vdup_n_s8(filter[15]);
    int8x8_t k31 = vdup_n_s8(filter[16]);
    int8x8_t k32 = vdup_n_s8(filter[17]);
    int8x8_t k33 = vdup_n_s8(filter[18]);
    int8x8_t k34 = vdup_n_s8(filter[19]);
    int8x8_t k40 = vdup_n_s8(filter[20]);
    int8x8_t k41 = vdup_n_s8(filter[21]);
    int8x8_t k42 = vdup_n_s8(filter[22]);
    int8x8_t k43 = vdup_n_s8(filter[23]);
    int8x8_t k44 = vdup_n_s8(filter[24]);

    int8x8_t _idx = {0, 2, 4, 6, 1, 3, 5, 7};
    int8x8_t _idxn = {4, 5, 6, 7, 0, 1, 2, 3};

    // block 2x8
    size_t oh = 0;
    for (; oh + 1 < OH; oh += 2) {
        size_t ih = oh * 2;
        for (size_t ow = 0; ow < OW; ow += 8) {
            size_t iw = ow * 2;
            int32_t* __restrict tptr = temp + oh * OW + ow;
            int8_t* __restrict dptr = dst + oh * OW + ow;
            const int8_t* __restrict sptr = src + ih * IW + iw;
            const int32_t* __restrict bptr = bias;
            int32x4_t sum00, sum01, sum10, sum11;
            int16x8_t d0, d1;
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
            }
            GET_R5(sptr);
            d0 = vmull_s8(_r00, k00);
            d0 = vmlal_s8(d0, _r01, k01);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r02, k02);
            d0 = vmlal_s8(d0, _r03, k03);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r04, k04);

            GET_R5(sptr + IW);
            d0 = vmlal_s8(d0, _r00, k10);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r01, k11);
            d0 = vmlal_s8(d0, _r02, k12);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r03, k13);
            d0 = vmlal_s8(d0, _r04, k14);
            ACC_S16_S32(sum00, sum01, d0);

            GET_R5(sptr + 2 * IW);
            d0 = vmull_s8(_r00, k20);
            d0 = vmlal_s8(d0, _r01, k21);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r02, k22);
            d0 = vmlal_s8(d0, _r03, k23);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r04, k24);
            d1 = vmull_s8(_r00, k00);
            d1 = vmlal_s8(d1, _r01, k01);
            ACC_S16_S32(sum10, sum11, d1);
            d1 = vmull_s8(_r02, k02);
            d1 = vmlal_s8(d1, _r03, k03);
            ACC_S16_S32(sum10, sum11, d1);
            d1 = vmull_s8(_r04, k04);

            GET_R5(sptr + 3 * IW);
            d0 = vmlal_s8(d0, _r00, k30);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r01, k31);
            d0 = vmlal_s8(d0, _r02, k32);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r03, k33);
            d0 = vmlal_s8(d0, _r04, k34);
            ACC_S16_S32(sum00, sum01, d0);
            d1 = vmlal_s8(d1, _r00, k10);
            ACC_S16_S32(sum10, sum11, d1);
            d1 = vmull_s8(_r01, k11);
            d1 = vmlal_s8(d1, _r02, k12);
            ACC_S16_S32(sum10, sum11, d1);
            d1 = vmull_s8(_r03, k13);
            d1 = vmlal_s8(d1, _r04, k14);
            ACC_S16_S32(sum10, sum11, d1);

            GET_R5(sptr + 4 * IW);
            d0 = vmull_s8(_r00, k40);
            d0 = vmlal_s8(d0, _r01, k41);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r02, k42);
            d0 = vmlal_s8(d0, _r03, k43);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r04, k44);
            ACC_S16_S32(sum00, sum01, d0);
            POSTPROCESS(sum00, sum01, tptr + 0 * OW, dptr + 0 * OW);
            d1 = vmull_s8(_r00, k20);
            d1 = vmlal_s8(d1, _r01, k21);
            ACC_S16_S32(sum10, sum11, d1);
            d1 = vmull_s8(_r02, k22);
            d1 = vmlal_s8(d1, _r03, k23);
            ACC_S16_S32(sum10, sum11, d1);
            d1 = vmull_s8(_r04, k24);

            GET_R5(sptr + 5 * IW);
            d1 = vmlal_s8(d1, _r00, k30);
            ACC_S16_S32(sum10, sum11, d1);
            d1 = vmull_s8(_r01, k31);
            d1 = vmlal_s8(d1, _r02, k32);
            ACC_S16_S32(sum10, sum11, d1);
            d1 = vmull_s8(_r03, k33);
            d1 = vmlal_s8(d1, _r04, k34);
            ACC_S16_S32(sum10, sum11, d1);

            GET_R5(sptr + 6 * IW);
            d1 = vmull_s8(_r00, k40);
            d1 = vmlal_s8(d1, _r01, k41);
            ACC_S16_S32(sum10, sum11, d1);
            d1 = vmull_s8(_r02, k42);
            d1 = vmlal_s8(d1, _r03, k43);
            ACC_S16_S32(sum10, sum11, d1);
            d1 = vmull_s8(_r04, k44);
            ACC_S16_S32(sum10, sum11, d1);
            POSTPROCESS(sum10, sum11, tptr + 1 * OW, dptr + 1 * OW);
        }
    }

    if (oh < OH) {
        size_t ih = oh * 2;
        for (size_t ow = 0; ow < OW; ow += 8) {
            size_t iw = ow * 2;
            int32_t* __restrict tptr = temp + oh * OW + ow;
            int8_t* __restrict dptr = dst + oh * OW + ow;
            const int8_t* __restrict sptr = src + ih * IW + iw;
            const int32_t* __restrict bptr = bias;
            int32x4_t sum00, sum01;
            int16x8_t d0;
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
            }
            GET_R5(sptr);
            d0 = vmull_s8(_r00, k00);
            d0 = vmlal_s8(d0, _r01, k01);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r02, k02);
            d0 = vmlal_s8(d0, _r03, k03);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r04, k04);

            GET_R5(sptr + IW);
            d0 = vmlal_s8(d0, _r00, k10);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r01, k11);
            d0 = vmlal_s8(d0, _r02, k12);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r03, k13);
            d0 = vmlal_s8(d0, _r04, k14);
            ACC_S16_S32(sum00, sum01, d0);

            GET_R5(sptr + 2 * IW);
            d0 = vmull_s8(_r00, k20);
            d0 = vmlal_s8(d0, _r01, k21);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r02, k22);
            d0 = vmlal_s8(d0, _r03, k23);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r04, k24);

            GET_R5(sptr + 3 * IW);
            d0 = vmlal_s8(d0, _r00, k30);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r01, k31);
            d0 = vmlal_s8(d0, _r02, k32);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r03, k33);
            d0 = vmlal_s8(d0, _r04, k34);
            ACC_S16_S32(sum00, sum01, d0);

            GET_R5(sptr + 4 * IW);
            d0 = vmull_s8(_r00, k40);
            d0 = vmlal_s8(d0, _r01, k41);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r02, k42);
            d0 = vmlal_s8(d0, _r03, k43);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r04, k44);
            ACC_S16_S32(sum00, sum01, d0);

            POSTPROCESS(sum00, sum01, tptr + 0 * OW, dptr + 0 * OW);
        }
    }
#undef GET_R5
}

template <bool first_ic, bool last_ic, BiasMode bias_mode, typename Op>
void conv_bias::conv_direct_stride2_7x7_int8_nchw(const int8_t* src,
                                             const int8_t* filter,
                                             const int32_t* bias, int32_t* temp,
                                             int8_t* dst, const size_t IH,
                                             const size_t IW, const size_t OH,
                                             const size_t OW, const Op& op) {
    MEGDNN_MARK_USED_VAR(IH);
#define GET_R7(sptr)                                                      \
    _r00 = vld1_s8(sptr);                                                 \
    _r00 = vtbl1_s8(_r00, _idx);                                          \
    _r01 = vld1_s8(sptr + 8);                                             \
    _r01 = vtbl1_s8(_r01, _idx);                                          \
    _rn = vzip_s32(vreinterpret_s32_s8(_r00), vreinterpret_s32_s8(_r01)); \
    _r00 = vreinterpret_s8_s32(_rn.val[0]);                               \
    _r01 = vreinterpret_s8_s32(_rn.val[1]);                               \
    _r05 = vld1_s8(sptr + 16);                                            \
    _r05 = vtbl1_s8(_r05, _idx);                                          \
    _r02 = vext_s8(_r00, _r05, 1);                                        \
    _r04 = vext_s8(_r00, _r05, 2);                                        \
    _r06 = vext_s8(_r00, _r05, 3);                                        \
    _r05 = vtbl1_s8(_r05, _idxn);                                         \
    _r03 = vext_s8(_r01, _r05, 1);                                        \
    _r05 = vext_s8(_r01, _r05, 2);

    int8x8_t k00 = vdup_n_s8(filter[0]);
    int8x8_t k01 = vdup_n_s8(filter[1]);
    int8x8_t k02 = vdup_n_s8(filter[2]);
    int8x8_t k03 = vdup_n_s8(filter[3]);
    int8x8_t k04 = vdup_n_s8(filter[4]);
    int8x8_t k05 = vdup_n_s8(filter[5]);
    int8x8_t k06 = vdup_n_s8(filter[6]);

    int8x8_t k10 = vdup_n_s8(filter[7]);
    int8x8_t k11 = vdup_n_s8(filter[8]);
    int8x8_t k12 = vdup_n_s8(filter[9]);
    int8x8_t k13 = vdup_n_s8(filter[10]);
    int8x8_t k14 = vdup_n_s8(filter[11]);
    int8x8_t k15 = vdup_n_s8(filter[12]);
    int8x8_t k16 = vdup_n_s8(filter[13]);

    int8x8_t k20 = vdup_n_s8(filter[14]);
    int8x8_t k21 = vdup_n_s8(filter[15]);
    int8x8_t k22 = vdup_n_s8(filter[16]);
    int8x8_t k23 = vdup_n_s8(filter[17]);
    int8x8_t k24 = vdup_n_s8(filter[18]);
    int8x8_t k25 = vdup_n_s8(filter[19]);
    int8x8_t k26 = vdup_n_s8(filter[20]);

    int8x8_t k30 = vdup_n_s8(filter[21]);
    int8x8_t k31 = vdup_n_s8(filter[22]);
    int8x8_t k32 = vdup_n_s8(filter[23]);
    int8x8_t k33 = vdup_n_s8(filter[24]);
    int8x8_t k34 = vdup_n_s8(filter[25]);
    int8x8_t k35 = vdup_n_s8(filter[26]);
    int8x8_t k36 = vdup_n_s8(filter[27]);

    int8x8_t k40 = vdup_n_s8(filter[28]);
    int8x8_t k41 = vdup_n_s8(filter[29]);
    int8x8_t k42 = vdup_n_s8(filter[30]);
    int8x8_t k43 = vdup_n_s8(filter[31]);
    int8x8_t k44 = vdup_n_s8(filter[32]);
    int8x8_t k45 = vdup_n_s8(filter[33]);
    int8x8_t k46 = vdup_n_s8(filter[34]);

    int8x8_t k50 = vdup_n_s8(filter[35]);
    int8x8_t k51 = vdup_n_s8(filter[36]);
    int8x8_t k52 = vdup_n_s8(filter[37]);
    int8x8_t k53 = vdup_n_s8(filter[38]);
    int8x8_t k54 = vdup_n_s8(filter[39]);
    int8x8_t k55 = vdup_n_s8(filter[40]);
    int8x8_t k56 = vdup_n_s8(filter[41]);

    int8x8_t k60 = vdup_n_s8(filter[42]);
    int8x8_t k61 = vdup_n_s8(filter[43]);
    int8x8_t k62 = vdup_n_s8(filter[44]);
    int8x8_t k63 = vdup_n_s8(filter[45]);
    int8x8_t k64 = vdup_n_s8(filter[46]);
    int8x8_t k65 = vdup_n_s8(filter[47]);
    int8x8_t k66 = vdup_n_s8(filter[48]);

    int8x8_t _idx = {0, 2, 4, 6, 1, 3, 5, 7};
    int8x8_t _idxn = {4, 5, 6, 7, 0, 1, 2, 3};

    // block 2x8
    size_t oh = 0;
    for (; oh + 1 < OH; oh += 2) {
        size_t ih = oh * 2;
        for (size_t ow = 0; ow < OW; ow += 8) {
            size_t iw = ow * 2;
            int32_t* __restrict tptr = temp + oh * OW + ow;
            int8_t* __restrict dptr = dst + oh * OW + ow;
            const int8_t* __restrict sptr = src + ih * IW + iw;
            const int32_t* __restrict bptr = bias;
            int32x4_t sum00, sum01, sum10, sum11;
            int16x8_t d0, d1;
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
            }
            GET_R7(sptr);
            d0 = vmull_s8(_r00, k00);
            d0 = vmlal_s8(d0, _r01, k01);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r02, k02);
            d0 = vmlal_s8(d0, _r03, k03);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r04, k04);
            d0 = vmlal_s8(d0, _r05, k05);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r06, k06);

            GET_R7(sptr + IW);
            d0 = vmlal_s8(d0, _r00, k10);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r01, k11);
            d0 = vmlal_s8(d0, _r02, k12);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r03, k13);
            d0 = vmlal_s8(d0, _r04, k14);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r05, k15);
            d0 = vmlal_s8(d0, _r06, k16);
            ACC_S16_S32(sum00, sum01, d0);

            GET_R7(sptr + 2 * IW);
            d0 = vmull_s8(_r00, k20);
            d0 = vmlal_s8(d0, _r01, k21);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r02, k22);
            d0 = vmlal_s8(d0, _r03, k23);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r04, k24);
            d0 = vmlal_s8(d0, _r05, k25);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r06, k26);
            d1 = vmull_s8(_r00, k00);
            d1 = vmlal_s8(d1, _r01, k01);
            ACC_S16_S32(sum10, sum11, d1);
            d1 = vmull_s8(_r02, k02);
            d1 = vmlal_s8(d1, _r03, k03);
            ACC_S16_S32(sum10, sum11, d1);
            d1 = vmull_s8(_r04, k04);
            d1 = vmlal_s8(d1, _r05, k05);
            ACC_S16_S32(sum10, sum11, d1);
            d1 = vmull_s8(_r06, k06);

            GET_R7(sptr + 3 * IW);
            d0 = vmlal_s8(d0, _r00, k30);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r01, k31);
            d0 = vmlal_s8(d0, _r02, k32);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r03, k33);
            d0 = vmlal_s8(d0, _r04, k34);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r05, k35);
            d0 = vmlal_s8(d0, _r06, k36);
            ACC_S16_S32(sum00, sum01, d0);
            d1 = vmlal_s8(d1, _r00, k10);
            ACC_S16_S32(sum10, sum11, d1);
            d1 = vmull_s8(_r01, k11);
            d1 = vmlal_s8(d1, _r02, k12);
            ACC_S16_S32(sum10, sum11, d1);
            d1 = vmull_s8(_r03, k13);
            d1 = vmlal_s8(d1, _r04, k14);
            ACC_S16_S32(sum10, sum11, d1);
            d1 = vmull_s8(_r05, k15);
            d1 = vmlal_s8(d1, _r06, k16);
            ACC_S16_S32(sum10, sum11, d1);

            GET_R7(sptr + 4 * IW);
            d0 = vmull_s8(_r00, k40);
            d0 = vmlal_s8(d0, _r01, k41);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r02, k42);
            d0 = vmlal_s8(d0, _r03, k43);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r04, k44);
            d0 = vmlal_s8(d0, _r05, k45);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r06, k46);
            d1 = vmull_s8(_r00, k20);
            d1 = vmlal_s8(d1, _r01, k21);
            ACC_S16_S32(sum10, sum11, d1);
            d1 = vmull_s8(_r02, k22);
            d1 = vmlal_s8(d1, _r03, k23);
            ACC_S16_S32(sum10, sum11, d1);
            d1 = vmull_s8(_r04, k24);
            d1 = vmlal_s8(d1, _r05, k25);
            ACC_S16_S32(sum10, sum11, d1);
            d1 = vmull_s8(_r06, k26);

            GET_R7(sptr + 5 * IW);
            d0 = vmlal_s8(d0, _r00, k50);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r01, k51);
            d0 = vmlal_s8(d0, _r02, k52);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r03, k53);
            d0 = vmlal_s8(d0, _r04, k54);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r05, k55);
            d0 = vmlal_s8(d0, _r06, k56);
            ACC_S16_S32(sum00, sum01, d0);
            d1 = vmlal_s8(d1, _r00, k30);
            ACC_S16_S32(sum10, sum11, d1);
            d1 = vmull_s8(_r01, k31);
            d1 = vmlal_s8(d1, _r02, k32);
            ACC_S16_S32(sum10, sum11, d1);
            d1 = vmull_s8(_r03, k33);
            d1 = vmlal_s8(d1, _r04, k34);
            ACC_S16_S32(sum10, sum11, d1);
            d1 = vmull_s8(_r05, k35);
            d1 = vmlal_s8(d1, _r06, k36);
            ACC_S16_S32(sum10, sum11, d1);

            GET_R7(sptr + 6 * IW);
            d0 = vmull_s8(_r00, k60);
            d0 = vmlal_s8(d0, _r01, k61);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r02, k62);
            d0 = vmlal_s8(d0, _r03, k63);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r04, k64);
            d0 = vmlal_s8(d0, _r05, k65);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r06, k66);
            ACC_S16_S32(sum00, sum01, d0);
            POSTPROCESS(sum00, sum01, tptr + 0 * OW, dptr + 0 * OW);
            d1 = vmull_s8(_r00, k40);
            d1 = vmlal_s8(d1, _r01, k41);
            ACC_S16_S32(sum10, sum11, d1);
            d1 = vmull_s8(_r02, k42);
            d1 = vmlal_s8(d1, _r03, k43);
            ACC_S16_S32(sum10, sum11, d1);
            d1 = vmull_s8(_r04, k44);
            d1 = vmlal_s8(d1, _r05, k45);
            ACC_S16_S32(sum10, sum11, d1);
            d1 = vmull_s8(_r06, k46);

            GET_R7(sptr + 7 * IW);
            d1 = vmlal_s8(d1, _r00, k50);
            ACC_S16_S32(sum10, sum11, d1);
            d1 = vmull_s8(_r01, k51);
            d1 = vmlal_s8(d1, _r02, k52);
            ACC_S16_S32(sum10, sum11, d1);
            d1 = vmull_s8(_r03, k53);
            d1 = vmlal_s8(d1, _r04, k54);
            ACC_S16_S32(sum10, sum11, d1);
            d1 = vmull_s8(_r05, k55);
            d1 = vmlal_s8(d1, _r06, k56);
            ACC_S16_S32(sum10, sum11, d1);

            GET_R7(sptr + 8 * IW);
            d1 = vmull_s8(_r00, k60);
            d1 = vmlal_s8(d1, _r01, k61);
            ACC_S16_S32(sum10, sum11, d1);
            d1 = vmull_s8(_r02, k62);
            d1 = vmlal_s8(d1, _r03, k63);
            ACC_S16_S32(sum10, sum11, d1);
            d1 = vmull_s8(_r04, k64);
            d1 = vmlal_s8(d1, _r05, k65);
            ACC_S16_S32(sum10, sum11, d1);
            d1 = vmull_s8(_r06, k66);
            ACC_S16_S32(sum10, sum11, d1);
            POSTPROCESS(sum10, sum11, tptr + 1 * OW, dptr + 1 * OW);
        }
    }

    if (oh < OH) {
        size_t ih = oh * 2;
        for (size_t ow = 0; ow < OW; ow += 8) {
            size_t iw = ow * 2;
            int32_t* __restrict tptr = temp + oh * OW + ow;
            int8_t* __restrict dptr = dst + oh * OW + ow;
            const int8_t* __restrict sptr = src + ih * IW + iw;
            const int32_t* __restrict bptr = bias;
            int32x4_t sum00, sum01;
            int16x8_t d0;
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
            }

            GET_R7(sptr);
            d0 = vmull_s8(_r00, k00);
            d0 = vmlal_s8(d0, _r01, k01);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r02, k02);
            d0 = vmlal_s8(d0, _r03, k03);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r04, k04);
            d0 = vmlal_s8(d0, _r05, k05);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r06, k06);

            GET_R7(sptr + IW);
            d0 = vmlal_s8(d0, _r00, k10);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r01, k11);
            d0 = vmlal_s8(d0, _r02, k12);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r03, k13);
            d0 = vmlal_s8(d0, _r04, k14);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r05, k15);
            d0 = vmlal_s8(d0, _r06, k16);
            ACC_S16_S32(sum00, sum01, d0);

            GET_R7(sptr + 2 * IW);
            d0 = vmull_s8(_r00, k20);
            d0 = vmlal_s8(d0, _r01, k21);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r02, k22);
            d0 = vmlal_s8(d0, _r03, k23);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r04, k24);
            d0 = vmlal_s8(d0, _r05, k25);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r06, k26);

            GET_R7(sptr + 3 * IW);
            d0 = vmlal_s8(d0, _r00, k30);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r01, k31);
            d0 = vmlal_s8(d0, _r02, k32);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r03, k33);
            d0 = vmlal_s8(d0, _r04, k34);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r05, k35);
            d0 = vmlal_s8(d0, _r06, k36);
            ACC_S16_S32(sum00, sum01, d0);

            GET_R7(sptr + 4 * IW);
            d0 = vmull_s8(_r00, k40);
            d0 = vmlal_s8(d0, _r01, k41);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r02, k42);
            d0 = vmlal_s8(d0, _r03, k43);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r04, k44);
            d0 = vmlal_s8(d0, _r05, k45);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r06, k46);

            GET_R7(sptr + 5 * IW);
            d0 = vmlal_s8(d0, _r00, k50);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r01, k51);
            d0 = vmlal_s8(d0, _r02, k52);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r03, k53);
            d0 = vmlal_s8(d0, _r04, k54);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r05, k55);
            d0 = vmlal_s8(d0, _r06, k56);
            ACC_S16_S32(sum00, sum01, d0);

            GET_R7(sptr + 6 * IW);
            d0 = vmull_s8(_r00, k60);
            d0 = vmlal_s8(d0, _r01, k61);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r02, k62);
            d0 = vmlal_s8(d0, _r03, k63);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r04, k64);
            d0 = vmlal_s8(d0, _r05, k65);
            ACC_S16_S32(sum00, sum01, d0);
            d0 = vmull_s8(_r06, k66);
            ACC_S16_S32(sum00, sum01, d0);

            POSTPROCESS(sum00, sum01, tptr + 0 * OW, dptr + 0 * OW);
        }
    }
#undef GET_R7
}

#undef POSTPROCESS
#undef ACC_S16_S32

#define INSTANTIATION(stride, i, first_ic, last_ic, bias, Op)                \
    template void conv_bias::conv_direct_##stride##_##i##x##i##_int8_nchw<   \
            first_ic, last_ic, bias, Op>(                                    \
            const int8_t*, const int8_t*, const int32_t*, int32_t*, int8_t*, \
            const size_t, const size_t, const size_t, const size_t,          \
            const Op&);

#define FOR_OP(stride, i, first_ic, last_ic, bias)            \
    INSTANTIATION(stride, i, first_ic, last_ic, bias,         \
                  TypeCvtOp<dt_qint32 MEGDNN_COMMA dt_qint8>) \
    INSTANTIATION(stride, i, first_ic, last_ic, bias,         \
                  ReluOp<dt_qint32 MEGDNN_COMMA dt_qint8>)    \
    INSTANTIATION(stride, i, first_ic, last_ic, bias,         \
                  HSwishOp<dt_qint32 MEGDNN_COMMA dt_qint8>)

#define FOR_BIAS(stride, i, first_ic, last_ic)              \
    FOR_OP(stride, i, first_ic, last_ic, BiasMode::NO_BIAS) \
    FOR_OP(stride, i, first_ic, last_ic, BiasMode::BROADCAST_CHANNEL_BIAS)

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
