/**
 * \file dnn/src/arm_common/conv_bias/quint8/direct_dotprod.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#if __ARM_FEATURE_DOTPROD
#include "src/arm_common/conv_bias/quint8/direct_dotprod.h"
#include "src/arm_common/elemwise_op.h"
#include "src/arm_common/simd_macro/marm_neon.h"
#include "src/common/utils.h"
#include "src/fallback/conv_bias/common.h"

using namespace megdnn;
using namespace arm_common;
using megdnn::arm_common::ReluOp;
using megdnn::arm_common::TypeCvtOp;

constexpr int32_t SHIFT = (1 << 30);

inline int8x16_t vqtbl1q_s8_v7(int8x16_t a, uint8x16_t index){
   int8x8x2_t src;
   src.val[0] = vget_low_s8(a);
   src.val[1] = vget_high_s8(a);
   uint8x8_t index_low  = vget_low_u8(index);
   uint8x8_t index_high = vget_high_u8(index);
   int8x8_t r00 = vtbl2_s8(src,vreinterpret_s8_u8(index_low)) ;
   int8x8_t r01 = vtbl2_s8(src,vreinterpret_s8_u8(index_high));
   int8x16_t r =  vcombine_s8(r00,r01);
   return r;
}

#define ST1_S32X4(dst0, tptr) vst1q_u32(tptr, dst0);

#define ST2_S32X4X2(dst0, tptr) vst2q_u32(tptr, dst0);

#define POSTPROCESS_1X8(dst0, dst1, tptr, dptr)                          \
    if (last_ic && fused_kern) {                                         \
        op({{vreinterpretq_u32_s32(dst0), vreinterpretq_u32_s32(dst1)}}, \
           reinterpret_cast<dt_quint8*>(dptr));                          \
    } else {                                                             \
        ST1_S32X4(dst0, tptr);                                           \
        ST1_S32X4(dst1, tptr + 4);                                       \
    }

#define POSTPROCESS2_1X8(dst0, tptr, dptr)                                  \
    if (last_ic && fused_kern) {                                            \
       uint32x4x2_t temp;                                                   \
       uint32x4_t temp00, temp11;                                           \
       temp = vzipq_u32(dst0.val[0], dst0.val[1]);                          \
       temp00 = temp.val[0];                                                \
       temp11 = temp.val[1];                                                \
       op({{temp00,temp11}},reinterpret_cast<dt_quint8*>(dptr));            \
    } else {                                                                \
        ST2_S32X4X2(dst0, tptr);                                            \
    }

#define POSTPROCESS_2X4(dst0, dst1, tptr1, tptr2, dptr1, dptr2)    \
    if (last_ic && fused_kern) {                                   \
        uint32x2_t res = reinterpret_cast<uint32x2_t>(             \
                op({{vreinterpretq_u32_s32(dst0),                  \
                     vreinterpretq_u32_s32(dst1)}}));              \
        vst1_lane_u32(reinterpret_cast<uint32_t*>(dptr1), res, 0); \
        vst1_lane_u32(reinterpret_cast<uint32_t*>(dptr2), res, 1); \
    } else {                                                       \
        ST1_S32X4(dst0, tptr1);                                    \
        ST1_S32X4(dst1, tptr2);                                    \
    }

#define POSTPROCESS_1X4(dst0, tptr, dptr)                         \
    if (last_ic && fused_kern) {                                  \
        int32x4_t dst1 = vdupq_n_s32(0);                          \
        uint32x2_t res = reinterpret_cast<uint32x2_t>(            \
                op({{vreinterpretq_u32_s32(dst0), dst1}}));       \
        vst1_lane_u32(reinterpret_cast<uint32_t*>(dptr), res, 0); \
    } else {                                                      \
        ST1_S32X4(dst0, tptr);                                    \
    }

#define POSTPROCESS_1X1(dst0, tptr, dptr)                          \
    if (last_ic && fused_kern) {                                   \
        int32x4_t dst1 = vdupq_n_s32(0);                           \
        uint8x8_t res = op({{vreinterpretq_u32_s32(dst0), dst1}}); \
        dptr = vget_lane_u8(res, 0);                               \
    } else {                                                       \
        tptr = vgetq_lane_u32(dst0, 0);                            \
    }

#define CALC_DST(_sum)            \
    _sum = vreinterpretq_u32_s32( \
            vaddq_s32(vreinterpretq_s32_u32(_sum), _shift_zp))

#define CALC_0(_k_idx, _c_idx)                                                 \
    _elem = vqtbl1q_s8_v7(_tmp, _idx##_c_idx);                                 \
    _sum0##_c_idx = vdotq_u32(_sum0##_c_idx, _k##_k_idx, _elem);               \
    _sum0##_c_idx = vsubq_u32(_sum0##_c_idx, vdotq2_u32(_src_zp, _k##_k_idx)); \
    _sum0##_c_idx = vsubq_u32(_sum0##_c_idx, vdotq2_u32(_filter_zp, _elem));

#define CALC_1(_k_idx, _c_idx)                                                 \
    _elem = vqtbl1q_s8_v7(_tmp, _idx##_c_idx);                                 \
    _sum1##_c_idx = vdotq_u32(_sum1##_c_idx, _k##_k_idx, _elem);               \
    _sum1##_c_idx = vsubq_u32(_sum1##_c_idx, vdotq2_u32(_src_zp, _k##_k_idx)); \
    _sum1##_c_idx = vsubq_u32(_sum1##_c_idx, vdotq2_u32(_filter_zp, _elem));

#define CALC_2(_k1_idx, _k2_idx, _c_idx)                                     \
    _elem = vqtbl1q_s8_v7(_tmp, _idx##_c_idx);                               \
    _sum0##_c_idx = vdotq_u32(_sum0##_c_idx, _k##_k1_idx, _elem);            \
    _sum0##_c_idx =                                                          \
            vsubq_u32(_sum0##_c_idx, vdotq2_u32(_src_zp, _k##_k1_idx));      \
    _sum0##_c_idx = vsubq_u32(_sum0##_c_idx, vdotq2_u32(_filter_zp, _elem)); \
    _sum1##_c_idx = vdotq_u32(_sum1##_c_idx, _k##_k2_idx, _elem);            \
    _sum1##_c_idx =                                                          \
            vsubq_u32(_sum1##_c_idx, vdotq2_u32(_src_zp, _k##_k2_idx));      \
    _sum1##_c_idx = vsubq_u32(_sum1##_c_idx, vdotq2_u32(_filter_zp, _elem));

template <bool first_ic, bool last_ic, bool fused_kern, BiasMode bias_mode,
          typename Op>
void conv_bias::conv_direct_stride1_2x2_quint8_dot(
        const uint8_t* src, const uint8_t* filter, const int32_t* bias,
        int32_t* temp, uint8_t* dst, const size_t IH, const size_t IW,
        const size_t OH, const size_t OW, const uint8_t src_zp,
        const uint8_t filter_zp, const int32_t src_filter_zp, const Op& op) {
    MEGDNN_MARK_USED_VAR(IH);
    const size_t tail_step = IW - OW;

    uint8x16_t _src_zp = vdupq_n_u8(src_zp);
    uint8x16_t _filter_zp = vdupq_n_u8(filter_zp);
    int32x4_t _shift_zp;
    if (bias_mode != BiasMode::NO_BIAS) {
        _shift_zp = vdupq_n_s32(src_filter_zp - SHIFT + bias[0]);
    } else {
        _shift_zp = vdupq_n_s32(src_filter_zp - SHIFT);
    }

    const uint8x16_t _idx0 = {0, 1, 16, 16, 1, 2, 16, 16,
                              2, 3, 16, 16, 3, 4, 16, 16};
    const uint8x16_t _idx1 = {4, 5, 16, 16, 5, 6, 16, 16,
                              6, 7, 16, 16, 7, 8, 16, 16};
    //! here we use uint32_t for calc
    uint32_t* outptr = reinterpret_cast<uint32_t*>(temp);
    uint32_t* outptr2 = outptr + OW;
    uint8_t* dstptr = dst;
    uint8_t* dstptr2 = dstptr + OW;

    const uint8_t* r0 = src;
    const uint8_t* r1 = src + IW;
    const uint8_t* r2 = src + 2 * IW;

    const uint8_t* k0 = filter;

    uint8x16_t _k = vreinterpretq_u8_u32(
            vdupq_n_u32(*reinterpret_cast<const int32_t*>(k0)));
    uint8x16_t _idx = {0, 1, 16, 16, 0, 1, 16, 16, 0, 1, 16, 16, 0, 1, 16, 16};
    uint8x16_t _k1 = vqtbl1q_s8_v7(_k, _idx);
    _idx = {2, 3, 16, 16, 2, 3, 16, 16, 2, 3, 16, 16, 2, 3, 16, 16};
    uint8x16_t _k23 = vqtbl1q_s8_v7(_k, _idx);

#define SUB_ZP(_sum, _r)                             \
    _sum = vdotq_u32(_sum, _k, _r);                  \
    _sum = vsubq_u32(_sum, vdotq2_u32(_src_zp, _k)); \
    _sum = vsubq_u32(_sum, vdotq2_u32(_filter_zp, _r));

    uint8x16_t _tmp, _elem;
    const int width = OW >> 2;
    size_t h = 0;
    for (; h + 1 < OH; h += 2) {
        int w = 0;
        for (; w + 4 < width; w += 4) {
            uint32x4x2_t _sum00, _sum01, _sum10, _sum11;
            if (!first_ic) {
                _sum00 = vld2q_u32(outptr);
                _sum01 = vld2q_u32(outptr + 8);
                _sum10 = vld2q_u32(outptr2);
                _sum11 = vld2q_u32(outptr2 + 8);
            } else {
                _sum00.val[0] = vdupq_n_u32(SHIFT);
                _sum01.val[0] = vdupq_n_u32(SHIFT);
                _sum10.val[0] = vdupq_n_u32(SHIFT);
                _sum11.val[0] = vdupq_n_u32(SHIFT);
                _sum00.val[1] = vdupq_n_u32(SHIFT);
                _sum01.val[1] = vdupq_n_u32(SHIFT);
                _sum10.val[1] = vdupq_n_u32(SHIFT);
                _sum11.val[1] = vdupq_n_u32(SHIFT);
            }

            uint8x16_t _r00 = vld1q_u8(r0);
            //! here will not not read out of bound
            uint8x16_t _r01_ = vdupq_n_u8(r0[16]);
            uint8x16_t _r10 = vld1q_u8(r1);
            uint8x16_t _r11_ = vdupq_n_u8(r1[16]);
            uint8x16_t _r20 = vld1q_u8(r2);
            uint8x16_t _r21_ = vdupq_n_u8(r2[16]);
            uint8x16_t _r01 = vextq_u8(_r00, _r01_, 1);
            uint8x16_t _r11 = vextq_u8(_r10, _r11_, 1);
            uint8x16_t _r21 = vextq_u8(_r20, _r21_, 1);

            int16x8x2_t r_0 = vzipq_s16(vreinterpretq_s16_u8(_r00),
                                         vreinterpretq_s16_u8(_r10));
            uint8x16_t _r0 = vreinterpretq_u8_s8(r_0.val[0]);
            uint8x16_t _r2 = vreinterpretq_u8_s8(r_0.val[1]);

            int16x8x2_t r_1 = vzipq_s16(vreinterpretq_s16_u8(_r01),
                                        vreinterpretq_s16_u8(_r11));
            int8x16_t _r1 = vreinterpretq_u8_s8(r_1.val[0]);
            int8x16_t _r3 = vreinterpretq_u8_s8(r_1.val[1]);

            SUB_ZP(_sum00.val[0], _r0);
            SUB_ZP(_sum00.val[1], _r1);
            SUB_ZP(_sum01.val[0], _r2);
            SUB_ZP(_sum01.val[1], _r3);

            r_0 = vzipq_s16(vreinterpretq_s16_u8(_r10),
                            vreinterpretq_s16_u8(_r20));
            _r0 = vreinterpretq_u8_s8(r_0.val[0]);
            _r2 = vreinterpretq_u8_s8(r_0.val[1]);

            r_1 = vzipq_s16(vreinterpretq_s16_u8(_r11),
                            vreinterpretq_s16_u8(_r21));
            _r1 = vreinterpretq_u8_s8(r_1.val[0]);
            _r3 = vreinterpretq_u8_s8(r_1.val[1]);

            SUB_ZP(_sum10.val[0], _r0);
            SUB_ZP(_sum10.val[1], _r1);
            SUB_ZP(_sum11.val[0], _r2);
            SUB_ZP(_sum11.val[1], _r3);

            if (last_ic) {
                CALC_DST(_sum00.val[0]);
                CALC_DST(_sum00.val[1]);
                CALC_DST(_sum01.val[0]);
                CALC_DST(_sum01.val[1]);
                CALC_DST(_sum10.val[0]);
                CALC_DST(_sum10.val[1]);
                CALC_DST(_sum11.val[0]);
                CALC_DST(_sum11.val[1]);
            }

            POSTPROCESS2_1X8(_sum00, outptr, dstptr);
            POSTPROCESS2_1X8(_sum01, outptr + 8, dstptr + 8);
            POSTPROCESS2_1X8(_sum10, outptr2, dstptr2);
            POSTPROCESS2_1X8(_sum11, outptr2 + 8, dstptr2 + 8);

            r0 += 16;
            r1 += 16;
            r2 += 16;
            outptr += 16;
            outptr2 += 16;
            dstptr += 16;
            dstptr2 += 16;
        }
        for (; w + 2 < width; w += 2) {
            uint32x4_t _sum00, _sum01, _sum10, _sum11;
            if (!first_ic) {
                _sum00 = vld1q_u32(outptr);
                _sum01 = vld1q_u32(outptr + 4);
                _sum10 = vld1q_u32(outptr2);
                _sum11 = vld1q_u32(outptr2 + 4);
            } else {
                _sum00 = vdupq_n_u32(SHIFT);
                _sum01 = vdupq_n_u32(SHIFT);
                _sum10 = vdupq_n_u32(SHIFT);
                _sum11 = vdupq_n_u32(SHIFT);
            }

            _tmp = vld1q_u8(r0);
            CALC_0(1, 0);
            CALC_0(1, 1);

            _tmp = vld1q_u8(r1);
            CALC_2(23, 1, 0);
            CALC_2(23, 1, 1);

            _tmp = vld1q_u8(r2);
            CALC_1(23, 0);
            CALC_1(23, 1);

            if (last_ic) {
                CALC_DST(_sum00);
                CALC_DST(_sum01);
                CALC_DST(_sum10);
                CALC_DST(_sum11);
            }
            POSTPROCESS_1X8(_sum00, _sum01, outptr, dstptr);
            POSTPROCESS_1X8(_sum10, _sum11, outptr2, dstptr2);

            r0 += 8;
            r1 += 8;
            r2 += 8;
            outptr += 8;
            outptr2 += 8;
            dstptr += 8;
            dstptr2 += 8;
        }

        for (; w < width; w++) {
            uint32x4_t _sum00, _sum10;
            if (!first_ic) {
                _sum00 = vld1q_u32(outptr);
                _sum10 = vld1q_u32(outptr2);
            } else {
                _sum00 = vdupq_n_u32(SHIFT);
                _sum10 = vdupq_n_u32(SHIFT);
            }

            _tmp = vtranslq_u8(vld1_u8(r0));
            CALC_0(1, 0);

            _tmp = vtranslq_u8(vld1_u8(r1));
            CALC_2(23, 1, 0);

            _tmp = vtranslq_u8(vld1_u8(r2));
            CALC_1(23, 0);

            if (last_ic) {
                CALC_DST(_sum00);
                CALC_DST(_sum10);
            }
            POSTPROCESS_2X4(_sum00, _sum10, outptr, outptr2, dstptr, dstptr2);

            r0 += 4;
            r1 += 4;
            r2 += 4;
            outptr += 4;
            outptr2 += 4;
            dstptr += 4;
            dstptr2 += 4;
        }
        r0 += tail_step + IW;
        r1 += tail_step + IW;
        r2 += tail_step + IW;
        outptr += OW;
        outptr2 += OW;
        dstptr += OW;
        dstptr2 += OW;
    }

    for (; h < OH; h++) {
        int w = 0;
        for (; w + 4 < width; w += 4) {
            uint32x4x2_t _sum0, _sum1;
            if (!first_ic) {
                _sum0 = vld2q_u32(outptr);
                _sum1 = vld2q_u32(outptr + 8);
            } else {
                _sum0.val[0] = vdupq_n_u32(SHIFT);
                _sum1.val[0] = vdupq_n_u32(SHIFT);
                _sum0.val[1] = vdupq_n_u32(SHIFT);
                _sum1.val[1] = vdupq_n_u32(SHIFT);
            }

            uint8x16_t _r00 = vld1q_u8(r0);
            //! here will not not read out of bound
            uint8x16_t _r01_ = vdupq_n_u8(r0[16]);
            uint8x16_t _r10 = vld1q_u8(r1);
            uint8x16_t _r11_ = vdupq_n_u8(r1[16]);
            uint8x16_t _r01 = vextq_u8(_r00, _r01_, 1);
            uint8x16_t _r11 = vextq_u8(_r10, _r11_, 1);

            int16x8x2_t r_0 = vzipq_s16(vreinterpretq_s16_u8(_r00),
                                         vreinterpretq_s16_u8(_r10));
            uint8x16_t _r0 = vreinterpretq_u8_s8(r_0.val[0]);
            uint8x16_t _r2 = vreinterpretq_u8_s8(r_0.val[1]);

            int16x8x2_t r_1 = vzipq_s16(vreinterpretq_s16_u8(_r01),
                                        vreinterpretq_s16_u8(_r11));
            int8x16_t _r1 = vreinterpretq_u8_s8(r_1.val[0]);
            int8x16_t _r3 = vreinterpretq_u8_s8(r_1.val[1]);

            SUB_ZP(_sum0.val[0], _r0);
            SUB_ZP(_sum0.val[1], _r1);
            SUB_ZP(_sum1.val[0], _r2);
            SUB_ZP(_sum1.val[1], _r3);

            if (last_ic) {
                CALC_DST(_sum0.val[0]);
                CALC_DST(_sum0.val[1]);
                CALC_DST(_sum1.val[0]);
                CALC_DST(_sum1.val[1]);
            }
            POSTPROCESS2_1X8(_sum0, outptr, dstptr);
            POSTPROCESS2_1X8(_sum1, outptr + 8, dstptr + 8);

            r0 += 16;
            r1 += 16;
            outptr += 16;
            dstptr += 16;
        }
        for (; w + 2 < width; w += 2) {
            uint32x4_t _sum00, _sum01;
            if (!first_ic) {
                _sum00 = vld1q_u32(outptr);
                _sum01 = vld1q_u32(outptr + 4);
            } else {
                _sum00 = vdupq_n_u32(SHIFT);
                _sum01 = vdupq_n_u32(SHIFT);
            }

            _tmp = vld1q_u8(r0);
            CALC_0(1, 0);
            CALC_0(1, 1);

            _tmp = vld1q_u8(r1);
            CALC_0(23, 0);
            CALC_0(23, 1);

            if (last_ic) {
                CALC_DST(_sum00);
                CALC_DST(_sum01);
            }
            POSTPROCESS_1X8(_sum00, _sum01, outptr, dstptr);

            r0 += 8;
            r1 += 8;
            outptr += 8;
            dstptr += 8;
        }

        for (; w < width; w++) {
            uint32x4_t _sum00;
            if (!first_ic) {
                _sum00 = vld1q_u32(outptr);
            } else {
                _sum00 = vdupq_n_u32(SHIFT);
            }

            _tmp = vtranslq_u8(vld1_u8(r0));
            CALC_0(1, 0);

            _tmp = vtranslq_u8(vld1_u8(r1));
            CALC_0(23, 0);

            if (last_ic) {
                CALC_DST(_sum00);
            }
            POSTPROCESS_1X4(_sum00, outptr, dstptr);

            r0 += 4;
            r1 += 4;
            outptr += 4;
            dstptr += 4;
        }
        r0 += tail_step;
        r1 += tail_step;
    }
#undef SUB_ZP
}

template <bool first_ic, bool last_ic, bool fused_kern, BiasMode bias_mode,
          typename Op>
void conv_bias::conv_direct_stride1_3x3_quint8_dot(
        const uint8_t* src, const uint8_t* filter, const int32_t* bias,
        int32_t* temp, uint8_t* dst, const size_t IH, const size_t IW,
        const size_t OH, const size_t OW, const uint8_t src_zp,
        const uint8_t filter_zp, const int32_t src_filter_zp, const Op& op) {
    MEGDNN_MARK_USED_VAR(IH);
    const size_t tail_step = IW - OW;

    uint8x16_t _src_zp = vdupq_n_u8(src_zp);
    uint8x16_t _filter_zp = vdupq_n_u8(filter_zp);
    int32x4_t _shift_zp;
    if (bias_mode != BiasMode::NO_BIAS) {
        _shift_zp = vdupq_n_s32(src_filter_zp - SHIFT + bias[0]);
    } else {
        _shift_zp = vdupq_n_s32(src_filter_zp - SHIFT);
    }

    const uint8x16_t _idx0 = {0, 1, 2, 16, 1, 2, 3, 16,
                              2, 3, 4, 16, 3, 4, 5, 16};
    const uint8x16_t _idx1 = {4, 5, 6, 16, 5, 6, 7, 16,
                              6, 7, 8, 16, 7, 8, 9, 16};
    const uint8x16_t _idx2 = {8,  9,  10, 16, 9,  10, 11, 16,
                              10, 11, 12, 16, 11, 12, 13, 16};
    uint32_t* outptr = reinterpret_cast<uint32_t*>(temp);
    uint32_t* outptr2 = outptr + OW;
    uint8_t* dstptr = dst;
    uint8_t* dstptr2 = dstptr + OW;

    const uint8_t* r0 = src;
    const uint8_t* r1 = src + IW;
    const uint8_t* r2 = src + IW * 2;
    const uint8_t* r3 = src + IW * 3;

    const uint8_t* k0 = filter;

    uint8x16_t _k_tmp = vcombine_u8(vld1_u8(k0), vdup_n_u8(k0[8]));
    uint8x16_t _idx = {0, 1, 2, 16, 0, 1, 2, 16, 0, 1, 2, 16, 0, 1, 2, 16};
    uint8x16_t _k12 = vqtbl1q_s8_v7(_k_tmp, _idx);
    _idx = {3, 4, 5, 16, 3, 4, 5, 16, 3, 4, 5, 16, 3, 4, 5, 16};
    uint8x16_t _k345 = vqtbl1q_s8_v7(_k_tmp, _idx);
    _idx = {6, 7, 8, 16, 6, 7, 8, 16, 6, 7, 8, 16, 6, 7, 8, 16};
    uint8x16_t _k678 = vqtbl1q_s8_v7(_k_tmp, _idx);

    uint8x16_t _tmp, _elem;
    size_t h = 0;
    for (; h + 1 < OH; h += 2) {
        int width = OW >> 2;

        int w = 0;
        for (; w + 3 < width; w += 3) {
            uint32x4_t _sum00, _sum01, _sum02, _sum10, _sum11, _sum12;
            if (!first_ic) {
                _sum00 = vld1q_u32(outptr);
                _sum01 = vld1q_u32(outptr + 4);
                _sum02 = vld1q_u32(outptr + 8);
                _sum10 = vld1q_u32(outptr2);
                _sum11 = vld1q_u32(outptr2 + 4);
                _sum12 = vld1q_u32(outptr2 + 8);
            } else {
                _sum00 = vdupq_n_u32(SHIFT);
                _sum01 = vdupq_n_u32(SHIFT);
                _sum02 = vdupq_n_u32(SHIFT);
                _sum10 = vdupq_n_u32(SHIFT);
                _sum11 = vdupq_n_u32(SHIFT);
                _sum12 = vdupq_n_u32(SHIFT);
            }

            _tmp = vld1q_u8(r0);
            CALC_0(12, 0);
            CALC_0(12, 1);
            CALC_0(12, 2);

            _tmp = vld1q_u8(r1);
            CALC_2(345, 12, 0);
            CALC_2(345, 12, 1);
            CALC_2(345, 12, 2);

            _tmp = vld1q_u8(r2);
            CALC_2(678, 345, 0);
            CALC_2(678, 345, 1);
            CALC_2(678, 345, 2);

            _tmp = vld1q_u8(r3);
            CALC_1(678, 0);
            CALC_1(678, 1);
            CALC_1(678, 2);

            if (last_ic) {
                CALC_DST(_sum00);
                CALC_DST(_sum01);
                CALC_DST(_sum02);
                CALC_DST(_sum10);
                CALC_DST(_sum11);
                CALC_DST(_sum12);
            }
            POSTPROCESS_1X8(_sum00, _sum01, outptr, dstptr);
            POSTPROCESS_1X4(_sum02, outptr + 8, dstptr + 8);
            POSTPROCESS_1X8(_sum10, _sum11, outptr2, dstptr2);
            POSTPROCESS_1X4(_sum12, outptr2 + 8, dstptr2 + 8);

            r0 += 12;
            r1 += 12;
            r2 += 12;
            r3 += 12;
            outptr += 12;
            outptr2 += 12;
            dstptr += 12;
            dstptr2 += 12;
        }
        for (; w < width; w++) {
            uint32x4_t _sum00, _sum10;
            if (!first_ic) {
                _sum00 = vld1q_u32(outptr);
                _sum10 = vld1q_u32(outptr2);
            } else {
                _sum00 = vdupq_n_u32(SHIFT);
                _sum10 = vdupq_n_u32(SHIFT);
            }

            _tmp = vtranslq_u8(vld1_u8(r0));
            CALC_0(12, 0);

            _tmp = vtranslq_u8(vld1_u8(r1));
            CALC_2(345, 12, 0);

            _tmp = vtranslq_u8(vld1_u8(r2));
            CALC_2(678, 345, 0);

            _tmp = vtranslq_u8(vld1_u8(r3));
            CALC_1(678, 0);

            if (last_ic) {
                CALC_DST(_sum00);
                CALC_DST(_sum10);
            }
            POSTPROCESS_2X4(_sum00, _sum10, outptr, outptr2, dstptr, dstptr2);

            r0 += 4;
            r1 += 4;
            r2 += 4;
            r3 += 4;
            outptr += 4;
            outptr2 += 4;
            dstptr += 4;
            dstptr2 += 4;
        }

        r0 += tail_step + IW;
        r1 += tail_step + IW;
        r2 += tail_step + IW;
        r3 += tail_step + IW;

        outptr += OW;
        outptr2 += OW;
        dstptr += OW;
        dstptr2 += OW;
    }

    for (; h < OH; h++) {
        int width = OW >> 2;

        int w = 0;
        for (; w + 3 < width; w += 3) {
            uint32x4_t _sum00, _sum01, _sum02;
            if (!first_ic) {
                _sum00 = vld1q_u32(outptr);
                _sum01 = vld1q_u32(outptr + 4);
                _sum02 = vld1q_u32(outptr + 8);
            } else {
                _sum00 = vdupq_n_u32(SHIFT);
                _sum01 = vdupq_n_u32(SHIFT);
                _sum02 = vdupq_n_u32(SHIFT);
            }

            _tmp = vld1q_u8(r0);
            CALC_0(12, 0);
            CALC_0(12, 1);
            CALC_0(12, 2);

            _tmp = vld1q_u8(r1);
            CALC_0(345, 0);
            CALC_0(345, 1);
            CALC_0(345, 2);

            _tmp = vld1q_u8(r2);
            CALC_0(678, 0);
            CALC_0(678, 1);
            CALC_0(678, 2);

            if (last_ic) {
                CALC_DST(_sum00);
                CALC_DST(_sum01);
                CALC_DST(_sum02);
            }
            POSTPROCESS_1X8(_sum00, _sum01, outptr, dstptr);
            POSTPROCESS_1X4(_sum02, outptr + 8, dstptr + 8);

            r0 += 12;
            r1 += 12;
            r2 += 12;
            outptr += 12;
            dstptr += 12;
        }
        for (; w < width; w++) {
            uint32x4_t _sum00;
            if (!first_ic) {
                _sum00 = vld1q_u32(outptr);
            } else {
                _sum00 = vdupq_n_u32(SHIFT);
            }

            _tmp = vtranslq_u8(vld1_u8(r0));
            CALC_0(12, 0);

            _tmp = vtranslq_u8(vld1_u8(r1));
            CALC_0(345, 0);

            _tmp = vtranslq_u8(vld1_u8(r2));
            CALC_0(678, 0);

            if (last_ic) {
                CALC_DST(_sum00);
            }
            POSTPROCESS_1X4(_sum00, outptr, dstptr);

            r0 += 4;
            r1 += 4;
            r2 += 4;
            outptr += 4;
            dstptr += 4;
        }
        r0 += tail_step;
        r1 += tail_step;
        r2 += tail_step;
    }
}

template <bool first_ic, bool last_ic, bool fused_kern, BiasMode bias_mode,
          typename Op>
void conv_bias::conv_direct_stride2_2x2_quint8_dot(
        const uint8_t* src, const uint8_t* filter, const int32_t* bias,
        int32_t* temp, uint8_t* dst, const size_t IH, const size_t IW,
        const size_t OH, const size_t OW, const uint8_t src_zp,
        const uint8_t filter_zp, const int32_t src_filter_zp, const Op& op) {
    MEGDNN_MARK_USED_VAR(IH);
    const size_t tail_step = IW - 2 * OW + IW;

    uint8x16_t _src_zp = vdupq_n_u8(src_zp);
    uint8x16_t _filter_zp = vdupq_n_u8(filter_zp);
    int32x4_t _shift_zp;
    if (bias_mode != BiasMode::NO_BIAS) {
        _shift_zp = vdupq_n_s32(src_filter_zp - SHIFT + bias[0]);
    } else {
        _shift_zp = vdupq_n_s32(src_filter_zp - SHIFT);
    }

    const uint8x16_t _idx0 = {0, 1, 16, 16, 2, 3, 16, 16,
                              4, 5, 16, 16, 6, 7, 16, 16};
    uint32_t* outptr = reinterpret_cast<uint32_t*>(temp);
    uint8_t* dstptr = dst;

    const uint8_t* r0 = src;
    const uint8_t* r1 = src + IW;

    const uint8_t* k0 = filter;

    uint8x16_t _k = vreinterpretq_u8_u32(
            vdupq_n_u32(*reinterpret_cast<const int32_t*>(k0)));
    uint8x16_t _idx = {0, 1, 16, 16, 0, 1, 16, 16, 0, 1, 16, 16, 0, 1, 16, 16};
    uint8x16_t _k1 = vqtbl1q_s8_v7(_k, _idx);
    _idx = {2, 3, 16, 16, 2, 3, 16, 16, 2, 3, 16, 16, 2, 3, 16, 16};
    uint8x16_t _k23 = vqtbl1q_s8_v7(_k, _idx);

#define SUB_ZP(_sum, _r)                             \
    _sum = vdotq_u32(_sum, _k, _r);                  \
    _sum = vsubq_u32(_sum, vdotq2_u32(_src_zp, _k)); \
    _sum = vsubq_u32(_sum, vdotq2_u32(_filter_zp, _r));

    uint8x16_t _tmp, _elem;
    const int width = OW >> 2;
    size_t h = 0;
    for (; h < OH; h++) {
        int w = 0;
        for (; w + 2 < width; w += 2) {
            uint32x4_t _sum0, _sum1;
            if (!first_ic) {
                _sum0 = vld1q_u32(outptr);
                _sum1 = vld1q_u32(outptr + 4);
            } else {
                _sum0 = vdupq_n_u32(SHIFT);
                _sum1 = vdupq_n_u32(SHIFT);
            }

            uint8x16_t _r00 = vld1q_u8(r0);
            //! here will not not read out of bound
            uint8x16_t _r10 = vld1q_u8(r1);

            int16x8x2_t r_0 = vzipq_s16(vreinterpretq_s16_u8(_r00),
                                         vreinterpretq_s16_u8(_r10));
            uint8x16_t _r0 = vreinterpretq_u8_s8(r_0.val[0]);
            uint8x16_t _r1 = vreinterpretq_u8_s8(r_0.val[1]);
            SUB_ZP(_sum0, _r0);
            SUB_ZP(_sum1, _r1);

            if (last_ic) {
                CALC_DST(_sum0);
                CALC_DST(_sum1);
            }

            POSTPROCESS_1X8(_sum0, _sum1, outptr, dstptr);

            r0 += 16;
            r1 += 16;
            outptr += 8;
            dstptr += 8;
        }

        for (; w < width; w++) {
            uint32x4_t _sum00;
            if (!first_ic) {
                _sum00 = vld1q_u32(outptr);
            } else {
                _sum00 = vdupq_n_u32(SHIFT);
            }

            _tmp = vtranslq_u8(vld1_u8(r0));
            CALC_0(1, 0);

            _tmp = vtranslq_u8(vld1_u8(r1));
            CALC_0(23, 0);

            if (last_ic) {
                CALC_DST(_sum00);
            }
            POSTPROCESS_1X4(_sum00, outptr, dstptr);

            r0 += 8;
            r1 += 8;
            outptr += 4;
            dstptr += 4;
        }
        r0 += tail_step;
        r1 += tail_step;
    }
#undef SUB_ZP
}

template <bool first_ic, bool last_ic, bool fused_kern, BiasMode bias_mode,
          typename Op>
void conv_bias::conv_direct_stride2_3x3_quint8_dot(
        const uint8_t* src, const uint8_t* filter, const int32_t* bias,
        int32_t* temp, uint8_t* dst, const size_t IH, const size_t IW,
        const size_t OH, const size_t OW, const uint8_t src_zp,
        const uint8_t filter_zp, const int32_t src_filter_zp, const Op& op) {
    MEGDNN_MARK_USED_VAR(IH);
    const size_t tail_step = IW - 2 * OW + IW;

    uint8x16_t _src_zp = vdupq_n_u8(src_zp);
    uint8x16_t _filter_zp = vdupq_n_u8(filter_zp);
    int32x4_t _shift_zp;
    if (bias_mode != BiasMode::NO_BIAS) {
        _shift_zp = vdupq_n_s32(src_filter_zp - SHIFT + bias[0]);
    } else {
        _shift_zp = vdupq_n_s32(src_filter_zp - SHIFT);
    }

    const uint8x16_t _idx0 = {0, 1, 2, 16, 2, 3, 4, 16,
                              4, 5, 6, 16, 6, 7, 8, 16};
    const uint8x16_t _idx1 = {8,  9,  10, 16, 10, 11, 12, 16,
                              12, 13, 14, 16, 16, 16, 16, 16};
    //! start from 12 13 14 15
    const uint8x16_t _idx2 = {2, 3, 4, 16, 4, 5, 6,  16,
                              6, 7, 8, 16, 8, 9, 10, 16};
    const uint8x16_t _idx3 = {10, 11, 12, 16, 16, 16, 16, 16,
                              16, 16, 16, 16, 16, 16, 16, 16};
    uint32_t* outptr = reinterpret_cast<uint32_t*>(temp);
    uint32_t* outptr2 = outptr + OW;
    uint8_t* dstptr = dst;
    uint8_t* dstptr2 = dstptr + OW;

    const uint8_t* r0 = src;
    const uint8_t* r1 = src + IW;
    const uint8_t* r2 = src + IW * 2;
    const uint8_t* r3 = src + IW * 3;
    const uint8_t* r4 = src + IW * 4;

    const uint8_t* k0 = filter;

    uint8x16_t _k_tmp = vcombine_u8(vld1_u8(k0), vdup_n_u8(k0[8]));
    uint8x16_t _idx = {0, 1, 2, 16, 0, 1, 2, 16, 0, 1, 2, 16, 0, 1, 2, 16};
    uint8x16_t _k12 = vqtbl1q_s8_v7(_k_tmp, _idx);
    _idx = {3, 4, 5, 16, 3, 4, 5, 16, 3, 4, 5, 16, 3, 4, 5, 16};
    uint8x16_t _k345 = vqtbl1q_s8_v7(_k_tmp, _idx);
    _idx = {6, 7, 8, 16, 6, 7, 8, 16, 6, 7, 8, 16, 6, 7, 8, 16};
    uint8x16_t _k678 = vqtbl1q_s8_v7(_k_tmp, _idx);

    uint8x16_t _tmp, _elem;
    size_t h = 0;
    for (; h + 1 < OH; h += 2) {
        int width = OW >> 2;

        int w = 0;
        for (; w + 3 < width; w += 3) {
            uint32x4_t _sum00, _sum01, _sum02, _sum03, _sum10, _sum11, _sum12,
                    _sum13;
            if (!first_ic) {
                _sum00 = vld1q_u32(outptr);
                _sum01 = vld1q_u32(outptr + 4);
                _sum02 = vld1q_u32(outptr + 7);
                _sum03 = vld1q_u32(outptr + 11);
                _sum10 = vld1q_u32(outptr2);
                _sum11 = vld1q_u32(outptr2 + 4);
                _sum12 = vld1q_u32(outptr2 + 7);
                _sum13 = vld1q_u32(outptr2 + 11);
            } else {
                _sum00 = vdupq_n_u32(SHIFT);
                _sum01 = vdupq_n_u32(SHIFT);
                _sum02 = vdupq_n_u32(SHIFT);
                _sum03 = vdupq_n_u32(SHIFT);
                _sum10 = vdupq_n_u32(SHIFT);
                _sum11 = vdupq_n_u32(SHIFT);
                _sum12 = vdupq_n_u32(SHIFT);
                _sum13 = vdupq_n_u32(SHIFT);
            }

            _tmp = vld1q_u8(r0);
            CALC_0(12, 0);
            CALC_0(12, 1);
            _tmp = vld1q_u8(r0 + 12);
            CALC_0(12, 2);
            CALC_0(12, 3);

            _tmp = vld1q_u8(r1);
            CALC_0(345, 0);
            CALC_0(345, 1);
            _tmp = vld1q_u8(r1 + 12);
            CALC_0(345, 2);
            CALC_0(345, 3);

            _tmp = vld1q_u8(r2);
            CALC_2(678, 12, 0);
            CALC_2(678, 12, 1);
            _tmp = vld1q_u8(r2 + 12);
            CALC_2(678, 12, 2);
            CALC_2(678, 12, 3);

            _tmp = vld1q_u8(r3);
            CALC_1(345, 0);
            CALC_1(345, 1);
            _tmp = vld1q_u8(r3 + 12);
            CALC_1(345, 2);
            CALC_1(345, 3);

            _tmp = vld1q_u8(r4);
            CALC_1(678, 0);
            CALC_1(678, 1);
            _tmp = vld1q_u8(r4 + 12);
            CALC_1(678, 2);
            CALC_1(678, 3);

            if (last_ic) {
                CALC_DST(_sum00);
                CALC_DST(_sum01);
                CALC_DST(_sum02);
                CALC_DST(_sum03);
                CALC_DST(_sum10);
                CALC_DST(_sum11);
                CALC_DST(_sum12);
                CALC_DST(_sum13);
            }

            POSTPROCESS_1X8(_sum00, _sum01, outptr, dstptr);
            POSTPROCESS_1X4(_sum02, outptr + 7, dstptr + 7);
            POSTPROCESS_1X1(_sum03, outptr[11], dstptr[11]);
            POSTPROCESS_1X8(_sum10, _sum11, outptr2, dstptr2);
            POSTPROCESS_1X4(_sum12, outptr2 + 7, dstptr2 + 7);
            POSTPROCESS_1X1(_sum13, outptr2[11], dstptr2[11]);

            r0 += 24;
            r1 += 24;
            r2 += 24;
            r3 += 24;
            r4 += 24;
            outptr += 12;
            outptr2 += 12;
            dstptr += 12;
            dstptr2 += 12;
        }
        for (; w < width; w++) {
            uint32x4_t _sum00, _sum10;
            if (!first_ic) {
                _sum00 = vld1q_u32(outptr);
                _sum10 = vld1q_u32(outptr2);
            } else {
                _sum00 = vdupq_n_u32(SHIFT);
                _sum10 = vdupq_n_u32(SHIFT);
            }

            _tmp = vld1q_u8(r0);
            CALC_0(12, 0);

            _tmp = vld1q_u8(r1);
            CALC_0(345, 0);

            _tmp = vld1q_u8(r2);
            CALC_2(678, 12, 0);

            _tmp = vld1q_u8(r3);
            CALC_1(345, 0);

            _tmp = vld1q_u8(r4);
            CALC_1(678, 0);

            if (last_ic) {
                CALC_DST(_sum00);
                CALC_DST(_sum10);
            }
            POSTPROCESS_2X4(_sum00, _sum10, outptr, outptr2, dstptr, dstptr2);

            r0 += 8;
            r1 += 8;
            r2 += 8;
            r3 += 8;
            r4 += 8;
            outptr += 4;
            outptr2 += 4;
            dstptr += 4;
            dstptr2 += 4;
        }

        r0 += tail_step + IW * 2;
        r1 += tail_step + IW * 2;
        r2 += tail_step + IW * 2;
        r3 += tail_step + IW * 2;
        r4 += tail_step + IW * 2;

        outptr += OW;
        outptr2 += OW;
        dstptr += OW;
        dstptr2 += OW;
    }

    for (; h < OH; h++) {
        int width = OW >> 2;

        int w = 0;
        for (; w + 3 < width; w += 3) {
            uint32x4_t _sum00, _sum01, _sum02, _sum03;
            if (!first_ic) {
                _sum00 = vld1q_u32(outptr);
                _sum01 = vld1q_u32(outptr + 4);
                _sum02 = vld1q_u32(outptr + 7);
                _sum03 = vld1q_u32(outptr + 11);
            } else {
                _sum00 = vdupq_n_u32(SHIFT);
                _sum01 = vdupq_n_u32(SHIFT);
                _sum02 = vdupq_n_u32(SHIFT);
                _sum03 = vdupq_n_u32(SHIFT);
            }

            _tmp = vld1q_u8(r0);
            CALC_0(12, 0);
            CALC_0(12, 1);
            _tmp = vld1q_u8(r0 + 12);
            CALC_0(12, 2);
            CALC_0(12, 3);

            _tmp = vld1q_u8(r1);
            CALC_0(345, 0);
            CALC_0(345, 1);
            _tmp = vld1q_u8(r1 + 12);
            CALC_0(345, 2);
            CALC_0(345, 3);

            _tmp = vld1q_u8(r2);
            CALC_0(678, 0);
            CALC_0(678, 1);
            _tmp = vld1q_u8(r2 + 12);
            CALC_0(678, 2);
            CALC_0(678, 3);

            if (last_ic) {
                CALC_DST(_sum00);
                CALC_DST(_sum01);
                CALC_DST(_sum02);
                CALC_DST(_sum03);
            }
            POSTPROCESS_1X8(_sum00, _sum01, outptr, dstptr);
            POSTPROCESS_1X4(_sum02, outptr + 7, dstptr + 7);
            POSTPROCESS_1X1(_sum03, outptr[11], dstptr[11]);

            r0 += 24;
            r1 += 24;
            r2 += 24;
            outptr += 12;
            dstptr += 12;
        }
        for (; w < width; w++) {
            uint32x4_t _sum00;
            if (!first_ic) {
                _sum00 = vld1q_u32(outptr);
            } else {
                _sum00 = vdupq_n_u32(SHIFT);
            }

            _tmp = vld1q_u8(r0);
            CALC_0(12, 0);

            _tmp = vld1q_u8(r1);
            CALC_0(345, 0);

            _tmp = vld1q_u8(r2);
            CALC_0(678, 0);

            if (last_ic) {
                CALC_DST(_sum00);
            }
            POSTPROCESS_1X4(_sum00, outptr, dstptr);

            r0 += 8;
            r1 += 8;
            r2 += 8;
            outptr += 4;
            dstptr += 4;
        }
        r0 += tail_step;
        r1 += tail_step;
        r2 += tail_step;
    }
}

#undef CALC_0
#undef CALC_1
#undef CALC_2

#define CALC_0(_k00_idx, _k01_idx, _c_idx)                                   \
    _elem = vqtbl1q_s8_v7(_tmp, _idx##_c_idx##0);                               \
    _sum0##_c_idx = vdotq_u32(_sum0##_c_idx, _k##_k00_idx, _elem);           \
    _sum0##_c_idx =                                                          \
            vsubq_u32(_sum0##_c_idx, vdotq2_u32(_src_zp, _k##_k00_idx));     \
    _sum0##_c_idx = vsubq_u32(_sum0##_c_idx, vdotq2_u32(_filter_zp, _elem)); \
    _elem = vqtbl1q_s8_v7(_tmp, _idx##_c_idx##1);                               \
    _sum0##_c_idx = vdotq_u32(_sum0##_c_idx, _k##_k01_idx, _elem);           \
    _sum0##_c_idx =                                                          \
            vsubq_u32(_sum0##_c_idx, vdotq2_u32(_src_zp, _k##_k01_idx));     \
    _sum0##_c_idx = vsubq_u32(_sum0##_c_idx, vdotq2_u32(_filter_zp, _elem));

#define CALC_1(_k00_idx, _k01_idx, _c_idx)                                   \
    _elem = vqtbl1q_s8_v7(_tmp, _idx##_c_idx##0);                               \
    _sum1##_c_idx = vdotq_u32(_sum1##_c_idx, _k##_k00_idx, _elem);           \
    _sum1##_c_idx =                                                          \
            vsubq_u32(_sum1##_c_idx, vdotq2_u32(_src_zp, _k##_k00_idx));     \
    _sum1##_c_idx = vsubq_u32(_sum1##_c_idx, vdotq2_u32(_filter_zp, _elem)); \
    _elem = vqtbl1q_s8_v7(_tmp, _idx##_c_idx##1);                               \
    _sum1##_c_idx = vdotq_u32(_sum1##_c_idx, _k##_k01_idx, _elem);           \
    _sum1##_c_idx =                                                          \
            vsubq_u32(_sum1##_c_idx, vdotq2_u32(_src_zp, _k##_k01_idx));     \
    _sum1##_c_idx = vsubq_u32(_sum1##_c_idx, vdotq2_u32(_filter_zp, _elem));

#define CALC_2(_k00_idx, _k01_idx, _k10_idx, _k11_idx, _c_idx)           \
    _elem = vqtbl1q_s8_v7(_tmp, _idx##_c_idx##0);                           \
    _sum0##_c_idx = vdotq_u32(_sum0##_c_idx, _k##_k00_idx, _elem);       \
    _sum0##_c_idx =                                                      \
            vsubq_u32(_sum0##_c_idx, vdotq2_u32(_src_zp, _k##_k00_idx)); \
    _elem2 = vdotq2_u32(_filter_zp, _elem);                              \
    _sum0##_c_idx = vsubq_u32(_sum0##_c_idx, _elem2);                    \
    _sum1##_c_idx = vdotq_u32(_sum1##_c_idx, _k##_k10_idx, _elem);       \
    _sum1##_c_idx =                                                      \
            vsubq_u32(_sum1##_c_idx, vdotq2_u32(_src_zp, _k##_k10_idx)); \
    _sum1##_c_idx = vsubq_u32(_sum1##_c_idx, _elem2);                    \
    _elem = vqtbl1q_s8_v7(_tmp, _idx##_c_idx##1);                           \
    _sum0##_c_idx = vdotq_u32(_sum0##_c_idx, _k##_k01_idx, _elem);       \
    _sum0##_c_idx =                                                      \
            vsubq_u32(_sum0##_c_idx, vdotq2_u32(_src_zp, _k##_k01_idx)); \
    _elem2 = vdotq2_u32(_filter_zp, _elem);                              \
    _sum0##_c_idx = vsubq_u32(_sum0##_c_idx, _elem2);                    \
    _sum1##_c_idx = vdotq_u32(_sum1##_c_idx, _k##_k11_idx, _elem);       \
    _sum1##_c_idx =                                                      \
            vsubq_u32(_sum1##_c_idx, vdotq2_u32(_src_zp, _k##_k11_idx)); \
    _sum1##_c_idx = vsubq_u32(_sum1##_c_idx, _elem2);

template <bool first_ic, bool last_ic, bool fused_kern, BiasMode bias_mode,
          typename Op>
void conv_bias::conv_direct_stride1_5x5_quint8_dot(
        const uint8_t* src, const uint8_t* filter, const int32_t* bias,
        int32_t* temp, uint8_t* dst, const size_t IH, const size_t IW,
        const size_t OH, const size_t OW, const uint8_t src_zp,
        const uint8_t filter_zp, const int32_t src_filter_zp, const Op& op) {
    MEGDNN_MARK_USED_VAR(IH);
    const size_t tail_step = IW - OW;

    uint8x16_t _src_zp = vdupq_n_u8(src_zp);
    uint8x16_t _filter_zp = vdupq_n_u8(filter_zp);
    int32x4_t _shift_zp;
    if (bias_mode != BiasMode::NO_BIAS) {
        _shift_zp = vdupq_n_s32(src_filter_zp - SHIFT + bias[0]);
    } else {
        _shift_zp = vdupq_n_s32(src_filter_zp - SHIFT);
    }

    const uint8x16_t _idx00 = {0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6};
    const uint8x16_t _idx01 = {4, 16, 16, 16, 5, 16, 16, 16,
                               6, 16, 16, 16, 7, 16, 16, 16};
    const uint8x16_t _idx10 = {4, 5, 6, 7, 5, 6, 7, 8, 6, 7, 8, 9, 7, 8, 9, 10};
    const uint8x16_t _idx11 = {8,  16, 16, 16, 9,  16, 16, 16,
                               10, 16, 16, 16, 11, 16, 16, 16};
    const uint8x16_t _idx20 = {8,  9,  10, 11, 9,  10, 11, 12,
                               10, 11, 12, 13, 11, 12, 13, 14};
    const uint8x16_t _idx21 = {12, 16, 16, 16, 13, 16, 16, 16,
                               14, 16, 16, 16, 15, 16, 16, 16};
    uint8x16_t _tmp, _elem;
    uint32x4_t _elem2;
    uint32_t* outptr = reinterpret_cast<uint32_t*>(temp);
    uint32_t* outptr2 = outptr + OW;
    uint8_t* dstptr = dst;
    uint8_t* dstptr2 = dstptr + OW;

    const uint8_t* r0 = src;
    const uint8_t* r1 = src + IW;
    const uint8_t* r2 = src + IW * 2;
    const uint8_t* r3 = src + IW * 3;
    const uint8_t* r4 = src + IW * 4;
    const uint8_t* r5 = src + IW * 5;

    const uint8_t* k0 = filter;

    uint8x16_t _k = vld1q_u8(k0);
    //! filter row 1
    uint8x16_t _idx = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    uint8x16_t _k123 = vqtbl1q_s8_v7(_k, _idx);
    _idx = {4, 16, 16, 16, 4, 16, 16, 16, 4, 16, 16, 16, 4, 16, 16, 16};
    uint8x16_t _k4 = vqtbl1q_s8_v7(_k, _idx);
    //! filter row 2
    _idx = {5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8};
    uint8x16_t _k5678 = vqtbl1q_s8_v7(_k, _idx);
    _idx = {9, 16, 16, 16, 9, 16, 16, 16, 9, 16, 16, 16, 9, 16, 16, 16};
    uint8x16_t _k9 = vqtbl1q_s8_v7(_k, _idx);
    //! filter row 3
    _idx = {10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13};
    uint8x16_t _k10111213 = vqtbl1q_s8_v7(_k, _idx);
    _idx = {14, 16, 16, 16, 14, 16, 16, 16, 14, 16, 16, 16, 14, 16, 16, 16};
    uint8x16_t _k14 = vqtbl1q_s8_v7(_k, _idx);
    //! 9 10 11 12 -> 13 14 15 16 -> 17 18 19 20 -> 21 22 23 24
    _k = vld1q_u8(k0 + 9);
    //! filter row 4
    _idx = {6, 7, 8, 9, 6, 7, 8, 9, 6, 7, 8, 9, 6, 7, 8, 9};
    uint8x16_t _k15161718 = vqtbl1q_s8_v7(_k, _idx);
    _idx = {10, 16, 16, 16, 10, 16, 16, 16, 10, 16, 16, 16, 10, 16, 16, 16};
    uint8x16_t _k19 = vqtbl1q_s8_v7(_k, _idx);
    //! filter row 5
    _idx = {11, 12, 13, 14, 11, 12, 13, 14, 11, 12, 13, 14, 11, 12, 13, 14};
    uint8x16_t _k20212223 = vqtbl1q_s8_v7(_k, _idx);
    _idx = {15, 16, 16, 16, 15, 16, 16, 16, 15, 16, 16, 16, 15, 16, 16, 16};
    uint8x16_t _k24 = vqtbl1q_s8_v7(_k, _idx);

    const int width = OW >> 2;
    size_t h = 0;
    for (; h + 1 < OH; h += 2) {
        int w = 0;
        for (; w + 3 < width; w += 3) {
            uint32x4_t _sum00, _sum01, _sum02, _sum10, _sum11, _sum12;
            if (!first_ic) {
                _sum00 = vld1q_u32(outptr);
                _sum01 = vld1q_u32(outptr + 4);
                _sum02 = vld1q_u32(outptr + 8);
                _sum10 = vld1q_u32(outptr2);
                _sum11 = vld1q_u32(outptr2 + 4);
                _sum12 = vld1q_u32(outptr2 + 8);
            } else {
                _sum00 = vdupq_n_u32(SHIFT);
                _sum01 = vdupq_n_u32(SHIFT);
                _sum02 = vdupq_n_u32(SHIFT);
                _sum10 = vdupq_n_u32(SHIFT);
                _sum11 = vdupq_n_u32(SHIFT);
                _sum12 = vdupq_n_u32(SHIFT);
            }

            _tmp = vld1q_u8(r0);
            CALC_0(123, 4, 0);
            CALC_0(123, 4, 1);
            CALC_0(123, 4, 2);

            _tmp = vld1q_u8(r1);
            CALC_2(5678, 9, 123, 4, 0);
            CALC_2(5678, 9, 123, 4, 1);
            CALC_2(5678, 9, 123, 4, 2);

            _tmp = vld1q_u8(r2);
            CALC_2(10111213, 14, 5678, 9, 0);
            CALC_2(10111213, 14, 5678, 9, 1);
            CALC_2(10111213, 14, 5678, 9, 2);

            _tmp = vld1q_u8(r3);
            CALC_2(15161718, 19, 10111213, 14, 0);
            CALC_2(15161718, 19, 10111213, 14, 1);
            CALC_2(15161718, 19, 10111213, 14, 2);

            _tmp = vld1q_u8(r4);
            CALC_2(20212223, 24, 15161718, 19, 0);
            CALC_2(20212223, 24, 15161718, 19, 1);
            CALC_2(20212223, 24, 15161718, 19, 2);

            _tmp = vld1q_u8(r5);
            CALC_1(20212223, 24, 0);
            CALC_1(20212223, 24, 1);
            CALC_1(20212223, 24, 2);

            if (last_ic) {
                CALC_DST(_sum00);
                CALC_DST(_sum01);
                CALC_DST(_sum02);
                CALC_DST(_sum10);
                CALC_DST(_sum11);
                CALC_DST(_sum12);
            }
            POSTPROCESS_1X8(_sum00, _sum01, outptr, dstptr);
            POSTPROCESS_1X4(_sum02, outptr + 8, dstptr + 8);
            POSTPROCESS_1X8(_sum10, _sum11, outptr2, dstptr2);
            POSTPROCESS_1X4(_sum12, outptr2 + 8, dstptr2 + 8);

            r0 += 12;
            r1 += 12;
            r2 += 12;
            r3 += 12;
            r4 += 12;
            r5 += 12;
            outptr += 12;
            outptr2 += 12;
            dstptr += 12;
            dstptr2 += 12;
        }
        for (; w < width; w++) {
            uint32x4_t _sum00, _sum10;
            if (!first_ic) {
                _sum00 = vld1q_u32(outptr);
                _sum10 = vld1q_u32(outptr2);
            } else {
                _sum00 = vdupq_n_u32(SHIFT);
                _sum10 = vdupq_n_u32(SHIFT);
            }

            _tmp = vtranslq_u8(vld1_u8(r0));
            CALC_0(123, 4, 0);

            _tmp = vtranslq_u8(vld1_u8(r1));
            CALC_2(5678, 9, 123, 4, 0);

            _tmp = vtranslq_u8(vld1_u8(r2));
            CALC_2(10111213, 14, 5678, 9, 0);

            _tmp = vtranslq_u8(vld1_u8(r3));
            CALC_2(15161718, 19, 10111213, 14, 0);

            _tmp = vtranslq_u8(vld1_u8(r4));
            CALC_2(20212223, 24, 15161718, 19, 0);

            _tmp = vtranslq_u8(vld1_u8(r5));
            CALC_1(20212223, 24, 0);

            if (last_ic) {
                CALC_DST(_sum00);
                CALC_DST(_sum10);
            }
            POSTPROCESS_2X4(_sum00, _sum10, outptr, outptr2, dstptr, dstptr2);

            r0 += 4;
            r1 += 4;
            r2 += 4;
            r3 += 4;
            r4 += 4;
            r5 += 4;
            outptr += 4;
            outptr2 += 4;
            dstptr += 4;
            dstptr2 += 4;
        }

        r0 += tail_step + IW;
        r1 += tail_step + IW;
        r2 += tail_step + IW;
        r3 += tail_step + IW;
        r4 += tail_step + IW;
        r5 += tail_step + IW;

        outptr += OW;
        outptr2 += OW;
        dstptr += OW;
        dstptr2 += OW;
    }

    for (; h < OH; h++) {
        int w = 0;
        for (; w + 3 < width; w += 3) {
            uint32x4_t _sum00, _sum01, _sum02;
            if (!first_ic) {
                _sum00 = vld1q_u32(outptr);
                _sum01 = vld1q_u32(outptr + 4);
                _sum02 = vld1q_u32(outptr + 8);
            } else {
                _sum00 = vdupq_n_u32(SHIFT);
                _sum01 = vdupq_n_u32(SHIFT);
                _sum02 = vdupq_n_u32(SHIFT);
            }

            _tmp = vld1q_u8(r0);
            CALC_0(123, 4, 0);
            CALC_0(123, 4, 1);
            CALC_0(123, 4, 2);

            _tmp = vld1q_u8(r1);
            CALC_0(5678, 9, 0);
            CALC_0(5678, 9, 1);
            CALC_0(5678, 9, 2);

            _tmp = vld1q_u8(r2);
            CALC_0(10111213, 14, 0);
            CALC_0(10111213, 14, 1);
            CALC_0(10111213, 14, 2);

            _tmp = vld1q_u8(r3);
            CALC_0(15161718, 19, 0);
            CALC_0(15161718, 19, 1);
            CALC_0(15161718, 19, 2);

            _tmp = vld1q_u8(r4);
            CALC_0(20212223, 24, 0);
            CALC_0(20212223, 24, 1);
            CALC_0(20212223, 24, 2);

            if (last_ic) {
                CALC_DST(_sum00);
                CALC_DST(_sum01);
                CALC_DST(_sum02);
            }
            POSTPROCESS_1X8(_sum00, _sum01, outptr, dstptr);
            POSTPROCESS_1X4(_sum02, outptr + 8, dstptr + 8);

            r0 += 12;
            r1 += 12;
            r2 += 12;
            r3 += 12;
            r4 += 12;
            outptr += 12;
            dstptr += 12;
        }
        for (; w < width; w++) {
            uint32x4_t _sum00;
            if (!first_ic) {
                _sum00 = vld1q_u32(outptr);
            } else {
                _sum00 = vdupq_n_u32(SHIFT);
            }

            _tmp = vtranslq_u8(vld1_u8(r0));
            CALC_0(123, 4, 0);

            _tmp = vtranslq_u8(vld1_u8(r1));
            CALC_0(5678, 9, 0);

            _tmp = vtranslq_u8(vld1_u8(r2));
            CALC_0(10111213, 14, 0);

            _tmp = vtranslq_u8(vld1_u8(r3));
            CALC_0(15161718, 19, 0);

            _tmp = vtranslq_u8(vld1_u8(r4));
            CALC_0(20212223, 24, 0);

            if (last_ic) {
                CALC_DST(_sum00);
            }
            POSTPROCESS_1X4(_sum00, outptr, dstptr);

            r0 += 4;
            r1 += 4;
            r2 += 4;
            r3 += 4;
            r4 += 4;
            outptr += 4;
            dstptr += 4;
        }
        r0 += tail_step;
        r1 += tail_step;
        r2 += tail_step;
        r3 += tail_step;
        r4 += tail_step;
    }
}

template <bool first_ic, bool last_ic, bool fused_kern, BiasMode bias_mode,
          typename Op>
void conv_bias::conv_direct_stride1_7x7_quint8_dot(
        const uint8_t* src, const uint8_t* filter, const int32_t* bias,
        int32_t* temp, uint8_t* dst, const size_t IH, const size_t IW,
        const size_t OH, const size_t OW, const uint8_t src_zp,
        const uint8_t filter_zp, const int32_t src_filter_zp, const Op& op) {
    MEGDNN_MARK_USED_VAR(IH);
    const size_t tail_step = IW - OW;

    uint8x16_t _src_zp = vdupq_n_u8(src_zp);
    uint8x16_t _filter_zp = vdupq_n_u8(filter_zp);
    int32x4_t _shift_zp;
    if (bias_mode != BiasMode::NO_BIAS) {
        _shift_zp = vdupq_n_s32(src_filter_zp - SHIFT + bias[0]);
    } else {
        _shift_zp = vdupq_n_s32(src_filter_zp - SHIFT);
    }

    const uint8x16_t _idx00 = {0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6};
    const uint8x16_t _idx01 = {4, 5, 6, 16, 5, 6, 7, 16,
                               6, 7, 8, 16, 7, 8, 9, 16};
    const uint8x16_t _idx10 = {4, 5, 6, 7, 5, 6, 7, 8, 6, 7, 8, 9, 7, 8, 9, 10};
    const uint8x16_t _idx11 = {8,  9,  10, 16, 9,  10, 11, 16,
                               10, 11, 12, 16, 11, 12, 13, 16};

    uint8x16_t _tmp, _elem;
    uint32x4_t _elem2;
    uint32_t* outptr = reinterpret_cast<uint32_t*>(temp);
    uint32_t* outptr2 = outptr + OW;
    uint8_t* dstptr = dst;
    uint8_t* dstptr2 = dstptr + OW;

    const uint8_t* r0 = src;
    const uint8_t* r1 = src + IW;
    const uint8_t* r2 = src + IW * 2;
    const uint8_t* r3 = src + IW * 3;
    const uint8_t* r4 = src + IW * 4;
    const uint8_t* r5 = src + IW * 5;
    const uint8_t* r6 = src + IW * 6;
    const uint8_t* r7 = src + IW * 7;

    const uint8_t* k0 = filter;

    uint8x16_t _k = vld1q_u8(k0);
    //! filter row 1
    uint8x16_t _idx = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    uint8x16_t _k123 = vqtbl1q_s8_v7(_k, _idx);
    _idx = {4, 5, 6, 16, 4, 5, 6, 16, 4, 5, 6, 16, 4, 5, 6, 16};
    uint8x16_t _k456 = vqtbl1q_s8_v7(_k, _idx);
    //! filter row 2
    _idx = {7, 8, 9, 10, 7, 8, 9, 10, 7, 8, 9, 10, 7, 8, 9, 10};
    uint8x16_t _k78910 = vqtbl1q_s8_v7(_k, _idx);
    _idx = {11, 12, 13, 16, 11, 12, 13, 16, 11, 12, 13, 16, 11, 12, 13, 16};
    uint8x16_t _k111213 = vqtbl1q_s8_v7(_k, _idx);

    //! 12 13 14 15 -> 16 17 18 19 -> 20 21 22 23 -> 24 25 26 27
    _k = vld1q_u8(k0 + 12);
    //! filter row 3
    _idx = {2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5};
    uint8x16_t _k14151617 = vqtbl1q_s8_v7(_k, _idx);
    _idx = {6, 7, 8, 16, 6, 7, 8, 16, 6, 7, 8, 16, 6, 7, 8, 16};
    uint8x16_t _k181920 = vqtbl1q_s8_v7(_k, _idx);
    //! filter row 4
    _idx = {9, 10, 11, 12, 9, 10, 11, 12, 9, 10, 11, 12, 9, 10, 11, 12};
    uint8x16_t _k21222324 = vqtbl1q_s8_v7(_k, _idx);
    _idx = {13, 14, 15, 16, 13, 14, 15, 16, 13, 14, 15, 16, 13, 14, 15, 16};
    uint8x16_t _k252627 = vqtbl1q_s8_v7(_k, _idx);

    //! 24 25 26 27->28 29 30 31 -> 32 33 34 35 -> 36 37 38 39
    _k = vld1q_u8(k0 + 24);
    //! filter row 5
    _idx = {4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7};
    uint8x16_t _k28293031 = vqtbl1q_s8_v7(_k, _idx);
    _idx = {8, 9, 10, 16, 8, 9, 10, 16, 8, 9, 10, 16, 8, 9, 10, 16};
    uint8x16_t _k323334 = vqtbl1q_s8_v7(_k, _idx);

    //! 33 34 35 36 -> 37 38 39 40 -> 41 42 43 44 -> 45 46 47 48
    _k = vld1q_u8(k0 + 33);
    //! filter row 6
    _idx = {2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5};
    uint8x16_t _k35363738 = vqtbl1q_s8_v7(_k, _idx);
    _idx = {6, 7, 8, 16, 6, 7, 8, 16, 6, 7, 8, 16, 6, 7, 8, 16};
    uint8x16_t _k394041 = vqtbl1q_s8_v7(_k, _idx);

    //! filter row 7
    _idx = {9, 10, 11, 12, 9, 10, 11, 12, 9, 10, 11, 12, 9, 10, 11, 12};
    uint8x16_t _k42434445 = vqtbl1q_s8_v7(_k, _idx);
    _idx = {13, 14, 15, 16, 13, 14, 15, 16, 13, 14, 15, 16, 13, 14, 15, 16};
    uint8x16_t _k464748 = vqtbl1q_s8_v7(_k, _idx);

    const int width = OW >> 2;
    size_t h = 0;
    for (; h + 1 < OH; h += 2) {
        int w = 0;
        for (; w + 2 < width; w += 2) {
            uint32x4_t _sum00, _sum01, _sum10, _sum11;
            if (!first_ic) {
                _sum00 = vld1q_u32(outptr);
                _sum01 = vld1q_u32(outptr + 4);
                _sum10 = vld1q_u32(outptr2);
                _sum11 = vld1q_u32(outptr2 + 4);
            } else {
                _sum00 = vdupq_n_u32(SHIFT);
                _sum01 = vdupq_n_u32(SHIFT);
                _sum10 = vdupq_n_u32(SHIFT);
                _sum11 = vdupq_n_u32(SHIFT);
            }

            _tmp = vld1q_u8(r0);
            CALC_0(123, 456, 0);
            CALC_0(123, 456, 1);

            _tmp = vld1q_u8(r1);
            CALC_2(78910, 111213, 123, 456, 0);
            CALC_2(78910, 111213, 123, 456, 1);

            _tmp = vld1q_u8(r2);
            CALC_2(14151617, 181920, 78910, 111213, 0);
            CALC_2(14151617, 181920, 78910, 111213, 1);

            _tmp = vld1q_u8(r3);
            CALC_2(21222324, 252627, 14151617, 181920, 0);
            CALC_2(21222324, 252627, 14151617, 181920, 1);

            _tmp = vld1q_u8(r4);
            CALC_2(28293031, 323334, 21222324, 252627, 0);
            CALC_2(28293031, 323334, 21222324, 252627, 1);

            _tmp = vld1q_u8(r5);
            CALC_2(35363738, 394041, 28293031, 323334, 0);
            CALC_2(35363738, 394041, 28293031, 323334, 1);

            _tmp = vld1q_u8(r6);
            CALC_2(42434445, 464748, 35363738, 394041, 0);
            CALC_2(42434445, 464748, 35363738, 394041, 1);

            _tmp = vld1q_u8(r7);
            CALC_1(42434445, 464748, 0);
            CALC_1(42434445, 464748, 1);

            if (last_ic) {
                CALC_DST(_sum00);
                CALC_DST(_sum01);
                CALC_DST(_sum10);
                CALC_DST(_sum11);
            }
            POSTPROCESS_1X8(_sum00, _sum01, outptr, dstptr);
            POSTPROCESS_1X8(_sum10, _sum11, outptr2, dstptr2);

            r0 += 8;
            r1 += 8;
            r2 += 8;
            r3 += 8;
            r4 += 8;
            r5 += 8;
            r6 += 8;
            r7 += 8;
            outptr += 8;
            outptr2 += 8;
            dstptr += 8;
            dstptr2 += 8;
        }
        for (; w < width; w++) {
            uint32x4_t _sum00, _sum10;
            if (!first_ic) {
                _sum00 = vld1q_u32(outptr);
                _sum10 = vld1q_u32(outptr2);
            } else {
                _sum00 = vdupq_n_u32(SHIFT);
                _sum10 = vdupq_n_u32(SHIFT);
            }

            _tmp = vld1q_u8(r0);
            CALC_0(123, 456, 0);

            _tmp = vld1q_u8(r1);
            CALC_2(78910, 111213, 123, 456, 0);

            _tmp = vld1q_u8(r2);
            CALC_2(14151617, 181920, 78910, 111213, 0);

            _tmp = vld1q_u8(r3);
            CALC_2(21222324, 252627, 14151617, 181920, 0);

            _tmp = vld1q_u8(r4);
            CALC_2(28293031, 323334, 21222324, 252627, 0);

            _tmp = vld1q_u8(r5);
            CALC_2(35363738, 394041, 28293031, 323334, 0);

            _tmp = vld1q_u8(r6);
            CALC_2(42434445, 464748, 35363738, 394041, 0);

            _tmp = vld1q_u8(r7);
            CALC_1(42434445, 464748, 0);

            if (last_ic) {
                CALC_DST(_sum00);
                CALC_DST(_sum10);
            }
            POSTPROCESS_2X4(_sum00, _sum10, outptr, outptr2, dstptr, dstptr2);

            r0 += 4;
            r1 += 4;
            r2 += 4;
            r3 += 4;
            r4 += 4;
            r5 += 4;
            r6 += 4;
            r7 += 4;
            outptr += 4;
            outptr2 += 4;
            dstptr += 4;
            dstptr2 += 4;
        }

        r0 += tail_step + IW;
        r1 += tail_step + IW;
        r2 += tail_step + IW;
        r3 += tail_step + IW;
        r4 += tail_step + IW;
        r5 += tail_step + IW;
        r6 += tail_step + IW;
        r7 += tail_step + IW;

        outptr += OW;
        outptr2 += OW;
        dstptr += OW;
        dstptr2 += OW;
    }

    for (; h < OH; h++) {
        int w = 0;
        for (; w + 2 < width; w += 2) {
            uint32x4_t _sum00, _sum01;
            if (!first_ic) {
                _sum00 = vld1q_u32(outptr);
                _sum01 = vld1q_u32(outptr + 4);
            } else {
                _sum00 = vdupq_n_u32(SHIFT);
                _sum01 = vdupq_n_u32(SHIFT);
            }

            _tmp = vld1q_u8(r0);
            CALC_0(123, 456, 0);
            CALC_0(123, 456, 1);

            _tmp = vld1q_u8(r1);
            CALC_0(78910, 111213, 0);
            CALC_0(78910, 111213, 1);

            _tmp = vld1q_u8(r2);
            CALC_0(14151617, 181920, 0);
            CALC_0(14151617, 181920, 1);

            _tmp = vld1q_u8(r3);
            CALC_0(21222324, 252627, 0);
            CALC_0(21222324, 252627, 1);

            _tmp = vld1q_u8(r4);
            CALC_0(28293031, 323334, 0);
            CALC_0(28293031, 323334, 1);

            _tmp = vld1q_u8(r5);
            CALC_0(35363738, 394041, 0);
            CALC_0(35363738, 394041, 1);

            _tmp = vld1q_u8(r6);
            CALC_0(42434445, 464748, 0);
            CALC_0(42434445, 464748, 1);

            if (last_ic) {
                CALC_DST(_sum00);
                CALC_DST(_sum01);
            }
            POSTPROCESS_1X8(_sum00, _sum01, outptr, dstptr);

            r0 += 8;
            r1 += 8;
            r2 += 8;
            r3 += 8;
            r4 += 8;
            r5 += 8;
            r6 += 8;
            outptr += 8;
            dstptr += 8;
        }
        for (; w < width; w++) {
            uint32x4_t _sum00;
            if (!first_ic) {
                _sum00 = vld1q_u32(outptr);
            } else {
                _sum00 = vdupq_n_u32(SHIFT);
            }

            _tmp = vld1q_u8(r0);
            CALC_0(123, 456, 0);

            _tmp = vld1q_u8(r1);
            CALC_0(78910, 111213, 0);

            _tmp = vld1q_u8(r2);
            CALC_0(14151617, 181920, 0);

            _tmp = vld1q_u8(r3);
            CALC_0(21222324, 252627, 0);

            _tmp = vld1q_u8(r4);
            CALC_0(28293031, 323334, 0);

            _tmp = vld1q_u8(r5);
            CALC_0(35363738, 394041, 0);

            _tmp = vld1q_u8(r6);
            CALC_0(42434445, 464748, 0);

            if (last_ic) {
                CALC_DST(_sum00);
            }
            POSTPROCESS_1X4(_sum00, outptr, dstptr);

            r0 += 4;
            r1 += 4;
            r2 += 4;
            r3 += 4;
            r4 += 4;
            r5 += 4;
            r6 += 4;
            outptr += 4;
            dstptr += 4;
        }
        r0 += tail_step;
        r1 += tail_step;
        r2 += tail_step;
        r3 += tail_step;
        r4 += tail_step;
        r5 += tail_step;
        r6 += tail_step;
    }
}

template <bool first_ic, bool last_ic, bool fused_kern, BiasMode bias_mode,
          typename Op>
void conv_bias::conv_direct_stride2_5x5_quint8_dot(
        const uint8_t* src, const uint8_t* filter, const int32_t* bias,
        int32_t* temp, uint8_t* dst, const size_t IH, const size_t IW,
        const size_t OH, const size_t OW, const uint8_t src_zp,
        const uint8_t filter_zp, const int32_t src_filter_zp, const Op& op) {
    MEGDNN_MARK_USED_VAR(IH);
    const size_t tail_step = IW - 2 * OW + IW;

    uint8x16_t _src_zp = vdupq_n_u8(src_zp);
    uint8x16_t _filter_zp = vdupq_n_u8(filter_zp);
    int32x4_t _shift_zp;
    if (bias_mode != BiasMode::NO_BIAS) {
        _shift_zp = vdupq_n_s32(src_filter_zp - SHIFT + bias[0]);
    } else {
        _shift_zp = vdupq_n_s32(src_filter_zp - SHIFT);
    }

    const uint8x16_t _idx00 = {0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, 7, 6, 7, 8, 9};
    const uint8x16_t _idx01 = {4, 16, 16, 16, 6,  16, 16, 16,
                               8, 16, 16, 16, 10, 16, 16, 16};
    //! start from 8
    const uint8x16_t& _idx10 = _idx00;
    const uint8x16_t& _idx11 = _idx01;

    uint8x16_t _tmp, _elem;
    uint32x4_t _elem2;
    uint32_t* outptr = reinterpret_cast<uint32_t*>(temp);
    uint32_t* outptr2 = outptr + OW;
    uint8_t* dstptr = dst;
    uint8_t* dstptr2 = dstptr + OW;

    const uint8_t* r0 = src;
    const uint8_t* r1 = src + IW;
    const uint8_t* r2 = src + IW * 2;
    const uint8_t* r3 = src + IW * 3;
    const uint8_t* r4 = src + IW * 4;
    const uint8_t* r5 = src + IW * 5;
    const uint8_t* r6 = src + IW * 6;

    const uint8_t* k0 = filter;

    uint8x16_t _k = vld1q_u8(k0);
    //! filter row 1
    uint8x16_t _idx = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    uint8x16_t _k123 = vqtbl1q_s8_v7(_k, _idx);
    _idx = {4, 16, 16, 16, 4, 16, 16, 16, 4, 16, 16, 16, 4, 16, 16, 16};
    uint8x16_t _k4 = vqtbl1q_s8_v7(_k, _idx);
    //! filter row 2
    _idx = {5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8};
    uint8x16_t _k5678 = vqtbl1q_s8_v7(_k, _idx);
    _idx = {9, 16, 16, 16, 9, 16, 16, 16, 9, 16, 16, 16, 9, 16, 16, 16};
    uint8x16_t _k9 = vqtbl1q_s8_v7(_k, _idx);
    //! filter row 3
    _idx = {10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13, 10, 11, 12, 13};
    uint8x16_t _k10111213 = vqtbl1q_s8_v7(_k, _idx);
    _idx = {14, 16, 16, 16, 14, 16, 16, 16, 14, 16, 16, 16, 14, 16, 16, 16};
    uint8x16_t _k14 = vqtbl1q_s8_v7(_k, _idx);
    //! 9 10 11 12 -> 13 14 15 16 -> 17 18 19 20 -> 21 22 23 24
    _k = vld1q_u8(k0 + 9);
    //! filter row 4
    _idx = {6, 7, 8, 9, 6, 7, 8, 9, 6, 7, 8, 9, 6, 7, 8, 9};
    uint8x16_t _k15161718 = vqtbl1q_s8_v7(_k, _idx);
    _idx = {10, 16, 16, 16, 10, 16, 16, 16, 10, 16, 16, 16, 10, 16, 16, 16};
    uint8x16_t _k19 = vqtbl1q_s8_v7(_k, _idx);
    //! filter row 5
    _idx = {11, 12, 13, 14, 11, 12, 13, 14, 11, 12, 13, 14, 11, 12, 13, 14};
    uint8x16_t _k20212223 = vqtbl1q_s8_v7(_k, _idx);
    _idx = {15, 16, 16, 16, 15, 16, 16, 16, 15, 16, 16, 16, 15, 16, 16, 16};
    uint8x16_t _k24 = vqtbl1q_s8_v7(_k, _idx);

    const int width = OW >> 2;
    size_t h = 0;
    for (; h + 1 < OH; h += 2) {
        int w = 0;
        for (; w + 2 < width; w += 2) {
            uint32x4_t _sum00, _sum01, _sum10, _sum11;
            if (!first_ic) {
                _sum00 = vld1q_u32(outptr);
                _sum01 = vld1q_u32(outptr + 4);
                _sum10 = vld1q_u32(outptr2);
                _sum11 = vld1q_u32(outptr2 + 4);
            } else {
                _sum00 = vdupq_n_u32(SHIFT);
                _sum01 = vdupq_n_u32(SHIFT);
                _sum10 = vdupq_n_u32(SHIFT);
                _sum11 = vdupq_n_u32(SHIFT);
            }

            _tmp = vld1q_u8(r0);
            CALC_0(123, 4, 0);
            _tmp = vld1q_u8(r0 + 8);
            CALC_0(123, 4, 1);

            _tmp = vld1q_u8(r1);
            CALC_0(5678, 9, 0);
            _tmp = vld1q_u8(r1 + 8);
            CALC_0(5678, 9, 1);

            _tmp = vld1q_u8(r2);
            CALC_2(10111213, 14, 123, 4, 0);
            _tmp = vld1q_u8(r2 + 8);
            CALC_2(10111213, 14, 123, 4, 1);

            _tmp = vld1q_u8(r3);
            CALC_2(15161718, 19, 5678, 9, 0);
            _tmp = vld1q_u8(r3 + 8);
            CALC_2(15161718, 19, 5678, 9, 1);

            _tmp = vld1q_u8(r4);
            CALC_2(20212223, 24, 10111213, 14, 0);
            _tmp = vld1q_u8(r4 + 8);
            CALC_2(20212223, 24, 10111213, 14, 1);

            _tmp = vld1q_u8(r5);
            CALC_1(15161718, 19, 0);
            _tmp = vld1q_u8(r5 + 8);
            CALC_1(15161718, 19, 1);

            _tmp = vld1q_u8(r6);
            CALC_1(20212223, 24, 0);
            _tmp = vld1q_u8(r6 + 8);
            CALC_1(20212223, 24, 1);

            if (last_ic) {
                CALC_DST(_sum00);
                CALC_DST(_sum01);
                CALC_DST(_sum10);
                CALC_DST(_sum11);
            }
            POSTPROCESS_1X8(_sum00, _sum01, outptr, dstptr);
            POSTPROCESS_1X8(_sum10, _sum11, outptr2, dstptr2);

            r0 += 16;
            r1 += 16;
            r2 += 16;
            r3 += 16;
            r4 += 16;
            r5 += 16;
            r6 += 16;
            outptr += 8;
            outptr2 += 8;
            dstptr += 8;
            dstptr2 += 8;
        }
        for (; w < width; w++) {
            uint32x4_t _sum00, _sum10;
            if (!first_ic) {
                _sum00 = vld1q_u32(outptr);
                _sum10 = vld1q_u32(outptr2);
            } else {
                _sum00 = vdupq_n_u32(SHIFT);
                _sum10 = vdupq_n_u32(SHIFT);
            }

            _tmp = vld1q_u8(r0);
            CALC_0(123, 4, 0);

            _tmp = vld1q_u8(r1);
            CALC_0(5678, 9, 0);

            _tmp = vld1q_u8(r2);
            CALC_2(10111213, 14, 123, 4, 0);

            _tmp = vld1q_u8(r3);
            CALC_2(15161718, 19, 5678, 9, 0);

            _tmp = vld1q_u8(r4);
            CALC_2(20212223, 24, 10111213, 14, 0);

            _tmp = vld1q_u8(r5);
            CALC_1(15161718, 19, 0);

            _tmp = vld1q_u8(r6);
            CALC_1(20212223, 24, 0);

            if (last_ic) {
                CALC_DST(_sum00);
                CALC_DST(_sum10);
            }
            POSTPROCESS_2X4(_sum00, _sum10, outptr, outptr2, dstptr, dstptr2);

            r0 += 8;
            r1 += 8;
            r2 += 8;
            r3 += 8;
            r4 += 8;
            r5 += 8;
            r6 += 8;
            outptr += 4;
            outptr2 += 4;
            dstptr += 4;
            dstptr2 += 4;
        }

        r0 += tail_step + IW * 2;
        r1 += tail_step + IW * 2;
        r2 += tail_step + IW * 2;
        r3 += tail_step + IW * 2;
        r4 += tail_step + IW * 2;
        r5 += tail_step + IW * 2;
        r6 += tail_step + IW * 2;

        outptr += OW;
        outptr2 += OW;
        dstptr += OW;
        dstptr2 += OW;
    }

    for (; h < OH; h++) {
        int w = 0;
        for (; w + 2 < width; w += 2) {
            uint32x4_t _sum00, _sum01;
            if (!first_ic) {
                _sum00 = vld1q_u32(outptr);
                _sum01 = vld1q_u32(outptr + 4);
            } else {
                _sum00 = vdupq_n_u32(SHIFT);
                _sum01 = vdupq_n_u32(SHIFT);
            }

            _tmp = vld1q_u8(r0);
            CALC_0(123, 4, 0);
            _tmp = vld1q_u8(r0 + 8);
            CALC_0(123, 4, 1);

            _tmp = vld1q_u8(r1);
            CALC_0(5678, 9, 0);
            _tmp = vld1q_u8(r1 + 8);
            CALC_0(5678, 9, 1);

            _tmp = vld1q_u8(r2);
            CALC_0(10111213, 14, 0);
            _tmp = vld1q_u8(r2 + 8);
            CALC_0(10111213, 14, 1);

            _tmp = vld1q_u8(r3);
            CALC_0(15161718, 19, 0);
            _tmp = vld1q_u8(r3 + 8);
            CALC_0(15161718, 19, 1);

            _tmp = vld1q_u8(r4);
            CALC_0(20212223, 24, 0);
            _tmp = vld1q_u8(r4 + 8);
            CALC_0(20212223, 24, 1);

            if (last_ic) {
                CALC_DST(_sum00);
                CALC_DST(_sum01);
            }
            POSTPROCESS_1X8(_sum00, _sum01, outptr, dstptr);

            r0 += 16;
            r1 += 16;
            r2 += 16;
            r3 += 16;
            r4 += 16;
            outptr += 8;
            dstptr += 8;
        }
        for (; w < width; w++) {
            uint32x4_t _sum00;
            if (!first_ic) {
                _sum00 = vld1q_u32(outptr);
            } else {
                _sum00 = vdupq_n_u32(SHIFT);
            }

            _tmp = vld1q_u8(r0);
            CALC_0(123, 4, 0);

            _tmp = vld1q_u8(r1);
            CALC_0(5678, 9, 0);

            _tmp = vld1q_u8(r2);
            CALC_0(10111213, 14, 0);

            _tmp = vld1q_u8(r3);
            CALC_0(15161718, 19, 0);

            _tmp = vld1q_u8(r4);
            CALC_0(20212223, 24, 0);

            if (last_ic) {
                CALC_DST(_sum00);
            }
            POSTPROCESS_1X4(_sum00, outptr, dstptr);

            r0 += 8;
            r1 += 8;
            r2 += 8;
            r3 += 8;
            r4 += 8;
            outptr += 4;
            dstptr += 4;
        }
        r0 += tail_step;
        r1 += tail_step;
        r2 += tail_step;
        r3 += tail_step;
        r4 += tail_step;
    }
}

template <bool first_ic, bool last_ic, bool fused_kern, BiasMode bias_mode,
          typename Op>
void conv_bias::conv_direct_stride2_7x7_quint8_dot(
        const uint8_t* src, const uint8_t* filter, const int32_t* bias,
        int32_t* temp, uint8_t* dst, const size_t IH, const size_t IW,
        const size_t OH, const size_t OW, const uint8_t src_zp,
        const uint8_t filter_zp, const int32_t src_filter_zp, const Op& op) {
    MEGDNN_MARK_USED_VAR(IH);
    const size_t tail_step = IW - 2 * OW + IW;

    uint8x16_t _src_zp = vdupq_n_u8(src_zp);
    uint8x16_t _filter_zp = vdupq_n_u8(filter_zp);
    int32x4_t _shift_zp;
    if (bias_mode != BiasMode::NO_BIAS) {
        _shift_zp = vdupq_n_s32(src_filter_zp - SHIFT + bias[0]);
    } else {
        _shift_zp = vdupq_n_s32(src_filter_zp - SHIFT);
    }

    const uint8x16_t _idx00 = {0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, 7, 6, 7, 8, 9};
    const uint8x16_t _idx01 = {4, 5, 6,  16, 6,  7,  8,  16,
                               8, 9, 10, 16, 10, 11, 12, 16};
    //! start from 8
    const uint8x16_t& _idx10 = _idx00;
    const uint8x16_t& _idx11 = _idx01;

    uint8x16_t _tmp, _elem;
    uint32x4_t _elem2;
    uint32_t* outptr = reinterpret_cast<uint32_t*>(temp);
    uint32_t* outptr2 = outptr + OW;
    uint8_t* dstptr = dst;
    uint8_t* dstptr2 = dstptr + OW;

    const uint8_t* r0 = src;
    const uint8_t* r1 = src + IW;
    const uint8_t* r2 = src + IW * 2;
    const uint8_t* r3 = src + IW * 3;
    const uint8_t* r4 = src + IW * 4;
    const uint8_t* r5 = src + IW * 5;
    const uint8_t* r6 = src + IW * 6;
    const uint8_t* r7 = src + IW * 7;
    const uint8_t* r8 = src + IW * 8;

    const uint8_t* k0 = filter;

    uint8x16_t _k = vld1q_u8(k0);
    //! filter row 1
    uint8x16_t _idx = {0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3};
    uint8x16_t _k123 = vqtbl1q_s8_v7(_k, _idx);
    _idx = {4, 5, 6, 16, 4, 5, 6, 16, 4, 5, 6, 16, 4, 5, 6, 16};
    uint8x16_t _k456 = vqtbl1q_s8_v7(_k, _idx);
    //! filter row 2
    _idx = {7, 8, 9, 10, 7, 8, 9, 10, 7, 8, 9, 10, 7, 8, 9, 10};
    uint8x16_t _k78910 = vqtbl1q_s8_v7(_k, _idx);
    _idx = {11, 12, 13, 16, 11, 12, 13, 16, 11, 12, 13, 16, 11, 12, 13, 16};
    uint8x16_t _k111213 = vqtbl1q_s8_v7(_k, _idx);

    //! 12 13 14 15 -> 16 17 18 19 -> 20 21 22 23 -> 24 25 26 27
    _k = vld1q_u8(k0 + 12);
    //! filter row 3
    _idx = {2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5};
    uint8x16_t _k14151617 = vqtbl1q_s8_v7(_k, _idx);
    _idx = {6, 7, 8, 16, 6, 7, 8, 16, 6, 7, 8, 16, 6, 7, 8, 16};
    uint8x16_t _k181920 = vqtbl1q_s8_v7(_k, _idx);
    //! filter row 4
    _idx = {9, 10, 11, 12, 9, 10, 11, 12, 9, 10, 11, 12, 9, 10, 11, 12};
    uint8x16_t _k21222324 = vqtbl1q_s8_v7(_k, _idx);
    _idx = {13, 14, 15, 16, 13, 14, 15, 16, 13, 14, 15, 16, 13, 14, 15, 16};
    uint8x16_t _k252627 = vqtbl1q_s8_v7(_k, _idx);

    //! 24 25 26 27->28 29 30 31 -> 32 33 34 35 -> 36 37 38 39
    _k = vld1q_u8(k0 + 24);
    //! filter row 5
    _idx = {4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7};
    uint8x16_t _k28293031 = vqtbl1q_s8_v7(_k, _idx);
    _idx = {8, 9, 10, 16, 8, 9, 10, 16, 8, 9, 10, 16, 8, 9, 10, 16};
    uint8x16_t _k323334 = vqtbl1q_s8_v7(_k, _idx);

    //! 33 34 35 36 -> 37 38 39 40 -> 41 42 43 44 -> 45 46 47 48
    _k = vld1q_u8(k0 + 33);
    //! filter row 6
    _idx = {2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5, 2, 3, 4, 5};
    uint8x16_t _k35363738 = vqtbl1q_s8_v7(_k, _idx);
    _idx = {6, 7, 8, 16, 6, 7, 8, 16, 6, 7, 8, 16, 6, 7, 8, 16};
    uint8x16_t _k394041 = vqtbl1q_s8_v7(_k, _idx);

    //! filter row 7
    _idx = {9, 10, 11, 12, 9, 10, 11, 12, 9, 10, 11, 12, 9, 10, 11, 12};
    uint8x16_t _k42434445 = vqtbl1q_s8_v7(_k, _idx);
    _idx = {13, 14, 15, 16, 13, 14, 15, 16, 13, 14, 15, 16, 13, 14, 15, 16};
    uint8x16_t _k464748 = vqtbl1q_s8_v7(_k, _idx);

    const int width = OW >> 2;
    size_t h = 0;
    for (; h + 1 < OH; h += 2) {
        int w = 0;
        for (; w + 2 < width; w += 2) {
            uint32x4_t _sum00, _sum01, _sum10, _sum11;
            if (!first_ic) {
                _sum00 = vld1q_u32(outptr);
                _sum01 = vld1q_u32(outptr + 4);
                _sum10 = vld1q_u32(outptr2);
                _sum11 = vld1q_u32(outptr2 + 4);
            } else {
                _sum00 = vdupq_n_u32(SHIFT);
                _sum01 = vdupq_n_u32(SHIFT);
                _sum10 = vdupq_n_u32(SHIFT);
                _sum11 = vdupq_n_u32(SHIFT);
            }

            _tmp = vld1q_u8(r0);
            CALC_0(123, 456, 0);
            _tmp = vld1q_u8(r0 + 8);
            CALC_0(123, 456, 1);

            _tmp = vld1q_u8(r1);
            CALC_0(78910, 111213, 0);
            _tmp = vld1q_u8(r1 + 8);
            CALC_0(78910, 111213, 1);

            _tmp = vld1q_u8(r2);
            CALC_2(14151617, 181920, 123, 456, 0);
            _tmp = vld1q_u8(r2 + 8);
            CALC_2(14151617, 181920, 123, 456, 1);

            _tmp = vld1q_u8(r3);
            CALC_2(21222324, 252627, 78910, 111213, 0);
            _tmp = vld1q_u8(r3 + 8);
            CALC_2(21222324, 252627, 78910, 111213, 1);

            _tmp = vld1q_u8(r4);
            CALC_2(28293031, 323334, 14151617, 181920, 0);
            _tmp = vld1q_u8(r4 + 8);
            CALC_2(28293031, 323334, 14151617, 181920, 1);

            _tmp = vld1q_u8(r5);
            CALC_2(35363738, 394041, 21222324, 252627, 0);
            _tmp = vld1q_u8(r5 + 8);
            CALC_2(35363738, 394041, 21222324, 252627, 1);

            _tmp = vld1q_u8(r6);
            CALC_2(42434445, 464748, 28293031, 323334, 0);
            _tmp = vld1q_u8(r6 + 8);
            CALC_2(42434445, 464748, 28293031, 323334, 1);

            _tmp = vld1q_u8(r7);
            CALC_1(35363738, 394041, 0);
            _tmp = vld1q_u8(r7 + 8);
            CALC_1(35363738, 394041, 1);

            _tmp = vld1q_u8(r8);
            CALC_1(42434445, 464748, 0);
            _tmp = vld1q_u8(r8 + 8);
            CALC_1(42434445, 464748, 1);

            if (last_ic) {
                CALC_DST(_sum00);
                CALC_DST(_sum01);
                CALC_DST(_sum10);
                CALC_DST(_sum11);
            }
            POSTPROCESS_1X8(_sum00, _sum01, outptr, dstptr);
            POSTPROCESS_1X8(_sum10, _sum11, outptr2, dstptr2);

            r0 += 16;
            r1 += 16;
            r2 += 16;
            r3 += 16;
            r4 += 16;
            r5 += 16;
            r6 += 16;
            r7 += 16;
            r8 += 16;
            outptr += 8;
            outptr2 += 8;
            dstptr += 8;
            dstptr2 += 8;
        }
        for (; w < width; w++) {
            uint32x4_t _sum00, _sum10;
            if (!first_ic) {
                _sum00 = vld1q_u32(outptr);
                _sum10 = vld1q_u32(outptr2);
            } else {
                _sum00 = vdupq_n_u32(SHIFT);
                _sum10 = vdupq_n_u32(SHIFT);
            }

            _tmp = vld1q_u8(r0);
            CALC_0(123, 456, 0);

            _tmp = vld1q_u8(r1);
            CALC_0(78910, 111213, 0);

            _tmp = vld1q_u8(r2);
            CALC_2(14151617, 181920, 123, 456, 0);

            _tmp = vld1q_u8(r3);
            CALC_2(21222324, 252627, 78910, 111213, 0);

            _tmp = vld1q_u8(r4);
            CALC_2(28293031, 323334, 14151617, 181920, 0);

            _tmp = vld1q_u8(r5);
            CALC_2(35363738, 394041, 21222324, 252627, 0);

            _tmp = vld1q_u8(r6);
            CALC_2(42434445, 464748, 28293031, 323334, 0);

            _tmp = vld1q_u8(r7);
            CALC_1(35363738, 394041, 0);

            _tmp = vld1q_u8(r8);
            CALC_1(42434445, 464748, 0);

            if (last_ic) {
                CALC_DST(_sum00);
                CALC_DST(_sum10);
            }
            POSTPROCESS_2X4(_sum00, _sum10, outptr, outptr2, dstptr, dstptr2);

            r0 += 8;
            r1 += 8;
            r2 += 8;
            r3 += 8;
            r4 += 8;
            r5 += 8;
            r6 += 8;
            r7 += 8;
            r8 += 8;
            outptr += 4;
            outptr2 += 4;
            dstptr += 4;
            dstptr2 += 4;
        }

        r0 += tail_step + IW * 2;
        r1 += tail_step + IW * 2;
        r2 += tail_step + IW * 2;
        r3 += tail_step + IW * 2;
        r4 += tail_step + IW * 2;
        r5 += tail_step + IW * 2;
        r6 += tail_step + IW * 2;
        r7 += tail_step + IW * 2;
        r8 += tail_step + IW * 2;

        outptr += OW;
        outptr2 += OW;
        dstptr += OW;
        dstptr2 += OW;
    }

    for (; h < OH; h++) {
        int w = 0;
        for (; w + 2 < width; w += 2) {
            uint32x4_t _sum00, _sum01;
            if (!first_ic) {
                _sum00 = vld1q_u32(outptr);
                _sum01 = vld1q_u32(outptr + 4);
            } else {
                _sum00 = vdupq_n_u32(SHIFT);
                _sum01 = vdupq_n_u32(SHIFT);
            }

            _tmp = vld1q_u8(r0);
            CALC_0(123, 456, 0);
            _tmp = vld1q_u8(r0 + 8);
            CALC_0(123, 456, 1);

            _tmp = vld1q_u8(r1);
            CALC_0(78910, 111213, 0);
            _tmp = vld1q_u8(r1 + 8);
            CALC_0(78910, 111213, 1);

            _tmp = vld1q_u8(r2);
            CALC_0(14151617, 181920, 0);
            _tmp = vld1q_u8(r2 + 8);
            CALC_0(14151617, 181920, 1);

            _tmp = vld1q_u8(r3);
            CALC_0(21222324, 252627, 0);
            _tmp = vld1q_u8(r3 + 8);
            CALC_0(21222324, 252627, 1);

            _tmp = vld1q_u8(r4);
            CALC_0(28293031, 323334, 0);
            _tmp = vld1q_u8(r4 + 8);
            CALC_0(28293031, 323334, 1);

            _tmp = vld1q_u8(r5);
            CALC_0(35363738, 394041, 0);
            _tmp = vld1q_u8(r5 + 8);
            CALC_0(35363738, 394041, 1);

            _tmp = vld1q_u8(r6);
            CALC_0(42434445, 464748, 0);
            _tmp = vld1q_u8(r6 + 8);
            CALC_0(42434445, 464748, 1);

            if (last_ic) {
                CALC_DST(_sum00);
                CALC_DST(_sum01);
            }
            POSTPROCESS_1X8(_sum00, _sum01, outptr, dstptr);

            r0 += 16;
            r1 += 16;
            r2 += 16;
            r3 += 16;
            r4 += 16;
            r5 += 16;
            r6 += 16;
            outptr += 8;
            dstptr += 8;
        }
        for (; w < width; w++) {
            uint32x4_t _sum00;
            if (!first_ic) {
                _sum00 = vld1q_u32(outptr);
            } else {
                _sum00 = vdupq_n_u32(SHIFT);
            }

            _tmp = vld1q_u8(r0);
            CALC_0(123, 456, 0);

            _tmp = vld1q_u8(r1);
            CALC_0(78910, 111213, 0);

            _tmp = vld1q_u8(r2);
            CALC_0(14151617, 181920, 0);

            _tmp = vld1q_u8(r3);
            CALC_0(21222324, 252627, 0);

            _tmp = vld1q_u8(r4);
            CALC_0(28293031, 323334, 0);

            _tmp = vld1q_u8(r5);
            CALC_0(35363738, 394041, 0);

            _tmp = vld1q_u8(r6);
            CALC_0(42434445, 464748, 0);

            if (last_ic) {
                CALC_DST(_sum00);
            }
            POSTPROCESS_1X4(_sum00, outptr, dstptr);

            r0 += 8;
            r1 += 8;
            r2 += 8;
            r3 += 8;
            r4 += 8;
            r5 += 8;
            r6 += 8;
            outptr += 4;
            dstptr += 4;
        }
        r0 += tail_step;
        r1 += tail_step;
        r2 += tail_step;
        r3 += tail_step;
        r4 += tail_step;
        r5 += tail_step;
        r6 += tail_step;
    }
}

#undef CALC_0
#undef CALC_1
#undef CALC_2

#undef POSTPROCESS_1X8
#undef POSTPROCESS2_1X8
#undef POSTPROCESS_2X4
#undef POSTPROCESS_1X4
#undef POSTPROCESS_1X1
#undef ST1_S32X4
#undef ST2_S32X4X2

#define INSTANTIATION(stride, i, first_ic, last_ic, fused_kern, bias, Op)     \
    template void conv_bias::conv_direct_##stride##_##i##x##i##_quint8_dot<       \
            first_ic, last_ic, fused_kern, bias, Op>(                         \
            const uint8_t*, const uint8_t*, const int32_t*, int32_t*,         \
            uint8_t*, const size_t, const size_t, const size_t, const size_t, \
            const uint8_t, const uint8_t, const int32_t, const Op&);

#define FOR_NONLINEAR(stride, i, first_ic, last_ic, fused_kern, bias) \
    INSTANTIATION(stride, i, first_ic, last_ic, fused_kern, bias,     \
                  TypeCvtOp<dt_qint32 MEGDNN_COMMA dt_quint8>)        \
    INSTANTIATION(stride, i, first_ic, last_ic, fused_kern, bias,     \
                  ReluOp<dt_qint32 MEGDNN_COMMA dt_quint8>)           \
    INSTANTIATION(stride, i, first_ic, last_ic, fused_kern, bias,     \
                  HSwishOp<dt_qint32 MEGDNN_COMMA dt_quint8>)

#define FOR_BIAS(stride, i, first_ic, last_ic, fused_kern)                     \
    FOR_NONLINEAR(stride, i, first_ic, last_ic, fused_kern, BiasMode::NO_BIAS) \
    FOR_NONLINEAR(stride, i, first_ic, last_ic, fused_kern,                    \
                  BiasMode::BROADCAST_CHANNEL_BIAS)

#define FOR_KERN(stride, i, first_ic, last_ic)   \
    FOR_BIAS(stride, i, first_ic, last_ic, true) \
    FOR_BIAS(stride, i, first_ic, last_ic, false)

#define FOR_IC(stride, i)             \
    FOR_KERN(stride, i, true, true)   \
    FOR_KERN(stride, i, true, false)  \
    FOR_KERN(stride, i, false, false) \
    FOR_KERN(stride, i, false, true)

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

#endif
// vim: syntax=cpp.doxygen
