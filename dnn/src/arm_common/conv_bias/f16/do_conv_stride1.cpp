/**
 * \file dnn/src/arm_common/conv_bias/f16/do_conv_stride1.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include <algorithm>

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#include "./do_conv_stride1.h"
#include "src/arm_common/simd_macro/marm_neon.h"
#include "src/arm_common/conv_bias/postprocess_helper.h"

using namespace megdnn;
using namespace arm_common;
using namespace fp16;
using namespace conv_stride1;

using NCBKernSizeParam = fallback::ConvBiasImpl::NCBKernSizeParam;
using NCBKernParam = fallback::ConvBiasImpl::NCBKernParam;


void conv_stride1::do_conv_2x2_stride1(const __fp16* src, const __fp16* filter, __fp16* dst,
                         size_t IH, size_t IW, size_t OH, size_t OW,
                         size_t IC) {
    const size_t tail_step = IW - OW;
    //! unroll of 2
    size_t ic = 0;
    for (; ic + 1 < IC; ic += 2) {
        const __fp16* src_ptr = src + IW * IH * ic;
        const __fp16* src_ptr1 = src_ptr + IW * IH;
        __fp16* outptr = dst;

        const __fp16* r00 = src_ptr;
        const __fp16* r01 = src_ptr + IW;
        const __fp16* r10 = src_ptr1;
        const __fp16* r11 = src_ptr1 + IW;

        const __fp16* k0 = filter + ic * 4;

        float16x8_t _k0 = vld1q_f16(k0);
        rep(h, OH) {
            int width = OW >> 3;

            rep(i, width) {
                float16x8_t _r000 = vld1q_f16(r00);
                float16x8_t _r010 = vld1q_f16(r01);
                float16x8_t _r001 = vld1q_f16(r00 + 1);
                float16x8_t _r011 = vld1q_f16(r01 + 1);

                float16x8_t _r100 = vld1q_f16(r10);
                float16x8_t _r110 = vld1q_f16(r11);
                float16x8_t _r101 = vld1q_f16(r10 + 1);
                float16x8_t _r111 = vld1q_f16(r11 + 1);

                float16x8_t _sum = vld1q_f16(outptr);

                _sum = vmlaq_lane_f16(_sum, _r000, vget_low_f16(_k0), 0);
                _sum = vmlaq_lane_f16(_sum, _r001, vget_low_f16(_k0), 1);
                _sum = vmlaq_lane_f16(_sum, _r010, vget_low_f16(_k0), 2);
                _sum = vmlaq_lane_f16(_sum, _r011, vget_low_f16(_k0), 3);

                _sum = vmlaq_lane_f16(_sum, _r100, vget_high_f16(_k0), 0);
                _sum = vmlaq_lane_f16(_sum, _r101, vget_high_f16(_k0), 1);
                _sum = vmlaq_lane_f16(_sum, _r110, vget_high_f16(_k0), 2);
                _sum = vmlaq_lane_f16(_sum, _r111, vget_high_f16(_k0), 3);

                vst1q_f16(outptr, _sum);

                r00 += 8;
                r01 += 8;
                r10 += 8;
                r11 += 8;
                outptr += 8;
            }

            r00 += tail_step;
            r01 += tail_step;
            r10 += tail_step;
            r11 += tail_step;
        }
    }
    for (; ic < IC; ic++) {
        const __fp16* src_ptr = src + IW * IH * ic;
        __fp16* outptr = dst;

        const __fp16* r0 = src_ptr;
        const __fp16* r1 = src_ptr + IW;

        const __fp16* k0 = filter + ic * 4;

        float16x8_t _k0 = vdupq_n_f16(k0[0]);
        float16x8_t _k1 = vdupq_n_f16(k0[1]);
        float16x8_t _k2 = vdupq_n_f16(k0[2]);
        float16x8_t _k3 = vdupq_n_f16(k0[3]);
        rep(h, OH) {
            int width = OW >> 3;

            rep(i, width) {
                float16x8_t _r00 = vld1q_f16(r0);
                float16x8_t _r10 = vld1q_f16(r1);
                float16x8_t _r01 = vld1q_f16(r0 + 1);
                float16x8_t _r11 = vld1q_f16(r1 + 1);

                float16x8_t _sum = vld1q_f16(outptr);
                float16x8_t _sum2;

                _sum = vmlaq_f16(_sum, _r00, _k0);
                _sum2 = vmulq_f16(_r01, _k1);
                _sum = vmlaq_f16(_sum, _r10, _k2);
                _sum2 = vmlaq_f16(_sum2, _r11, _k3);

                _sum = vaddq_f16(_sum, _sum2);

                vst1q_f16(outptr, _sum);

                r0 += 8;
                r1 += 8;
                outptr += 8;
            }

            r0 += tail_step;
            r1 += tail_step;
        }
    }
}

void conv_stride1::do_conv_3x3_stride1(const __fp16* src, const __fp16* filter, __fp16* dst,
                         size_t IH, size_t IW, size_t OH, size_t OW,
                         size_t IC) {
    const size_t tail_step = IW - OW;

    rep(ic, IC) {
        const __fp16* src_ptr = src + IW * IH * ic;
        __fp16* outptr = dst;
        __fp16* outptr2 = outptr + OW;

        const __fp16* r0 = src_ptr;
        const __fp16* r1 = src_ptr + IW;
        const __fp16* r2 = src_ptr + IW * 2;
        const __fp16* r3 = src_ptr + IW * 3;

        float16x8_t _k01234567 = vld1q_f16(filter);
        float16x8_t _k12345678 = vld1q_f16(filter + 1);

        size_t h = 0;
        for (; h + 1 < OH; h += 2) {
            int width = OW >> 3;

            rep(i, width) {
                float16x8_t _sum1 = vld1q_f16(outptr);
                float16x8_t _sum2 = vdupq_n_f16(0.f);
                float16x8_t _sum3 = vld1q_f16(outptr2);
                float16x8_t _sum4 = vdupq_n_f16(0.f);

                float16x8_t _r00 = vld1q_f16(r0);
                float16x8_t _r00n = vld1q_f16(r0 + 8);
                float16x8_t _r01 = vextq_f16(_r00, _r00n, 1);
                float16x8_t _r02 = vextq_f16(_r00, _r00n, 2);

                float16x8_t _r10 = vld1q_f16(r1);
                float16x8_t _r10n = vld1q_f16(r1 + 8);
                float16x8_t _r11 = vextq_f16(_r10, _r10n, 1);
                float16x8_t _r12 = vextq_f16(_r10, _r10n, 2);

                float16x8_t _r20 = vld1q_f16(r2);
                float16x8_t _r20n = vld1q_f16(r2 + 8);
                float16x8_t _r21 = vextq_f16(_r20, _r20n, 1);
                float16x8_t _r22 = vextq_f16(_r20, _r20n, 2);

                float16x8_t _r30 = vld1q_f16(r3);
                float16x8_t _r30n = vld1q_f16(r3 + 8);
                float16x8_t _r31 = vextq_f16(_r30, _r30n, 1);
                float16x8_t _r32 = vextq_f16(_r30, _r30n, 2);

                _sum1 = vmlaq_low_lane_f16(_sum1, _r00, _k01234567, 0);
                _sum2 = vmlaq_low_lane_f16(_sum2, _r01, _k01234567, 1);
                _sum1 = vmlaq_low_lane_f16(_sum1, _r02, _k01234567, 2);
                _sum2 = vmlaq_low_lane_f16(_sum2, _r10, _k01234567, 3);
                _sum1 = vmlaq_high_lane_f16(_sum1, _r11, _k01234567, 4);
                _sum2 = vmlaq_high_lane_f16(_sum2, _r12, _k01234567, 5);
                _sum1 = vmlaq_high_lane_f16(_sum1, _r20, _k01234567, 6);
                _sum2 = vmlaq_high_lane_f16(_sum2, _r21, _k01234567, 7);
                _sum1 = vmlaq_high_lane_f16(_sum1, _r22, _k12345678, 7);

                _sum3 = vmlaq_low_lane_f16(_sum3, _r10, _k01234567, 0);
                _sum4 = vmlaq_low_lane_f16(_sum4, _r11, _k01234567, 1);
                _sum3 = vmlaq_low_lane_f16(_sum3, _r12, _k01234567, 2);
                _sum4 = vmlaq_low_lane_f16(_sum4, _r20, _k01234567, 3);
                _sum3 = vmlaq_high_lane_f16(_sum3, _r21, _k01234567, 4);
                _sum4 = vmlaq_high_lane_f16(_sum4, _r22, _k01234567, 5);
                _sum3 = vmlaq_high_lane_f16(_sum3, _r30, _k01234567, 6);
                _sum4 = vmlaq_high_lane_f16(_sum4, _r31, _k01234567, 7);
                _sum3 = vmlaq_high_lane_f16(_sum3, _r32, _k12345678, 7);

                _sum1 = vaddq_f16(_sum1, _sum2);
                _sum3 = vaddq_f16(_sum3, _sum4);

                vst1q_f16(outptr, _sum1);
                vst1q_f16(outptr2, _sum3);

                r0 += 8;
                r1 += 8;
                r2 += 8;
                r3 += 8;
                outptr += 8;
                outptr2 += 8;
            }

            r0 += tail_step + IW;
            r1 += tail_step + IW;
            r2 += tail_step + IW;
            r3 += tail_step + IW;

            outptr += OW;
            outptr2 += OW;
        }

        for (; h < OH; h++) {
            int width = OW >> 3;

            rep(i, width) {
                float16x8_t _sum1 = vld1q_f16(outptr);
                float16x8_t _sum2 = vdupq_n_f16(0.f);

                float16x8_t _r00 = vld1q_f16(r0);
                float16x8_t _r00n = vld1q_f16(r0 + 8);
                float16x8_t _r01 = vextq_f16(_r00, _r00n, 1);
                float16x8_t _r02 = vextq_f16(_r00, _r00n, 2);

                float16x8_t _r10 = vld1q_f16(r1);
                float16x8_t _r10n = vld1q_f16(r1 + 8);
                float16x8_t _r11 = vextq_f16(_r10, _r10n, 1);
                float16x8_t _r12 = vextq_f16(_r10, _r10n, 2);

                float16x8_t _r20 = vld1q_f16(r2);
                float16x8_t _r20n = vld1q_f16(r2 + 8);
                float16x8_t _r21 = vextq_f16(_r20, _r20n, 1);
                float16x8_t _r22 = vextq_f16(_r20, _r20n, 2);

                _sum1 = vmlaq_low_lane_f16(_sum1, _r00, _k01234567, 0);
                _sum2 = vmlaq_low_lane_f16(_sum2, _r01, _k01234567, 1);
                _sum1 = vmlaq_low_lane_f16(_sum1, _r02, _k01234567, 2);
                _sum2 = vmlaq_low_lane_f16(_sum2, _r10, _k01234567, 3);
                _sum1 = vmlaq_high_lane_f16(_sum1, _r11, _k01234567, 4);
                _sum2 = vmlaq_high_lane_f16(_sum2, _r12, _k01234567, 5);
                _sum1 = vmlaq_high_lane_f16(_sum1, _r20, _k01234567, 6);
                _sum2 = vmlaq_high_lane_f16(_sum2, _r21, _k01234567, 7);
                _sum1 = vmlaq_high_lane_f16(_sum1, _r22, _k12345678, 7);

                _sum1 = vaddq_f16(_sum1, _sum2);

                vst1q_f16(outptr, _sum1);

                r0 += 8;
                r1 += 8;
                r2 += 8;
                outptr += 8;
            }
            r0 += tail_step;
            r1 += tail_step;
            r2 += tail_step;
        }

        filter += 9;
    }
}

void conv_stride1::do_conv_5x5_stride1(const __fp16* src, const __fp16* filter, __fp16* dst,
                         size_t IH, size_t IW, size_t OH, size_t OW,
                         size_t IC) {
    const size_t tail_step = IW - OW;

    rep(ic, IC) {
        const __fp16* src_ptr = src + IW * IH * ic;
        __fp16* outptr = dst;
        __fp16* outptr2 = outptr + OW;

        const __fp16* r0 = src_ptr;
        const __fp16* r1 = src_ptr + IW;
        const __fp16* r2 = src_ptr + IW * 2;
        const __fp16* r3 = src_ptr + IW * 3;
        const __fp16* r4 = src_ptr + IW * 4;
        const __fp16* r5 = src_ptr + IW * 5;

        float16x8_t _k0 = vld1q_f16(filter);
        float16x8_t _k1 = vld1q_f16(filter + 8);
        float16x8_t _k2 = vld1q_f16(filter + 16);
        float16x8_t _k3 = vld1q_f16(filter + 17);

        size_t h = 0;
        for (; h + 1 < OH; h += 2) {
            int width = OW >> 3;

            rep(i, width) {
                float16x8_t _sum = vld1q_f16(outptr);
                float16x8_t _sum2 = vld1q_f16(outptr2);

                float16x8_t _r00 = vld1q_f16(r0);
                float16x8_t _r05 = vld1q_f16(r0 + 8);
                float16x8_t _r01 = vextq_f16(_r00, _r05, 1);
                float16x8_t _r02 = vextq_f16(_r00, _r05, 2);
                float16x8_t _r03 = vextq_f16(_r00, _r05, 3);
                float16x8_t _r04 = vextq_f16(_r00, _r05, 4);

                float16x8_t _r10 = vld1q_f16(r1);
                float16x8_t _r15 = vld1q_f16(r1 + 8);
                float16x8_t _r11 = vextq_f16(_r10, _r15, 1);
                float16x8_t _r12 = vextq_f16(_r10, _r15, 2);
                float16x8_t _r13 = vextq_f16(_r10, _r15, 3);
                float16x8_t _r14 = vextq_f16(_r10, _r15, 4);

                float16x8_t _r20 = vld1q_f16(r2);
                float16x8_t _r25 = vld1q_f16(r2 + 8);
                float16x8_t _r21 = vextq_f16(_r20, _r25, 1);
                float16x8_t _r22 = vextq_f16(_r20, _r25, 2);
                float16x8_t _r23 = vextq_f16(_r20, _r25, 3);
                float16x8_t _r24 = vextq_f16(_r20, _r25, 4);

                float16x8_t _r30 = vld1q_f16(r3);
                float16x8_t _r35 = vld1q_f16(r3 + 8);
                float16x8_t _r31 = vextq_f16(_r30, _r35, 1);
                float16x8_t _r32 = vextq_f16(_r30, _r35, 2);
                float16x8_t _r33 = vextq_f16(_r30, _r35, 3);
                float16x8_t _r34 = vextq_f16(_r30, _r35, 4);

                float16x8_t _r40 = vld1q_f16(r4);
                float16x8_t _r45 = vld1q_f16(r4 + 8);
                float16x8_t _r41 = vextq_f16(_r40, _r45, 1);
                float16x8_t _r42 = vextq_f16(_r40, _r45, 2);
                float16x8_t _r43 = vextq_f16(_r40, _r45, 3);
                float16x8_t _r44 = vextq_f16(_r40, _r45, 4);

                float16x8_t _r50 = vld1q_f16(r5);
                float16x8_t _r55 = vld1q_f16(r5 + 8);
                float16x8_t _r51 = vextq_f16(_r50, _r55, 1);
                float16x8_t _r52 = vextq_f16(_r50, _r55, 2);
                float16x8_t _r53 = vextq_f16(_r50, _r55, 3);
                float16x8_t _r54 = vextq_f16(_r50, _r55, 4);

                _sum = vmlaq_low_lane_f16(_sum, _r00, _k0, 0);
                _sum = vmlaq_low_lane_f16(_sum, _r01, _k0, 1);
                _sum = vmlaq_low_lane_f16(_sum, _r02, _k0, 2);
                _sum = vmlaq_low_lane_f16(_sum, _r03, _k0, 3);
                _sum = vmlaq_high_lane_f16(_sum, _r04, _k0, 4);

                _sum = vmlaq_high_lane_f16(_sum, _r10, _k0, 5);
                _sum = vmlaq_high_lane_f16(_sum, _r11, _k0, 6);
                _sum = vmlaq_high_lane_f16(_sum, _r12, _k0, 7);
                _sum = vmlaq_low_lane_f16(_sum, _r13, _k1, 0);
                _sum = vmlaq_low_lane_f16(_sum, _r14, _k1, 1);

                _sum = vmlaq_low_lane_f16(_sum, _r20, _k1, 2);
                _sum = vmlaq_low_lane_f16(_sum, _r21, _k1, 3);
                _sum = vmlaq_high_lane_f16(_sum, _r22, _k1, 4);
                _sum = vmlaq_high_lane_f16(_sum, _r23, _k1, 5);
                _sum = vmlaq_high_lane_f16(_sum, _r24, _k1, 6);

                _sum = vmlaq_high_lane_f16(_sum, _r30, _k1, 7);
                _sum = vmlaq_low_lane_f16(_sum, _r31, _k2, 0);
                _sum = vmlaq_low_lane_f16(_sum, _r32, _k2, 1);
                _sum = vmlaq_low_lane_f16(_sum, _r33, _k2, 2);
                _sum = vmlaq_low_lane_f16(_sum, _r34, _k2, 3);

                _sum = vmlaq_high_lane_f16(_sum, _r40, _k2, 4);
                _sum = vmlaq_high_lane_f16(_sum, _r41, _k2, 5);
                _sum = vmlaq_high_lane_f16(_sum, _r42, _k2, 6);
                _sum = vmlaq_high_lane_f16(_sum, _r43, _k2, 7);
                _sum = vmlaq_high_lane_f16(_sum, _r44, _k3, 7);

                _sum2 = vmlaq_low_lane_f16(_sum2, _r10, _k0, 0);
                _sum2 = vmlaq_low_lane_f16(_sum2, _r11, _k0, 1);
                _sum2 = vmlaq_low_lane_f16(_sum2, _r12, _k0, 2);
                _sum2 = vmlaq_low_lane_f16(_sum2, _r13, _k0, 3);
                _sum2 = vmlaq_high_lane_f16(_sum2, _r14, _k0, 4);

                _sum2 = vmlaq_high_lane_f16(_sum2, _r20, _k0, 5);
                _sum2 = vmlaq_high_lane_f16(_sum2, _r21, _k0, 6);
                _sum2 = vmlaq_high_lane_f16(_sum2, _r22, _k0, 7);
                _sum2 = vmlaq_low_lane_f16(_sum2, _r23, _k1, 0);
                _sum2 = vmlaq_low_lane_f16(_sum2, _r24, _k1, 1);

                _sum2 = vmlaq_low_lane_f16(_sum2, _r30, _k1, 2);
                _sum2 = vmlaq_low_lane_f16(_sum2, _r31, _k1, 3);
                _sum2 = vmlaq_high_lane_f16(_sum2, _r32, _k1, 4);
                _sum2 = vmlaq_high_lane_f16(_sum2, _r33, _k1, 5);
                _sum2 = vmlaq_high_lane_f16(_sum2, _r34, _k1, 6);

                _sum2 = vmlaq_high_lane_f16(_sum2, _r40, _k1, 7);
                _sum2 = vmlaq_low_lane_f16(_sum2, _r41, _k2, 0);
                _sum2 = vmlaq_low_lane_f16(_sum2, _r42, _k2, 1);
                _sum2 = vmlaq_low_lane_f16(_sum2, _r43, _k2, 2);
                _sum2 = vmlaq_low_lane_f16(_sum2, _r44, _k2, 3);

                _sum2 = vmlaq_high_lane_f16(_sum2, _r50, _k2, 4);
                _sum2 = vmlaq_high_lane_f16(_sum2, _r51, _k2, 5);
                _sum2 = vmlaq_high_lane_f16(_sum2, _r52, _k2, 6);
                _sum2 = vmlaq_high_lane_f16(_sum2, _r53, _k2, 7);
                _sum2 = vmlaq_high_lane_f16(_sum2, _r54, _k3, 7);

                vst1q_f16(outptr, _sum);
                vst1q_f16(outptr2, _sum2);

                r0 += 8;
                r1 += 8;
                r2 += 8;
                r3 += 8;
                r4 += 8;
                r5 += 8;
                outptr += 8;
                outptr2 += 8;
            }

            r0 += tail_step + IW;
            r1 += tail_step + IW;
            r2 += tail_step + IW;
            r3 += tail_step + IW;
            r4 += tail_step + IW;
            r5 += tail_step + IW;

            outptr += OW;
            outptr2 += OW;
        }

        for (; h < OH; h++) {
            int width = OW >> 3;

            rep(i, width) {
                float16x8_t _sum = vld1q_f16(outptr);

                float16x8_t _r00 = vld1q_f16(r0);
                float16x8_t _r05 = vld1q_f16(r0 + 8);
                float16x8_t _r01 = vextq_f16(_r00, _r05, 1);
                float16x8_t _r02 = vextq_f16(_r00, _r05, 2);
                float16x8_t _r03 = vextq_f16(_r00, _r05, 3);
                float16x8_t _r04 = vextq_f16(_r00, _r05, 4);

                float16x8_t _r10 = vld1q_f16(r1);
                float16x8_t _r15 = vld1q_f16(r1 + 8);
                float16x8_t _r11 = vextq_f16(_r10, _r15, 1);
                float16x8_t _r12 = vextq_f16(_r10, _r15, 2);
                float16x8_t _r13 = vextq_f16(_r10, _r15, 3);
                float16x8_t _r14 = vextq_f16(_r10, _r15, 4);

                float16x8_t _r20 = vld1q_f16(r2);
                float16x8_t _r25 = vld1q_f16(r2 + 8);
                float16x8_t _r21 = vextq_f16(_r20, _r25, 1);
                float16x8_t _r22 = vextq_f16(_r20, _r25, 2);
                float16x8_t _r23 = vextq_f16(_r20, _r25, 3);
                float16x8_t _r24 = vextq_f16(_r20, _r25, 4);

                float16x8_t _r30 = vld1q_f16(r3);
                float16x8_t _r35 = vld1q_f16(r3 + 8);
                float16x8_t _r31 = vextq_f16(_r30, _r35, 1);
                float16x8_t _r32 = vextq_f16(_r30, _r35, 2);
                float16x8_t _r33 = vextq_f16(_r30, _r35, 3);
                float16x8_t _r34 = vextq_f16(_r30, _r35, 4);

                float16x8_t _r40 = vld1q_f16(r4);
                float16x8_t _r45 = vld1q_f16(r4 + 8);
                float16x8_t _r41 = vextq_f16(_r40, _r45, 1);
                float16x8_t _r42 = vextq_f16(_r40, _r45, 2);
                float16x8_t _r43 = vextq_f16(_r40, _r45, 3);
                float16x8_t _r44 = vextq_f16(_r40, _r45, 4);

                _sum = vmlaq_low_lane_f16(_sum, _r00, _k0, 0);
                _sum = vmlaq_low_lane_f16(_sum, _r01, _k0, 1);
                _sum = vmlaq_low_lane_f16(_sum, _r02, _k0, 2);
                _sum = vmlaq_low_lane_f16(_sum, _r03, _k0, 3);
                _sum = vmlaq_high_lane_f16(_sum, _r04, _k0, 4);

                _sum = vmlaq_high_lane_f16(_sum, _r10, _k0, 5);
                _sum = vmlaq_high_lane_f16(_sum, _r11, _k0, 6);
                _sum = vmlaq_high_lane_f16(_sum, _r12, _k0, 7);
                _sum = vmlaq_low_lane_f16(_sum, _r13, _k1, 0);
                _sum = vmlaq_low_lane_f16(_sum, _r14, _k1, 1);

                _sum = vmlaq_low_lane_f16(_sum, _r20, _k1, 2);
                _sum = vmlaq_low_lane_f16(_sum, _r21, _k1, 3);
                _sum = vmlaq_high_lane_f16(_sum, _r22, _k1, 4);
                _sum = vmlaq_high_lane_f16(_sum, _r23, _k1, 5);
                _sum = vmlaq_high_lane_f16(_sum, _r24, _k1, 6);

                _sum = vmlaq_high_lane_f16(_sum, _r30, _k1, 7);
                _sum = vmlaq_low_lane_f16(_sum, _r31, _k2, 0);
                _sum = vmlaq_low_lane_f16(_sum, _r32, _k2, 1);
                _sum = vmlaq_low_lane_f16(_sum, _r33, _k2, 2);
                _sum = vmlaq_low_lane_f16(_sum, _r34, _k2, 3);

                _sum = vmlaq_high_lane_f16(_sum, _r40, _k2, 4);
                _sum = vmlaq_high_lane_f16(_sum, _r41, _k2, 5);
                _sum = vmlaq_high_lane_f16(_sum, _r42, _k2, 6);
                _sum = vmlaq_high_lane_f16(_sum, _r43, _k2, 7);
                _sum = vmlaq_high_lane_f16(_sum, _r44, _k3, 7);

                vst1q_f16(outptr, _sum);

                r0 += 8;
                r1 += 8;
                r2 += 8;
                r3 += 8;
                r4 += 8;
                outptr += 8;
            }

            r0 += tail_step;
            r1 += tail_step;
            r2 += tail_step;
            r3 += tail_step;
            r4 += tail_step;
        }

        filter += 25;
    }
}
#endif
// vim: syntax=cpp.doxygen
