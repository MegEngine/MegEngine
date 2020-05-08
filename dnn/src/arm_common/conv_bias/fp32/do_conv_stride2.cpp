/**
 * \file dnn/src/arm_common/conv_bias/fp32/do_conv_stride2.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include <algorithm>

#include "./do_conv_stride2.h"
#include "midout.h"
#include "src/arm_common/simd_macro/neon_helper.h"
#include "src/arm_common/conv_bias/postprocess_helper.h"

MIDOUT_DECL(megdnn_arm_common_conv_bias_f32_convs2)

using namespace megdnn;
using namespace arm_common;
using namespace fp32;
using namespace conv_stride2;

using NCBKernSizeParam = fallback::ConvBiasImpl::NCBKernSizeParam;
using NCBKernParam = fallback::ConvBiasImpl::NCBKernParam;


void conv_stride2::do_conv_2x2_stride2(const float* src, const float* filter, float* dst,
                         size_t IH, size_t IW, size_t OH, size_t OW,
                         size_t IC) {
    const size_t tail_step = IW - 2 * OW + IW;

    rep(ic, IC) {
        const float* src_ptr = src + IW * IH * ic;
        float* outptr = dst;

        const float* r0 = src_ptr;
        const float* r1 = src_ptr + IW;

        const float* k0 = filter;

        MEGDNN_SIMD_TYPE _k0123 = MEGDNN_SIMD_LOADU(k0);
        rep(h, OH) {
            int nn = OW >> 2;

            rep(i, nn) {
                MEGDNN_SIMD_TYPE _outp = MEGDNN_SIMD_LOADU(outptr);

                MEGDNN_SIMD_TYPE2 _r0 = MEGDNN_SIMD_LOAD2(r0);

                MEGDNN_SIMD_TYPE _r00 = _r0.val[0];  // 0 2 4 6
                MEGDNN_SIMD_TYPE _r01 = _r0.val[1];  // 1 3 5 7

                _outp = MEGDNN_SIMD_FMA_LANE(_outp, _r00, _k0123, 0);
                _outp = MEGDNN_SIMD_FMA_LANE(_outp, _r01, _k0123, 1);

                MEGDNN_SIMD_TYPE2 _r1 = MEGDNN_SIMD_LOAD2(r1);

                MEGDNN_SIMD_TYPE _r10 = _r1.val[0];
                MEGDNN_SIMD_TYPE _r11 = _r1.val[1];

                _outp = MEGDNN_SIMD_FMA_LANE(_outp, _r10, _k0123, 2);
                _outp = MEGDNN_SIMD_FMA_LANE(_outp, _r11, _k0123, 3);

                MEGDNN_SIMD_STOREU(outptr, _outp);

                r0 += 8;
                r1 += 8;
                outptr += 4;
            }

            r0 += tail_step;
            r1 += tail_step;
        }

        filter += 4;
    }
}

void conv_stride2::do_conv_3x3_stride2(const float* src, const float* filter, float* dst,
                         size_t IH, size_t IW, size_t OH, size_t OW,
                         size_t IC) {
    const size_t tail_step = IW - 2 * OW + IW;

    rep(ic, IC) {
        const float* src_ptr = src + IW * IH * ic;
        float* outptr = dst;

        const float* r0 = src_ptr;
        const float* r1 = src_ptr + IW;
        const float* r2 = src_ptr + IW * 2;

        const float* k0 = filter;
        const float* k1 = filter + 3;
        const float* k2 = filter + 5;

        MEGDNN_SIMD_TYPE _k0123 = MEGDNN_SIMD_LOADU(k0);
        MEGDNN_SIMD_TYPE _k3456 = MEGDNN_SIMD_LOADU(k1);
        MEGDNN_SIMD_TYPE _k5678 = MEGDNN_SIMD_LOADU(k2);
        MEGDNN_SIMD_TYPE _k6789 = MEGDNN_SIMD_EXT(_k5678, _k5678, 1);
        rep(h, OH) {
            int nn = OW >> 2;

            rep(i, nn) {
                MEGDNN_SIMD_TYPE _outp = MEGDNN_SIMD_LOADU(outptr);

                MEGDNN_SIMD_TYPE2 _r0 = MEGDNN_SIMD_LOAD2(r0);
                MEGDNN_SIMD_TYPE2 _r0n = MEGDNN_SIMD_LOAD2(r0 + 8);

                MEGDNN_SIMD_TYPE _r00 = _r0.val[0];  // 0 2 4 6
                MEGDNN_SIMD_TYPE _r01 = _r0.val[1];  // 1 3 5 7
                MEGDNN_SIMD_TYPE _r02 =
                        MEGDNN_SIMD_EXT(_r00, _r0n.val[0], 1);  // 2 4 6 8

                _outp = MEGDNN_SIMD_FMA_LANE(_outp, _r00, _k0123, 0);
                _outp = MEGDNN_SIMD_FMA_LANE(_outp, _r01, _k0123, 1);
                _outp = MEGDNN_SIMD_FMA_LANE(_outp, _r02, _k0123, 2);

                MEGDNN_SIMD_TYPE2 _r1 = MEGDNN_SIMD_LOAD2(r1);
                MEGDNN_SIMD_TYPE2 _r1n = MEGDNN_SIMD_LOAD2(r1 + 8);

                MEGDNN_SIMD_TYPE _r10 = _r1.val[0];
                MEGDNN_SIMD_TYPE _r11 = _r1.val[1];
                MEGDNN_SIMD_TYPE _r12 = MEGDNN_SIMD_EXT(_r10, _r1n.val[0], 1);

                _outp = MEGDNN_SIMD_FMA_LANE(_outp, _r10, _k3456, 0);
                _outp = MEGDNN_SIMD_FMA_LANE(_outp, _r11, _k3456, 1);
                _outp = MEGDNN_SIMD_FMA_LANE(_outp, _r12, _k3456, 2);

                MEGDNN_SIMD_TYPE2 _r2 = MEGDNN_SIMD_LOAD2(r2);
                MEGDNN_SIMD_TYPE2 _r2n = MEGDNN_SIMD_LOAD2(r2 + 8);

                MEGDNN_SIMD_TYPE _r20 = _r2.val[0];
                MEGDNN_SIMD_TYPE _r21 = _r2.val[1];
                MEGDNN_SIMD_TYPE _r22 = MEGDNN_SIMD_EXT(_r20, _r2n.val[0], 1);

                _outp = MEGDNN_SIMD_FMA_LANE(_outp, _r20, _k6789, 0);
                _outp = MEGDNN_SIMD_FMA_LANE(_outp, _r21, _k6789, 1);
                _outp = MEGDNN_SIMD_FMA_LANE(_outp, _r22, _k6789, 2);

                MEGDNN_SIMD_STOREU(outptr, _outp);

                r0 += 8;
                r1 += 8;
                r2 += 8;
                outptr += 4;
            }

            r0 += tail_step;
            r1 += tail_step;
            r2 += tail_step;
        }

        filter += 9;
    }
}

void conv_stride2::do_conv_5x5_stride2(const float* src, const float* filter, float* dst,
                         size_t IH, size_t IW, size_t OH, size_t OW,
                         size_t IC) {
    const size_t tail_step = IW - 2 * OW + IW;

    rep(ic, IC) {
        const float* src_ptr = src + IW * IH * ic;
        float* outptr = dst;

        const float* r0 = src_ptr;
        const float* r1 = src_ptr + IW;
        const float* r2 = src_ptr + IW * 2;
        const float* r3 = src_ptr + IW * 3;
        const float* r4 = src_ptr + IW * 4;

        MEGDNN_SIMD_TYPE _k0123 = MEGDNN_SIMD_LOADU(filter);
        MEGDNN_SIMD_TYPE _k4567 = MEGDNN_SIMD_LOADU(filter + 4);
        MEGDNN_SIMD_TYPE _k891011 = MEGDNN_SIMD_LOADU(filter + 8);
        MEGDNN_SIMD_TYPE _k12131415 = MEGDNN_SIMD_LOADU(filter + 12);
        MEGDNN_SIMD_TYPE _k16171819 = MEGDNN_SIMD_LOADU(filter + 16);
        MEGDNN_SIMD_TYPE _k20212223 = MEGDNN_SIMD_LOADU(filter + 20);
        MEGDNN_SIMD_TYPE _k24242424 = MEGDNN_SIMD_SET1(filter[24]);

        for (size_t i = 0; i < OH; i++) {
            int nn = OW >> 2;

            rep(i, nn) {
                MEGDNN_SIMD_TYPE _sum = MEGDNN_SIMD_LOADU(outptr);

                MEGDNN_SIMD_TYPE2 _r00_02461357 = MEGDNN_SIMD_LOAD2(r0);
                MEGDNN_SIMD_TYPE2 _r00nx2 = MEGDNN_SIMD_LOAD2(r0 + 8);
                MEGDNN_SIMD_TYPE _r0_8101214 = _r00nx2.val[0];  // 8 10 12 14
                MEGDNN_SIMD_TYPE _r0_9111315 = _r00nx2.val[1];  // 9 11 13 15
                MEGDNN_SIMD_TYPE _r00 = _r00_02461357.val[0];   // 0 2 4 6
                MEGDNN_SIMD_TYPE _r01 = _r00_02461357.val[1];   // 1 3 5 7
                MEGDNN_SIMD_TYPE _r02 =
                        MEGDNN_SIMD_EXT(_r00, _r0_8101214, 1);  // 2 4 6 8
                MEGDNN_SIMD_TYPE _r03 =
                        MEGDNN_SIMD_EXT(_r01, _r0_9111315, 1);  // 3 5 7 9
                MEGDNN_SIMD_TYPE _r04 =
                        MEGDNN_SIMD_EXT(_r00, _r0_8101214, 2);  // 4 6 8 10

                MEGDNN_SIMD_TYPE2 _r10_02461357 = MEGDNN_SIMD_LOAD2(r1);
                MEGDNN_SIMD_TYPE2 _r10nx2 = MEGDNN_SIMD_LOAD2(r1 + 8);
                MEGDNN_SIMD_TYPE _r1_8101214 = _r10nx2.val[0];
                MEGDNN_SIMD_TYPE _r1_9111315 = _r10nx2.val[1];
                MEGDNN_SIMD_TYPE _r10 = _r10_02461357.val[0];
                MEGDNN_SIMD_TYPE _r11 = _r10_02461357.val[1];
                MEGDNN_SIMD_TYPE _r12 = MEGDNN_SIMD_EXT(_r10, _r1_8101214, 1);
                MEGDNN_SIMD_TYPE _r13 = MEGDNN_SIMD_EXT(_r11, _r1_9111315, 1);
                MEGDNN_SIMD_TYPE _r14 = MEGDNN_SIMD_EXT(_r10, _r1_8101214, 2);

                MEGDNN_SIMD_TYPE2 _r20_02461357 = MEGDNN_SIMD_LOAD2(r2);
                MEGDNN_SIMD_TYPE2 _r20nx2 = MEGDNN_SIMD_LOAD2(r2 + 8);
                MEGDNN_SIMD_TYPE _r2_8101214 = _r20nx2.val[0];
                MEGDNN_SIMD_TYPE _r2_9111315 = _r20nx2.val[1];
                MEGDNN_SIMD_TYPE _r20 = _r20_02461357.val[0];
                MEGDNN_SIMD_TYPE _r21 = _r20_02461357.val[1];
                MEGDNN_SIMD_TYPE _r22 = MEGDNN_SIMD_EXT(_r20, _r2_8101214, 1);
                MEGDNN_SIMD_TYPE _r23 = MEGDNN_SIMD_EXT(_r21, _r2_9111315, 1);
                MEGDNN_SIMD_TYPE _r24 = MEGDNN_SIMD_EXT(_r20, _r2_8101214, 2);

                MEGDNN_SIMD_TYPE2 _r30_02461357 = MEGDNN_SIMD_LOAD2(r3);
                MEGDNN_SIMD_TYPE2 _r30nx2 = MEGDNN_SIMD_LOAD2(r3 + 8);
                MEGDNN_SIMD_TYPE _r3_8101214 = _r30nx2.val[0];
                MEGDNN_SIMD_TYPE _r3_9111315 = _r30nx2.val[1];
                MEGDNN_SIMD_TYPE _r30 = _r30_02461357.val[0];
                MEGDNN_SIMD_TYPE _r31 = _r30_02461357.val[1];
                MEGDNN_SIMD_TYPE _r32 = MEGDNN_SIMD_EXT(_r30, _r3_8101214, 1);
                MEGDNN_SIMD_TYPE _r33 = MEGDNN_SIMD_EXT(_r31, _r3_9111315, 1);
                MEGDNN_SIMD_TYPE _r34 = MEGDNN_SIMD_EXT(_r30, _r3_8101214, 2);

                MEGDNN_SIMD_TYPE2 _r40_02461357 = MEGDNN_SIMD_LOAD2(r4);
                MEGDNN_SIMD_TYPE2 _r40nx2 = MEGDNN_SIMD_LOAD2(r4 + 8);
                MEGDNN_SIMD_TYPE _r4_8101214 = _r40nx2.val[0];
                MEGDNN_SIMD_TYPE _r4_9111315 = _r40nx2.val[1];
                MEGDNN_SIMD_TYPE _r40 = _r40_02461357.val[0];
                MEGDNN_SIMD_TYPE _r41 = _r40_02461357.val[1];
                MEGDNN_SIMD_TYPE _r42 = MEGDNN_SIMD_EXT(_r40, _r4_8101214, 1);
                MEGDNN_SIMD_TYPE _r43 = MEGDNN_SIMD_EXT(_r41, _r4_9111315, 1);
                MEGDNN_SIMD_TYPE _r44 = MEGDNN_SIMD_EXT(_r40, _r4_8101214, 2);

                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r00, _k0123, 0);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r01, _k0123, 1);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r02, _k0123, 2);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r03, _k0123, 3);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r04, _k4567, 0);

                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r10, _k4567, 1);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r11, _k4567, 2);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r12, _k4567, 3);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r13, _k891011, 0);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r14, _k891011, 1);

                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r20, _k891011, 2);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r21, _k891011, 3);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r22, _k12131415, 0);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r23, _k12131415, 1);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r24, _k12131415, 2);

                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r30, _k12131415, 3);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r31, _k16171819, 0);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r32, _k16171819, 1);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r33, _k16171819, 2);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r34, _k16171819, 3);

                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r40, _k20212223, 0);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r41, _k20212223, 1);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r42, _k20212223, 2);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r43, _k20212223, 3);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r44, _k24242424, 0);

                MEGDNN_SIMD_STOREU(outptr, _sum);

                r0 += 8;
                r1 += 8;
                r2 += 8;
                r3 += 8;
                r4 += 8;
                outptr += 4;
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

void conv_stride2::do_conv_7x7_stride2(const float* src, const float* filter, float* dst,
                         size_t IH, size_t IW, size_t OH, size_t OW,
                         size_t IC) {
    const size_t tail_step = IW - 2 * OW + IW;

    rep(ic, IC) {
        const float* src_ptr = src + IW * IH * ic;
        float* outptr = dst;

        const float* r0 = src_ptr;
        const float* r1 = src_ptr + IW;
        const float* r2 = src_ptr + IW * 2;
        const float* r3 = src_ptr + IW * 3;
        const float* r4 = src_ptr + IW * 4;
        const float* r5 = src_ptr + IW * 5;
        const float* r6 = src_ptr + IW * 6;

        const float* k0 = filter;
        const float* k1 = filter + 7;
        const float* k2 = filter + 14;
        const float* k3 = filter + 21;
        const float* k4 = filter + 28;
        const float* k5 = filter + 35;
        const float* k6 = filter + 42;

        for (size_t i = 0; i < OH; i++) {
            int nn = OW >> 2;

            rep(i, nn) {
                MEGDNN_SIMD_TYPE _sum = MEGDNN_SIMD_LOADU(outptr);

                MEGDNN_SIMD_TYPE _k0123 = MEGDNN_SIMD_LOADU(k0);
                MEGDNN_SIMD_TYPE _k4567 = MEGDNN_SIMD_LOADU(k0 + 4);

                MEGDNN_SIMD_TYPE2 _r00_02461357 = MEGDNN_SIMD_LOAD2(r0);
                MEGDNN_SIMD_TYPE2 _r00nx2 = MEGDNN_SIMD_LOAD2(r0 + 8);
                MEGDNN_SIMD_TYPE _r0_8101214 = _r00nx2.val[0];  // 8 10 12 14
                MEGDNN_SIMD_TYPE _r0_9111315 = _r00nx2.val[1];  // 9 11 13 15
                MEGDNN_SIMD_TYPE _r00 = _r00_02461357.val[0];   // 0 2 4 6
                MEGDNN_SIMD_TYPE _r01 = _r00_02461357.val[1];   // 1 3 5 7
                MEGDNN_SIMD_TYPE _r02 =
                        MEGDNN_SIMD_EXT(_r00, _r0_8101214, 1);  // 2 4 6 8
                MEGDNN_SIMD_TYPE _r03 =
                        MEGDNN_SIMD_EXT(_r01, _r0_9111315, 1);  // 3 5 7 9
                MEGDNN_SIMD_TYPE _r04 =
                        MEGDNN_SIMD_EXT(_r00, _r0_8101214, 2);  // 4 6 8 10
                MEGDNN_SIMD_TYPE _r05 =
                        MEGDNN_SIMD_EXT(_r01, _r0_9111315, 2);  // 5 7 9 11
                MEGDNN_SIMD_TYPE _r06 =
                        MEGDNN_SIMD_EXT(_r00, _r0_8101214, 3);  // 6 8 10 12

                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r00, _k0123, 0);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r01, _k0123, 1);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r02, _k0123, 2);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r03, _k0123, 3);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r04, _k4567, 0);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r05, _k4567, 1);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r06, _k4567, 2);

                MEGDNN_SIMD_TYPE _k78910 = MEGDNN_SIMD_LOADU(k1);
                MEGDNN_SIMD_TYPE _k11121314 = MEGDNN_SIMD_LOADU(k1 + 4);

                MEGDNN_SIMD_TYPE2 _r10_02461357 = MEGDNN_SIMD_LOAD2(r1);
                MEGDNN_SIMD_TYPE2 _r10nx2 = MEGDNN_SIMD_LOAD2(r1 + 8);
                MEGDNN_SIMD_TYPE _r1_8101214 = _r10nx2.val[0];
                MEGDNN_SIMD_TYPE _r1_9111315 = _r10nx2.val[1];
                MEGDNN_SIMD_TYPE _r10 = _r10_02461357.val[0];
                MEGDNN_SIMD_TYPE _r11 = _r10_02461357.val[1];
                MEGDNN_SIMD_TYPE _r12 = MEGDNN_SIMD_EXT(_r10, _r1_8101214, 1);
                MEGDNN_SIMD_TYPE _r13 = MEGDNN_SIMD_EXT(_r11, _r1_9111315, 1);
                MEGDNN_SIMD_TYPE _r14 = MEGDNN_SIMD_EXT(_r10, _r1_8101214, 2);
                MEGDNN_SIMD_TYPE _r15 = MEGDNN_SIMD_EXT(_r11, _r1_9111315, 2);
                MEGDNN_SIMD_TYPE _r16 = MEGDNN_SIMD_EXT(_r10, _r1_8101214, 3);

                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r10, _k78910, 0);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r11, _k78910, 1);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r12, _k78910, 2);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r13, _k78910, 3);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r14, _k11121314, 0);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r15, _k11121314, 1);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r16, _k11121314, 2);

                MEGDNN_SIMD_TYPE _k14151617 = MEGDNN_SIMD_LOADU(k2);
                MEGDNN_SIMD_TYPE _k18192021 = MEGDNN_SIMD_LOADU(k2 + 4);

                MEGDNN_SIMD_TYPE2 _r20_02461357 = MEGDNN_SIMD_LOAD2(r2);
                MEGDNN_SIMD_TYPE2 _r20nx2 = MEGDNN_SIMD_LOAD2(r2 + 8);
                MEGDNN_SIMD_TYPE _r2_8101214 = _r20nx2.val[0];
                MEGDNN_SIMD_TYPE _r2_9111315 = _r20nx2.val[1];
                MEGDNN_SIMD_TYPE _r20 = _r20_02461357.val[0];
                MEGDNN_SIMD_TYPE _r21 = _r20_02461357.val[1];
                MEGDNN_SIMD_TYPE _r22 = MEGDNN_SIMD_EXT(_r20, _r2_8101214, 1);
                MEGDNN_SIMD_TYPE _r23 = MEGDNN_SIMD_EXT(_r21, _r2_9111315, 1);
                MEGDNN_SIMD_TYPE _r24 = MEGDNN_SIMD_EXT(_r20, _r2_8101214, 2);
                MEGDNN_SIMD_TYPE _r25 = MEGDNN_SIMD_EXT(_r21, _r2_9111315, 2);
                MEGDNN_SIMD_TYPE _r26 = MEGDNN_SIMD_EXT(_r20, _r2_8101214, 3);

                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r20, _k14151617, 0);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r21, _k14151617, 1);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r22, _k14151617, 2);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r23, _k14151617, 3);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r24, _k18192021, 0);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r25, _k18192021, 1);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r26, _k18192021, 2);

                MEGDNN_SIMD_TYPE _k21222324 = MEGDNN_SIMD_LOADU(k3);
                MEGDNN_SIMD_TYPE _k25262728 = MEGDNN_SIMD_LOADU(k3 + 4);

                MEGDNN_SIMD_TYPE2 _r30_02461357 = MEGDNN_SIMD_LOAD2(r3);
                MEGDNN_SIMD_TYPE2 _r30nx2 = MEGDNN_SIMD_LOAD2(r3 + 8);
                MEGDNN_SIMD_TYPE _r3_8101214 = _r30nx2.val[0];
                MEGDNN_SIMD_TYPE _r3_9111315 = _r30nx2.val[1];
                MEGDNN_SIMD_TYPE _r30 = _r30_02461357.val[0];
                MEGDNN_SIMD_TYPE _r31 = _r30_02461357.val[1];
                MEGDNN_SIMD_TYPE _r32 = MEGDNN_SIMD_EXT(_r30, _r3_8101214, 1);
                MEGDNN_SIMD_TYPE _r33 = MEGDNN_SIMD_EXT(_r31, _r3_9111315, 1);
                MEGDNN_SIMD_TYPE _r34 = MEGDNN_SIMD_EXT(_r30, _r3_8101214, 2);
                MEGDNN_SIMD_TYPE _r35 = MEGDNN_SIMD_EXT(_r31, _r3_9111315, 2);
                MEGDNN_SIMD_TYPE _r36 = MEGDNN_SIMD_EXT(_r30, _r3_8101214, 3);

                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r30, _k21222324, 0);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r31, _k21222324, 1);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r32, _k21222324, 2);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r33, _k21222324, 3);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r34, _k25262728, 0);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r35, _k25262728, 1);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r36, _k25262728, 2);

                MEGDNN_SIMD_TYPE _k28293031 = MEGDNN_SIMD_LOADU(k4);
                MEGDNN_SIMD_TYPE _k32333435 = MEGDNN_SIMD_LOADU(k4 + 4);

                MEGDNN_SIMD_TYPE2 _r40_02461357 = MEGDNN_SIMD_LOAD2(r4);
                MEGDNN_SIMD_TYPE2 _r40nx2 = MEGDNN_SIMD_LOAD2(r4 + 8);
                MEGDNN_SIMD_TYPE _r4_8101214 = _r40nx2.val[0];
                MEGDNN_SIMD_TYPE _r4_9111315 = _r40nx2.val[1];
                MEGDNN_SIMD_TYPE _r40 = _r40_02461357.val[0];
                MEGDNN_SIMD_TYPE _r41 = _r40_02461357.val[1];
                MEGDNN_SIMD_TYPE _r42 = MEGDNN_SIMD_EXT(_r40, _r4_8101214, 1);
                MEGDNN_SIMD_TYPE _r43 = MEGDNN_SIMD_EXT(_r41, _r4_9111315, 1);
                MEGDNN_SIMD_TYPE _r44 = MEGDNN_SIMD_EXT(_r40, _r4_8101214, 2);
                MEGDNN_SIMD_TYPE _r45 = MEGDNN_SIMD_EXT(_r41, _r4_9111315, 2);
                MEGDNN_SIMD_TYPE _r46 = MEGDNN_SIMD_EXT(_r40, _r4_8101214, 3);

                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r40, _k28293031, 0);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r41, _k28293031, 1);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r42, _k28293031, 2);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r43, _k28293031, 3);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r44, _k32333435, 0);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r45, _k32333435, 1);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r46, _k32333435, 2);

                MEGDNN_SIMD_TYPE _k35363738 = MEGDNN_SIMD_LOADU(k5);
                MEGDNN_SIMD_TYPE _k39404142 = MEGDNN_SIMD_LOADU(k5 + 4);

                MEGDNN_SIMD_TYPE2 _r50_02461357 = MEGDNN_SIMD_LOAD2(r5);
                MEGDNN_SIMD_TYPE2 _r50nx2 = MEGDNN_SIMD_LOAD2(r5 + 8);
                MEGDNN_SIMD_TYPE _r5_8101214 = _r50nx2.val[0];
                MEGDNN_SIMD_TYPE _r5_9111315 = _r50nx2.val[1];
                MEGDNN_SIMD_TYPE _r50 = _r50_02461357.val[0];
                MEGDNN_SIMD_TYPE _r51 = _r50_02461357.val[1];
                MEGDNN_SIMD_TYPE _r52 = MEGDNN_SIMD_EXT(_r50, _r5_8101214, 1);
                MEGDNN_SIMD_TYPE _r53 = MEGDNN_SIMD_EXT(_r51, _r5_9111315, 1);
                MEGDNN_SIMD_TYPE _r54 = MEGDNN_SIMD_EXT(_r50, _r5_8101214, 2);
                MEGDNN_SIMD_TYPE _r55 = MEGDNN_SIMD_EXT(_r51, _r5_9111315, 2);
                MEGDNN_SIMD_TYPE _r56 = MEGDNN_SIMD_EXT(_r50, _r5_8101214, 3);

                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r50, _k35363738, 0);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r51, _k35363738, 1);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r52, _k35363738, 2);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r53, _k35363738, 3);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r54, _k39404142, 0);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r55, _k39404142, 1);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r56, _k39404142, 2);

                MEGDNN_SIMD_TYPE _k42434445 = MEGDNN_SIMD_LOADU(k6);
                MEGDNN_SIMD_TYPE _k45464748 = MEGDNN_SIMD_LOADU(k6 + 3);

                MEGDNN_SIMD_TYPE2 _r60_02461357 = MEGDNN_SIMD_LOAD2(r6);
                MEGDNN_SIMD_TYPE2 _r60nx2 = MEGDNN_SIMD_LOAD2(r6 + 8);
                MEGDNN_SIMD_TYPE _r6_8101214 = _r60nx2.val[0];
                MEGDNN_SIMD_TYPE _r6_9111315 = _r60nx2.val[1];
                MEGDNN_SIMD_TYPE _r60 = _r60_02461357.val[0];
                MEGDNN_SIMD_TYPE _r61 = _r60_02461357.val[1];
                MEGDNN_SIMD_TYPE _r62 = MEGDNN_SIMD_EXT(_r60, _r6_8101214, 1);
                MEGDNN_SIMD_TYPE _r63 = MEGDNN_SIMD_EXT(_r61, _r6_9111315, 1);
                MEGDNN_SIMD_TYPE _r64 = MEGDNN_SIMD_EXT(_r60, _r6_8101214, 2);
                MEGDNN_SIMD_TYPE _r65 = MEGDNN_SIMD_EXT(_r61, _r6_9111315, 2);
                MEGDNN_SIMD_TYPE _r66 = MEGDNN_SIMD_EXT(_r60, _r6_8101214, 3);

                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r60, _k42434445, 0);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r61, _k42434445, 1);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r62, _k42434445, 2);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r63, _k42434445, 3);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r64, _k45464748, 1);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r65, _k45464748, 2);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r66, _k45464748, 3);

                MEGDNN_SIMD_STOREU(outptr, _sum);

                r0 += 8;
                r1 += 8;
                r2 += 8;
                r3 += 8;
                r4 += 8;
                r5 += 8;
                r6 += 8;
                outptr += 4;
            }

            r0 += tail_step;
            r1 += tail_step;
            r2 += tail_step;
            r3 += tail_step;
            r4 += tail_step;
            r5 += tail_step;
            r6 += tail_step;
        }
        filter += 49;
    }
}
// vim: syntax=cpp.doxygen
