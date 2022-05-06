/**
 * \file dnn/src/fallback/conv_bias/gi/fp32/do_conv_stride2.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include <algorithm>

#include "./do_conv_stride2.h"
#include "midout.h"
#include "src/fallback/conv_bias/gi/postprocess_helper.h"
#include "src/fallback/general_intrinsic/gi_float.h"

MIDOUT_DECL(megdnn_fallback_conv_bias_f32_convs2)

using namespace megdnn;
using namespace fallback;
using namespace fp32;
using namespace conv_stride2;

using NCBKernSizeParam = fallback::ConvBiasImpl::NCBKernSizeParam;
using NCBKernParam = fallback::ConvBiasImpl::NCBKernParam;

void conv_stride2::do_conv_2x2_stride2(
        const float* src, const float* filter, float* dst, size_t IH, size_t IW,
        size_t OH, size_t OW, size_t IC) {
    const size_t tail_step = IW - 2 * OW + IW;

    rep(ic, IC) {
        const float* src_ptr = src + IW * IH * ic;
        float* outptr = dst;

        const float* r0 = src_ptr;
        const float* r1 = src_ptr + IW;

        const float* k0 = filter;

        GI_FLOAT32_t _k0123 = GiLoadFloat32(k0);
        rep(h, OH) {
            int nn = OW >> 2;

            rep(i, nn) {
                GI_FLOAT32_t _outp = GiLoadFloat32(outptr);

                GI_FLOAT32_V2_t _r0 = GiLd2qFloat32(r0);

                GI_FLOAT32_t _r00 = _r0.val[0];  // 0 2 4 6
                GI_FLOAT32_t _r01 = _r0.val[1];  // 1 3 5 7

                _outp = GiSimdFmaLane(_outp, _r00, _k0123, 0);
                _outp = GiSimdFmaLane(_outp, _r01, _k0123, 1);

                GI_FLOAT32_V2_t _r1 = GiLd2qFloat32(r1);

                GI_FLOAT32_t _r10 = _r1.val[0];
                GI_FLOAT32_t _r11 = _r1.val[1];

                _outp = GiSimdFmaLane(_outp, _r10, _k0123, 2);
                _outp = GiSimdFmaLane(_outp, _r11, _k0123, 3);

                GiStoreFloat32(outptr, _outp);

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

void conv_stride2::do_conv_3x3_stride2(
        const float* src, const float* filter, float* dst, size_t IH, size_t IW,
        size_t OH, size_t OW, size_t IC) {
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

        GI_FLOAT32_t _k0123 = GiLoadFloat32(k0);
        GI_FLOAT32_t _k3456 = GiLoadFloat32(k1);
        GI_FLOAT32_t _k5678 = GiLoadFloat32(k2);
        GI_FLOAT32_t _k6789 = GiExtqFloat32(_k5678, _k5678, 1);
        rep(h, OH) {
            int nn = OW >> 2;

            rep(i, nn) {
                GI_FLOAT32_t _outp = GiLoadFloat32(outptr);

                GI_FLOAT32_V2_t _r0 = GiLd2qFloat32(r0);
                GI_FLOAT32_V2_t _r0n = GiLd2qFloat32(r0 + 8);

                GI_FLOAT32_t _r00 = _r0.val[0];                           // 0 2 4 6
                GI_FLOAT32_t _r01 = _r0.val[1];                           // 1 3 5 7
                GI_FLOAT32_t _r02 = GiExtqFloat32(_r00, _r0n.val[0], 1);  // 2 4 6 8

                _outp = GiSimdFmaLane(_outp, _r00, _k0123, 0);
                _outp = GiSimdFmaLane(_outp, _r01, _k0123, 1);
                _outp = GiSimdFmaLane(_outp, _r02, _k0123, 2);

                GI_FLOAT32_V2_t _r1 = GiLd2qFloat32(r1);
                GI_FLOAT32_V2_t _r1n = GiLd2qFloat32(r1 + 8);

                GI_FLOAT32_t _r10 = _r1.val[0];
                GI_FLOAT32_t _r11 = _r1.val[1];
                GI_FLOAT32_t _r12 = GiExtqFloat32(_r10, _r1n.val[0], 1);

                _outp = GiSimdFmaLane(_outp, _r10, _k3456, 0);
                _outp = GiSimdFmaLane(_outp, _r11, _k3456, 1);
                _outp = GiSimdFmaLane(_outp, _r12, _k3456, 2);

                GI_FLOAT32_V2_t _r2 = GiLd2qFloat32(r2);
                GI_FLOAT32_V2_t _r2n = GiLd2qFloat32(r2 + 8);

                GI_FLOAT32_t _r20 = _r2.val[0];
                GI_FLOAT32_t _r21 = _r2.val[1];
                GI_FLOAT32_t _r22 = GiExtqFloat32(_r20, _r2n.val[0], 1);

                _outp = GiSimdFmaLane(_outp, _r20, _k6789, 0);
                _outp = GiSimdFmaLane(_outp, _r21, _k6789, 1);
                _outp = GiSimdFmaLane(_outp, _r22, _k6789, 2);

                GiStoreFloat32(outptr, _outp);

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

void conv_stride2::do_conv_5x5_stride2(
        const float* src, const float* filter, float* dst, size_t IH, size_t IW,
        size_t OH, size_t OW, size_t IC) {
    const size_t tail_step = IW - 2 * OW + IW;

    rep(ic, IC) {
        const float* src_ptr = src + IW * IH * ic;
        float* outptr = dst;

        const float* r0 = src_ptr;
        const float* r1 = src_ptr + IW;
        const float* r2 = src_ptr + IW * 2;
        const float* r3 = src_ptr + IW * 3;
        const float* r4 = src_ptr + IW * 4;

        GI_FLOAT32_t _k0123 = GiLoadFloat32(filter);
        GI_FLOAT32_t _k4567 = GiLoadFloat32(filter + 4);
        GI_FLOAT32_t _k891011 = GiLoadFloat32(filter + 8);
        GI_FLOAT32_t _k12131415 = GiLoadFloat32(filter + 12);
        GI_FLOAT32_t _k16171819 = GiLoadFloat32(filter + 16);
        GI_FLOAT32_t _k20212223 = GiLoadFloat32(filter + 20);
        GI_FLOAT32_t _k24242424 = GiBroadcastFloat32(filter[24]);

        for (size_t i = 0; i < OH; i++) {
            int nn = OW >> 2;

            rep(i, nn) {
                GI_FLOAT32_t _sum = GiLoadFloat32(outptr);

                GI_FLOAT32_V2_t _r00_02461357 = GiLd2qFloat32(r0);
                GI_FLOAT32_V2_t _r00nx2 = GiLd2qFloat32(r0 + 8);
                GI_FLOAT32_t _r0_8101214 = _r00nx2.val[0];                // 8 10 12 14
                GI_FLOAT32_t _r0_9111315 = _r00nx2.val[1];                // 9 11 13 15
                GI_FLOAT32_t _r00 = _r00_02461357.val[0];                 // 0 2 4 6
                GI_FLOAT32_t _r01 = _r00_02461357.val[1];                 // 1 3 5 7
                GI_FLOAT32_t _r02 = GiExtqFloat32(_r00, _r0_8101214, 1);  // 2 4 6 8
                GI_FLOAT32_t _r03 = GiExtqFloat32(_r01, _r0_9111315, 1);  // 3 5 7 9
                GI_FLOAT32_t _r04 = GiExtqFloat32(_r00, _r0_8101214, 2);  // 4 6 8 10

                GI_FLOAT32_V2_t _r10_02461357 = GiLd2qFloat32(r1);
                GI_FLOAT32_V2_t _r10nx2 = GiLd2qFloat32(r1 + 8);
                GI_FLOAT32_t _r1_8101214 = _r10nx2.val[0];
                GI_FLOAT32_t _r1_9111315 = _r10nx2.val[1];
                GI_FLOAT32_t _r10 = _r10_02461357.val[0];
                GI_FLOAT32_t _r11 = _r10_02461357.val[1];
                GI_FLOAT32_t _r12 = GiExtqFloat32(_r10, _r1_8101214, 1);
                GI_FLOAT32_t _r13 = GiExtqFloat32(_r11, _r1_9111315, 1);
                GI_FLOAT32_t _r14 = GiExtqFloat32(_r10, _r1_8101214, 2);

                GI_FLOAT32_V2_t _r20_02461357 = GiLd2qFloat32(r2);
                GI_FLOAT32_V2_t _r20nx2 = GiLd2qFloat32(r2 + 8);
                GI_FLOAT32_t _r2_8101214 = _r20nx2.val[0];
                GI_FLOAT32_t _r2_9111315 = _r20nx2.val[1];
                GI_FLOAT32_t _r20 = _r20_02461357.val[0];
                GI_FLOAT32_t _r21 = _r20_02461357.val[1];
                GI_FLOAT32_t _r22 = GiExtqFloat32(_r20, _r2_8101214, 1);
                GI_FLOAT32_t _r23 = GiExtqFloat32(_r21, _r2_9111315, 1);
                GI_FLOAT32_t _r24 = GiExtqFloat32(_r20, _r2_8101214, 2);

                GI_FLOAT32_V2_t _r30_02461357 = GiLd2qFloat32(r3);
                GI_FLOAT32_V2_t _r30nx2 = GiLd2qFloat32(r3 + 8);
                GI_FLOAT32_t _r3_8101214 = _r30nx2.val[0];
                GI_FLOAT32_t _r3_9111315 = _r30nx2.val[1];
                GI_FLOAT32_t _r30 = _r30_02461357.val[0];
                GI_FLOAT32_t _r31 = _r30_02461357.val[1];
                GI_FLOAT32_t _r32 = GiExtqFloat32(_r30, _r3_8101214, 1);
                GI_FLOAT32_t _r33 = GiExtqFloat32(_r31, _r3_9111315, 1);
                GI_FLOAT32_t _r34 = GiExtqFloat32(_r30, _r3_8101214, 2);

                GI_FLOAT32_V2_t _r40_02461357 = GiLd2qFloat32(r4);
                GI_FLOAT32_V2_t _r40nx2 = GiLd2qFloat32(r4 + 8);
                GI_FLOAT32_t _r4_8101214 = _r40nx2.val[0];
                GI_FLOAT32_t _r4_9111315 = _r40nx2.val[1];
                GI_FLOAT32_t _r40 = _r40_02461357.val[0];
                GI_FLOAT32_t _r41 = _r40_02461357.val[1];
                GI_FLOAT32_t _r42 = GiExtqFloat32(_r40, _r4_8101214, 1);
                GI_FLOAT32_t _r43 = GiExtqFloat32(_r41, _r4_9111315, 1);
                GI_FLOAT32_t _r44 = GiExtqFloat32(_r40, _r4_8101214, 2);

                _sum = GiSimdFmaLane(_sum, _r00, _k0123, 0);
                _sum = GiSimdFmaLane(_sum, _r01, _k0123, 1);
                _sum = GiSimdFmaLane(_sum, _r02, _k0123, 2);
                _sum = GiSimdFmaLane(_sum, _r03, _k0123, 3);
                _sum = GiSimdFmaLane(_sum, _r04, _k4567, 0);

                _sum = GiSimdFmaLane(_sum, _r10, _k4567, 1);
                _sum = GiSimdFmaLane(_sum, _r11, _k4567, 2);
                _sum = GiSimdFmaLane(_sum, _r12, _k4567, 3);
                _sum = GiSimdFmaLane(_sum, _r13, _k891011, 0);
                _sum = GiSimdFmaLane(_sum, _r14, _k891011, 1);

                _sum = GiSimdFmaLane(_sum, _r20, _k891011, 2);
                _sum = GiSimdFmaLane(_sum, _r21, _k891011, 3);
                _sum = GiSimdFmaLane(_sum, _r22, _k12131415, 0);
                _sum = GiSimdFmaLane(_sum, _r23, _k12131415, 1);
                _sum = GiSimdFmaLane(_sum, _r24, _k12131415, 2);

                _sum = GiSimdFmaLane(_sum, _r30, _k12131415, 3);
                _sum = GiSimdFmaLane(_sum, _r31, _k16171819, 0);
                _sum = GiSimdFmaLane(_sum, _r32, _k16171819, 1);
                _sum = GiSimdFmaLane(_sum, _r33, _k16171819, 2);
                _sum = GiSimdFmaLane(_sum, _r34, _k16171819, 3);

                _sum = GiSimdFmaLane(_sum, _r40, _k20212223, 0);
                _sum = GiSimdFmaLane(_sum, _r41, _k20212223, 1);
                _sum = GiSimdFmaLane(_sum, _r42, _k20212223, 2);
                _sum = GiSimdFmaLane(_sum, _r43, _k20212223, 3);
                _sum = GiSimdFmaLane(_sum, _r44, _k24242424, 0);

                GiStoreFloat32(outptr, _sum);

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

void conv_stride2::do_conv_7x7_stride2(
        const float* src, const float* filter, float* dst, size_t IH, size_t IW,
        size_t OH, size_t OW, size_t IC) {
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
                GI_FLOAT32_t _sum = GiLoadFloat32(outptr);

                GI_FLOAT32_t _k0123 = GiLoadFloat32(k0);
                GI_FLOAT32_t _k4567 = GiLoadFloat32(k0 + 4);

                GI_FLOAT32_V2_t _r00_02461357 = GiLd2qFloat32(r0);
                GI_FLOAT32_V2_t _r00nx2 = GiLd2qFloat32(r0 + 8);
                GI_FLOAT32_t _r0_8101214 = _r00nx2.val[0];                // 8 10 12 14
                GI_FLOAT32_t _r0_9111315 = _r00nx2.val[1];                // 9 11 13 15
                GI_FLOAT32_t _r00 = _r00_02461357.val[0];                 // 0 2 4 6
                GI_FLOAT32_t _r01 = _r00_02461357.val[1];                 // 1 3 5 7
                GI_FLOAT32_t _r02 = GiExtqFloat32(_r00, _r0_8101214, 1);  // 2 4 6 8
                GI_FLOAT32_t _r03 = GiExtqFloat32(_r01, _r0_9111315, 1);  // 3 5 7 9
                GI_FLOAT32_t _r04 = GiExtqFloat32(_r00, _r0_8101214, 2);  // 4 6 8 10
                GI_FLOAT32_t _r05 = GiExtqFloat32(_r01, _r0_9111315, 2);  // 5 7 9 11
                GI_FLOAT32_t _r06 = GiExtqFloat32(_r00, _r0_8101214, 3);  // 6 8 10 12

                _sum = GiSimdFmaLane(_sum, _r00, _k0123, 0);
                _sum = GiSimdFmaLane(_sum, _r01, _k0123, 1);
                _sum = GiSimdFmaLane(_sum, _r02, _k0123, 2);
                _sum = GiSimdFmaLane(_sum, _r03, _k0123, 3);
                _sum = GiSimdFmaLane(_sum, _r04, _k4567, 0);
                _sum = GiSimdFmaLane(_sum, _r05, _k4567, 1);
                _sum = GiSimdFmaLane(_sum, _r06, _k4567, 2);

                GI_FLOAT32_t _k78910 = GiLoadFloat32(k1);
                GI_FLOAT32_t _k11121314 = GiLoadFloat32(k1 + 4);

                GI_FLOAT32_V2_t _r10_02461357 = GiLd2qFloat32(r1);
                GI_FLOAT32_V2_t _r10nx2 = GiLd2qFloat32(r1 + 8);
                GI_FLOAT32_t _r1_8101214 = _r10nx2.val[0];
                GI_FLOAT32_t _r1_9111315 = _r10nx2.val[1];
                GI_FLOAT32_t _r10 = _r10_02461357.val[0];
                GI_FLOAT32_t _r11 = _r10_02461357.val[1];
                GI_FLOAT32_t _r12 = GiExtqFloat32(_r10, _r1_8101214, 1);
                GI_FLOAT32_t _r13 = GiExtqFloat32(_r11, _r1_9111315, 1);
                GI_FLOAT32_t _r14 = GiExtqFloat32(_r10, _r1_8101214, 2);
                GI_FLOAT32_t _r15 = GiExtqFloat32(_r11, _r1_9111315, 2);
                GI_FLOAT32_t _r16 = GiExtqFloat32(_r10, _r1_8101214, 3);

                _sum = GiSimdFmaLane(_sum, _r10, _k78910, 0);
                _sum = GiSimdFmaLane(_sum, _r11, _k78910, 1);
                _sum = GiSimdFmaLane(_sum, _r12, _k78910, 2);
                _sum = GiSimdFmaLane(_sum, _r13, _k78910, 3);
                _sum = GiSimdFmaLane(_sum, _r14, _k11121314, 0);
                _sum = GiSimdFmaLane(_sum, _r15, _k11121314, 1);
                _sum = GiSimdFmaLane(_sum, _r16, _k11121314, 2);

                GI_FLOAT32_t _k14151617 = GiLoadFloat32(k2);
                GI_FLOAT32_t _k18192021 = GiLoadFloat32(k2 + 4);

                GI_FLOAT32_V2_t _r20_02461357 = GiLd2qFloat32(r2);
                GI_FLOAT32_V2_t _r20nx2 = GiLd2qFloat32(r2 + 8);
                GI_FLOAT32_t _r2_8101214 = _r20nx2.val[0];
                GI_FLOAT32_t _r2_9111315 = _r20nx2.val[1];
                GI_FLOAT32_t _r20 = _r20_02461357.val[0];
                GI_FLOAT32_t _r21 = _r20_02461357.val[1];
                GI_FLOAT32_t _r22 = GiExtqFloat32(_r20, _r2_8101214, 1);
                GI_FLOAT32_t _r23 = GiExtqFloat32(_r21, _r2_9111315, 1);
                GI_FLOAT32_t _r24 = GiExtqFloat32(_r20, _r2_8101214, 2);
                GI_FLOAT32_t _r25 = GiExtqFloat32(_r21, _r2_9111315, 2);
                GI_FLOAT32_t _r26 = GiExtqFloat32(_r20, _r2_8101214, 3);

                _sum = GiSimdFmaLane(_sum, _r20, _k14151617, 0);
                _sum = GiSimdFmaLane(_sum, _r21, _k14151617, 1);
                _sum = GiSimdFmaLane(_sum, _r22, _k14151617, 2);
                _sum = GiSimdFmaLane(_sum, _r23, _k14151617, 3);
                _sum = GiSimdFmaLane(_sum, _r24, _k18192021, 0);
                _sum = GiSimdFmaLane(_sum, _r25, _k18192021, 1);
                _sum = GiSimdFmaLane(_sum, _r26, _k18192021, 2);

                GI_FLOAT32_t _k21222324 = GiLoadFloat32(k3);
                GI_FLOAT32_t _k25262728 = GiLoadFloat32(k3 + 4);

                GI_FLOAT32_V2_t _r30_02461357 = GiLd2qFloat32(r3);
                GI_FLOAT32_V2_t _r30nx2 = GiLd2qFloat32(r3 + 8);
                GI_FLOAT32_t _r3_8101214 = _r30nx2.val[0];
                GI_FLOAT32_t _r3_9111315 = _r30nx2.val[1];
                GI_FLOAT32_t _r30 = _r30_02461357.val[0];
                GI_FLOAT32_t _r31 = _r30_02461357.val[1];
                GI_FLOAT32_t _r32 = GiExtqFloat32(_r30, _r3_8101214, 1);
                GI_FLOAT32_t _r33 = GiExtqFloat32(_r31, _r3_9111315, 1);
                GI_FLOAT32_t _r34 = GiExtqFloat32(_r30, _r3_8101214, 2);
                GI_FLOAT32_t _r35 = GiExtqFloat32(_r31, _r3_9111315, 2);
                GI_FLOAT32_t _r36 = GiExtqFloat32(_r30, _r3_8101214, 3);

                _sum = GiSimdFmaLane(_sum, _r30, _k21222324, 0);
                _sum = GiSimdFmaLane(_sum, _r31, _k21222324, 1);
                _sum = GiSimdFmaLane(_sum, _r32, _k21222324, 2);
                _sum = GiSimdFmaLane(_sum, _r33, _k21222324, 3);
                _sum = GiSimdFmaLane(_sum, _r34, _k25262728, 0);
                _sum = GiSimdFmaLane(_sum, _r35, _k25262728, 1);
                _sum = GiSimdFmaLane(_sum, _r36, _k25262728, 2);

                GI_FLOAT32_t _k28293031 = GiLoadFloat32(k4);
                GI_FLOAT32_t _k32333435 = GiLoadFloat32(k4 + 4);

                GI_FLOAT32_V2_t _r40_02461357 = GiLd2qFloat32(r4);
                GI_FLOAT32_V2_t _r40nx2 = GiLd2qFloat32(r4 + 8);
                GI_FLOAT32_t _r4_8101214 = _r40nx2.val[0];
                GI_FLOAT32_t _r4_9111315 = _r40nx2.val[1];
                GI_FLOAT32_t _r40 = _r40_02461357.val[0];
                GI_FLOAT32_t _r41 = _r40_02461357.val[1];
                GI_FLOAT32_t _r42 = GiExtqFloat32(_r40, _r4_8101214, 1);
                GI_FLOAT32_t _r43 = GiExtqFloat32(_r41, _r4_9111315, 1);
                GI_FLOAT32_t _r44 = GiExtqFloat32(_r40, _r4_8101214, 2);
                GI_FLOAT32_t _r45 = GiExtqFloat32(_r41, _r4_9111315, 2);
                GI_FLOAT32_t _r46 = GiExtqFloat32(_r40, _r4_8101214, 3);

                _sum = GiSimdFmaLane(_sum, _r40, _k28293031, 0);
                _sum = GiSimdFmaLane(_sum, _r41, _k28293031, 1);
                _sum = GiSimdFmaLane(_sum, _r42, _k28293031, 2);
                _sum = GiSimdFmaLane(_sum, _r43, _k28293031, 3);
                _sum = GiSimdFmaLane(_sum, _r44, _k32333435, 0);
                _sum = GiSimdFmaLane(_sum, _r45, _k32333435, 1);
                _sum = GiSimdFmaLane(_sum, _r46, _k32333435, 2);

                GI_FLOAT32_t _k35363738 = GiLoadFloat32(k5);
                GI_FLOAT32_t _k39404142 = GiLoadFloat32(k5 + 4);

                GI_FLOAT32_V2_t _r50_02461357 = GiLd2qFloat32(r5);
                GI_FLOAT32_V2_t _r50nx2 = GiLd2qFloat32(r5 + 8);
                GI_FLOAT32_t _r5_8101214 = _r50nx2.val[0];
                GI_FLOAT32_t _r5_9111315 = _r50nx2.val[1];
                GI_FLOAT32_t _r50 = _r50_02461357.val[0];
                GI_FLOAT32_t _r51 = _r50_02461357.val[1];
                GI_FLOAT32_t _r52 = GiExtqFloat32(_r50, _r5_8101214, 1);
                GI_FLOAT32_t _r53 = GiExtqFloat32(_r51, _r5_9111315, 1);
                GI_FLOAT32_t _r54 = GiExtqFloat32(_r50, _r5_8101214, 2);
                GI_FLOAT32_t _r55 = GiExtqFloat32(_r51, _r5_9111315, 2);
                GI_FLOAT32_t _r56 = GiExtqFloat32(_r50, _r5_8101214, 3);

                _sum = GiSimdFmaLane(_sum, _r50, _k35363738, 0);
                _sum = GiSimdFmaLane(_sum, _r51, _k35363738, 1);
                _sum = GiSimdFmaLane(_sum, _r52, _k35363738, 2);
                _sum = GiSimdFmaLane(_sum, _r53, _k35363738, 3);
                _sum = GiSimdFmaLane(_sum, _r54, _k39404142, 0);
                _sum = GiSimdFmaLane(_sum, _r55, _k39404142, 1);
                _sum = GiSimdFmaLane(_sum, _r56, _k39404142, 2);

                GI_FLOAT32_t _k42434445 = GiLoadFloat32(k6);
                GI_FLOAT32_t _k45464748 = GiLoadFloat32(k6 + 3);

                GI_FLOAT32_V2_t _r60_02461357 = GiLd2qFloat32(r6);
                GI_FLOAT32_V2_t _r60nx2 = GiLd2qFloat32(r6 + 8);
                GI_FLOAT32_t _r6_8101214 = _r60nx2.val[0];
                GI_FLOAT32_t _r6_9111315 = _r60nx2.val[1];
                GI_FLOAT32_t _r60 = _r60_02461357.val[0];
                GI_FLOAT32_t _r61 = _r60_02461357.val[1];
                GI_FLOAT32_t _r62 = GiExtqFloat32(_r60, _r6_8101214, 1);
                GI_FLOAT32_t _r63 = GiExtqFloat32(_r61, _r6_9111315, 1);
                GI_FLOAT32_t _r64 = GiExtqFloat32(_r60, _r6_8101214, 2);
                GI_FLOAT32_t _r65 = GiExtqFloat32(_r61, _r6_9111315, 2);
                GI_FLOAT32_t _r66 = GiExtqFloat32(_r60, _r6_8101214, 3);

                _sum = GiSimdFmaLane(_sum, _r60, _k42434445, 0);
                _sum = GiSimdFmaLane(_sum, _r61, _k42434445, 1);
                _sum = GiSimdFmaLane(_sum, _r62, _k42434445, 2);
                _sum = GiSimdFmaLane(_sum, _r63, _k42434445, 3);
                _sum = GiSimdFmaLane(_sum, _r64, _k45464748, 1);
                _sum = GiSimdFmaLane(_sum, _r65, _k45464748, 2);
                _sum = GiSimdFmaLane(_sum, _r66, _k45464748, 3);

                GiStoreFloat32(outptr, _sum);

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
