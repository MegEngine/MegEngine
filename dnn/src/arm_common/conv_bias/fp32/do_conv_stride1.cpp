/**
 * \file dnn/src/arm_common/conv_bias/fp32/do_conv_stride1.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include <algorithm>

#include "src/arm_common/conv_bias/fp32/do_conv_stride1.h"
#include "src/arm_common/conv_bias/postprocess_helper.h"
#include "src/arm_common/simd_macro/neon_helper.h"

#include "midout.h"

MIDOUT_DECL(megdnn_arm_common_conv_bias_f32_convs1)

using namespace megdnn;
using namespace arm_common;
using namespace fp32;
using namespace conv_stride1;

using NCBKernSizeParam = fallback::ConvBiasImpl::NCBKernSizeParam;
using NCBKernParam = fallback::ConvBiasImpl::NCBKernParam;

void conv_stride1::do_conv_2x2_stride1(const float* src, const float* filter,
                                       float* dst, size_t IH, size_t IW,
                                       size_t OH, size_t OW, size_t IC) {
    const size_t tail_step = IW - OW;
    //! unroll of 2
    size_t ic = 0;
    for (; ic + 1 < IC; ic += 2) {
        const float* src_ptr = src + IW * IH * ic;
        const float* src_ptr1 = src_ptr + IW * IH;
        float* outptr = dst;

        const float* r00 = src_ptr;
        const float* r01 = src_ptr + IW;
        const float* r10 = src_ptr1;
        const float* r11 = src_ptr1 + IW;

        const float* k0 = filter + ic * 4;
        const float* k1 = k0 + 4;

        MEGDNN_SIMD_TYPE _k0 = MEGDNN_SIMD_LOADU(k0);
        MEGDNN_SIMD_TYPE _k1 = MEGDNN_SIMD_LOADU(k1);
        rep(h, OH) {
            int width = OW >> 2;

            rep(i, width) {
                MEGDNN_SIMD_TYPE _r000 = MEGDNN_SIMD_LOADU(r00);
                MEGDNN_SIMD_TYPE _r010 = MEGDNN_SIMD_LOADU(r01);
                MEGDNN_SIMD_TYPE _r001 = MEGDNN_SIMD_LOADU(r00 + 1);
                MEGDNN_SIMD_TYPE _r011 = MEGDNN_SIMD_LOADU(r01 + 1);

                MEGDNN_SIMD_TYPE _r100 = MEGDNN_SIMD_LOADU(r10);
                MEGDNN_SIMD_TYPE _r110 = MEGDNN_SIMD_LOADU(r11);
                MEGDNN_SIMD_TYPE _r101 = MEGDNN_SIMD_LOADU(r10 + 1);
                MEGDNN_SIMD_TYPE _r111 = MEGDNN_SIMD_LOADU(r11 + 1);

                MEGDNN_SIMD_TYPE _sum = MEGDNN_SIMD_LOADU(outptr);

                _sum = MEGDNN_SIMD_VMLAQ_LANE(_sum, _r000,
                                              MEGDNN_SIMD_GET_LOW(_k0), 0);
                _sum = MEGDNN_SIMD_VMLAQ_LANE(_sum, _r001,
                                              MEGDNN_SIMD_GET_LOW(_k0), 1);
                _sum = MEGDNN_SIMD_VMLAQ_LANE(_sum, _r010,
                                              MEGDNN_SIMD_GET_HIGH(_k0), 0);
                _sum = MEGDNN_SIMD_VMLAQ_LANE(_sum, _r011,
                                              MEGDNN_SIMD_GET_HIGH(_k0), 1);

                _sum = MEGDNN_SIMD_VMLAQ_LANE(_sum, _r100,
                                              MEGDNN_SIMD_GET_LOW(_k1), 0);
                _sum = MEGDNN_SIMD_VMLAQ_LANE(_sum, _r101,
                                              MEGDNN_SIMD_GET_LOW(_k1), 1);
                _sum = MEGDNN_SIMD_VMLAQ_LANE(_sum, _r110,
                                              MEGDNN_SIMD_GET_HIGH(_k1), 0);
                _sum = MEGDNN_SIMD_VMLAQ_LANE(_sum, _r111,
                                              MEGDNN_SIMD_GET_HIGH(_k1), 1);

                MEGDNN_SIMD_STOREU(outptr, _sum);

                r00 += 4;
                r01 += 4;
                r10 += 4;
                r11 += 4;
                outptr += 4;
            }

            r00 += tail_step;
            r01 += tail_step;
            r10 += tail_step;
            r11 += tail_step;
        }
    }
    for (; ic < IC; ic++) {
        const float* src_ptr = src + IW * IH * ic;
        float* outptr = dst;

        const float* r0 = src_ptr;
        const float* r1 = src_ptr + IW;

        const float* k0 = filter + ic * 4;

        MEGDNN_SIMD_TYPE _k0 = MEGDNN_SIMD_SET1(k0[0]);
        MEGDNN_SIMD_TYPE _k1 = MEGDNN_SIMD_SET1(k0[1]);
        MEGDNN_SIMD_TYPE _k2 = MEGDNN_SIMD_SET1(k0[2]);
        MEGDNN_SIMD_TYPE _k3 = MEGDNN_SIMD_SET1(k0[3]);
        rep(h, OH) {
            int width = OW >> 2;

            rep(i, width) {
                MEGDNN_SIMD_TYPE _r00 = MEGDNN_SIMD_LOADU(r0);
                MEGDNN_SIMD_TYPE _r10 = MEGDNN_SIMD_LOADU(r1);
                MEGDNN_SIMD_TYPE _r01 = MEGDNN_SIMD_LOADU(r0 + 1);
                MEGDNN_SIMD_TYPE _r11 = MEGDNN_SIMD_LOADU(r1 + 1);

                MEGDNN_SIMD_TYPE _sum = MEGDNN_SIMD_LOADU(outptr);
                MEGDNN_SIMD_TYPE _sum2;

                _sum = MEGDNN_SIMD_FMADD(_r00, _k0, _sum);
                _sum2 = MEGDNN_SIMD_MUL(_r01, _k1);
                _sum = MEGDNN_SIMD_FMADD(_r10, _k2, _sum);
                _sum2 = MEGDNN_SIMD_FMADD(_r11, _k3, _sum2);

                _sum = MEGDNN_SIMD_ADD(_sum, _sum2);

                MEGDNN_SIMD_STOREU(outptr, _sum);

                r0 += 4;
                r1 += 4;
                outptr += 4;
            }

            r0 += tail_step;
            r1 += tail_step;
        }
    }
}

void conv_stride1::do_conv_3x3_stride1(const float* src, const float* filter,
                                       float* dst, size_t IH, size_t IW,
                                       size_t OH, size_t OW, size_t IC) {
    const size_t tail_step = IW - OW;

    rep(ic, IC) {
        const float* src_ptr = src + IW * IH * ic;
        float* outptr = dst;
        float* outptr2 = outptr + OW;

        const float* r0 = src_ptr;
        const float* r1 = src_ptr + IW;
        const float* r2 = src_ptr + IW * 2;
        const float* r3 = src_ptr + IW * 3;

        const float* k0 = filter;
        const float* k1 = filter + 3;
        const float* k2 = filter + 5;

        MEGDNN_SIMD_TYPE _k0123 = MEGDNN_SIMD_LOADU(k0);
        MEGDNN_SIMD_TYPE _k3456 = MEGDNN_SIMD_LOADU(k1);
        MEGDNN_SIMD_TYPE _k5678 = MEGDNN_SIMD_LOADU(k2);
        MEGDNN_SIMD_TYPE _k6789 = MEGDNN_SIMD_EXT(_k5678, _k5678, 1);

        size_t h = 0;
        for (; h + 1 < OH; h += 2) {
            int width = OW >> 2;

            rep(i, width) {
                MEGDNN_SIMD_TYPE _sum1 = MEGDNN_SIMD_LOADU(outptr);
                MEGDNN_SIMD_TYPE _sum2 = MEGDNN_SIMD_SET1(0.f);
                MEGDNN_SIMD_TYPE _sum3 = MEGDNN_SIMD_LOADU(outptr2);
                MEGDNN_SIMD_TYPE _sum4 = MEGDNN_SIMD_SET1(0.f);

                MEGDNN_SIMD_TYPE _r00 = MEGDNN_SIMD_LOADU(r0);
                MEGDNN_SIMD_TYPE _r00n = MEGDNN_SIMD_LOADU(r0 + 4);
                MEGDNN_SIMD_TYPE _r01 = MEGDNN_SIMD_EXT(_r00, _r00n, 1);
                MEGDNN_SIMD_TYPE _r02 = MEGDNN_SIMD_EXT(_r00, _r00n, 2);

                MEGDNN_SIMD_TYPE _r10 = MEGDNN_SIMD_LOADU(r1);
                MEGDNN_SIMD_TYPE _r10n = MEGDNN_SIMD_LOADU(r1 + 4);
                MEGDNN_SIMD_TYPE _r11 = MEGDNN_SIMD_EXT(_r10, _r10n, 1);
                MEGDNN_SIMD_TYPE _r12 = MEGDNN_SIMD_EXT(_r10, _r10n, 2);

                MEGDNN_SIMD_TYPE _r20 = MEGDNN_SIMD_LOADU(r2);
                MEGDNN_SIMD_TYPE _r20n = MEGDNN_SIMD_LOADU(r2 + 4);
                MEGDNN_SIMD_TYPE _r21 = MEGDNN_SIMD_EXT(_r20, _r20n, 1);
                MEGDNN_SIMD_TYPE _r22 = MEGDNN_SIMD_EXT(_r20, _r20n, 2);

                MEGDNN_SIMD_TYPE _r30 = MEGDNN_SIMD_LOADU(r3);
                MEGDNN_SIMD_TYPE _r30n = MEGDNN_SIMD_LOADU_2(r3 + 4);
                MEGDNN_SIMD_TYPE _r31 = MEGDNN_SIMD_EXT(_r30, _r30n, 1);
                MEGDNN_SIMD_TYPE _r32 = MEGDNN_SIMD_EXT(_r30, _r30n, 2);

                _sum1 = MEGDNN_SIMD_FMA_LANE(_sum1, _r00, _k0123, 0);
                _sum2 = MEGDNN_SIMD_FMA_LANE(_sum2, _r01, _k0123, 1);
                _sum1 = MEGDNN_SIMD_FMA_LANE(_sum1, _r02, _k0123, 2);
                _sum2 = MEGDNN_SIMD_FMA_LANE(_sum2, _r10, _k3456, 0);
                _sum1 = MEGDNN_SIMD_FMA_LANE(_sum1, _r11, _k3456, 1);
                _sum2 = MEGDNN_SIMD_FMA_LANE(_sum2, _r12, _k3456, 2);
                _sum1 = MEGDNN_SIMD_FMA_LANE(_sum1, _r20, _k6789, 0);
                _sum2 = MEGDNN_SIMD_FMA_LANE(_sum2, _r21, _k6789, 1);
                _sum1 = MEGDNN_SIMD_FMA_LANE(_sum1, _r22, _k6789, 2);

                _sum3 = MEGDNN_SIMD_FMA_LANE(_sum3, _r10, _k0123, 0);
                _sum4 = MEGDNN_SIMD_FMA_LANE(_sum4, _r11, _k0123, 1);
                _sum3 = MEGDNN_SIMD_FMA_LANE(_sum3, _r12, _k0123, 2);
                _sum4 = MEGDNN_SIMD_FMA_LANE(_sum4, _r20, _k3456, 0);
                _sum3 = MEGDNN_SIMD_FMA_LANE(_sum3, _r21, _k3456, 1);
                _sum4 = MEGDNN_SIMD_FMA_LANE(_sum4, _r22, _k3456, 2);
                _sum3 = MEGDNN_SIMD_FMA_LANE(_sum3, _r30, _k6789, 0);
                _sum4 = MEGDNN_SIMD_FMA_LANE(_sum4, _r31, _k6789, 1);
                _sum3 = MEGDNN_SIMD_FMA_LANE(_sum3, _r32, _k6789, 2);

                _sum1 = MEGDNN_SIMD_ADD(_sum1, _sum2);
                _sum3 = MEGDNN_SIMD_ADD(_sum3, _sum4);

                MEGDNN_SIMD_STOREU(outptr, _sum1);
                MEGDNN_SIMD_STOREU(outptr2, _sum3);

                r0 += 4;
                r1 += 4;
                r2 += 4;
                r3 += 4;
                outptr += 4;
                outptr2 += 4;
            }

            r0 += tail_step + IW;
            r1 += tail_step + IW;
            r2 += tail_step + IW;
            r3 += tail_step + IW;

            outptr += OW;
            outptr2 += OW;
        }

        for (; h < OH; h++) {
            int width = OW >> 2;

            rep(i, width) {
                MEGDNN_SIMD_TYPE _sum1 = MEGDNN_SIMD_LOADU(outptr);
                MEGDNN_SIMD_TYPE _sum2 = MEGDNN_SIMD_SET1(0.f);

                MEGDNN_SIMD_TYPE _r00 = MEGDNN_SIMD_LOADU(r0);
                MEGDNN_SIMD_TYPE _r00n = MEGDNN_SIMD_LOADU(r0 + 4);
                MEGDNN_SIMD_TYPE _r01 = MEGDNN_SIMD_EXT(_r00, _r00n, 1);
                MEGDNN_SIMD_TYPE _r02 = MEGDNN_SIMD_EXT(_r00, _r00n, 2);

                MEGDNN_SIMD_TYPE _r10 = MEGDNN_SIMD_LOADU(r1);
                MEGDNN_SIMD_TYPE _r10n = MEGDNN_SIMD_LOADU(r1 + 4);
                MEGDNN_SIMD_TYPE _r11 = MEGDNN_SIMD_EXT(_r10, _r10n, 1);
                MEGDNN_SIMD_TYPE _r12 = MEGDNN_SIMD_EXT(_r10, _r10n, 2);

                MEGDNN_SIMD_TYPE _r20 = MEGDNN_SIMD_LOADU(r2);
                MEGDNN_SIMD_TYPE _r20n = MEGDNN_SIMD_LOADU(r2 + 4);
                MEGDNN_SIMD_TYPE _r21 = MEGDNN_SIMD_EXT(_r20, _r20n, 1);
                MEGDNN_SIMD_TYPE _r22 = MEGDNN_SIMD_EXT(_r20, _r20n, 2);

                _sum1 = MEGDNN_SIMD_FMA_LANE(_sum1, _r00, _k0123, 0);
                _sum2 = MEGDNN_SIMD_FMA_LANE(_sum2, _r01, _k0123, 1);
                _sum1 = MEGDNN_SIMD_FMA_LANE(_sum1, _r02, _k0123, 2);
                _sum2 = MEGDNN_SIMD_FMA_LANE(_sum2, _r10, _k3456, 0);
                _sum1 = MEGDNN_SIMD_FMA_LANE(_sum1, _r11, _k3456, 1);
                _sum2 = MEGDNN_SIMD_FMA_LANE(_sum2, _r12, _k3456, 2);
                _sum1 = MEGDNN_SIMD_FMA_LANE(_sum1, _r20, _k6789, 0);
                _sum2 = MEGDNN_SIMD_FMA_LANE(_sum2, _r21, _k6789, 1);
                _sum1 = MEGDNN_SIMD_FMA_LANE(_sum1, _r22, _k6789, 2);

                _sum1 = MEGDNN_SIMD_ADD(_sum1, _sum2);

                MEGDNN_SIMD_STOREU(outptr, _sum1);

                r0 += 4;
                r1 += 4;
                r2 += 4;
                outptr += 4;
            }
            r0 += tail_step;
            r1 += tail_step;
            r2 += tail_step;
        }

        filter += 9;
    }
}

void conv_stride1::do_conv_5x5_stride1(const float* src, const float* filter,
                                       float* dst, size_t IH, size_t IW,
                                       size_t OH, size_t OW, size_t IC) {
    const size_t tail_step = IW - OW;

    rep(ic, IC) {
        const float* src_ptr = src + IW * IH * ic;
        float* outptr = dst;
        float* outptr2 = outptr + OW;

        const float* r0 = src_ptr;
        const float* r1 = src_ptr + IW;
        const float* r2 = src_ptr + IW * 2;
        const float* r3 = src_ptr + IW * 3;
        const float* r4 = src_ptr + IW * 4;
        const float* r5 = src_ptr + IW * 5;

        MEGDNN_SIMD_TYPE _k0123 = MEGDNN_SIMD_LOADU(filter);
        MEGDNN_SIMD_TYPE _k4567 = MEGDNN_SIMD_LOADU(filter + 4);
        MEGDNN_SIMD_TYPE _k891011 = MEGDNN_SIMD_LOADU(filter + 8);
        MEGDNN_SIMD_TYPE _k12131415 = MEGDNN_SIMD_LOADU(filter + 12);
        MEGDNN_SIMD_TYPE _k16171819 = MEGDNN_SIMD_LOADU(filter + 16);
        MEGDNN_SIMD_TYPE _k20212223 = MEGDNN_SIMD_LOADU(filter + 20);
        MEGDNN_SIMD_TYPE _k24242424 = MEGDNN_SIMD_SET1(filter[24]);

        size_t h = 0;
        for (; h + 1 < OH; h += 2) {
            int width = OW >> 2;

            rep(i, width) {
                MEGDNN_SIMD_TYPE _sum = MEGDNN_SIMD_LOADU(outptr);
                MEGDNN_SIMD_TYPE _sum2 = MEGDNN_SIMD_LOADU(outptr2);

                MEGDNN_SIMD_TYPE _r00 = MEGDNN_SIMD_LOADU(r0);
                MEGDNN_SIMD_TYPE _r04 = MEGDNN_SIMD_LOADU(r0 + 4);
                MEGDNN_SIMD_TYPE _r01 = MEGDNN_SIMD_EXT(_r00, _r04, 1);
                MEGDNN_SIMD_TYPE _r02 = MEGDNN_SIMD_EXT(_r00, _r04, 2);
                MEGDNN_SIMD_TYPE _r03 = MEGDNN_SIMD_EXT(_r00, _r04, 3);

                MEGDNN_SIMD_TYPE _r10 = MEGDNN_SIMD_LOADU(r1);
                MEGDNN_SIMD_TYPE _r14 = MEGDNN_SIMD_LOADU(r1 + 4);
                MEGDNN_SIMD_TYPE _r11 = MEGDNN_SIMD_EXT(_r10, _r14, 1);
                MEGDNN_SIMD_TYPE _r12 = MEGDNN_SIMD_EXT(_r10, _r14, 2);
                MEGDNN_SIMD_TYPE _r13 = MEGDNN_SIMD_EXT(_r10, _r14, 3);

                MEGDNN_SIMD_TYPE _r20 = MEGDNN_SIMD_LOADU(r2);
                MEGDNN_SIMD_TYPE _r24 = MEGDNN_SIMD_LOADU(r2 + 4);
                MEGDNN_SIMD_TYPE _r21 = MEGDNN_SIMD_EXT(_r20, _r24, 1);
                MEGDNN_SIMD_TYPE _r22 = MEGDNN_SIMD_EXT(_r20, _r24, 2);
                MEGDNN_SIMD_TYPE _r23 = MEGDNN_SIMD_EXT(_r20, _r24, 3);

                MEGDNN_SIMD_TYPE _r30 = MEGDNN_SIMD_LOADU(r3);
                MEGDNN_SIMD_TYPE _r34 = MEGDNN_SIMD_LOADU(r3 + 4);
                MEGDNN_SIMD_TYPE _r31 = MEGDNN_SIMD_EXT(_r30, _r34, 1);
                MEGDNN_SIMD_TYPE _r32 = MEGDNN_SIMD_EXT(_r30, _r34, 2);
                MEGDNN_SIMD_TYPE _r33 = MEGDNN_SIMD_EXT(_r30, _r34, 3);

                MEGDNN_SIMD_TYPE _r40 = MEGDNN_SIMD_LOADU(r4);
                MEGDNN_SIMD_TYPE _r44 = MEGDNN_SIMD_LOADU(r4 + 4);
                MEGDNN_SIMD_TYPE _r41 = MEGDNN_SIMD_EXT(_r40, _r44, 1);
                MEGDNN_SIMD_TYPE _r42 = MEGDNN_SIMD_EXT(_r40, _r44, 2);
                MEGDNN_SIMD_TYPE _r43 = MEGDNN_SIMD_EXT(_r40, _r44, 3);

                MEGDNN_SIMD_TYPE _r50 = MEGDNN_SIMD_LOADU(r5);
                MEGDNN_SIMD_TYPE _r54 = MEGDNN_SIMD_LOADU(r5 + 4);
                MEGDNN_SIMD_TYPE _r51 = MEGDNN_SIMD_EXT(_r50, _r54, 1);
                MEGDNN_SIMD_TYPE _r52 = MEGDNN_SIMD_EXT(_r50, _r54, 2);
                MEGDNN_SIMD_TYPE _r53 = MEGDNN_SIMD_EXT(_r50, _r54, 3);

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

                _sum2 = MEGDNN_SIMD_FMA_LANE(_sum2, _r10, _k0123, 0);
                _sum2 = MEGDNN_SIMD_FMA_LANE(_sum2, _r11, _k0123, 1);
                _sum2 = MEGDNN_SIMD_FMA_LANE(_sum2, _r12, _k0123, 2);
                _sum2 = MEGDNN_SIMD_FMA_LANE(_sum2, _r13, _k0123, 3);
                _sum2 = MEGDNN_SIMD_FMA_LANE(_sum2, _r14, _k4567, 0);

                _sum2 = MEGDNN_SIMD_FMA_LANE(_sum2, _r20, _k4567, 1);
                _sum2 = MEGDNN_SIMD_FMA_LANE(_sum2, _r21, _k4567, 2);
                _sum2 = MEGDNN_SIMD_FMA_LANE(_sum2, _r22, _k4567, 3);
                _sum2 = MEGDNN_SIMD_FMA_LANE(_sum2, _r23, _k891011, 0);
                _sum2 = MEGDNN_SIMD_FMA_LANE(_sum2, _r24, _k891011, 1);

                _sum2 = MEGDNN_SIMD_FMA_LANE(_sum2, _r30, _k891011, 2);
                _sum2 = MEGDNN_SIMD_FMA_LANE(_sum2, _r31, _k891011, 3);
                _sum2 = MEGDNN_SIMD_FMA_LANE(_sum2, _r32, _k12131415, 0);
                _sum2 = MEGDNN_SIMD_FMA_LANE(_sum2, _r33, _k12131415, 1);
                _sum2 = MEGDNN_SIMD_FMA_LANE(_sum2, _r34, _k12131415, 2);

                _sum2 = MEGDNN_SIMD_FMA_LANE(_sum2, _r40, _k12131415, 3);
                _sum2 = MEGDNN_SIMD_FMA_LANE(_sum2, _r41, _k16171819, 0);
                _sum2 = MEGDNN_SIMD_FMA_LANE(_sum2, _r42, _k16171819, 1);
                _sum2 = MEGDNN_SIMD_FMA_LANE(_sum2, _r43, _k16171819, 2);
                _sum2 = MEGDNN_SIMD_FMA_LANE(_sum2, _r44, _k16171819, 3);

                _sum2 = MEGDNN_SIMD_FMA_LANE(_sum2, _r50, _k20212223, 0);
                _sum2 = MEGDNN_SIMD_FMA_LANE(_sum2, _r51, _k20212223, 1);
                _sum2 = MEGDNN_SIMD_FMA_LANE(_sum2, _r52, _k20212223, 2);
                _sum2 = MEGDNN_SIMD_FMA_LANE(_sum2, _r53, _k20212223, 3);
                _sum2 = MEGDNN_SIMD_FMA_LANE(_sum2, _r54, _k24242424, 0);

                MEGDNN_SIMD_STOREU(outptr, _sum);
                MEGDNN_SIMD_STOREU(outptr2, _sum2);

                r0 += 4;
                r1 += 4;
                r2 += 4;
                r3 += 4;
                r4 += 4;
                r5 += 4;
                outptr += 4;
                outptr2 += 4;
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
            int width = OW >> 2;

            rep(i, width) {
                MEGDNN_SIMD_TYPE _sum = MEGDNN_SIMD_LOADU(outptr);

                MEGDNN_SIMD_TYPE _r00 = MEGDNN_SIMD_LOADU(r0);
                MEGDNN_SIMD_TYPE _r04 = MEGDNN_SIMD_LOADU(r0 + 4);
                MEGDNN_SIMD_TYPE _r01 = MEGDNN_SIMD_EXT(_r00, _r04, 1);
                MEGDNN_SIMD_TYPE _r02 = MEGDNN_SIMD_EXT(_r00, _r04, 2);
                MEGDNN_SIMD_TYPE _r03 = MEGDNN_SIMD_EXT(_r00, _r04, 3);

                MEGDNN_SIMD_TYPE _r10 = MEGDNN_SIMD_LOADU(r1);
                MEGDNN_SIMD_TYPE _r14 = MEGDNN_SIMD_LOADU(r1 + 4);
                MEGDNN_SIMD_TYPE _r11 = MEGDNN_SIMD_EXT(_r10, _r14, 1);
                MEGDNN_SIMD_TYPE _r12 = MEGDNN_SIMD_EXT(_r10, _r14, 2);
                MEGDNN_SIMD_TYPE _r13 = MEGDNN_SIMD_EXT(_r10, _r14, 3);

                MEGDNN_SIMD_TYPE _r20 = MEGDNN_SIMD_LOADU(r2);
                MEGDNN_SIMD_TYPE _r24 = MEGDNN_SIMD_LOADU(r2 + 4);
                MEGDNN_SIMD_TYPE _r21 = MEGDNN_SIMD_EXT(_r20, _r24, 1);
                MEGDNN_SIMD_TYPE _r22 = MEGDNN_SIMD_EXT(_r20, _r24, 2);
                MEGDNN_SIMD_TYPE _r23 = MEGDNN_SIMD_EXT(_r20, _r24, 3);

                MEGDNN_SIMD_TYPE _r30 = MEGDNN_SIMD_LOADU(r3);
                MEGDNN_SIMD_TYPE _r34 = MEGDNN_SIMD_LOADU(r3 + 4);
                MEGDNN_SIMD_TYPE _r31 = MEGDNN_SIMD_EXT(_r30, _r34, 1);
                MEGDNN_SIMD_TYPE _r32 = MEGDNN_SIMD_EXT(_r30, _r34, 2);
                MEGDNN_SIMD_TYPE _r33 = MEGDNN_SIMD_EXT(_r30, _r34, 3);

                MEGDNN_SIMD_TYPE _r40 = MEGDNN_SIMD_LOADU(r4);
                MEGDNN_SIMD_TYPE _r44 = MEGDNN_SIMD_LOADU(r4 + 4);
                MEGDNN_SIMD_TYPE _r41 = MEGDNN_SIMD_EXT(_r40, _r44, 1);
                MEGDNN_SIMD_TYPE _r42 = MEGDNN_SIMD_EXT(_r40, _r44, 2);
                MEGDNN_SIMD_TYPE _r43 = MEGDNN_SIMD_EXT(_r40, _r44, 3);

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

                r0 += 4;
                r1 += 4;
                r2 += 4;
                r3 += 4;
                r4 += 4;
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

void conv_stride1::do_conv_7x7_stride1(const float* src, const float* filter,
                                       float* dst, size_t IH, size_t IW,
                                       size_t OH, size_t OW, size_t IC) {
    const size_t tail_step = IW - OW;

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
            int width = OW >> 2;

            rep(i, width) {
                MEGDNN_SIMD_TYPE _sum = MEGDNN_SIMD_LOADU(outptr);

                MEGDNN_SIMD_TYPE _k0123 = MEGDNN_SIMD_LOADU(k0);
                MEGDNN_SIMD_TYPE _k4567 = MEGDNN_SIMD_LOADU(k0 + 4);

                MEGDNN_SIMD_TYPE _r00 = MEGDNN_SIMD_LOADU(r0);      // 0 1 2 3
                MEGDNN_SIMD_TYPE _r04 = MEGDNN_SIMD_LOADU(r0 + 4);  // 4 5 6 7
                MEGDNN_SIMD_TYPE _r00n =
                        MEGDNN_SIMD_LOADU(r0 + 8);  // 8 9 10 11
                MEGDNN_SIMD_TYPE _r01 =
                        MEGDNN_SIMD_EXT(_r00, _r04, 1);  // 1 2 3 4
                MEGDNN_SIMD_TYPE _r02 =
                        MEGDNN_SIMD_EXT(_r00, _r04, 2);  // 2 3 4 5
                MEGDNN_SIMD_TYPE _r03 =
                        MEGDNN_SIMD_EXT(_r00, _r04, 3);  // 3 4 5 6
                MEGDNN_SIMD_TYPE _r05 =
                        MEGDNN_SIMD_EXT(_r04, _r00n, 1);  // 5 6 7 8
                MEGDNN_SIMD_TYPE _r06 =
                        MEGDNN_SIMD_EXT(_r04, _r00n, 2);  // 6 7 8 9

                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r00, _k0123, 0);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r01, _k0123, 1);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r02, _k0123, 2);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r03, _k0123, 3);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r04, _k4567, 0);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r05, _k4567, 1);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r06, _k4567, 2);

                MEGDNN_SIMD_TYPE _k78910 = MEGDNN_SIMD_LOADU(k1);
                MEGDNN_SIMD_TYPE _k11121314 = MEGDNN_SIMD_LOADU(k1 + 4);

                MEGDNN_SIMD_TYPE _r10 = MEGDNN_SIMD_LOADU(r1);
                MEGDNN_SIMD_TYPE _r14 = MEGDNN_SIMD_LOADU(r1 + 4);
                MEGDNN_SIMD_TYPE _r10n = MEGDNN_SIMD_LOADU(r1 + 8);
                MEGDNN_SIMD_TYPE _r11 = MEGDNN_SIMD_EXT(_r10, _r14, 1);
                MEGDNN_SIMD_TYPE _r12 = MEGDNN_SIMD_EXT(_r10, _r14, 2);
                MEGDNN_SIMD_TYPE _r13 = MEGDNN_SIMD_EXT(_r10, _r14, 3);
                MEGDNN_SIMD_TYPE _r15 = MEGDNN_SIMD_EXT(_r14, _r10n, 1);
                MEGDNN_SIMD_TYPE _r16 = MEGDNN_SIMD_EXT(_r14, _r10n, 2);

                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r10, _k78910, 0);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r11, _k78910, 1);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r12, _k78910, 2);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r13, _k78910, 3);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r14, _k11121314, 0);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r15, _k11121314, 1);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r16, _k11121314, 2);

                MEGDNN_SIMD_TYPE _k14151617 = MEGDNN_SIMD_LOADU(k2);
                MEGDNN_SIMD_TYPE _k18192021 = MEGDNN_SIMD_LOADU(k2 + 4);

                MEGDNN_SIMD_TYPE _r20 = MEGDNN_SIMD_LOADU(r2);
                MEGDNN_SIMD_TYPE _r24 = MEGDNN_SIMD_LOADU(r2 + 4);
                MEGDNN_SIMD_TYPE _r20n = MEGDNN_SIMD_LOADU(r2 + 8);
                MEGDNN_SIMD_TYPE _r21 = MEGDNN_SIMD_EXT(_r20, _r24, 1);
                MEGDNN_SIMD_TYPE _r22 = MEGDNN_SIMD_EXT(_r20, _r24, 2);
                MEGDNN_SIMD_TYPE _r23 = MEGDNN_SIMD_EXT(_r20, _r24, 3);
                MEGDNN_SIMD_TYPE _r25 = MEGDNN_SIMD_EXT(_r24, _r20n, 1);
                MEGDNN_SIMD_TYPE _r26 = MEGDNN_SIMD_EXT(_r24, _r20n, 2);

                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r20, _k14151617, 0);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r21, _k14151617, 1);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r22, _k14151617, 2);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r23, _k14151617, 3);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r24, _k18192021, 0);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r25, _k18192021, 1);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r26, _k18192021, 2);

                MEGDNN_SIMD_TYPE _k21222324 = MEGDNN_SIMD_LOADU(k3);
                MEGDNN_SIMD_TYPE _k25262728 = MEGDNN_SIMD_LOADU(k3 + 4);

                MEGDNN_SIMD_TYPE _r30 = MEGDNN_SIMD_LOADU(r3);
                MEGDNN_SIMD_TYPE _r34 = MEGDNN_SIMD_LOADU(r3 + 4);
                MEGDNN_SIMD_TYPE _r30n = MEGDNN_SIMD_LOADU(r3 + 8);
                MEGDNN_SIMD_TYPE _r31 = MEGDNN_SIMD_EXT(_r30, _r34, 1);
                MEGDNN_SIMD_TYPE _r32 = MEGDNN_SIMD_EXT(_r30, _r34, 2);
                MEGDNN_SIMD_TYPE _r33 = MEGDNN_SIMD_EXT(_r30, _r34, 3);
                MEGDNN_SIMD_TYPE _r35 = MEGDNN_SIMD_EXT(_r34, _r30n, 1);
                MEGDNN_SIMD_TYPE _r36 = MEGDNN_SIMD_EXT(_r34, _r30n, 2);

                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r30, _k21222324, 0);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r31, _k21222324, 1);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r32, _k21222324, 2);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r33, _k21222324, 3);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r34, _k25262728, 0);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r35, _k25262728, 1);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r36, _k25262728, 2);

                MEGDNN_SIMD_TYPE _k28293031 = MEGDNN_SIMD_LOADU(k4);
                MEGDNN_SIMD_TYPE _k32333435 = MEGDNN_SIMD_LOADU(k4 + 4);

                MEGDNN_SIMD_TYPE _r40 = MEGDNN_SIMD_LOADU(r4);
                MEGDNN_SIMD_TYPE _r44 = MEGDNN_SIMD_LOADU(r4 + 4);
                MEGDNN_SIMD_TYPE _r40n = MEGDNN_SIMD_LOADU(r4 + 8);
                MEGDNN_SIMD_TYPE _r41 = MEGDNN_SIMD_EXT(_r40, _r44, 1);
                MEGDNN_SIMD_TYPE _r42 = MEGDNN_SIMD_EXT(_r40, _r44, 2);
                MEGDNN_SIMD_TYPE _r43 = MEGDNN_SIMD_EXT(_r40, _r44, 3);
                MEGDNN_SIMD_TYPE _r45 = MEGDNN_SIMD_EXT(_r44, _r40n, 1);
                MEGDNN_SIMD_TYPE _r46 = MEGDNN_SIMD_EXT(_r44, _r40n, 2);

                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r40, _k28293031, 0);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r41, _k28293031, 1);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r42, _k28293031, 2);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r43, _k28293031, 3);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r44, _k32333435, 0);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r45, _k32333435, 1);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r46, _k32333435, 2);

                MEGDNN_SIMD_TYPE _k35363738 = MEGDNN_SIMD_LOADU(k5);
                MEGDNN_SIMD_TYPE _k39404142 = MEGDNN_SIMD_LOADU(k5 + 4);

                MEGDNN_SIMD_TYPE _r50 = MEGDNN_SIMD_LOADU(r5);
                MEGDNN_SIMD_TYPE _r54 = MEGDNN_SIMD_LOADU(r5 + 4);
                MEGDNN_SIMD_TYPE _r50n = MEGDNN_SIMD_LOADU(r5 + 8);
                MEGDNN_SIMD_TYPE _r51 = MEGDNN_SIMD_EXT(_r50, _r54, 1);
                MEGDNN_SIMD_TYPE _r52 = MEGDNN_SIMD_EXT(_r50, _r54, 2);
                MEGDNN_SIMD_TYPE _r53 = MEGDNN_SIMD_EXT(_r50, _r54, 3);
                MEGDNN_SIMD_TYPE _r55 = MEGDNN_SIMD_EXT(_r54, _r50n, 1);
                MEGDNN_SIMD_TYPE _r56 = MEGDNN_SIMD_EXT(_r54, _r50n, 2);

                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r50, _k35363738, 0);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r51, _k35363738, 1);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r52, _k35363738, 2);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r53, _k35363738, 3);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r54, _k39404142, 0);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r55, _k39404142, 1);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r56, _k39404142, 2);

                MEGDNN_SIMD_TYPE _k42434445 = MEGDNN_SIMD_LOADU(k6);
                MEGDNN_SIMD_TYPE _k46474849 = MEGDNN_SIMD_LOADU_3(k6 + 4);

                MEGDNN_SIMD_TYPE _r60 = MEGDNN_SIMD_LOADU(r6);
                MEGDNN_SIMD_TYPE _r64 = MEGDNN_SIMD_LOADU(r6 + 4);
                MEGDNN_SIMD_TYPE _r60n = MEGDNN_SIMD_LOADU(r6 + 8);
                MEGDNN_SIMD_TYPE _r61 = MEGDNN_SIMD_EXT(_r60, _r64, 1);
                MEGDNN_SIMD_TYPE _r62 = MEGDNN_SIMD_EXT(_r60, _r64, 2);
                MEGDNN_SIMD_TYPE _r63 = MEGDNN_SIMD_EXT(_r60, _r64, 3);
                MEGDNN_SIMD_TYPE _r65 = MEGDNN_SIMD_EXT(_r64, _r60n, 1);
                MEGDNN_SIMD_TYPE _r66 = MEGDNN_SIMD_EXT(_r64, _r60n, 2);

                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r60, _k42434445, 0);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r61, _k42434445, 1);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r62, _k42434445, 2);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r63, _k42434445, 3);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r64, _k46474849, 0);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r65, _k46474849, 1);
                _sum = MEGDNN_SIMD_FMA_LANE(_sum, _r66, _k46474849, 2);

                MEGDNN_SIMD_STOREU(outptr, _sum);

                r0 += 4;
                r1 += 4;
                r2 += 4;
                r3 += 4;
                r4 += 4;
                r5 += 4;
                r6 += 4;
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

#include "src/common/simd_macro/epilogue.h"
// vim: syntax=cpp.doxygen
