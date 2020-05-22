/**
 * By downloading, copying, installing or using the software you agree to this license.
 * If you do not agree to this license, do not download, install,
 * copy or use the software.
 *
 *
 *                           License Agreement
 *                For Open Source Computer Vision Library
 *                        (3-clause BSD License)
 *
 * Copyright (C) 2000-2020, Intel Corporation, all rights reserved.
 * Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
 * Copyright (C) 2009-2016, NVIDIA Corporation, all rights reserved.
 * Copyright (C) 2010-2013, Advanced Micro Devices, Inc., all rights reserved.
 * Copyright (C) 2015-2016, OpenCV Foundation, all rights reserved.
 * Copyright (C) 2015-2016, Itseez Inc., all rights reserved.
 * Copyright (C) 2019-2020, Xperience AI, all rights reserved.
 * Third party copyrights are property of their respective owners.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 *   * Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *   * Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *
 *   * Neither the names of the copyright holders nor the names of the contributors
 *     may be used to endorse or promote products derived from this software
 *     without specific prior written permission.
 *
 * This software is provided by the copyright holders and contributors "as is" and
 * any express or implied warranties, including, but not limited to, the implied
 * warranties of merchantability and fitness for a particular purpose are disclaimed.
 * In no event shall copyright holders or contributors be liable for any direct,
 * indirect, incidental, special, exemplary, or consequential damages
 * (including, but not limited to, procurement of substitute goods or services;
 * loss of use, data, or profits; or business interruption) however caused
 * and on any theory of liability, whether in contract, strict liability,
 * or tort (including negligence or otherwise) arising in any way out of
 * the use of this software, even if advised of the possibility of such damage.
 *
 * ---------------------------------------------------------------------------
 * \file dnn/src/arm_common/separable_filter/filter.h
 *
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *
 * This file has been modified by Megvii ("Megvii Modifications").
 * All Megvii Modifications are Copyright (C) 2014-2019 Megvii Inc. All rights reserved.
 *
 * ---------------------------------------------------------------------------
 */

#pragma once
#include "src/common/cv/filter.h"
#include "src/arm_common/simd_macro/marm_neon.h"
#include <cfloat>
#include <cmath>

namespace megdnn {
namespace megcv {
namespace sep_filter {

using namespace filter_common;

struct SymmRowSmallVec_8u32s {
    SymmRowSmallVec_8u32s() {}
    SymmRowSmallVec_8u32s(const uchar* _kernel, int _len) {
        kernel = (int*)_kernel;
        ksize = _len;
    }

    int operator()(const uchar* src, uchar* _dst, int width, int cn) const {
        int i = 0, _ksize = ksize;
        int* dst = (int*)_dst;
        const int* kx = kernel + _ksize / 2;

        src += (_ksize / 2) * cn;
        width *= cn;

        if (_ksize == 1)
            return 0;
        if (_ksize == 3) {
            if (kx[0] == 2 && kx[1] == 1) {
                uint16x8_t zq = vdupq_n_u16(0);

                for (; i <= width - 8; i += 8, src += 8) {
                    uint8x8_t x0, x1, x2;
                    x0 = vld1_u8((uint8_t*)(src - cn));
                    x1 = vld1_u8((uint8_t*)(src));
                    x2 = vld1_u8((uint8_t*)(src + cn));

                    uint16x8_t y0, y1, y2;
                    y0 = vaddl_u8(x0, x2);
                    y1 = vshll_n_u8(x1, 1);
                    y2 = vaddq_u16(y0, y1);

                    uint16x8x2_t str;
                    str.val[0] = y2;
                    str.val[1] = zq;
                    vst2q_u16((uint16_t*)(dst + i), str);
                }
            } else if (kx[0] == -2 && kx[1] == 1)
                return 0;
            else {
                int32x4_t k32 = vdupq_n_s32(0);
                k32 = vld1q_lane_s32(kx, k32, 0);
                k32 = vld1q_lane_s32(kx + 1, k32, 1);

                int16x4_t k = vqmovn_s32(k32);

                uint8x8_t z = vdup_n_u8(0);

                for (; i <= width - 8; i += 8, src += 8) {
                    uint8x8_t x0, x1, x2;
                    x0 = vld1_u8((uint8_t*)(src - cn));
                    x1 = vld1_u8((uint8_t*)(src));
                    x2 = vld1_u8((uint8_t*)(src + cn));

                    int16x8_t y0, y1;
                    int32x4_t y2, y3;
                    y0 = vreinterpretq_s16_u16(vaddl_u8(x1, z));
                    y1 = vreinterpretq_s16_u16(vaddl_u8(x0, x2));
                    y2 = vmull_lane_s16(vget_low_s16(y0), k, 0);
                    y2 = vmlal_lane_s16(y2, vget_low_s16(y1), k, 1);
                    y3 = vmull_lane_s16(vget_high_s16(y0), k, 0);
                    y3 = vmlal_lane_s16(y3, vget_high_s16(y1), k, 1);

                    vst1q_s32((int32_t*)(dst + i), y2);
                    vst1q_s32((int32_t*)(dst + i + 4), y3);
                }
            }
        } else if (_ksize == 5) {
            if (kx[0] == -2 && kx[1] == 0 && kx[2] == 1)
                return 0;
            else {
                int32x4_t k32 = vdupq_n_s32(0);
                k32 = vld1q_lane_s32(kx, k32, 0);
                k32 = vld1q_lane_s32(kx + 1, k32, 1);
                k32 = vld1q_lane_s32(kx + 2, k32, 2);

                int16x4_t k = vqmovn_s32(k32);

                uint8x8_t z = vdup_n_u8(0);

                for (; i <= width - 8; i += 8, src += 8) {
                    uint8x8_t x0, x1, x2, x3, x4;
                    x0 = vld1_u8((uint8_t*)(src - cn));
                    x1 = vld1_u8((uint8_t*)(src));
                    x2 = vld1_u8((uint8_t*)(src + cn));

                    int16x8_t y0, y1;
                    int32x4_t accl, acch;
                    y0 = vreinterpretq_s16_u16(vaddl_u8(x1, z));
                    y1 = vreinterpretq_s16_u16(vaddl_u8(x0, x2));
                    accl = vmull_lane_s16(vget_low_s16(y0), k, 0);
                    accl = vmlal_lane_s16(accl, vget_low_s16(y1), k, 1);
                    acch = vmull_lane_s16(vget_high_s16(y0), k, 0);
                    acch = vmlal_lane_s16(acch, vget_high_s16(y1), k, 1);

                    int16x8_t y2;
                    x3 = vld1_u8((uint8_t*)(src - cn * 2));
                    x4 = vld1_u8((uint8_t*)(src + cn * 2));
                    y2 = vreinterpretq_s16_u16(vaddl_u8(x3, x4));
                    accl = vmlal_lane_s16(accl, vget_low_s16(y2), k, 2);
                    acch = vmlal_lane_s16(acch, vget_high_s16(y2), k, 2);

                    vst1q_s32((int32_t*)(dst + i), accl);
                    vst1q_s32((int32_t*)(dst + i + 4), acch);
                }
            }
        }

        return i;
    }

    int* kernel;
    size_t ksize;
};

struct SymmColumnVec_32s8u {
    SymmColumnVec_32s8u() {}
    SymmColumnVec_32s8u(const uchar* _kernel, int _len, int _bits) {
        ksize = _len;
        kernel = (float*)malloc(sizeof(float) * ksize);

        for (size_t i = 0; i < ksize; i++)
            kernel[i] = (float)(((int*)_kernel)[i]) * (1. / (1 << _bits));
    }

    ~SymmColumnVec_32s8u() { free(kernel); }

    int operator()(const uchar** _src, uchar* dst, int& count,
                   int width) const {
        MEGDNN_MARK_USED_VAR(count);
        int _ksize = ksize;
        int ksize2 = _ksize / 2;
        const float* ky = kernel + ksize2;
        const int** src = (const int**)_src;
        const int *S, *S2;
        int i = 0, k;

        float32x4_t d4 = vdupq_n_f32(0);

        if (_ksize == 1)
            return 0;

        float32x2_t k32;
        k32 = vdup_n_f32(0);
        k32 = vld1_lane_f32(ky, k32, 0);
        k32 = vld1_lane_f32(ky + 1, k32, 1);

        for (; i <= width - 8; i += 8) {
            float32x4_t accl, acch;
            float32x4_t f0l, f0h, f1l, f1h, f2l, f2h;

            S = src[0] + i;

            f0l = vcvtq_f32_s32(vld1q_s32(S));
            f0h = vcvtq_f32_s32(vld1q_s32(S + 4));

            S = src[1] + i;
            S2 = src[-1] + i;

            f1l = vcvtq_f32_s32(vld1q_s32(S));
            f1h = vcvtq_f32_s32(vld1q_s32(S + 4));
            f2l = vcvtq_f32_s32(vld1q_s32(S2));
            f2h = vcvtq_f32_s32(vld1q_s32(S2 + 4));

            accl = acch = d4;
            accl = vmlaq_lane_f32(accl, f0l, k32, 0);
            acch = vmlaq_lane_f32(acch, f0h, k32, 0);
            accl = vmlaq_lane_f32(accl, vaddq_f32(f1l, f2l), k32, 1);
            acch = vmlaq_lane_f32(acch, vaddq_f32(f1h, f2h), k32, 1);

            for (k = 2; k <= ksize2; k++) {
                S = src[k] + i;
                S2 = src[-k] + i;

                float32x4_t f3l, f3h, f4l, f4h;
                f3l = vcvtq_f32_s32(vld1q_s32(S));
                f3h = vcvtq_f32_s32(vld1q_s32(S + 4));
                f4l = vcvtq_f32_s32(vld1q_s32(S2));
                f4h = vcvtq_f32_s32(vld1q_s32(S2 + 4));

                accl = vmlaq_n_f32(accl, vaddq_f32(f3l, f4l), ky[k]);
                acch = vmlaq_n_f32(acch, vaddq_f32(f3h, f4h), ky[k]);
            }

            int32x4_t s32l, s32h;
            s32l = vcvtq_s32_f32(accl);
            s32h = vcvtq_s32_f32(acch);

            int16x4_t s16l, s16h;
            s16l = vqmovn_s32(s32l);
            s16h = vqmovn_s32(s32h);

            uint8x8_t u8;
            u8 = vqmovun_s16(vcombine_s16(s16l, s16h));

            vst1_u8((uint8_t*)(dst + i), u8);
        }

        return i;
    }

    float* kernel;
    size_t ksize;
};

//! 32f

struct RowVec_32f {
    RowVec_32f() {}

    RowVec_32f(const uchar* _kernel, int _len) {
        ksize = _len;
        kernel = (float*)_kernel;
    }

    int operator()(const uchar* _src, uchar* _dst, int width, int cn) const {
        int _ksize = ksize;
        const float* src0 = (const float*)_src;
        float* dst = (float*)_dst;
        const float* _kx = (float*)kernel;

        int i = 0, k;
        width *= cn;

        for (; i <= width - 8; i += 8) {
            const float* src = src0 + i;
            float32x4_t f, s0 = vdupq_n_f32(0), s1 = s0, x0, x1;
            for (k = 0; k < _ksize; k++, src += cn) {
                f = vdupq_n_f32(_kx[k]);
                x0 = vld1q_f32(src);
                x1 = vld1q_f32(src + 4);
                s0 = vmlaq_f32(s0, x0, f);
                s1 = vmlaq_f32(s1, x1, f);
            }
            vst1q_f32(dst + i, s0);
            vst1q_f32(dst + i + 4, s1);
        }
        for (; i <= width - 4; i += 4) {
            const float* src = src0 + i;
            float32x4_t f, s0 = vdupq_n_f32(0), x0;
            for (k = 0; k < _ksize; k++, src += cn) {
                f = vdupq_n_f32(_kx[k]);

                x0 = vld1q_f32(src);
                s0 = vmlaq_f32(s0, x0, f);
            }
            vst1q_f32(dst + i, s0);
        }
        return i;
    }

    float* kernel;
    int ksize;
};

struct SymmRowSmallVec_32f {
    SymmRowSmallVec_32f() {}
    SymmRowSmallVec_32f(const uchar* _kernel, int _len) {
        ksize = _len;
        kernel = (float*)_kernel;
    }

    int operator()(const uchar* _src, uchar* _dst, int width, int cn) const {
        int i = 0, _ksize = ksize;
        float* dst = (float*)_dst;
        const float* src = (const float*)_src + (_ksize / 2) * cn;
        const float* kx = (float*)kernel + _ksize / 2;
        width *= cn;

        {
            if (_ksize == 1)
                return 0;
            if (_ksize == 3) {
                float32x4_t k0 = vdupq_n_f32(kx[0]), k1 = vdupq_n_f32(kx[1]);
                for (; i <= width - 8; i += 8, src += 8) {
                    float32x4_t x0, x1, x2, y0, y1, y2;
                    x0 = vld1q_f32(src - cn);
                    x1 = vld1q_f32(src);
                    x2 = vld1q_f32(src + cn);
                    y0 = vld1q_f32(src - cn + 4);
                    y1 = vld1q_f32(src + 4);
                    y2 = vld1q_f32(src + cn + 4);

                    x0 = vmulq_f32(vaddq_f32(x0, x2), k1);
                    y0 = vmulq_f32(vaddq_f32(y0, y2), k1);
                    x0 = vmlaq_f32(x0, x1, k0);
                    y0 = vmlaq_f32(y0, y1, k0);
                    vst1q_f32(dst + i, x0);
                    vst1q_f32(dst + i + 4, y0);
                }
            } else if (_ksize == 5) {
                float32x4_t k0 = vdupq_n_f32(kx[0]), k1 = vdupq_n_f32(kx[1]),
                            k2 = vdupq_n_f32(kx[2]);
                for (; i <= width - 8; i += 8, src += 8) {
                    float32x4_t x0, x1, x2, y0, y1, y2;
                    x0 = vld1q_f32(src - cn);
                    x1 = vld1q_f32(src);
                    x2 = vld1q_f32(src + cn);
                    y0 = vld1q_f32(src - cn + 4);
                    y1 = vld1q_f32(src + 4);
                    y2 = vld1q_f32(src + cn + 4);

                    x0 = vmulq_f32(vaddq_f32(x0, x2), k1);
                    y0 = vmulq_f32(vaddq_f32(y0, y2), k1);
                    x0 = vmlaq_f32(x0, x1, k0);
                    y0 = vmlaq_f32(y0, y1, k0);

                    x2 = vaddq_f32(vld1q_f32(src + cn * 2),
                                   vld1q_f32(src - cn * 2));
                    y2 = vaddq_f32(vld1q_f32(src + cn * 2 + 4),
                                   vld1q_f32(src - cn * 2 + 4));
                    x0 = vmlaq_f32(x0, x2, k2);
                    y0 = vmlaq_f32(y0, y2, k2);

                    vst1q_f32(dst + i, x0);
                    vst1q_f32(dst + i + 4, y0);
                }
            }
        }
        return i;
    }

    float* kernel;
    int ksize;
};

struct ColumnVec_32f {
    ColumnVec_32f() {}
    ColumnVec_32f(const uchar* _kernel, int _len, int) {
        ksize = _len;
        kernel = (float*)_kernel;
    }

    int operator()(const uchar** _src, uchar* _dst, int&, int width) const {
        const float* ky = (const float*)kernel;
        int i = 0, k;
        const float** src = (const float**)_src;
        const float* S;
        float* dst = (float*)_dst;

        {
            for (; i <= width - 16; i += 16) {
                float32x4_t f = vdupq_n_f32(ky[0]);

                float32x4_t s0, s1, s2, s3;
                float32x4_t x0, x1;
                S = src[0] + i;
                s0 = vld1q_f32(S);
                s1 = vld1q_f32(S + 4);
                s0 = vmulq_f32(s0, f);
                s1 = vmulq_f32(s1, f);
                s2 = vld1q_f32(S + 8);
                s3 = vld1q_f32(S + 12);
                s2 = vmulq_f32(s2, f);
                s3 = vmulq_f32(s3, f);

                for (k = 1; k < ksize; k++) {
                    S = src[k] + i;
                    float32x4_t f = vdupq_n_f32(ky[k]);
                    x0 = vld1q_f32(S);
                    x1 = vld1q_f32(S + 4);
                    s0 = vmlaq_f32(s0, f, x0);
                    s1 = vmlaq_f32(s1, f, x1);

                    x0 = vld1q_f32(S + 8);
                    x1 = vld1q_f32(S + 12);
                    s2 = vmlaq_f32(s2, f, x0);
                    s3 = vmlaq_f32(s3, f, x1);
                }
                vst1q_f32(dst + i, s0);
                vst1q_f32(dst + i + 4, s1);
                vst1q_f32(dst + i + 8, s2);
                vst1q_f32(dst + i + 12, s3);
            }

            for (; i <= width - 4; i += 4) {
                float32x4_t f = vdupq_n_f32(ky[0]);

                float32x4_t x0, s0 = vld1q_f32(src[0] + i);
                s0 = vmulq_f32(s0, f);

                for (k = 1; k < ksize; k++) {
                    float32x4_t f = vdupq_n_f32(ky[k]);
                    S = src[k] + i;
                    x0 = vld1q_f32(S);
                    s0 = vmlaq_f32(s0, f, x0);
                }
                vst1q_f32(dst + i, s0);
            }
        }

        return i;
    }

    float* kernel;
    int ksize;
};

struct SymmColumnVec_32f {
    SymmColumnVec_32f() {}
    SymmColumnVec_32f(const uchar* _kernel, int _len, int) {
        ksize = _len;
        kernel = (float*)_kernel;
    }

    int operator()(const uchar** _src, uchar* _dst, int&, int width) const {
        int ksize2 = (ksize) / 2;
        const float* ky = (const float*)kernel + ksize2;
        int i = 0, k;
        const float** src = (const float**)_src;
        const float *S, *S2;
        float* dst = (float*)_dst;

        {
            for (; i <= width - 16; i += 16) {
                float32x4_t f = vdupq_n_f32(ky[0]);

                float32x4_t s0, s1, s2, s3;
                float32x4_t x0, x1;
                S = src[0] + i;
                s0 = vld1q_f32(S);
                s1 = vld1q_f32(S + 4);
                s0 = vmulq_f32(s0, f);
                s1 = vmulq_f32(s1, f);
                s2 = vld1q_f32(S + 8);
                s3 = vld1q_f32(S + 12);
                s2 = vmulq_f32(s2, f);
                s3 = vmulq_f32(s3, f);

                for (k = 1; k <= ksize2; k++) {
                    S = src[k] + i;
                    S2 = src[-k] + i;
                    float32x4_t f = vdupq_n_f32(ky[k]);

                    x0 = vaddq_f32(vld1q_f32(S), vld1q_f32(S2));
                    x1 = vaddq_f32(vld1q_f32(S + 4), vld1q_f32(S2 + 4));
                    s0 = vmlaq_f32(s0, x0, f);
                    s1 = vmlaq_f32(s1, x1, f);
                    x0 = vaddq_f32(vld1q_f32(S + 8), vld1q_f32(S2 + 8));
                    x1 = vaddq_f32(vld1q_f32(S + 12), vld1q_f32(S2 + 12));
                    s2 = vmlaq_f32(s2, x0, f);
                    s3 = vmlaq_f32(s3, x1, f);
                }

                vst1q_f32(dst + i, s0);
                vst1q_f32(dst + i + 4, s1);
                vst1q_f32(dst + i + 8, s2);
                vst1q_f32(dst + i + 12, s3);
            }

            for (; i <= width - 4; i += 4) {
                float32x4_t f = vdupq_n_f32(ky[0]);
                float32x4_t x0, s0 = vld1q_f32(src[0] + i);
                s0 = vmulq_f32(s0, f);

                for (k = 1; k <= ksize2; k++) {
                    float32x4_t f = vdupq_n_f32(ky[k]);
                    S = src[k] + i;
                    S2 = src[-k] + i;
                    x0 = vaddq_f32(vld1q_f32(S), vld1q_f32(S2));
                    s0 = vmlaq_f32(s0, x0, f);
                }
                vst1q_f32(dst + i, s0);
            }
        }

        return i;
    }

    float* kernel;
    int ksize;
};

struct SymmColumnSmallVec_32f {
    SymmColumnSmallVec_32f() {}
    SymmColumnSmallVec_32f(const uchar* _kernel, int _len, int) {
        ksize = _len;
        kernel = (float*)_kernel;
    }

    int operator()(const uchar** _src, uchar* _dst, int& count,
                   int width) const {
        MEGDNN_MARK_USED_VAR(count);
        int ksize2 = (ksize) / 2;
        const float* ky = (float*)kernel + ksize2;
        int i = 0;
        const float** src = (const float**)_src;
        const float *S0 = src[-1], *S1 = src[0], *S2 = src[1];
        float* dst = (float*)_dst;
        {
            float32x4_t k0 = vdupq_n_f32(ky[0]), k1 = vdupq_n_f32(ky[1]);
            for (; i <= width - 8; i += 8) {
                float32x4_t s0, s1, x0, x1;
                s0 = vld1q_f32(S1 + i);
                s1 = vld1q_f32(S1 + i + 4);
                s0 = vmulq_f32(s0, k0);
                s1 = vmulq_f32(s1, k0);
                x0 = vaddq_f32(vld1q_f32(S0 + i), vld1q_f32(S2 + i));
                x1 = vaddq_f32(vld1q_f32(S0 + i + 4), vld1q_f32(S2 + i + 4));
                s0 = vmlaq_f32(s0, x0, k1);
                s1 = vmlaq_f32(s1, x1, k1);
                vst1q_f32(dst + i, s0);
                vst1q_f32(dst + i + 4, s1);
            }
        }

        return i;
    }

    float* kernel;
    int ksize;
};

/*!
 * \brief get the column filter
 * \tparam FT The inner buffer type, used to store the product of src and filter
 * \tparam DT The dst image type.
 */
template <typename FT, typename DT, bool SYMM>
static BaseColumnFilter* getLinearColumnFilter(Mat<FT>& kernel, int bits) {
    MEGDNN_MARK_USED_VAR(bits);
    int ksize = kernel.cols();
    int anchor = ksize / 2;
    uchar* kernel_str = static_cast<uchar*>(kernel.raw_ptr());
    if (SYMM && ksize == 3) {
        if (std::is_same<DT, uchar>::value && std::is_same<FT, int>::value)
            return new SymmColumnSmallFilter<FixedPtCastEx<FT, DT>,
                                             SymmColumnVec_32s8u>(
                    kernel, anchor, FixedPtCastEx<FT, DT>(bits),
                    SymmColumnVec_32s8u(kernel_str, ksize, bits));
        if (std::is_same<DT, float>::value && std::is_same<FT, float>::value)
            return new SymmColumnSmallFilter<FixedPtCastEx<FT, DT>,
                                             SymmColumnSmallVec_32f>(
                    kernel, anchor, FixedPtCastEx<FT, DT>(0),
                    SymmColumnSmallVec_32f(kernel_str, ksize, 0));
    }

    if (std::is_same<DT, uchar>::value && std::is_same<FT, int>::value)
        return new ColumnFilter<FixedPtCastEx<FT, DT>, ColumnNoVec>(
                kernel, anchor, FixedPtCastEx<FT, DT>(bits),
                ColumnNoVec(kernel_str, ksize, bits));

    if (std::is_same<DT, float>::value && std::is_same<FT, float>::value)
        return new ColumnFilter<FixedPtCastEx<FT, DT>, ColumnVec_32f>(
                kernel, anchor, FixedPtCastEx<FT, DT>(),
                ColumnVec_32f(kernel_str, ksize, 0));

    MegCVException(
            "Unsupported combination of source format and buffer format\n");
}

/*!
 * \brief get the row filter
 * \tparam ST The src image type
 * \tparam FT The inner buffer type, used to store the product of src and filter
 */
template <typename ST, typename FT, bool SYMM>
static BaseRowFilter* getLinearRowFilter(Mat<FT>& kernel) {
    int ksize = kernel.cols();
    int anchor = ksize / 2;

    uchar* kernel_str = static_cast<uchar*>(kernel.raw_ptr());

    if (SYMM && (ksize == 1 || ksize == 3 || ksize == 5)) {
        if (std::is_same<ST, uchar>::value && std::is_same<FT, int>::value)
            return new SymmRowSmallFilter<ST, FT, SymmRowSmallVec_8u32s>(
                    kernel, anchor, SymmRowSmallVec_8u32s(kernel_str, ksize));
        if (std::is_same<ST, float>::value && std::is_same<FT, float>::value)
            return new SymmRowSmallFilter<ST, FT, SymmRowSmallVec_32f>(
                    kernel, anchor, SymmRowSmallVec_32f(kernel_str, ksize));
    }

    if (std::is_same<ST, uchar>::value && std::is_same<FT, int>::value)
        return new RowFilter<ST, FT, RowNoVec>(kernel, anchor,
                                               RowNoVec(kernel_str, ksize));

    if (std::is_same<ST, float>::value && std::is_same<FT, float>::value)
        return new RowFilter<ST, FT, RowVec_32f>(kernel, anchor,
                                                 RowVec_32f(kernel_str, ksize));

    MegCVException(
            "Unsupported combination of source format and buffer format\n");
}

}  // namespace sep_filter
}  // namespace x86
}  // namespace megdnn

// vim: syntax=cpp.doxygen
