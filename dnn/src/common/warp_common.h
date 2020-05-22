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
 * \file dnn/src/common/warp_common.h
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
#include "megdnn/dtype.h"
#include "src/common/cv/common.h"
#include "src/common/cv/helper.h"
#include "src/common/cv/interp_helper.h"
#include "src/common/rounding_converter.cuh"
#include "src/common/utils.h"

#include "include/megdnn/oprs.h"
#include "midout.h"

#if MEGDNN_X86
#include <xmmintrin.h>
#elif MEGDNN_AARCH64 || MEGDNN_ARMV7
#include "src/arm_common/simd_macro/marm_neon.h"
#endif

MIDOUT_DECL(megdnn_warp)
MIDOUT_DECL(remapBilinear_bmode)
MIDOUT_DECL(remapBilinear_ch)

namespace megdnn {
namespace warp {

bool is_cv_available(const TensorLayout& src, const TensorLayout& mat,
                     const TensorLayout& dst,
                     param::WarpAffine::InterpolationMode imode,
                     param::WarpAffine::Format format);

bool is_dnn_available(const TensorLayout&, const TensorLayout&,
                      const TensorLayout&,
                      param::WarpAffine::InterpolationMode imode,
                      param::WarpAffine::Format format);

using namespace megcv;
using IMode = InterpolationMode;
using BMode = BorderMode;
using InterpTable = InterpolationTable<>;
constexpr int INTER_REMAP_COEF_BITS = InterpTable::INTER_REMAP_COEF_BITS;
constexpr int INTER_BITS = InterpTable::INTER_BITS;
constexpr int INTER_TAB_SIZE = InterpTable::INTER_TAB_SIZE;
constexpr int INTER_TAB_SIZE2 = InterpTable::INTER_TAB_SIZE2;
constexpr int INTER_REMAP_COEF_SCALE = InterpTable::INTER_REMAP_COEF_SCALE;

template <typename T, size_t CH>
struct RemapVec {
    int operator()(const Mat<T>&, void*, const short*, const ushort*,
                   const void*, int) const {
        return 0;
    }
};

#if MEGDNN_X86

template <size_t CH>
struct RemapVec<uchar, CH> {
    int operator()(const Mat8u& _src, void* _dst, const short* XY,
                   const ushort* FXY, const void* _wtab, int width) const {
        int x = 0, sstep = (int)_src.step();

        if ((CH != 1 && CH != 3) || sstep > 0x8000)
            return 0;

        const uchar *S0 = _src.ptr(), *S1 = _src.ptr(1);
        const short* wtab = CH == 1 ? (const short*)_wtab
                                    : InterpTable::get_linear_ic4_table();
        uchar* D = (uchar*)_dst;
        __m128i delta = _mm_set1_epi32(INTER_REMAP_COEF_SCALE / 2);
        __m128i xy2ofs = _mm_set1_epi32(CH + (sstep << 16));
        __m128i z = _mm_setzero_si128();
        alignas(16) int iofs0[4];
        alignas(16) int iofs1[4];

        if (CH == 1) {
            for (; x <= width - 8; x += 8) {
                __m128i xy0 = _mm_loadu_si128((const __m128i*)(XY + x * 2));
                __m128i xy1 = _mm_loadu_si128((const __m128i*)(XY + x * 2 + 8));
                __m128i v0, v1, v2, v3, a0, a1, b0, b1;
                unsigned i0, i1;

                xy0 = _mm_madd_epi16(xy0, xy2ofs);
                xy1 = _mm_madd_epi16(xy1, xy2ofs);
                _mm_store_si128((__m128i*)iofs0, xy0);
                _mm_store_si128((__m128i*)iofs1, xy1);

                i0 = *(ushort*)(S0 + iofs0[0]) +
                     (*(ushort*)(S0 + iofs0[1]) << 16);
                i1 = *(ushort*)(S0 + iofs0[2]) +
                     (*(ushort*)(S0 + iofs0[3]) << 16);
                v0 = _mm_unpacklo_epi32(_mm_cvtsi32_si128(i0),
                                        _mm_cvtsi32_si128(i1));
                i0 = *(ushort*)(S1 + iofs0[0]) +
                     (*(ushort*)(S1 + iofs0[1]) << 16);
                i1 = *(ushort*)(S1 + iofs0[2]) +
                     (*(ushort*)(S1 + iofs0[3]) << 16);
                v1 = _mm_unpacklo_epi32(_mm_cvtsi32_si128(i0),
                                        _mm_cvtsi32_si128(i1));
                v0 = _mm_unpacklo_epi8(v0, z);
                v1 = _mm_unpacklo_epi8(v1, z);

                a0 = _mm_unpacklo_epi32(
                        _mm_loadl_epi64((__m128i*)(wtab + FXY[x] * 4)),
                        _mm_loadl_epi64((__m128i*)(wtab + FXY[x + 1] * 4)));
                a1 = _mm_unpacklo_epi32(
                        _mm_loadl_epi64((__m128i*)(wtab + FXY[x + 2] * 4)),
                        _mm_loadl_epi64((__m128i*)(wtab + FXY[x + 3] * 4)));
                b0 = _mm_unpacklo_epi64(a0, a1);
                b1 = _mm_unpackhi_epi64(a0, a1);
                v0 = _mm_madd_epi16(v0, b0);
                v1 = _mm_madd_epi16(v1, b1);
                v0 = _mm_add_epi32(_mm_add_epi32(v0, v1), delta);

                i0 = *(ushort*)(S0 + iofs1[0]) +
                     (*(ushort*)(S0 + iofs1[1]) << 16);
                i1 = *(ushort*)(S0 + iofs1[2]) +
                     (*(ushort*)(S0 + iofs1[3]) << 16);
                v2 = _mm_unpacklo_epi32(_mm_cvtsi32_si128(i0),
                                        _mm_cvtsi32_si128(i1));
                i0 = *(ushort*)(S1 + iofs1[0]) +
                     (*(ushort*)(S1 + iofs1[1]) << 16);
                i1 = *(ushort*)(S1 + iofs1[2]) +
                     (*(ushort*)(S1 + iofs1[3]) << 16);
                v3 = _mm_unpacklo_epi32(_mm_cvtsi32_si128(i0),
                                        _mm_cvtsi32_si128(i1));
                v2 = _mm_unpacklo_epi8(v2, z);
                v3 = _mm_unpacklo_epi8(v3, z);

                a0 = _mm_unpacklo_epi32(
                        _mm_loadl_epi64((__m128i*)(wtab + FXY[x + 4] * 4)),
                        _mm_loadl_epi64((__m128i*)(wtab + FXY[x + 5] * 4)));
                a1 = _mm_unpacklo_epi32(
                        _mm_loadl_epi64((__m128i*)(wtab + FXY[x + 6] * 4)),
                        _mm_loadl_epi64((__m128i*)(wtab + FXY[x + 7] * 4)));
                b0 = _mm_unpacklo_epi64(a0, a1);
                b1 = _mm_unpackhi_epi64(a0, a1);
                v2 = _mm_madd_epi16(v2, b0);
                v3 = _mm_madd_epi16(v3, b1);
                v2 = _mm_add_epi32(_mm_add_epi32(v2, v3), delta);

                v0 = _mm_srai_epi32(v0, INTER_REMAP_COEF_BITS);
                v2 = _mm_srai_epi32(v2, INTER_REMAP_COEF_BITS);
                v0 = _mm_packus_epi16(_mm_packs_epi32(v0, v2), z);
                _mm_storel_epi64((__m128i*)(D + x), v0);
            }
        } else if (CH == 3) {
            for (; x <= width - 5; x += 4, D += 12) {
                __m128i xy0 = _mm_loadu_si128((const __m128i*)(XY + x * 2));
                __m128i u0, v0, u1, v1;

                xy0 = _mm_madd_epi16(xy0, xy2ofs);
                _mm_store_si128((__m128i*)iofs0, xy0);
                const __m128i *w0, *w1;
                w0 = (const __m128i*)(wtab + FXY[x] * 16);
                w1 = (const __m128i*)(wtab + FXY[x + 1] * 16);

                u0 = _mm_unpacklo_epi8(
                        _mm_cvtsi32_si128(*(int*)(S0 + iofs0[0])),
                        _mm_cvtsi32_si128(*(int*)(S0 + iofs0[0] + 3)));
                v0 = _mm_unpacklo_epi8(
                        _mm_cvtsi32_si128(*(int*)(S1 + iofs0[0])),
                        _mm_cvtsi32_si128(*(int*)(S1 + iofs0[0] + 3)));
                u1 = _mm_unpacklo_epi8(
                        _mm_cvtsi32_si128(*(int*)(S0 + iofs0[1])),
                        _mm_cvtsi32_si128(*(int*)(S0 + iofs0[1] + 3)));
                v1 = _mm_unpacklo_epi8(
                        _mm_cvtsi32_si128(*(int*)(S1 + iofs0[1])),
                        _mm_cvtsi32_si128(*(int*)(S1 + iofs0[1] + 3)));
                u0 = _mm_unpacklo_epi8(u0, z);
                v0 = _mm_unpacklo_epi8(v0, z);
                u1 = _mm_unpacklo_epi8(u1, z);
                v1 = _mm_unpacklo_epi8(v1, z);
                u0 = _mm_add_epi32(_mm_madd_epi16(u0, w0[0]),
                                   _mm_madd_epi16(v0, w0[1]));
                u1 = _mm_add_epi32(_mm_madd_epi16(u1, w1[0]),
                                   _mm_madd_epi16(v1, w1[1]));
                u0 = _mm_srai_epi32(_mm_add_epi32(u0, delta),
                                    INTER_REMAP_COEF_BITS);
                u1 = _mm_srai_epi32(_mm_add_epi32(u1, delta),
                                    INTER_REMAP_COEF_BITS);
                u0 = _mm_slli_si128(u0, 4);
                u0 = _mm_packs_epi32(u0, u1);
                u0 = _mm_packus_epi16(u0, u0);
                _mm_storel_epi64((__m128i*)D, _mm_srli_si128(u0, 1));

                w0 = (const __m128i*)(wtab + FXY[x + 2] * 16);
                w1 = (const __m128i*)(wtab + FXY[x + 3] * 16);

                u0 = _mm_unpacklo_epi8(
                        _mm_cvtsi32_si128(*(int*)(S0 + iofs0[2])),
                        _mm_cvtsi32_si128(*(int*)(S0 + iofs0[2] + 3)));
                v0 = _mm_unpacklo_epi8(
                        _mm_cvtsi32_si128(*(int*)(S1 + iofs0[2])),
                        _mm_cvtsi32_si128(*(int*)(S1 + iofs0[2] + 3)));
                u1 = _mm_unpacklo_epi8(
                        _mm_cvtsi32_si128(*(int*)(S0 + iofs0[3])),
                        _mm_cvtsi32_si128(*(int*)(S0 + iofs0[3] + 3)));
                v1 = _mm_unpacklo_epi8(
                        _mm_cvtsi32_si128(*(int*)(S1 + iofs0[3])),
                        _mm_cvtsi32_si128(*(int*)(S1 + iofs0[3] + 3)));
                u0 = _mm_unpacklo_epi8(u0, z);
                v0 = _mm_unpacklo_epi8(v0, z);
                u1 = _mm_unpacklo_epi8(u1, z);
                v1 = _mm_unpacklo_epi8(v1, z);
                u0 = _mm_add_epi32(_mm_madd_epi16(u0, w0[0]),
                                   _mm_madd_epi16(v0, w0[1]));
                u1 = _mm_add_epi32(_mm_madd_epi16(u1, w1[0]),
                                   _mm_madd_epi16(v1, w1[1]));
                u0 = _mm_srai_epi32(_mm_add_epi32(u0, delta),
                                    INTER_REMAP_COEF_BITS);
                u1 = _mm_srai_epi32(_mm_add_epi32(u1, delta),
                                    INTER_REMAP_COEF_BITS);
                u0 = _mm_slli_si128(u0, 4);
                u0 = _mm_packs_epi32(u0, u1);
                u0 = _mm_packus_epi16(u0, u0);
                _mm_storel_epi64((__m128i*)(D + 6), _mm_srli_si128(u0, 1));
            }
        }

        return x;
    }
};
#endif

template <typename T, BorderMode bmode>
using RemapNNFunc = void (*)(const Mat<T>& _src, Mat<T>& _dst,
                             const Mat<short>& _xy, const T* bvalue);
template <typename T, BorderMode bmode>
using RemapFunc = void (*)(const Mat<T>& _src, Mat<T>& _dst,
                           const Mat<short>& _xy, const Mat<ushort>& _fxy,
                           const void* _wtab, const T* bvalue);

template <typename T, BorderMode bmode, size_t CH>
static void remapNearest(const Mat<T>& _src, Mat<T>& _dst,
                         const Mat<short>& _xy, const T* bvalue) {
    const T* S0 = _src.ptr();
    size_t sstep = _src.step();
    int dx, dy;
    int width1 = _src.width(), height1 = _src.height();
    int swidth = _src.width(), sheight = _src.height();
    int dwidth = _dst.width(), dheight = _dst.height();
    if (_dst.is_continuous() && _xy.is_continuous()) {
        dwidth *= dheight;
        dheight = 1;
    }
    for (dy = 0; dy < dheight; dy++) {
        T* D = _dst.ptr(dy);
        const short* XY = _xy.ptr(dy);
        if (CH == 1) {
            for (dx = 0; dx < dwidth; dx++) {
                int sx = XY[dx * 2], sy = XY[dx * 2 + 1];
                if ((unsigned)sx < (unsigned)width1 &&
                    (unsigned)sy < (unsigned)height1) {
                    D[dx] = S0[sy * sstep + sx];
                } else {
                    if (bmode == BMode::BORDER_REPLICATE) {
                        sx = saturate(sx, 0, swidth);
                        sy = saturate(sy, 0, sheight);
                        D[dx] = S0[sy * sstep + sx];
                    } else if (bmode == BMode::BORDER_CONSTANT)
                        D[dx] = bvalue[0];
                    else if (bmode != BMode::BORDER_TRANSPARENT) {
                        sx = border_interpolate<bmode>(sx, swidth);
                        sy = border_interpolate<bmode>(sy, sheight);
                        D[dx] = S0[sy * sstep + sx];
                    }
                }
            }
        } else {
            for (dx = 0; dx < dwidth; dx++, D += CH) {
                int sx = XY[dx * 2], sy = XY[dx * 2 + 1];
                const T* S;
                if ((unsigned)sx < (unsigned)width1 &&
                    (unsigned)sy < (unsigned)height1) {
                    S = S0 + sy * sstep + sx * CH;
                    for (size_t i = 0; i < CH; i++) {
                        D[i] = S[i];
                    }
                } else if (bmode != BMode::BORDER_TRANSPARENT) {
                    if (bmode == BMode::BORDER_REPLICATE) {
                        sx = saturate(sx, 0, swidth);
                        sy = saturate(sy, 0, sheight);
                        S = S0 + sy * sstep + sx * CH;
                    } else if (bmode == BMode::BORDER_CONSTANT)
                        S = bvalue;
                    else {
                        sx = border_interpolate<bmode>(sx, swidth);
                        sy = border_interpolate<bmode>(sy, sheight);
                        S = S0 + sy * sstep + sx * CH;
                    }
                    for (size_t i = 0; i < CH; i++) {
                        D[i] = S[i];
                    }
                }
            }
        }
    }
}

template <class CastOp, typename AT, int ONE, typename T, BorderMode bmode,
          size_t CH>
static void remapBicubic(const Mat<T>& _src, Mat<T>& _dst,
                         const Mat<short>& _xy, const Mat<ushort>& _fxy,
                         const void* _wtab, const T* bvalue) {
    typedef typename CastOp::type1 WT;
    const AT* wtab = (const AT*)_wtab;
    const T* S0 = _src.ptr();
    size_t sstep = _src.step();
    int dx, dy;
    CastOp castOp;
    int swidth = _src.width(), sheight = _src.height();
    int dwidth = _dst.width(), dheight = _dst.height();
    unsigned width1 = std::max(swidth - 3, 0),
             height1 = std::max(sheight - 3, 0);
    if (_dst.is_continuous() && _xy.is_continuous() && _fxy.is_continuous()) {
        dwidth *= dheight;
        dheight = 1;
    }
    for (dy = 0; dy < dheight; dy++) {
        T* D = _dst.ptr(dy);
        const short* XY = _xy.ptr(dy);
        const ushort* FXY = _fxy.ptr(dy);
        for (dx = 0; dx < dwidth; dx++, D += CH) {
            int sx = XY[dx * 2] - 1, sy = XY[dx * 2 + 1] - 1;
            const AT* w = wtab + FXY[dx] * 16;
            size_t i, k;
            if ((unsigned)sx < width1 && (unsigned)sy < height1) {
                const T* S = S0 + sy * sstep + sx * CH;
                for (k = 0; k < CH; k++) {
                    WT sum = S[0] * w[0] + S[CH] * w[1] + S[CH * 2] * w[2] +
                             S[CH * 3] * w[3];
                    S += sstep;
                    sum += S[0] * w[4] + S[CH] * w[5] + S[CH * 2] * w[6] +
                           S[CH * 3] * w[7];
                    S += sstep;
                    sum += S[0] * w[8] + S[CH] * w[9] + S[CH * 2] * w[10] +
                           S[CH * 3] * w[11];
                    S += sstep;
                    sum += S[0] * w[12] + S[CH] * w[13] + S[CH * 2] * w[14] +
                           S[CH * 3] * w[15];
                    S += 1 - sstep * 3;
                    D[k] = castOp(sum);
                }
            } else {
                int x[4], y[4];
                if (bmode == BMode::BORDER_TRANSPARENT &&
                    ((unsigned)(sx + 1) >= (unsigned)swidth ||
                     (unsigned)(sy + 1) >= (unsigned)sheight))
                    continue;
                if (bmode == BMode::BORDER_CONSTANT &&
                    (sx >= swidth || sx + 4 <= 0 || sy >= sheight ||
                     sy + 4 <= 0)) {
                    for (size_t i = 0; i < CH; i++) {
                        D[i] = bvalue[i];
                    }
                    continue;
                }
                for (i = 0; i < 4; i++) {
                    x[i] = border_interpolate<bmode>(sx + i, swidth) * CH;
                    y[i] = border_interpolate<bmode>(sy + i, sheight);
                }
                for (k = 0; k < CH; k++, S0++, w -= 16) {
                    WT cv = bvalue[k], sum = cv * ONE;
                    for (i = 0; i < 4; i++, w += 4) {
                        int yi = y[i];
                        const T* S = S0 + yi * sstep;
                        if (yi < 0)
                            continue;
                        if (x[0] >= 0)
                            sum += (S[x[0]] - cv) * w[0];
                        if (x[1] >= 0)
                            sum += (S[x[1]] - cv) * w[1];
                        if (x[2] >= 0)
                            sum += (S[x[2]] - cv) * w[2];
                        if (x[3] >= 0)
                            sum += (S[x[3]] - cv) * w[3];
                    }
                    D[k] = castOp(sum);
                }
                S0 -= CH;
            }
        }
    }
}

template <class CastOp, class VecOp, typename AT, typename T, BorderMode bmode,
          size_t CH>
static void remapBilinear(const Mat<T>& _src, Mat<T>& _dst,
                          const Mat<short>& _xy, const Mat<ushort>& _fxy,
                          const void* _wtab, const T* bvalue) {
    MIDOUT_BEGIN(remapBilinear_bmode, midout_iv(bmode)) {
    typedef typename CastOp::type1 WT;
    const AT* wtab = (const AT*)_wtab;
    const T* S0 = _src.ptr();
    size_t sstep = _src.step();
    int dx, dy;
    CastOp castOp;
    VecOp vecOp;
    int swidth = _src.width(), sheight = _src.height();
    int dwidth = _dst.width(), dheight = _dst.height();
    unsigned width1 = std::max(swidth - 1, 0),
             height1 = std::max(sheight - 1, 0);
    for (dy = 0; dy < dheight; dy++) {
        T* D = _dst.ptr(dy);
        const short* XY = _xy.ptr(dy);
        const ushort* FXY = _fxy.ptr(dy);
        int X0 = 0;
        bool prevInlier = false;

        for (dx = 0; dx <= dwidth; dx++) {
            bool curInlier =
                    dx < dwidth ? (unsigned)XY[dx * 2] < width1 &&
                                          (unsigned)XY[dx * 2 + 1] < height1
                                : !prevInlier;
            if (curInlier == prevInlier)
                continue;

            int X1 = dx;
            dx = X0;
            X0 = X1;
            prevInlier = curInlier;

            if (!curInlier) {
                int len = vecOp(_src, D, XY + dx * 2, FXY + dx, wtab, X1 - dx);
                D += len * CH;
                dx += len;

                if (CH == 1) {
                    MIDOUT_BEGIN(remapBilinear_bmode, 0, 1) {
                        for (; dx < X1; dx++, D++) {
                            int sx = XY[dx * 2], sy = XY[dx * 2 + 1];
                            const AT* w = wtab + FXY[dx] * 4;
                            const T* S = S0 + sy * sstep + sx;
                            *D = castOp(WT(S[0] * w[0] + S[1] * w[1] +
                                           S[sstep] * w[2] + S[sstep + 1] * w[3]));
                        }
                    }
                    MIDOUT_END();
                } else if (CH == 2) {
                    MIDOUT_BEGIN(remapBilinear_bmode, 0, 2) {
                        for (; dx < X1; dx++, D += 2) {
                            int sx = XY[dx * 2], sy = XY[dx * 2 + 1];
                            const AT* w = wtab + FXY[dx] * 4;
                            const T* S = S0 + sy * sstep + sx * 2;
                            WT t0 = S[0] * w[0] + S[2] * w[1] + S[sstep] * w[2] +
                                    S[sstep + 2] * w[3];
                            WT t1 = S[1] * w[0] + S[3] * w[1] +
                                    S[sstep + 1] * w[2] + S[sstep + 3] * w[3];
                            D[0] = castOp(t0);
                            D[1] = castOp(t1);
                        }
                    }
                    MIDOUT_END();
                } else if (CH == 3)
                    MIDOUT_BEGIN(remapBilinear_bmode, 0, 3) {
                        for (; dx < X1; dx++, D += 3) {
                            int sx = XY[dx * 2], sy = XY[dx * 2 + 1];
                            const AT* w = wtab + FXY[dx] * 4;
                            const T* S = S0 + sy * sstep + sx * 3;
                            WT t0 = S[0] * w[0] + S[3] * w[1] + S[sstep] * w[2] +
                                    S[sstep + 3] * w[3];
                            WT t1 = S[1] * w[0] + S[4] * w[1] +
                                    S[sstep + 1] * w[2] + S[sstep + 4] * w[3];
                            WT t2 = S[2] * w[0] + S[5] * w[1] +
                                    S[sstep + 2] * w[2] + S[sstep + 5] * w[3];
                            D[0] = castOp(t0);
                            D[1] = castOp(t1);
                            D[2] = castOp(t2);
                        }
                    }
                    MIDOUT_END();
                else
                    megdnn_throw("nr. of channels must be 1/2/3.");

            } else {
                if (bmode == BMode::BORDER_TRANSPARENT && CH != 3) {
                    megdnn_throw(
                            "unsupported Linear InterpolationMode"
                            " with BORDER_TRANSPARENT and channel size 1");
                    continue;
                }
                if (CH == 1) {
                    MIDOUT_BEGIN(remapBilinear_bmode, 1, 1) {
                        for (; dx < X1; dx++, D++) {
                            int sx = XY[dx * 2], sy = XY[dx * 2 + 1];
                            if (bmode == BMode::BORDER_CONSTANT &&
                                (sx >= swidth || sx + 1 < 0 || sy >= sheight ||
                                 sy + 1 < 0)) {
                                D[0] = bvalue[0];
                            } else {
                                int sx0, sx1, sy0, sy1;
                                T v0, v1, v2, v3;
                                const AT* w = wtab + FXY[dx] * 4;
                                if (bmode == BMode::BORDER_REPLICATE) {
                                    sx0 = saturate(sx, 0, swidth);
                                    sx1 = saturate(sx + 1, 0, swidth);
                                    sy0 = saturate(sy, 0, sheight);
                                    sy1 = saturate(sy + 1, 0, sheight);
                                    v0 = S0[sy0 * sstep + sx0];
                                    v1 = S0[sy0 * sstep + sx1];
                                    v2 = S0[sy1 * sstep + sx0];
                                    v3 = S0[sy1 * sstep + sx1];
                                } else {
                                    sx0 = border_interpolate<bmode>(sx, swidth);
                                    sx1 = border_interpolate<bmode>(sx + 1, swidth);
                                    sy0 = border_interpolate<bmode>(sy, sheight);
                                    sy1 = border_interpolate<bmode>(sy + 1,
                                                                    sheight);
                                    v0 = sx0 >= 0 && sy0 >= 0
                                                 ? S0[sy0 * sstep + sx0]
                                                 : bvalue[0];
                                    v1 = sx1 >= 0 && sy0 >= 0
                                                 ? S0[sy0 * sstep + sx1]
                                                 : bvalue[0];
                                    v2 = sx0 >= 0 && sy1 >= 0
                                                 ? S0[sy1 * sstep + sx0]
                                                 : bvalue[0];
                                    v3 = sx1 >= 0 && sy1 >= 0
                                                 ? S0[sy1 * sstep + sx1]
                                                 : bvalue[0];
                                }
                                D[0] = castOp(WT(v0 * w[0] + v1 * w[1] + v2 * w[2] +
                                                 v3 * w[3]));
                            }
                        }
                    }
                    MIDOUT_END();
                } else {
                    for (; dx < X1; dx++, D += CH) {
                        int sx = XY[dx * 2], sy = XY[dx * 2 + 1];
                        if (bmode == BMode::BORDER_CONSTANT &&
                            (sx >= swidth || sx + 1 < 0 || sy >= sheight ||
                             sy + 1 < 0)) {
                            for (size_t k = 0; k < CH; k++)
                                D[k] = bvalue[k];
                        } else {
                            int sx0, sx1, sy0, sy1;
                            const T *v0, *v1, *v2, *v3;
                            const AT* w = wtab + FXY[dx] * 4;
                            if (bmode == BMode::BORDER_REPLICATE) {
                                sx0 = saturate(sx, 0, swidth);
                                sx1 = saturate(sx + 1, 0, swidth);
                                sy0 = saturate(sy, 0, sheight);
                                sy1 = saturate(sy + 1, 0, sheight);
                                v0 = S0 + sy0 * sstep + sx0 * CH;
                                v1 = S0 + sy0 * sstep + sx1 * CH;
                                v2 = S0 + sy1 * sstep + sx0 * CH;
                                v3 = S0 + sy1 * sstep + sx1 * CH;
                            } else if (bmode == BMode::BORDER_TRANSPARENT &&
                                       ((unsigned)sx >=
                                                (unsigned)(swidth - 1) ||
                                        (unsigned)sy >=
                                                (unsigned)(sheight - 1)))
                                continue;
                            else {
                                sx0 = border_interpolate<bmode>(sx, swidth);
                                sx1 = border_interpolate<bmode>(sx + 1, swidth);
                                sy0 = border_interpolate<bmode>(sy, sheight);
                                sy1 = border_interpolate<bmode>(sy + 1,
                                                                sheight);
                                v0 = sx0 >= 0 && sy0 >= 0
                                             ? S0 + sy0 * sstep + sx0 * CH
                                             : &bvalue[0];
                                v1 = sx1 >= 0 && sy0 >= 0
                                             ? S0 + sy0 * sstep + sx1 * CH
                                             : &bvalue[0];
                                v2 = sx0 >= 0 && sy1 >= 0
                                             ? S0 + sy1 * sstep + sx0 * CH
                                             : &bvalue[0];
                                v3 = sx1 >= 0 && sy1 >= 0
                                             ? S0 + sy1 * sstep + sx1 * CH
                                             : &bvalue[0];
                            }

                            for (size_t k = 0; k < CH; k++) {
                                D[k] = castOp(WT(v0[k] * w[0] + v1[k] * w[1] +
                                                 v2[k] * w[2] + v3[k] * w[3]));
                            }
                        }
                    }
                }
            }
        }
    }
    }
    MIDOUT_END();
}

template <class CastOp, typename AT, int ONE, typename T, BorderMode bmode,
          size_t CH>
static void remapLanczos4(const Mat<T>& _src, Mat<T>& _dst,
                          const Mat<short>& _xy, const Mat<ushort>& _fxy,
                          const void* _wtab, const T* bvalue) {
    typedef typename CastOp::type1 WT;
    const AT* wtab = (const AT*)_wtab;
    const T* S0 = _src.ptr();
    size_t sstep = _src.step();
    int dx, dy;
    CastOp castOp;
    int swidth = _src.width(), sheight = _src.height();
    int dwidth = _dst.width(), dheight = _dst.height();
    unsigned width1 = std::max(swidth - 7, 0),
             height1 = std::max(sheight - 7, 0);
    if (_dst.is_continuous() && _xy.is_continuous() && _fxy.is_continuous()) {
        dwidth *= dheight;
        dheight = 1;
    }
    for (dy = 0; dy < dheight; dy++) {
        T* D = _dst.ptr(dy);
        const short* XY = _xy.ptr(dy);
        const ushort* FXY = _fxy.ptr(dy);
        for (dx = 0; dx < dwidth; dx++, D += CH) {
            int sx = XY[dx * 2] - 3, sy = XY[dx * 2 + 1] - 3;
            const AT* w = wtab + FXY[dx] * 64;
            const T* S = S0 + sy * sstep + sx * CH;
            size_t i, k;
            if ((unsigned)sx < width1 && (unsigned)sy < height1) {
                for (k = 0; k < CH; k++) {
                    WT sum = 0;
                    for (int r = 0; r < 8; r++, S += sstep, w += 8)
                        sum += S[0] * w[0] + S[CH] * w[1] + S[CH * 2] * w[2] +
                               S[CH * 3] * w[3] + S[CH * 4] * w[4] +
                               S[CH * 5] * w[5] + S[CH * 6] * w[6] +
                               S[CH * 7] * w[7];
                    w -= 64;
                    S -= sstep * 8 - 1;
                    D[k] = castOp(sum);
                }
            } else {
                int x[8], y[8];
                if (bmode == BMode::BORDER_TRANSPARENT &&
                    ((unsigned)(sx + 3) >= (unsigned)swidth ||
                     (unsigned)(sy + 3) >= (unsigned)sheight))
                    continue;
                if (bmode == BMode::BORDER_CONSTANT &&
                    (sx >= swidth || sx + 8 <= 0 || sy >= sheight ||
                     sy + 8 <= 0)) {
                    for (size_t i = 0; i < CH; i++) {
                        D[i] = bvalue[i];
                    }
                    continue;
                }
                for (i = 0; i < 8; i++) {
                    x[i] = border_interpolate<bmode>(sx + i, swidth) * CH;
                    y[i] = border_interpolate<bmode>(sy + i, sheight);
                }
                for (k = 0; k < CH; k++, S0++, w -= 64) {
                    WT cv = bvalue[k], sum = cv * ONE;
                    for (i = 0; i < 8; i++, w += 8) {
                        int yi = y[i];
                        const T* S1 = S0 + yi * sstep;
                        if (yi < 0)
                            continue;
                        if (x[0] >= 0)
                            sum += (S1[x[0]] - cv) * w[0];
                        if (x[1] >= 0)
                            sum += (S1[x[1]] - cv) * w[1];
                        if (x[2] >= 0)
                            sum += (S1[x[2]] - cv) * w[2];
                        if (x[3] >= 0)
                            sum += (S1[x[3]] - cv) * w[3];
                        if (x[4] >= 0)
                            sum += (S1[x[4]] - cv) * w[4];
                        if (x[5] >= 0)
                            sum += (S1[x[5]] - cv) * w[5];
                        if (x[6] >= 0)
                            sum += (S1[x[6]] - cv) * w[6];
                        if (x[7] >= 0)
                            sum += (S1[x[7]] - cv) * w[7];
                    }
                    D[k] = castOp(sum);
                }
                S0 -= CH;
            }
        }
    }
}

template <typename T, InterpolationMode imode, BorderMode bmode, size_t CH,
          typename RemapVec>
struct RemapFuncHolder;

template <InterpolationMode imode, BorderMode bmode, size_t CH,
          typename RemapVec>
struct RemapFuncHolder<uchar, imode, bmode, CH, RemapVec> {
    static void get_funcs(RemapNNFunc<uchar, bmode>& nnfunc,
                          RemapFunc<uchar, bmode>& ifunc) {
        switch (imode) {
            case IMode::INTER_NEAREST:
                MIDOUT_BEGIN(megdnn_warp, midout_iv(0)) {
                    nnfunc = remapNearest<uchar, bmode, CH>;
                }
                MIDOUT_END();
                break;
            case IMode::INTER_LINEAR:
                MIDOUT_BEGIN(megdnn_warp, midout_iv(1)) {
                    ifunc = remapBilinear<
                            FixedPtCast<int, uchar, INTER_REMAP_COEF_BITS>,
                            RemapVec, short, uchar, bmode, CH>;
                }
                MIDOUT_END();
                break;
            case IMode::INTER_CUBIC:
                MIDOUT_BEGIN(megdnn_warp, midout_iv(2)) {
                    ifunc = remapBicubic<
                            FixedPtCast<int, uchar, INTER_REMAP_COEF_BITS>,
                            short, INTER_REMAP_COEF_SCALE, uchar, bmode, CH>;
                }
                MIDOUT_END();
                break;
            case IMode::INTER_LANCZOS4:
                MIDOUT_BEGIN(megdnn_warp, midout_iv(3)) {
                    ifunc = remapLanczos4<
                            FixedPtCast<int, uchar, INTER_REMAP_COEF_BITS>,
                            short, INTER_REMAP_COEF_SCALE, uchar, bmode, CH>;
                }
                MIDOUT_END();
                break;
            default:
                megdnn_throw(("unrecognized interpolation mode"));
        }
    }
};

template <InterpolationMode imode, BorderMode bmode, size_t CH,
          typename RemapVec>
struct RemapFuncHolder<float, imode, bmode, CH, RemapVec> {
    static void get_funcs(RemapNNFunc<float, bmode>& nnfunc,
                          RemapFunc<float, bmode>& ifunc) {
        switch (imode) {
            case IMode::INTER_NEAREST:
                MIDOUT_BEGIN(megdnn_warp, midout_iv(0)) {
                    nnfunc = remapNearest<float, bmode, CH>;
                }
                MIDOUT_END();
                break;
            case IMode::INTER_LINEAR:
                MIDOUT_BEGIN(megdnn_warp, midout_iv(1)) {
                    ifunc = remapBilinear<Cast<float, float>, RemapVec, float,
                                          float, bmode, CH>;
                }
                MIDOUT_END();
                break;
            case IMode::INTER_CUBIC:
                MIDOUT_BEGIN(megdnn_warp, midout_iv(2)) {
                    ifunc = remapBicubic<Cast<float, float>, float, 1, float,
                                         bmode, CH>;
                }
                MIDOUT_END();
                break;
            case IMode::INTER_LANCZOS4:
                MIDOUT_BEGIN(megdnn_warp, midout_iv(3)) {
                    ifunc = remapLanczos4<Cast<float, float>, float, 1, float,
                                          bmode, CH>;
                }
                MIDOUT_END();
                break;
            default:
                megdnn_throw(("unrecognized interpolation mode"));
        }
    }
};

template <typename T, InterpolationMode imode, BorderMode bmode, size_t CH,
          typename RemapVec>
#if MEGDNN_X86
MEGDNN_ATTRIBUTE_TARGET("sse3")
#endif
void remap(const Mat<T>& src, Mat<T>& dst, Mat<short>& map1, Mat<ushort>& map2,
           const T* bvalue) {
    RemapNNFunc<T, bmode> nnfunc = 0;
    RemapFunc<T, bmode> ifunc = 0;
    bool fixpt = std::is_same<T, uchar>::value;
    const void* ctab = 0;
    RemapFuncHolder<T, imode, bmode, CH, RemapVec>::get_funcs(nnfunc, ifunc);
    if (imode != IMode::INTER_NEAREST) {
        ctab = InterpTable::get_table(imode, fixpt);
    }
    {
        // remap invoker
        int x, y, x1, y1;
        const int buf_size = 1 << 14;
        int dstcols = dst.cols(), dstrows = dst.rows();
        int brows0 = std::min(128, dstrows);
        int bcols0 = std::min(buf_size / brows0, dstcols);
        brows0 = std::min(buf_size / bcols0, dstrows);
        Mat<short> _bufxy(brows0, bcols0, 2);
        Mat<ushort> _bufa(brows0, bcols0, 1);
        for (y = 0; y < dstrows; y += brows0)
            for (x = 0; x < dstcols; x += bcols0) {
                int brows = std::min(brows0, dstrows - y);
                int bcols = std::min(bcols0, dstcols - x);
                Mat<T> dpart(dst, y, brows, x, bcols);
                Mat<short> bufxy(_bufxy, 0, brows, 0, bcols);
                if (nnfunc) {
                    bufxy = Mat<short>(map1, y, brows, x, bcols);
                    nnfunc(src, dpart, bufxy, bvalue);
                    continue;
                }
                Mat<ushort> bufa(_bufa, 0, brows, 0, bcols);
                for (y1 = 0; y1 < brows; ++y1) {
                    ushort* A = bufa.ptr(y1);
                    bufxy = Mat<short>(map1, y, brows, x, bcols);
                    const ushort* sA = map2.ptr(y + y1) + x;
                    x1 = 0;
#if MEGDNN_X86
                    __m128i sA_data, d_data;
                    __m128i v_INTER_TAB_SIZE2 =
                            _mm_set1_epi16(INTER_TAB_SIZE2 - 1);

                    for (; x1 <= bcols - 8; x1 += 8) {
                        __m128i const* src = (__m128i const*)(sA + x1);
                        __m128i* dst = (__m128i*)(A + x1);

                        sA_data = _mm_loadu_si128(src);
                        d_data = _mm_and_si128(sA_data, v_INTER_TAB_SIZE2);
                        _mm_storeu_si128(dst, d_data);
                    }
#elif MEGDNN_AARCH64 || MEGDNN_ARMV7
                    uint16x8_t v_scale = vdupq_n_u16(INTER_TAB_SIZE2 - 1);
                    for (; x1 <= bcols - 8; x1 += 8)
                        vst1q_u16(A + x1,
                                  vandq_u16(vld1q_u16(sA + x1), v_scale));

#endif
                    for (; x1 < bcols; ++x1)
                        A[x1] = (ushort)(sA[x1] & (INTER_TAB_SIZE2 - 1));
                }
                ifunc(src, dpart, bufxy, bufa, ctab, bvalue);
            }
    }
}

#define DISPATCH_CHANNEL(_imode, _bmode, _ch, _cb)                         \
    switch (_ch) {                                                         \
        case 1: {                                                          \
            _cb(_imode, _bmode, 1);                                        \
            break;                                                         \
        }                                                                  \
        case 2: {                                                          \
            _cb(_imode, _bmode, 2);                                        \
            break;                                                         \
        }                                                                  \
        case 3: {                                                          \
            _cb(_imode, _bmode, 3);                                        \
            break;                                                         \
        }                                                                  \
        default: {                                                         \
            megdnn_assert(0, "unsupport channels: %zu, only supprt 1/2/3", \
                          _ch);                                            \
        }                                                                  \
    }

#define DISPATCH_BMODE(_imode, _bmode, _ch, _cb)                         \
    switch (_bmode) {                                                    \
        case BorderMode::REPLICATE: {                                    \
            DISPATCH_CHANNEL(_imode, BorderMode::REPLICATE, _ch, _cb);   \
            break;                                                       \
        }                                                                \
        case BorderMode::REFLECT: {                                      \
            DISPATCH_CHANNEL(_imode, BorderMode::REFLECT, _ch, _cb);     \
            break;                                                       \
        }                                                                \
        case BorderMode::REFLECT_101: {                                  \
            DISPATCH_CHANNEL(_imode, BorderMode::REFLECT_101, _ch, _cb); \
            break;                                                       \
        }                                                                \
        case BorderMode::WRAP: {                                         \
            DISPATCH_CHANNEL(_imode, BorderMode::WRAP, _ch, _cb);        \
            break;                                                       \
        }                                                                \
        case BorderMode::CONSTANT: {                                     \
            DISPATCH_CHANNEL(_imode, BorderMode::CONSTANT, _ch, _cb);    \
            break;                                                       \
        }                                                                \
        default: { megdnn_assert(0, "unsupport border mode for cv"); }   \
    }

#define DISPATCH_IMODE(_imode, _bmode, _ch, _cb)                              \
    switch (_imode) {                                                         \
        case InterpolationMode::NEAREST: {                                    \
            DISPATCH_BMODE(InterpolationMode::NEAREST, _bmode, _ch, _cb);     \
            break;                                                            \
        }                                                                     \
        case InterpolationMode::LINEAR: {                                     \
            DISPATCH_BMODE(InterpolationMode::LINEAR, _bmode, _ch, _cb);      \
            break;                                                            \
        }                                                                     \
        case InterpolationMode::AREA: {                                       \
            DISPATCH_BMODE(InterpolationMode::AREA, _bmode, _ch, _cb);        \
            break;                                                            \
        }                                                                     \
        case InterpolationMode::CUBIC: {                                      \
            DISPATCH_BMODE(InterpolationMode::CUBIC, _bmode, _ch, _cb);       \
            break;                                                            \
        }                                                                     \
        case InterpolationMode::LANCZOS4: {                                   \
            DISPATCH_BMODE(InterpolationMode::LANCZOS4, _bmode, _ch, _cb);    \
            break;                                                            \
        }                                                                     \
        default: { megdnn_assert(0, "unsupport interpolation mode for cv"); } \
    }

}  // namespace warp
}  // namespace megdnn

// vim: syntax=cpp.doxygen
