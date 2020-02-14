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
 * \file dnn/src/x86/warp_affine/warp_affine_cv.cpp
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



#include "src/x86/warp_affine/warp_affine_cv.h"
#include "src/x86/handle.h"

#include <cstring>
#include <mutex>
#include "src/common/cv/common.h"
#include "src/common/cv/helper.h"
#include "src/common/cv/interp_helper.h"
#include "src/common/utils.h"
#include "src/common/warp_common.h"

#include <pmmintrin.h>
#include <smmintrin.h>
#include <tmmintrin.h>
#include <xmmintrin.h>

using namespace megdnn;
using namespace x86;
using namespace megcv;
using namespace warp;

namespace {
constexpr size_t BLOCK_SZ = 64_z;
template <typename T, InterpolationMode imode, BorderMode bmode, size_t CH>
MEGDNN_ATTRIBUTE_TARGET("sse3")
void warp_affine_cv(const Mat<T>& src, Mat<T>& dst, const float* trans,
                    const float border_value, size_t task_id) {
    // no extra padding

    double M[6];
    rep(i, 6) M[i] = trans[i];
    T bvalue[3] = {(T)border_value, (T)border_value, (T)border_value};

    std::vector<int> _adelta(dst.cols() * 2);
    int *adelta = _adelta.data(), *bdelta = adelta + dst.cols();

    // clang 3.6 can not deduce that `std::max(10, (int)INTER_BITS)' is a
    // constant, which will cause compilation error in subsequent vshrq_n_s32.
    const int AB_BITS = 10 > INTER_BITS ? 10 : INTER_BITS;
    const int AB_SCALE = 1 << AB_BITS;
    size_t dstcols = dst.cols();

    for (size_t x = 0; x < dstcols; ++x) {
        adelta[x] = saturate_cast<int>(M[0] * x * AB_SCALE);
        bdelta[x] = saturate_cast<int>(M[3] * x * AB_SCALE);
    }
    size_t x1, y1, dstrows = dst.rows();
    size_t BLOCK_SZ_H = std::min(BLOCK_SZ / 2, dstrows);
    size_t BLOCK_SZ_W = std::min(BLOCK_SZ * BLOCK_SZ / BLOCK_SZ_H, dstcols);
    BLOCK_SZ_H = std::min(BLOCK_SZ * BLOCK_SZ / BLOCK_SZ_W, dstrows);

    size_t width_block_size = div_ceil<size_t>(dstcols, BLOCK_SZ_W);
    size_t y = (task_id / width_block_size) * BLOCK_SZ_H;
    size_t x = (task_id % width_block_size) * BLOCK_SZ_W;

    short XY[BLOCK_SZ * BLOCK_SZ * 2 + 16], A[BLOCK_SZ * BLOCK_SZ];
    int round_delta =
            (imode == IMode::INTER_NEAREST ? AB_SCALE / 2
                                           : AB_SCALE / INTER_TAB_SIZE / 2);
    size_t bw = std::min(BLOCK_SZ_W, dstcols - x);
    size_t bh = std::min(BLOCK_SZ_H, dstrows - y);
    Mat<short> _XY(bh, bw, 2, XY);
    Mat<T> dpart(dst, y, bh, x, bw);
    for (y1 = 0; y1 < bh; ++y1) {
        short* xy = XY + y1 * bw * 2;
        int X0 = saturate_cast<int>((M[1] * (y + y1) + M[2]) * AB_SCALE) +
                 round_delta;
        int Y0 = saturate_cast<int>((M[4] * (y + y1) + M[5]) * AB_SCALE) +
                 round_delta;

        if (imode == IMode::INTER_NEAREST) {
            x1 = 0;

            __m128i v_X0 = _mm_set1_epi32(X0);
            __m128i v_Y0 = _mm_set1_epi32(Y0);

            __m128i adelta_data;
            __m128i bdelta_data;

            __m128i v_X;
            __m128i v_Y;

            for (; x1 + 4 <= bw; x1++) {
                __m128i const* src1 = (__m128i const*)(adelta + x + x1);
                __m128i const* src2 = (__m128i const*)(bdelta + x + x1);

                adelta_data = _mm_lddqu_si128(src1);
                bdelta_data = _mm_lddqu_si128(src2);

                v_X = _mm_srai_epi32(_mm_add_epi32(v_X0, adelta_data), AB_BITS);
                v_Y = _mm_srai_epi32(_mm_add_epi32(v_Y0, bdelta_data), AB_BITS);

                int* x_data = (int*)(&v_X);
                int* y_data = (int*)(&v_Y);

                xy[x1 * 2] = saturate_cast<short>(x_data[0]);
                xy[x1 * 2 + 1] = saturate_cast<short>(y_data[0]);

                x1++;
                xy[x1 * 2] = saturate_cast<short>(x_data[1]);
                xy[x1 * 2 + 1] = saturate_cast<short>(y_data[1]);

                x1++;
                xy[x1 * 2] = saturate_cast<short>(x_data[2]);
                xy[x1 * 2 + 1] = saturate_cast<short>(y_data[2]);

                x1++;
                xy[x1 * 2] = saturate_cast<short>(x_data[3]);
                xy[x1 * 2 + 1] = saturate_cast<short>(y_data[3]);
            }
            for (; x1 < bw; x1++) {
                int X = (X0 + adelta[x + x1]) >> AB_BITS;
                int Y = (Y0 + bdelta[x + x1]) >> AB_BITS;
                xy[x1 * 2] = saturate_cast<short>(X);
                xy[x1 * 2 + 1] = saturate_cast<short>(Y);
            }
        } else {
            // if imode is not INTER_NEAREST
            short* alpha = A + y1 * bw;
            x1 = 0;

            __m128i fxy_mask = _mm_set1_epi32(INTER_TAB_SIZE - 1);
            __m128i XX = _mm_set1_epi32(X0), YY = _mm_set1_epi32(Y0);
            for (; x1 + 8 <= bw; x1 += 8) {
                __m128i tx0, tx1, ty0, ty1;
                tx0 = _mm_add_epi32(
                        _mm_loadu_si128((const __m128i*)(adelta + x + x1)), XX);
                ty0 = _mm_add_epi32(
                        _mm_loadu_si128((const __m128i*)(bdelta + x + x1)), YY);
                tx1 = _mm_add_epi32(
                        _mm_loadu_si128((const __m128i*)(adelta + x + x1 + 4)),
                        XX);
                ty1 = _mm_add_epi32(
                        _mm_loadu_si128((const __m128i*)(bdelta + x + x1 + 4)),
                        YY);

                tx0 = _mm_srai_epi32(tx0, AB_BITS - INTER_BITS);
                ty0 = _mm_srai_epi32(ty0, AB_BITS - INTER_BITS);
                tx1 = _mm_srai_epi32(tx1, AB_BITS - INTER_BITS);
                ty1 = _mm_srai_epi32(ty1, AB_BITS - INTER_BITS);

                __m128i fx_ = _mm_packs_epi32(_mm_and_si128(tx0, fxy_mask),
                                              _mm_and_si128(tx1, fxy_mask));
                __m128i fy_ = _mm_packs_epi32(_mm_and_si128(ty0, fxy_mask),
                                              _mm_and_si128(ty1, fxy_mask));
                tx0 = _mm_packs_epi32(_mm_srai_epi32(tx0, INTER_BITS),
                                      _mm_srai_epi32(tx1, INTER_BITS));
                ty0 = _mm_packs_epi32(_mm_srai_epi32(ty0, INTER_BITS),
                                      _mm_srai_epi32(ty1, INTER_BITS));
                fx_ = _mm_adds_epi16(fx_, _mm_slli_epi16(fy_, INTER_BITS));

                _mm_storeu_si128((__m128i*)(xy + x1 * 2),
                                 _mm_unpacklo_epi16(tx0, ty0));
                _mm_storeu_si128((__m128i*)(xy + x1 * 2 + 8),
                                 _mm_unpackhi_epi16(tx0, ty0));
                _mm_storeu_si128((__m128i*)(alpha + x1), fx_);
            }
            for (; x1 < bw; x1++) {
                int X = (X0 + adelta[x + x1]) >> (AB_BITS - INTER_BITS);
                int Y = (Y0 + bdelta[x + x1]) >> (AB_BITS - INTER_BITS);
                xy[x1 * 2] = saturate_cast<short>(X >> INTER_BITS);
                xy[x1 * 2 + 1] = saturate_cast<short>(Y >> INTER_BITS);
                alpha[x1] =
                        (short)((Y & (INTER_TAB_SIZE - 1)) * INTER_TAB_SIZE +
                                (X & (INTER_TAB_SIZE - 1)));
            }
        }
    }
    Mat<ushort> _matA(bh, bw, 1, (ushort*)(A));
    remap<T, imode, bmode, CH, RemapVec<T, CH>>(src, dpart, _XY, _matA, bvalue);
}

}  // anonymous namespace

void megdnn::x86::warp_affine_cv_exec(_megdnn_tensor_in src,
                                      _megdnn_tensor_in trans,
                                      _megdnn_tensor_in dst, float border_value,
                                      BorderMode bmode, InterpolationMode imode,
                                      Handle* handle) {
    size_t ch = dst.layout[3];
    size_t width = dst.layout[2];
    size_t height = dst.layout[1];
    const size_t batch = dst.layout.shape[0];

    size_t BLOCK_SZ_H = std::min(BLOCK_SZ / 2, height);
    size_t BLOCK_SZ_W = std::min(BLOCK_SZ * BLOCK_SZ / BLOCK_SZ_H, width);
    BLOCK_SZ_H = std::min(BLOCK_SZ * BLOCK_SZ / BLOCK_SZ_W, height);

    size_t parallelism_batch = div_ceil<size_t>(height, BLOCK_SZ_H) *
                               div_ceil<size_t>(width, BLOCK_SZ_W);

    megdnn_assert(ch == 1 || ch == 3 || ch == 2,
                  "unsupported src channel: %zu, avaiable channel size: 1/2/3",
                  ch);
    const float* trans_ptr = trans.ptr<dt_float32>();
    if (dst.layout.dtype.enumv() == DTypeEnum::Float32) {
#define cb(_imode, _bmode, _ch)                                                \
    auto task = [src, trans_ptr, dst, border_value, parallelism_batch](        \
                        size_t index, size_t) {                                \
        size_t batch_id = index / parallelism_batch;                           \
        size_t task_id = index % parallelism_batch;                            \
        Mat<float> src_mat = TensorND2Mat<float>(src, batch_id);               \
        Mat<float> dst_mat = TensorND2Mat<float>(dst, batch_id);               \
        const float* task_trans_ptr = trans_ptr + batch_id * 2 * 3;            \
        warp_affine_cv<float MEGDNN_COMMA _imode MEGDNN_COMMA _bmode           \
                               MEGDNN_COMMA _ch>(                              \
                src_mat MEGDNN_COMMA const_cast<Mat<float>&>(dst_mat)          \
                        MEGDNN_COMMA task_trans_ptr MEGDNN_COMMA border_value, \
                task_id);                                                      \
    };                                                                         \
    MEGDNN_DISPATCH_MULTI_THREAD_CPU_KERN(                                     \
            static_cast<naive::HandleImpl*>(handle), batch* parallelism_batch, \
            task);
        DISPATCH_IMODE(imode, bmode, ch, cb)
    } else if (dst.layout.dtype.enumv() == DTypeEnum::Uint8) {
#undef cb
#define cb(_imode, _bmode, _ch)                                                \
    auto task = [src, trans_ptr, dst, border_value, parallelism_batch](        \
                        size_t index, size_t) {                                \
        size_t batch_id = index / parallelism_batch;                           \
        size_t task_id = index % parallelism_batch;                            \
        Mat<uchar> src_mat = TensorND2Mat<uchar>(src, batch_id);               \
        Mat<uchar> dst_mat = TensorND2Mat<uchar>(dst, batch_id);               \
        const float* task_trans_ptr = trans_ptr + batch_id * 2 * 3;            \
        warp_affine_cv<uchar MEGDNN_COMMA _imode MEGDNN_COMMA _bmode           \
                               MEGDNN_COMMA _ch>(                              \
                src_mat MEGDNN_COMMA const_cast<Mat<uchar>&>(dst_mat)          \
                        MEGDNN_COMMA task_trans_ptr MEGDNN_COMMA border_value, \
                task_id);                                                      \
    };                                                                         \
    MEGDNN_DISPATCH_MULTI_THREAD_CPU_KERN(                                     \
            static_cast<naive::HandleImpl*>(handle), batch* parallelism_batch, \
            task);
        DISPATCH_IMODE(imode, bmode, ch, cb)
#undef cb
    } else {
        megdnn_throw(megdnn_mangle("Unsupported datatype of WarpAffine optr."));
    }
}

// vim: syntax=cpp.doxygen
