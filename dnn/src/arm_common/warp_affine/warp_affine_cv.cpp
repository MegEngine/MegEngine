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
 * \file dnn/src/arm_common/warp_affine/warp_affine_cv.cpp
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

#include "src/arm_common/warp_affine/warp_affine_cv.h"
#include "src/arm_common/handle.h"

#include "src/common/cv/common.h"
#include "src/common/cv/helper.h"
#include "src/common/cv/interp_helper.h"
#include "src/common/utils.h"
#include "src/common/warp_common.h"

#include <climits>
#include <cstring>

#include "src/arm_common/simd_macro/marm_neon.h"

using namespace megdnn;
using namespace arm_common;
using namespace megcv;
using namespace warp;

#include "midout.h"
MIDOUT_DECL(megdnn_arm_common_warp_affine_cv)

namespace {

constexpr size_t BLOCK_SZ = 64_z;
template <typename T, InterpolationMode imode, BorderMode bmode, size_t CH>
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
    constexpr int AB_BITS = 10 > INTER_BITS ? 10 : INTER_BITS;
    constexpr int AB_SCALE = 1 << AB_BITS;
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

    short XY[BLOCK_SZ * BLOCK_SZ * 2], A[BLOCK_SZ * BLOCK_SZ];
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

            int32x4_t v_X0 = vdupq_n_s32(X0), v_Y0 = vdupq_n_s32(Y0);
            for (; x1 + 8 <= bw; x1 += 8) {
                int16x8x2_t v_dst;
                v_dst.val[0] = vcombine_s16(
                        vqmovn_s32(vshrq_n_s32(
                                vaddq_s32(v_X0, vld1q_s32(adelta + x + x1)),
                                AB_BITS)),
                        vqmovn_s32(vshrq_n_s32(
                                vaddq_s32(v_X0, vld1q_s32(adelta + x + x1 + 4)),
                                AB_BITS)));
                v_dst.val[1] = vcombine_s16(
                        vqmovn_s32(vshrq_n_s32(
                                vaddq_s32(v_Y0, vld1q_s32(bdelta + x + x1)),
                                AB_BITS)),
                        vqmovn_s32(vshrq_n_s32(
                                vaddq_s32(v_Y0, vld1q_s32(bdelta + x + x1 + 4)),
                                AB_BITS)));

                vst2q_s16(xy + (x1 << 1), v_dst);
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

            int32x4_t v__X0 = vdupq_n_s32(X0), v__Y0 = vdupq_n_s32(Y0),
                      v_mask = vdupq_n_s32(INTER_TAB_SIZE - 1);
            for (; x1 + 8 <= bw; x1 += 8) {
                int32x4_t v_X0 = vshrq_n_s32(
                        vaddq_s32(v__X0, vld1q_s32(adelta + x + x1)),
                        AB_BITS - INTER_BITS);
                int32x4_t v_Y0 = vshrq_n_s32(
                        vaddq_s32(v__Y0, vld1q_s32(bdelta + x + x1)),
                        AB_BITS - INTER_BITS);
                int32x4_t v_X1 = vshrq_n_s32(
                        vaddq_s32(v__X0, vld1q_s32(adelta + x + x1 + 4)),
                        AB_BITS - INTER_BITS);
                int32x4_t v_Y1 = vshrq_n_s32(
                        vaddq_s32(v__Y0, vld1q_s32(bdelta + x + x1 + 4)),
                        AB_BITS - INTER_BITS);

                int16x8x2_t v_xy;
                v_xy.val[0] =
                        vcombine_s16(vqmovn_s32(vshrq_n_s32(v_X0, INTER_BITS)),
                                     vqmovn_s32(vshrq_n_s32(v_X1, INTER_BITS)));
                v_xy.val[1] =
                        vcombine_s16(vqmovn_s32(vshrq_n_s32(v_Y0, INTER_BITS)),
                                     vqmovn_s32(vshrq_n_s32(v_Y1, INTER_BITS)));

                vst2q_s16(xy + (x1 << 1), v_xy);

                int16x4_t v_alpha0 = vmovn_s32(vaddq_s32(
                        vshlq_n_s32(vandq_s32(v_Y0, v_mask), INTER_BITS),
                        vandq_s32(v_X0, v_mask)));
                int16x4_t v_alpha1 = vmovn_s32(vaddq_s32(
                        vshlq_n_s32(vandq_s32(v_Y1, v_mask), INTER_BITS),
                        vandq_s32(v_X1, v_mask)));
                vst1q_s16(alpha + x1, vcombine_s16(v_alpha0, v_alpha1));
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

void megdnn::arm_common::warp_affine_cv_exec(
        _megdnn_tensor_in src, _megdnn_tensor_in trans, _megdnn_tensor_in dst,
        float border_value, BorderMode bmode, InterpolationMode imode,
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
#define cb(_imode, _bmode, _ch)                                             \
    MIDOUT_BEGIN(megdnn_arm_common_warp_affine_cv, midout_iv(_imode),       \
                 midout_iv(_bmode), midout_iv(_ch), float) {                \
        auto task = [src, trans_ptr, dst, border_value, parallelism_batch]( \
                            size_t index, size_t) {                         \
            size_t batch_id = index / parallelism_batch;                    \
            size_t task_id = index % parallelism_batch;                     \
            Mat<float> src_mat = TensorND2Mat<float>(src, batch_id);        \
            Mat<float> dst_mat = TensorND2Mat<float>(dst, batch_id);        \
            const float* task_trans_ptr = trans_ptr + batch_id * 2 * 3;     \
            warp_affine_cv<float MEGDNN_COMMA _imode MEGDNN_COMMA _bmode    \
                                   MEGDNN_COMMA _ch>(                       \
                    src_mat MEGDNN_COMMA const_cast<Mat<float>&>(dst_mat)   \
                            MEGDNN_COMMA task_trans_ptr MEGDNN_COMMA        \
                                    border_value,                           \
                    task_id);                                               \
        };                                                                  \
        MEGDNN_DISPATCH_MULTI_THREAD_CPU_KERN(                              \
                static_cast<naive::HandleImpl*>(handle),                    \
                batch* parallelism_batch, task);                            \
    }                                                                       \
    MIDOUT_END();
        DISPATCH_IMODE(imode, bmode, ch, cb)
    } else if (dst.layout.dtype.enumv() == DTypeEnum::Uint8) {
#undef cb
#define cb(_imode, _bmode, _ch)                                             \
    MIDOUT_BEGIN(megdnn_arm_common_warp_affine_cv, midout_iv(_imode),       \
                 midout_iv(_bmode), midout_iv(_ch), uchar) {                \
        auto task = [src, trans_ptr, dst, border_value, parallelism_batch]( \
                            size_t index, size_t) {                         \
            size_t batch_id = index / parallelism_batch;                    \
            size_t task_id = index % parallelism_batch;                     \
            Mat<uchar> src_mat = TensorND2Mat<uchar>(src, batch_id);        \
            Mat<uchar> dst_mat = TensorND2Mat<uchar>(dst, batch_id);        \
            const float* task_trans_ptr = trans_ptr + batch_id * 2 * 3;     \
            warp_affine_cv<uchar MEGDNN_COMMA _imode MEGDNN_COMMA _bmode    \
                                   MEGDNN_COMMA _ch>(                       \
                    src_mat MEGDNN_COMMA const_cast<Mat<uchar>&>(dst_mat)   \
                            MEGDNN_COMMA task_trans_ptr MEGDNN_COMMA        \
                                    border_value,                           \
                    task_id);                                               \
        };                                                                  \
        MEGDNN_DISPATCH_MULTI_THREAD_CPU_KERN(                              \
                static_cast<naive::HandleImpl*>(handle),                    \
                batch* parallelism_batch, task);                            \
    }                                                                       \
    MIDOUT_END();
        DISPATCH_IMODE(imode, bmode, ch, cb)
#undef cb
    } else {
        megdnn_throw(megdnn_mangle("Unsupported datatype of WarpAffine optr."));
    }
}

// vim: syntax=cpp.doxygen
