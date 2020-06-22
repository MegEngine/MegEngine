/**
 * \file dnn/src/aarch64/warp_perspective/warp_perspective_cv.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/aarch64/handle.h"
#include "src/aarch64/warp_perspective/warp_perspective_cv.h"
#include "src/arm_common/simd_macro/marm_neon.h"
#include "src/common/cv/common.h"
#include "src/common/cv/helper.h"
#include "src/common/cv/interp_helper.h"
#include "src/common/utils.h"
#include "src/common/warp_common.h"

using namespace megdnn;
using namespace aarch64;
using namespace megcv;
using namespace warp;
namespace {

constexpr size_t BLOCK_SZ = 32u;
template <typename T, InterpolationMode imode, BorderMode bmode, size_t CH>
void warp_perspective_cv(const Mat<T>& src, Mat<T>& dst, const float* trans,
                         const float border_value, size_t task_id) {
    // no extra padding
    double M[9];
    rep(i, 9) M[i] = trans[i];
    T bvalue[3] = {(T)border_value, (T)border_value, (T)border_value};

    size_t x1, y1, width = dst.cols(), height = dst.rows();
    size_t BLOCK_SZ_H = std::min(BLOCK_SZ / 2, height);
    size_t BLOCK_SZ_W = std::min(BLOCK_SZ * BLOCK_SZ / BLOCK_SZ_H, width);
    BLOCK_SZ_H = std::min(BLOCK_SZ * BLOCK_SZ / BLOCK_SZ_W, height);

    size_t width_block_size = div_ceil<size_t>(width, BLOCK_SZ_W);
    size_t y = (task_id / width_block_size) * BLOCK_SZ_H;
    size_t x = (task_id % width_block_size) * BLOCK_SZ_W;
    // start invoke
    short XY[BLOCK_SZ * BLOCK_SZ * 2], A[BLOCK_SZ * BLOCK_SZ];

    float64x2_t vM6 = vdupq_n_f64(M[6]);
    float64x2_t vM0 = vdupq_n_f64(M[0]);
    float64x2_t vM3 = vdupq_n_f64(M[3]);
    float64x2_t v2M6 = vdupq_n_f64(M[6] * 2);
    float64x2_t v2M0 = vdupq_n_f64(M[0] * 2);
    float64x2_t v2M3 = vdupq_n_f64(M[3] * 2);
    float64x2_t v4f = vdupq_n_f64(4);
    float64x2_t v1f = vdupq_n_f64(1);
    float64x2_t v0f = vdupq_n_f64(0);
    float64x2_t vTABLE_SIZE = vdupq_n_f64(INTER_TAB_SIZE);
    float64x2_t vmin = vdupq_n_f64((double)INT_MIN);
    float64x2_t vmax = vdupq_n_f64((double)INT_MAX);
    int32x4_t vtabmask = vdupq_n_s32(INTER_TAB_SIZE - 1);

    size_t bw = std::min(BLOCK_SZ_W, width - x);
    size_t bh = std::min(BLOCK_SZ_H, height - y);  // height
    Mat<short> _XY(bh, bw, 2, XY);
    Mat<T> dpart(dst, y, bh, x, bw);

    for (y1 = 0; y1 < bh; y1++) {
        short* xy = XY + y1 * bw * 2;
        double X0 = M[0] * x + M[1] * (y + y1) + M[2];
        double Y0 = M[3] * x + M[4] * (y + y1) + M[5];
        double W0 = M[6] * x + M[7] * (y + y1) + M[8];
        float64x2_t vW0 = vdupq_n_f64(W0);
        float64x2_t vidx = {0.f, 1.f};
        float64x2_t vX0 = vdupq_n_f64(X0);
        float64x2_t vY0 = vdupq_n_f64(Y0);
        if (imode == IMode::NEAREST) {
            for (x1 = 0; x1 + 4 <= bw; x1 += 4) {
                float64x2_t vw0 = vaddq_f64(vW0, vmulq_f64(vM6, vidx));
                float64x2_t vw1 = vaddq_f64(vw0, v2M6);

                vw0 = vbitq_f64(vdivq_f64(v1f, vw0), v0f, vceqq_f64(vw0, v0f));
                vw1 = vbitq_f64(vdivq_f64(v1f, vw1), v0f, vceqq_f64(vw1, v0f));

                float64x2_t vtmp0 = vmlaq_f64(vX0, vM0, vidx);
                float64x2_t vtmp1 = vaddq_f64(vtmp0, v2M0);
                float64x2_t vfx0 = vmulq_f64(vtmp0, vw0);
                float64x2_t vfx1 = vmulq_f64(vtmp1, vw1);
                vfx0 = vmaxq_f64(vminq_f64(vfx0, vmax), vmin);
                vfx1 = vmaxq_f64(vminq_f64(vfx1, vmax), vmin);

                vtmp0 = vmlaq_f64(vY0, vM3, vidx);
                vtmp1 = vaddq_f64(vtmp0, v2M3);
                float64x2_t vfy0 = vmulq_f64(vtmp0, vw0);
                float64x2_t vfy1 = vmulq_f64(vtmp1, vw1);
                vfy0 = vmaxq_f64(vminq_f64(vfy0, vmax), vmin);
                vfy1 = vmaxq_f64(vminq_f64(vfy1, vmax), vmin);

                int32x2_t vx0 = vqmovn_s64(vcvtaq_s64_f64(vfx0));
                int32x2_t vx1 = vqmovn_s64(vcvtaq_s64_f64(vfx1));
                int32x2_t vy0 = vqmovn_s64(vcvtaq_s64_f64(vfy0));
                int32x2_t vy1 = vqmovn_s64(vcvtaq_s64_f64(vfy1));

                int32x4_t vx = vcombine_s32(vx0, vx1);
                int32x4_t vy = vcombine_s32(vy0, vy1);

                int16x4x2_t ret = {{vqmovn_s32(vx), vqmovn_s32(vy)}};
                vst2_s16(xy + x1 * 2, ret);

                vidx = vaddq_f64(vidx, v4f);
            }

            for (; x1 < bw; x1++) {
                double W = W0 + M[6] * x1;
                W = W ? 1. / W : 0;
                double fX = std::max(
                        (double)INT_MIN,
                        std::min((double)INT_MAX, (X0 + M[0] * x1) * W));
                double fY = std::max(
                        (double)INT_MIN,
                        std::min((double)INT_MAX, (Y0 + M[3] * x1) * W));
                int X = saturate_cast<int>(fX);
                int Y = saturate_cast<int>(fY);
                xy[x1 * 2] = saturate_cast<short>(X);
                xy[x1 * 2 + 1] = saturate_cast<short>(Y);
            }
        } else {
            short* alpha = A + y1 * bw;
            for (x1 = 0; x1 + 4 <= bw; x1 += 4) {
                float64x2_t vw0 = vaddq_f64(vW0, vmulq_f64(vM6, vidx));
                float64x2_t vw1 = vaddq_f64(vw0, v2M6);

                vw0 = vbitq_f64(vdivq_f64(vTABLE_SIZE, vw0), v0f,
                                vceqq_f64(vw0, v0f));
                vw1 = vbitq_f64(vdivq_f64(vTABLE_SIZE, vw1), v0f,
                                vceqq_f64(vw1, v0f));

                float64x2_t vtmp0 = vmlaq_f64(vX0, vM0, vidx);
                float64x2_t vtmp1 = vaddq_f64(vtmp0, v2M0);
                float64x2_t vfx0 = vmulq_f64(vtmp0, vw0);
                float64x2_t vfx1 = vmulq_f64(vtmp1, vw1);
                vfx0 = vmaxq_f64(vminq_f64(vfx0, vmax), vmin);
                vfx1 = vmaxq_f64(vminq_f64(vfx1, vmax), vmin);

                vtmp0 = vmlaq_f64(vY0, vM3, vidx);
                vtmp1 = vaddq_f64(vtmp0, v2M3);
                float64x2_t vfy0 = vmulq_f64(vtmp0, vw0);
                float64x2_t vfy1 = vmulq_f64(vtmp1, vw1);
                vfy0 = vmaxq_f64(vminq_f64(vfy0, vmax), vmin);
                vfy1 = vmaxq_f64(vminq_f64(vfy1, vmax), vmin);

                int32x2_t vx0 = vqmovn_s64(vcvtaq_s64_f64(vfx0));
                int32x2_t vx1 = vqmovn_s64(vcvtaq_s64_f64(vfx1));
                int32x2_t vy0 = vqmovn_s64(vcvtaq_s64_f64(vfy0));
                int32x2_t vy1 = vqmovn_s64(vcvtaq_s64_f64(vfy1));

                int32x4_t vx = vcombine_s32(vx0, vx1);
                int32x4_t vy = vcombine_s32(vy0, vy1);

                int16x4x2_t ret = {{vqshrn_n_s32(vx, INTER_BITS),
                                    vqshrn_n_s32(vy, INTER_BITS)}};
                vst2_s16(xy + x1 * 2, ret);

                vidx = vaddq_f64(vidx, v4f);

                vx = vandq_s32(vx, vtabmask);
                vy = vandq_s32(vy, vtabmask);

                vst1_s16(&alpha[x1],
                         vqmovn_s32(vmlaq_n_s32(vx, vy, INTER_TAB_SIZE)));
            }
            for (; x1 < bw; x1++) {
                double W = W0 + M[6] * x1;
                W = W ? INTER_TAB_SIZE / W : 0;
                double fX = std::max(
                        (double)INT_MIN,
                        std::min((double)INT_MAX, (X0 + M[0] * x1) * W));
                double fY = std::max(
                        (double)INT_MIN,
                        std::min((double)INT_MAX, (Y0 + M[3] * x1) * W));
                int X = saturate_cast<int>(fX);
                int Y = saturate_cast<int>(fY);
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
void megdnn::aarch64::warp_perspective_cv_exec(
        _megdnn_tensor_in src, _megdnn_tensor_in trans,
        _megdnn_tensor_in mat_idx, _megdnn_tensor_in dst, float border_value,
        BorderMode bmode, InterpolationMode imode, Handle* handle) {
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
    const int* midx_ptr = nullptr;
    if (mat_idx.raw_ptr) {
        megdnn_assert(mat_idx.layout.ndim == 1);
        midx_ptr = mat_idx.ptr<int>();
    }
    if (dst.layout.dtype.enumv() == DTypeEnum::Float32) {
#define cb(_imode, _bmode, _ch)                                                \
    auto task = [src, trans_ptr, midx_ptr, dst, border_value,                  \
                 parallelism_batch](size_t index, size_t) {                    \
        size_t batch_id = index / parallelism_batch;                           \
        size_t task_id = index % parallelism_batch;                            \
        size_t src_id = batch_id;                                              \
        if (midx_ptr) {                                                        \
            src_id = midx_ptr[batch_id];                                       \
            megdnn_assert(                                                     \
                    src_id < src.layout.shape[0],                              \
                    "mat_idx out of bound: mat_idx[%zu]=%zu src_batch=%zu",    \
                    batch_id, src_id, src.layout.shape[0]);                    \
        }                                                                      \
        Mat<float> src_mat = TensorND2Mat<float>(src, src_id);                 \
        Mat<float> dst_mat = TensorND2Mat<float>(dst, batch_id);               \
        const float* task_trans_ptr = trans_ptr + batch_id * 3 * 3;            \
        warp_perspective_cv<float MEGDNN_COMMA _imode MEGDNN_COMMA _bmode      \
                                    MEGDNN_COMMA _ch>(                         \
                src_mat MEGDNN_COMMA const_cast<Mat<float>&>(dst_mat)          \
                        MEGDNN_COMMA task_trans_ptr MEGDNN_COMMA border_value, \
                task_id);                                                      \
    };                                                                         \
    MEGDNN_DISPATCH_MULTI_THREAD_CPU_KERN(                                     \
            static_cast<naive::HandleImpl*>(handle), batch* parallelism_batch, \
            task);
            DISPATCH_IMODE(imode, bmode, ch, cb)
#undef cb
        } else if (dst.layout.dtype.enumv() == DTypeEnum::Uint8) {
#define cb(_imode, _bmode, _ch)                                                \
    auto task = [src, trans_ptr, midx_ptr, dst, border_value,                  \
                 parallelism_batch](size_t index, size_t) {                    \
        size_t batch_id = index / parallelism_batch;                           \
        size_t task_id = index % parallelism_batch;                            \
        size_t src_id = batch_id;                                              \
        if (midx_ptr) {                                                        \
            src_id = midx_ptr[batch_id];                                       \
            megdnn_assert(                                                     \
                    src_id < src.layout.shape[0],                              \
                    "mat_idx out of bound: mat_idx[%zu]=%zu src_batch=%zu",    \
                    batch_id, src_id, src.layout.shape[0]);                    \
        }                                                                      \
        Mat<uchar> src_mat = TensorND2Mat<uchar>(src, src_id);                 \
        Mat<uchar> dst_mat = TensorND2Mat<uchar>(dst, batch_id);               \
        const float* task_trans_ptr = trans_ptr + batch_id * 3 * 3;            \
        warp_perspective_cv<uchar MEGDNN_COMMA _imode MEGDNN_COMMA _bmode      \
                                    MEGDNN_COMMA _ch>(                         \
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
            megdnn_throw(
                    megdnn_mangle("Unsupported datatype of WarpPerspective optr."));
        }
}
// vim: syntax=cpp.doxygen
