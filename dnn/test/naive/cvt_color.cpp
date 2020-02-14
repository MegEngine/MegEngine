/**
 * \file dnn/test/naive/cvt_color.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/common/cvt_color.h"
#include "test/common/checker.h"
#include "test/common/rng.h"
#include "test/common/tensor.h"

#include "test/naive/fixture.h"

namespace megdnn {
namespace test {

namespace {

static __inline int32_t clamp0(int32_t v) {
  return ((-(v) >> 31) & (v));
}

static __inline int32_t clamp255(int32_t v) {
  return (((255 - (v)) >> 31) | (v)) & 255;
}

static __inline uint32_t Clamp(int32_t val) {
  int v = clamp0(val);
  return (uint32_t)(clamp255(v));
}

void naive_row(const uint8_t* src_y, const uint8_t* src_vu, uint8_t* rgb_buf,
               int width) {
#define YG 18997  /* round(1.164 * 64 * 256 * 256 / 257) */
#define YGB -1160 /* 1.164 * 64 * -16 + 64 / 2 */

// U and V contributions to R,G,B.
#define UB -128 /* max(-128, round(-2.018 * 64)) */
#define UG 25   /* round(0.391 * 64) */
#define VG 52   /* round(0.813 * 64) */
#define VR -102 /* round(-1.596 * 64) */

// Bias values to subtract 16 from Y and 128 from U and V.
#define BB (UB * 128 + YGB)
#define BG (UG * 128 + VG * 128 + YGB)
#define BR (VR * 128 + YGB)

    for (int x = 0; x < width - 1; x += 2) {
        uint8_t y = src_y[0];
        uint8_t u = src_vu[1];
        uint8_t v = src_vu[0];
        uint32_t y1 = (uint32_t)(y * 0x0101 * YG) >> 16;
        uint8_t B = Clamp((int32_t)(-(u * UB) + y1 + BB) >> 6);
        uint8_t G = Clamp((int32_t)(-(u * UG + v * VG) + y1 + BG) >> 6);
        uint8_t R = Clamp((int32_t)(-(v * VR) + y1 + BR) >> 6);
        rgb_buf[0] = B;
        rgb_buf[1] = G;
        rgb_buf[2] = R;

        y = src_y[1];
        y1 = (uint32_t)(y * 0x0101 * YG) >> 16;
        B = Clamp((int32_t)(-(u * UB) + y1 + BB) >> 6);
        G = Clamp((int32_t)(-(u * UG + v * VG) + y1 + BG) >> 6);
        R = Clamp((int32_t)(-(v * VR) + y1 + BR) >> 6);
        rgb_buf[3] = B;
        rgb_buf[4] = G;
        rgb_buf[5] = R;
        src_y += 2;
        src_vu += 2;
        rgb_buf += 6;  // Advance 2 pixels.
    }
#undef BB
#undef BG
#undef BR
#undef YGB
#undef UB
#undef UG
#undef VG
#undef VR
#undef YG

}

//! refer to libyuv
//! https://github.com/lemenkov/libyuv/blob/7e936044d154b9fe159a67f9562e10b1ef1cb590/source/convert_argb.cc#L1079
void naive(const uint8_t* src_y, int src_stride_y, const uint8_t* src_uv,
           int src_stride_uv, uint8_t* dst_argb, int dst_stride_argb, int width,
           int height) {
    rep(y, height) {
        naive_row(src_y, src_uv, dst_argb, width);
        dst_argb += dst_stride_argb;
        src_y += src_stride_y;
        if (y & 1) {
            src_uv += src_stride_uv;
        }
    }
}

//! check real yuv
void run_check(Handle* handle, const size_t IH, const size_t IW) {
    const size_t OH = IH / 3 * 2;
    const size_t OW = IW;
    const size_t OC = 3;
    SyncedTensor<uint8_t> src(handle, {1, IH, IW, 1}),
        dst(handle, {1, OH, OW, OC}),
        expect(handle, {1, OH, OW, OC});
    auto opr = handle->create_operator<CvtColor>();
    opr->param().mode = param::CvtColor::Mode::BT601_YUV2BGR_NV21;
    opr->exec(src.tensornd_dev(), dst.tensornd_dev(), {});
    naive(src.ptr_host(), IW, src.ptr_host() + OH * IW, IW,
          expect.ptr_mutable_host(), OW * OC, OW, OH);

    rep(i, OH) rep(j, OW) rep(c, OC) {
        uint8_t dst_value = dst.ptr_host()[i * OW * OC + j * OC + c];
        uint8_t expect_value = expect.ptr_host()[i * OW * OC + j * OC + c];
        megdnn_assert(dst_value == expect_value,
                      "Error: %d(actual) != %d(expect) at(%zu,%zu,%zu)",
                      static_cast<int>(dst_value),
                      static_cast<int>(expect_value), i, j, c);
    }
#undef rep

}

}  // namespace

TEST_F(NAIVE, CVTCOLOR_BT601_YUV)
{
    run_check(handle(), 150, 100);
    run_check(handle(), 180, 100);
}

} // namespace test
} // namespace megdnn

// vim: syntax=cpp.doxygen
