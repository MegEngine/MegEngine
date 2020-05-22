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
 * \file dnn/src/arm_common/cvt_color/opr_impl.cpp
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
#include <cstring>
#include "src/arm_common/cvt_color/opr_impl.h"
#include "src/arm_common/handle.h"
#include "src/arm_common/simd_macro/marm_neon.h"
#include "src/common/cv/common.h"
#include "src/common/cv/cvt_color.h"
#include "src/common/cv/helper.h"
#include "src/common/utils.h"
#include "midout.h"

MIDOUT_DECL(megdnn_arm_cvtcolor)
MIDOUT_DECL(megdnn_arm_cvtcolor_cases)
MIDOUT_DECL(megdnn_arm_cvt_bt601_yuv)

namespace megdnn {
namespace arm_common {

GENERATE_CVT_OPR_DECL_FOREACH(GENERATE_CVT_OPR_DECL)
GENERATE_UNSUPPORT_CVT_OPR_FOR_FLOAT(GENERATE_UNSUPPORT_CVT_OPR)

using namespace megcv;

namespace {
/**
 * \brief yuv to rgb or bgr.
 *
 * \tparam rgb, is convert to rgb or bgr
 * \tparam is_planar, if true, the layout is YYYYUUVV or YYYYVVUU, otherwise
 *     YYYYYUVUV or YYYYYVUVU
 * \tparam is_uv, if true, U is before V, otherwise V is before U
 */
template <bool rgb = true, bool is_planar = true, bool is_uv = true>
void cvt_yuv_transform(const Mat8u& src, Mat8u& dst) {
    uint8x16_t v_y;
    int32x4_t v_y_s32_0, v_y_s32_1, v_y_s32_2, v_y_s32_3;
    uint8x8x2_t v_vu;
    int32x4_t v_RV0, v_RV1, v_RV2, v_RV3;
    int32x4_t v_GVU0, v_GVU1, v_GVU2, v_GVU3;
    int32x4_t v_BU0, v_BU1, v_BU2, v_BU3;

    int32x4x4_t v_R;
    int32x4x4_t v_G;
    int32x4x4_t v_B;
    uint8x16x3_t v_RGB, v_BGR;

    int16x8_t v_128;
    v_128 = vdupq_n_s16(128);

    int16x4_t v_359, v_88, v_183, v_454;
    v_359 = vdup_n_s16(359);
    v_88 = vdup_n_s16(88);
    v_183 = vdup_n_s16(183);
    v_454 = vdup_n_s16(454);
    size_t height = dst.rows();
    size_t width = dst.cols();
    int src_step = src.step();
    const unsigned char* pY = src.ptr();
    const unsigned char* pU;
    const unsigned char* pV;
    if (is_uv) {
        pU = src.ptr(height);
        //! only used if is_planar is false
        pV = src.ptr(height + height / 4);
    } else {
        pV = src.ptr(height);
        //! only used if is_planar is false
        pU = src.ptr(height + height / 4);
    }

#define SET_COLOR(out, index) \
    if (rgb) {                \
        out[index++] = R;     \
        out[index++] = G;     \
        out[index++] = B;     \
    } else {                  \
        out[index++] = B;     \
        out[index++] = G;     \
        out[index++] = R;     \
    }

    for (size_t r = 0; r < height; r += 2, pY += (src_step << 1)) {
        unsigned char* dst0 = dst.ptr(r);
        unsigned char* dst1 = dst.ptr(r + 1);
        size_t index0 = 0;
        size_t index1 = 0;
        int c = 0;
        for (; c <= (int)(width - 16); c += 16, index0 += 48, index1 += 48) {
            int16x8x2_t v_vu_s16;
            if (is_planar) {
                v_vu_s16.val[0] =
                        vreinterpretq_s16_u16(vmovl_u8(vld1_u8(pV + c / 2)));
                v_vu_s16.val[1] =
                        vreinterpretq_s16_u16(vmovl_u8(vld1_u8(pU + c / 2)));
            } else {
                if (is_uv) {
                    v_vu = vld2_u8(pU + c);
                    v_vu_s16.val[0] =
                            vreinterpretq_s16_u16(vmovl_u8(v_vu.val[1]));
                    v_vu_s16.val[1] =
                            vreinterpretq_s16_u16(vmovl_u8(v_vu.val[0]));
                } else {
                    v_vu = vld2_u8(pV + c);
                    v_vu_s16.val[0] =
                            vreinterpretq_s16_u16(vmovl_u8(v_vu.val[0]));
                    v_vu_s16.val[1] =
                            vreinterpretq_s16_u16(vmovl_u8(v_vu.val[1]));
                }
            }

            v_vu_s16.val[0] = vsubq_s16(v_vu_s16.val[0], v_128);
            v_vu_s16.val[1] = vsubq_s16(v_vu_s16.val[1], v_128);

            int16x4_t v_v0, v_u0;
            int16x4_t v_v1, v_u1;
            v_v0 = vget_low_s16(v_vu_s16.val[0]);
            v_v1 = vget_high_s16(v_vu_s16.val[0]);
            v_u0 = vget_low_s16(v_vu_s16.val[1]);
            v_u1 = vget_high_s16(v_vu_s16.val[1]);

            v_RV1 = vshrq_n_s32(vmull_s16(v_v0, v_359), 8);
            v_RV3 = vshrq_n_s32(vmull_s16(v_v1, v_359), 8);
            v_GVU1 = vshrq_n_s32(
                    vaddq_s32(vmull_s16(v_u0, v_88), vmull_s16(v_v0, v_183)),
                    8);
            v_GVU3 = vshrq_n_s32(
                    vaddq_s32(vmull_s16(v_u1, v_88), vmull_s16(v_v1, v_183)),
                    8);
            v_BU1 = vshrq_n_s32(vmull_s16(v_u0, v_454), 8);
            v_BU3 = vshrq_n_s32(vmull_s16(v_u1, v_454), 8);

            int32x4x2_t temp;
            temp = vzipq_s32(v_RV1, v_RV1);
            v_RV0 = temp.val[0];
            v_RV1 = temp.val[1];
            temp = vzipq_s32(v_RV3, v_RV3);
            v_RV2 = temp.val[0];
            v_RV3 = temp.val[1];

            temp = vzipq_s32(v_GVU1, v_GVU1);
            v_GVU0 = temp.val[0];
            v_GVU1 = temp.val[1];
            temp = vzipq_s32(v_GVU3, v_GVU3);
            v_GVU2 = temp.val[0];
            v_GVU3 = temp.val[1];

            temp = vzipq_s32(v_BU1, v_BU1);
            v_BU0 = temp.val[0];
            v_BU1 = temp.val[1];
            temp = vzipq_s32(v_BU3, v_BU3);
            v_BU2 = temp.val[0];
            v_BU3 = temp.val[1];

            v_y = vld1q_u8(pY + c);
            uint8x8_t v_y_half;
            v_y_half = vget_low_u8(v_y);
            int16x8_t v_y_2quarter = vreinterpretq_s16_u16(vmovl_u8(v_y_half));
            v_y_s32_0 = vmovl_s16(vget_low_s16(v_y_2quarter));
            v_y_s32_1 = vmovl_s16(vget_high_s16(v_y_2quarter));

            v_y_half = vget_high_u8(v_y);
            v_y_2quarter = vreinterpretq_s16_u16(vmovl_u8(v_y_half));
            v_y_s32_2 = vmovl_s16(vget_low_s16(v_y_2quarter));
            v_y_s32_3 = vmovl_s16(vget_high_s16(v_y_2quarter));

            v_R.val[0] = vaddq_s32(v_y_s32_0, v_RV0);
            v_R.val[1] = vaddq_s32(v_y_s32_1, v_RV1);
            v_R.val[2] = vaddq_s32(v_y_s32_2, v_RV2);
            v_R.val[3] = vaddq_s32(v_y_s32_3, v_RV3);

            v_G.val[0] = vsubq_s32(v_y_s32_0, v_GVU0);
            v_G.val[1] = vsubq_s32(v_y_s32_1, v_GVU1);
            v_G.val[2] = vsubq_s32(v_y_s32_2, v_GVU2);
            v_G.val[3] = vsubq_s32(v_y_s32_3, v_GVU3);

            v_B.val[0] = vaddq_s32(v_y_s32_0, v_BU0);
            v_B.val[1] = vaddq_s32(v_y_s32_1, v_BU1);
            v_B.val[2] = vaddq_s32(v_y_s32_2, v_BU2);
            v_B.val[3] = vaddq_s32(v_y_s32_3, v_BU3);

            if (rgb) {
                v_RGB.val[0] = vcombine_u8(
                        vqmovun_s16(vcombine_s16(vmovn_s32(v_R.val[0]),
                                                 vmovn_s32(v_R.val[1]))),
                        vqmovun_s16(vcombine_s16(vmovn_s32(v_R.val[2]),
                                                 vmovn_s32(v_R.val[3]))));
                v_RGB.val[1] = vcombine_u8(
                        vqmovun_s16(vcombine_s16(vmovn_s32(v_G.val[0]),
                                                 vmovn_s32(v_G.val[1]))),
                        vqmovun_s16(vcombine_s16(vmovn_s32(v_G.val[2]),
                                                 vmovn_s32(v_G.val[3]))));
                v_RGB.val[2] = vcombine_u8(
                        vqmovun_s16(vcombine_s16(vmovn_s32(v_B.val[0]),
                                                 vmovn_s32(v_B.val[1]))),
                        vqmovun_s16(vcombine_s16(vmovn_s32(v_B.val[2]),
                                                 vmovn_s32(v_B.val[3]))));

                vst3q_u8((dst0 + c * 3), v_RGB);
            } else {
                v_BGR.val[0] = vcombine_u8(
                        vqmovun_s16(vcombine_s16(vmovn_s32(v_B.val[0]),
                                                 vmovn_s32(v_B.val[1]))),
                        vqmovun_s16(vcombine_s16(vmovn_s32(v_B.val[2]),
                                                 vmovn_s32(v_B.val[3]))));
                v_BGR.val[1] = vcombine_u8(
                        vqmovun_s16(vcombine_s16(vmovn_s32(v_G.val[0]),
                                                 vmovn_s32(v_G.val[1]))),
                        vqmovun_s16(vcombine_s16(vmovn_s32(v_G.val[2]),
                                                 vmovn_s32(v_G.val[3]))));
                v_BGR.val[2] = vcombine_u8(
                        vqmovun_s16(vcombine_s16(vmovn_s32(v_R.val[0]),
                                                 vmovn_s32(v_R.val[1]))),
                        vqmovun_s16(vcombine_s16(vmovn_s32(v_R.val[2]),
                                                 vmovn_s32(v_R.val[3]))));
                vst3q_u8((dst0 + c * 3), v_BGR);
            }

            v_y = vld1q_u8(pY + src_step + c);
            v_y_half = vget_low_u8(v_y);
            v_y_2quarter = vreinterpretq_s16_u16(vmovl_u8(v_y_half));
            v_y_s32_0 = vmovl_s16(vget_low_s16(v_y_2quarter));
            v_y_s32_1 = vmovl_s16(vget_high_s16(v_y_2quarter));

            v_y_half = vget_high_u8(v_y);
            v_y_2quarter = vreinterpretq_s16_u16(vmovl_u8(v_y_half));
            v_y_s32_2 = vmovl_s16(vget_low_s16(v_y_2quarter));
            v_y_s32_3 = vmovl_s16(vget_high_s16(v_y_2quarter));

            v_R.val[0] = vaddq_s32(v_y_s32_0, v_RV0);
            v_R.val[1] = vaddq_s32(v_y_s32_1, v_RV1);
            v_R.val[2] = vaddq_s32(v_y_s32_2, v_RV2);
            v_R.val[3] = vaddq_s32(v_y_s32_3, v_RV3);

            v_G.val[0] = vsubq_s32(v_y_s32_0, v_GVU0);
            v_G.val[1] = vsubq_s32(v_y_s32_1, v_GVU1);
            v_G.val[2] = vsubq_s32(v_y_s32_2, v_GVU2);
            v_G.val[3] = vsubq_s32(v_y_s32_3, v_GVU3);

            v_B.val[0] = vaddq_s32(v_y_s32_0, v_BU0);
            v_B.val[1] = vaddq_s32(v_y_s32_1, v_BU1);
            v_B.val[2] = vaddq_s32(v_y_s32_2, v_BU2);
            v_B.val[3] = vaddq_s32(v_y_s32_3, v_BU3);

            if (rgb) {
                v_RGB.val[0] = vcombine_u8(
                        vqmovun_s16(vcombine_s16(vmovn_s32(v_R.val[0]),
                                                 vmovn_s32(v_R.val[1]))),
                        vqmovun_s16(vcombine_s16(vmovn_s32(v_R.val[2]),
                                                 vmovn_s32(v_R.val[3]))));
                v_RGB.val[1] = vcombine_u8(
                        vqmovun_s16(vcombine_s16(vmovn_s32(v_G.val[0]),
                                                 vmovn_s32(v_G.val[1]))),
                        vqmovun_s16(vcombine_s16(vmovn_s32(v_G.val[2]),
                                                 vmovn_s32(v_G.val[3]))));
                v_RGB.val[2] = vcombine_u8(
                        vqmovun_s16(vcombine_s16(vmovn_s32(v_B.val[0]),
                                                 vmovn_s32(v_B.val[1]))),
                        vqmovun_s16(vcombine_s16(vmovn_s32(v_B.val[2]),
                                                 vmovn_s32(v_B.val[3]))));

                vst3q_u8((dst1 + c * 3), v_RGB);
            } else {
                v_BGR.val[0] = vcombine_u8(
                        vqmovun_s16(vcombine_s16(vmovn_s32(v_B.val[0]),
                                                 vmovn_s32(v_B.val[1]))),
                        vqmovun_s16(vcombine_s16(vmovn_s32(v_B.val[2]),
                                                 vmovn_s32(v_B.val[3]))));
                v_BGR.val[1] = vcombine_u8(
                        vqmovun_s16(vcombine_s16(vmovn_s32(v_G.val[0]),
                                                 vmovn_s32(v_G.val[1]))),
                        vqmovun_s16(vcombine_s16(vmovn_s32(v_G.val[2]),
                                                 vmovn_s32(v_G.val[3]))));
                v_BGR.val[2] = vcombine_u8(
                        vqmovun_s16(vcombine_s16(vmovn_s32(v_R.val[0]),
                                                 vmovn_s32(v_R.val[1]))),
                        vqmovun_s16(vcombine_s16(vmovn_s32(v_R.val[2]),
                                                 vmovn_s32(v_R.val[3]))));
                vst3q_u8((dst1 + c * 3), v_BGR);
            }
        }

        for (; c < (int)width; c += 2) {
            int Y00, Y01, Y10, Y11, U, V;
            int R, G, B;
            Y00 = *((pY) + c);
            Y01 = *((pY) + c + 1);
            Y10 = *((pY) + src_step + c);
            Y11 = *((pY) + src_step + c + 1);
            if (is_planar) {
                V = *(pV + c / 2);
                U = *(pU + c / 2);
            } else {
                if (is_uv) {
                    U = *(pU + c);
                    V = *(pU + c + 1);
                } else {
                    V = *(pV + c);
                    U = *(pV + c + 1);
                }
            }

            int ruv, guv, buv;
            ruv = ((359 * (V - 128)) >> 8);
            guv = -1 * ((88 * (U - 128) + 183 * (V - 128)) >> 8);
            buv = ((454 * (U - 128)) >> 8);

            R = Y00 + ruv;
            G = Y00 + guv;
            B = Y00 + buv;
            R = (R > 255) ? 255 : ((R < 0) ? 0 : R);
            G = (G > 255) ? 255 : ((G < 0) ? 0 : G);
            B = (B > 255) ? 255 : ((B < 0) ? 0 : B);

            SET_COLOR(dst0, index0);

            R = Y01 + ruv;
            G = Y01 + guv;
            B = Y01 + buv;
            R = (R > 255) ? 255 : ((R < 0) ? 0 : R);
            G = (G > 255) ? 255 : ((G < 0) ? 0 : G);
            B = (B > 255) ? 255 : ((B < 0) ? 0 : B);

            SET_COLOR(dst0, index0);

            ruv = ((359 * (V - 128)) >> 8);
            guv = -1 * ((88 * (U - 128) + 183 * (V - 128)) >> 8);
            buv = ((454 * (U - 128)) >> 8);
            R = Y10 + ruv;
            G = Y10 + guv;
            B = Y10 + buv;
            R = (R > 255) ? 255 : ((R < 0) ? 0 : R);
            G = (G > 255) ? 255 : ((G < 0) ? 0 : G);
            B = (B > 255) ? 255 : ((B < 0) ? 0 : B);

            SET_COLOR(dst1, index1);

            R = Y11 + ruv;
            G = Y11 + guv;
            B = Y11 + buv;
            R = (R > 255) ? 255 : ((R < 0) ? 0 : R);
            G = (G > 255) ? 255 : ((G < 0) ? 0 : G);
            B = (B > 255) ? 255 : ((B < 0) ? 0 : B);

            SET_COLOR(dst1, index1);
        }
        if (is_planar) {
            pV += src_step / 2;
            pU += src_step / 2;
        } else {
            if (is_uv) {
                pU += src_step;
            } else {
                pV += src_step;
            }
        }
    }
#undef SET_COLOR
}

/**
 * \brief real yuv to rgb or bgr.
 *
 * \tparam rgb, is convert to rgb or bgr
 * \tparam is_planar, if true, the layout is YYYYUUVV or YYYYVVUU, otherwise
 *     YYYYYUVUV or YYYYYVUVU
 * \tparam is_uv, if true, U is before V, otherwise V is before U
 *
 * \note it is BT.601 YUV to RGB reference, it refer to
 * https://github.com/opencv/opencv/blob/1b53a4fccc1a61541b71340af9a04b59484ec2cf/modules/imgproc/src/opencl/color_yuv.cl#L253
 *     R = (Y - 16) * 1.164              - (V - 128) * -1.596
 *     G = (Y - 16) * 1.164 - (U - 128) *  0.391 - (V - 128) *  0.813
 *     B = (Y - 16) * 1.164 - (U - 128) * -2.018
 * The Numerical approximations refers to libyuv
 * implementation(https://github.com/lemenkov/libyuv/blob/7e936044d154b9fe159a67f9562e10b1ef1cb590/source/row_common.cc#L1002),
 */
template <bool rgb = true, bool is_planar = true, bool is_uv = true>
void cvt_BT601_yuv_transform(const Mat8u& src, Mat8u& dst) {
    typedef unsigned char uint8;
    const uint8* pY;
    const uint8* pU;
    const uint8* pV;

#define SET_COLOR(out, index) \
    if (rgb) {                \
        out[index++] = R;     \
        out[index++] = G;     \
        out[index++] = B;     \
    } else {                  \
        out[index++] = B;     \
        out[index++] = G;     \
        out[index++] = R;     \
    }

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

    int32x4_t v_UB = vdupq_n_s32(UB);
    int32x4_t v_YG = vdupq_n_s32(YG);
    int32x4_t v_UG = vdupq_n_s32(UG);
    int32x4_t v_VG = vdupq_n_s32(VG);
    int32x4_t v_VR = vdupq_n_s32(VR);
    int32x4_t v_BB = vdupq_n_s32(UB * 128 + YGB);
    int32x4_t v_BG = vdupq_n_s32(UG * 128 + VG * 128 + YGB);
    int32x4_t v_BR = vdupq_n_s32(VR * 128 + YGB);
    int32x4_t v_0101 = vdupq_n_s32(0x0101);

    uint8x8x2_t v_vu;
    int32x4x4_t v_R;
    int32x4x4_t v_G;
    int32x4x4_t v_B;
    uint8x16x3_t v_RGB, v_BGR;
    int32x4_t v_Y1;

    int width = dst.cols();
    int height = dst.rows();
    int src_step = src.step();
    pY = src.ptr();
    if (is_uv) {
        pU = src.ptr(height);
        pV = src.ptr(height + height / 4);
    } else {
        pV = src.ptr(height);
        pU = src.ptr(height + height / 4);
    }
    for (int i = 0; i < height; i += 2, pY += src_step * 2) {
        size_t index = 0;
        size_t index1 = 0;
        uint8* out = dst.ptr(i);
        uint8* out1 = dst.ptr(i + 1);
        int j = 0;
        int jV = 0;

        for (; j <= (int)(width - 16); j += 16, index += 48, index1 += 48) {
            int16x8x2_t v_vu_s16;
            if (is_planar) {
                v_vu_s16.val[0] =
                        vreinterpretq_s16_u16(vmovl_u8(vld1_u8(pV + jV)));
                v_vu_s16.val[1] =
                        vreinterpretq_s16_u16(vmovl_u8(vld1_u8(pU + jV)));
                jV += 8;
            } else {
                if (is_uv) {
                    v_vu = vld2_u8(pU + j);
                    v_vu_s16.val[0] =
                            vreinterpretq_s16_u16(vmovl_u8(v_vu.val[1]));
                    v_vu_s16.val[1] =
                            vreinterpretq_s16_u16(vmovl_u8(v_vu.val[0]));
                } else {
                    v_vu = vld2_u8(pV + j);
                    v_vu_s16.val[0] =
                            vreinterpretq_s16_u16(vmovl_u8(v_vu.val[0]));
                    v_vu_s16.val[1] =
                            vreinterpretq_s16_u16(vmovl_u8(v_vu.val[1]));
                }
            }

            int32x4_t v_v0, v_u0;
            int32x4_t v_v1, v_u1;
            int32x4_t v_v2, v_u2;
            int32x4_t v_v3, v_u3;
            v_v0 = vmovl_s16(vget_low_s16(v_vu_s16.val[0]));
            v_v2 = vmovl_s16(vget_high_s16(v_vu_s16.val[0]));
            v_u0 = vmovl_s16(vget_low_s16(v_vu_s16.val[1]));
            v_u2 = vmovl_s16(vget_high_s16(v_vu_s16.val[1]));

            //! zip the v0 to 0011/2233, as two y value share the shape u/v
            int32x4x2_t temp;
            temp = vzipq_s32(v_v0, v_v0);
            v_v0 = temp.val[0];
            v_v1 = temp.val[1];
            temp = vzipq_s32(v_v2, v_v2);
            v_v2 = temp.val[0];
            v_v3 = temp.val[1];

            temp = vzipq_s32(v_u0, v_u0);
            v_u0 = temp.val[0];
            v_u1 = temp.val[1];
            temp = vzipq_s32(v_u2, v_u2);
            v_u2 = temp.val[0];
            v_u3 = temp.val[1];

            uint8x16_t v_y = vld1q_u8(pY + j);
            uint8x8_t v_y_half = vget_low_u8(v_y);
            int16x8_t v_y_2quarter = vreinterpretq_s16_u16(vmovl_u8(v_y_half));
            int32x4_t v_y0 = vmovl_s16(vget_low_s16(v_y_2quarter));
            int32x4_t v_y1 = vmovl_s16(vget_high_s16(v_y_2quarter));
            v_y_half = vget_high_u8(v_y);
            v_y_2quarter = vreinterpretq_s16_u16(vmovl_u8(v_y_half));
            int32x4_t v_y2 = vmovl_s16(vget_low_s16(v_y_2quarter));
            int32x4_t v_y3 = vmovl_s16(vget_high_s16(v_y_2quarter));

            //! calc
#define CALC(_idx)                                                            \
    v_Y1 = vshrq_n_s32(vmulq_s32(vmulq_s32(v_y##_idx, v_0101), v_YG), 16);    \
    v_B.val[_idx] = vshrq_n_s32(                                              \
            vsubq_s32(vaddq_s32(v_Y1, v_BB), vmulq_s32(v_u##_idx, v_UB)), 6); \
    v_G.val[_idx] =                                                           \
            vshrq_n_s32(vsubq_s32(vaddq_s32(v_Y1, v_BG),                      \
                                  vaddq_s32(vmulq_s32(v_u##_idx, v_UG),       \
                                            vmulq_s32(v_v##_idx, v_VG))),     \
                        6);                                                   \
    v_R.val[_idx] = vshrq_n_s32(                                              \
            vsubq_s32(vaddq_s32(v_Y1, v_BR), vmulq_s32(v_v##_idx, v_VR)), 6);

            CALC(0);
            CALC(1);
            CALC(2);
            CALC(3);

            if (rgb) {
                v_RGB.val[0] = vcombine_u8(
                        vqmovun_s16(vcombine_s16(vmovn_s32(v_R.val[0]),
                                                 vmovn_s32(v_R.val[1]))),
                        vqmovun_s16(vcombine_s16(vmovn_s32(v_R.val[2]),
                                                 vmovn_s32(v_R.val[3]))));
                v_RGB.val[1] = vcombine_u8(
                        vqmovun_s16(vcombine_s16(vmovn_s32(v_G.val[0]),
                                                 vmovn_s32(v_G.val[1]))),
                        vqmovun_s16(vcombine_s16(vmovn_s32(v_G.val[2]),
                                                 vmovn_s32(v_G.val[3]))));
                v_RGB.val[2] = vcombine_u8(
                        vqmovun_s16(vcombine_s16(vmovn_s32(v_B.val[0]),
                                                 vmovn_s32(v_B.val[1]))),
                        vqmovun_s16(vcombine_s16(vmovn_s32(v_B.val[2]),
                                                 vmovn_s32(v_B.val[3]))));
                vst3q_u8((out + index), v_RGB);
            } else {
                v_BGR.val[0] = vcombine_u8(
                        vqmovun_s16(vcombine_s16(vmovn_s32(v_B.val[0]),
                                                 vmovn_s32(v_B.val[1]))),
                        vqmovun_s16(vcombine_s16(vmovn_s32(v_B.val[2]),
                                                 vmovn_s32(v_B.val[3]))));
                v_BGR.val[1] = vcombine_u8(
                        vqmovun_s16(vcombine_s16(vmovn_s32(v_G.val[0]),
                                                 vmovn_s32(v_G.val[1]))),
                        vqmovun_s16(vcombine_s16(vmovn_s32(v_G.val[2]),
                                                 vmovn_s32(v_G.val[3]))));
                v_BGR.val[2] = vcombine_u8(
                        vqmovun_s16(vcombine_s16(vmovn_s32(v_R.val[0]),
                                                 vmovn_s32(v_R.val[1]))),
                        vqmovun_s16(vcombine_s16(vmovn_s32(v_R.val[2]),
                                                 vmovn_s32(v_R.val[3]))));
                vst3q_u8((out + index), v_BGR);
            }

            v_y = vld1q_u8(pY + src_step + j);
            v_y_half = vget_low_u8(v_y);
            v_y_2quarter = vreinterpretq_s16_u16(vmovl_u8(v_y_half));
            v_y0 = vmovl_s16(vget_low_s16(v_y_2quarter));
            v_y1 = vmovl_s16(vget_high_s16(v_y_2quarter));
            v_y_half = vget_high_u8(v_y);
            v_y_2quarter = vreinterpretq_s16_u16(vmovl_u8(v_y_half));
            v_y2 = vmovl_s16(vget_low_s16(v_y_2quarter));
            v_y3 = vmovl_s16(vget_high_s16(v_y_2quarter));

            CALC(0);
            CALC(1);
            CALC(2);
            CALC(3);

            if (rgb) {
                v_RGB.val[0] = vcombine_u8(
                        vqmovun_s16(vcombine_s16(vmovn_s32(v_R.val[0]),
                                                 vmovn_s32(v_R.val[1]))),
                        vqmovun_s16(vcombine_s16(vmovn_s32(v_R.val[2]),
                                                 vmovn_s32(v_R.val[3]))));
                v_RGB.val[1] = vcombine_u8(
                        vqmovun_s16(vcombine_s16(vmovn_s32(v_G.val[0]),
                                                 vmovn_s32(v_G.val[1]))),
                        vqmovun_s16(vcombine_s16(vmovn_s32(v_G.val[2]),
                                                 vmovn_s32(v_G.val[3]))));
                v_RGB.val[2] = vcombine_u8(
                        vqmovun_s16(vcombine_s16(vmovn_s32(v_B.val[0]),
                                                 vmovn_s32(v_B.val[1]))),
                        vqmovun_s16(vcombine_s16(vmovn_s32(v_B.val[2]),
                                                 vmovn_s32(v_B.val[3]))));
                vst3q_u8((out1 + index1), v_RGB);
            } else {
                v_BGR.val[0] = vcombine_u8(
                        vqmovun_s16(vcombine_s16(vmovn_s32(v_B.val[0]),
                                                 vmovn_s32(v_B.val[1]))),
                        vqmovun_s16(vcombine_s16(vmovn_s32(v_B.val[2]),
                                                 vmovn_s32(v_B.val[3]))));
                v_BGR.val[1] = vcombine_u8(
                        vqmovun_s16(vcombine_s16(vmovn_s32(v_G.val[0]),
                                                 vmovn_s32(v_G.val[1]))),
                        vqmovun_s16(vcombine_s16(vmovn_s32(v_G.val[2]),
                                                 vmovn_s32(v_G.val[3]))));
                v_BGR.val[2] = vcombine_u8(
                        vqmovun_s16(vcombine_s16(vmovn_s32(v_R.val[0]),
                                                 vmovn_s32(v_R.val[1]))),
                        vqmovun_s16(vcombine_s16(vmovn_s32(v_R.val[2]),
                                                 vmovn_s32(v_R.val[3]))));
                vst3q_u8((out1 + index1), v_BGR);
            }
#undef CALC
        }

        for (; j < width; j += 2) {
            int U = 0, V = 0, Y0 = 0;
            if (is_planar) {
                V = *(pV + jV);
                U = *(pU + jV);
                jV++;
            } else {
                if (is_uv) {
                    U = *(pU + j);
                    V = *(pU + j + 1);
                } else {
                    V = *(pV + j);
                    U = *(pV + j + 1);
                }
            }

            Y0 = *((pY) + j);
            uint32_t Y1 = static_cast<uint32_t>(Y0 * 0x0101 * YG) >> 16;
            uint8_t B = saturate_cast<unsigned char>(
                    static_cast<int32_t>(-(U * UB) + Y1 + BB) >> 6);
            uint8_t G = saturate_cast<unsigned char>(
                    static_cast<int32_t>(-(U * UG + V * VG) + Y1 + BG) >> 6);
            uint8_t R = saturate_cast<unsigned char>(
                    static_cast<int32_t>(-(V * VR) + Y1 + BR) >> 6);
            SET_COLOR(out, index)

            Y0 = *((pY) + j + 1);
            Y1 = static_cast<uint32_t>(Y0 * 0x0101 * YG) >> 16;
            B = saturate_cast<unsigned char>(
                    static_cast<int32_t>(-(U * UB) + Y1 + BB) >> 6);
            G = saturate_cast<unsigned char>(
                    static_cast<int32_t>(-(U * UG + V * VG) + Y1 + BG) >> 6);
            R = saturate_cast<unsigned char>(
                    static_cast<int32_t>(-(V * VR) + Y1 + BR) >> 6);
            SET_COLOR(out, index)

            Y0 = *((pY) + src_step + j);
            Y1 = static_cast<uint32_t>(Y0 * 0x0101 * YG) >> 16;
            B = saturate_cast<unsigned char>(
                    static_cast<int32_t>(-(U * UB) + Y1 + BB) >> 6);
            G = saturate_cast<unsigned char>(
                    static_cast<int32_t>(-(U * UG + V * VG) + Y1 + BG) >> 6);
            R = saturate_cast<unsigned char>(
                    static_cast<int32_t>(-(V * VR) + Y1 + BR) >> 6);
            SET_COLOR(out1, index1)

            Y0 = *((pY) + src_step + j + 1);
            Y1 = static_cast<uint32_t>(Y0 * 0x0101 * YG) >> 16;
            B = saturate_cast<unsigned char>(
                    static_cast<int32_t>(-(U * UB) + Y1 + BB) >> 6);
            G = saturate_cast<unsigned char>(
                    static_cast<int32_t>(-(U * UG + V * VG) + Y1 + BG) >> 6);
            R = saturate_cast<unsigned char>(
                    static_cast<int32_t>(-(V * VR) + Y1 + BR) >> 6);
            SET_COLOR(out1, index1)
        }

        if (is_planar) {
            pV += src_step / 2;
            pU += src_step / 2;
        } else {
            if (is_uv) {
                pU += src_step;
            } else {
                pV += src_step;
            }
        }
    }
#undef SET_COLOR
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

}  // namespace

void cvt_rgb2gray_32f_neon(const Mat32f& src, Mat32f& dst) {
    static const float coef[] = {0.299f, 0.587f, 0.114f};
    // load coef into neon types
    const float32x4_t v_cr(vdupq_n_f32(coef[0])), v_cg(vdupq_n_f32(coef[1])),
            v_cb(vdupq_n_f32(coef[2]));

#define EXPAND(offset)                                                         \
    v_src = vld3q_f32(psrc + offset * 3);                                      \
    vst1q_f32(pdst + offset,                                                   \
              vmlaq_f32(vmlaq_f32(vmulq_f32(v_src.val[0], v_cr), v_src.val[1], \
                                  v_cg),                                       \
                        v_src.val[2], v_cb));
    for (size_t r = 0; r < src.rows(); ++r) {
        const float* psrc = src.ptr(r);
        float* pdst = dst.ptr(r);

        const float* pend = psrc + src.cols() * 3;
        // pack 48 float at a time (16 pixels)

        for (; psrc <= pend - 16 * 3; psrc += 16 * 3, pdst += 16) {
            float32x4x3_t v_src;

            EXPAND(0);
            EXPAND(4);
            EXPAND(8);
            EXPAND(12);
        }
        // if more than 8 pixels left, do an extra pack
        if (psrc <= pend - 8 * 3) {
            float32x4x3_t v_src;

            EXPAND(0);
            EXPAND(4);

            psrc += 8 * 3;
            pdst += 8;
        }
        // if more than 4 pixels left, do an extra pack
        if (psrc <= pend - 4 * 3) {
            float32x4x3_t v_src;

            EXPAND(0);

            psrc += 4 * 3;
            pdst += 4;
        }
        // loop over left pixels
        for (; psrc < pend; psrc += 3, pdst += 1) {
            *pdst = psrc[0] * coef[0] + psrc[1] * coef[1] + psrc[2] * coef[2];
        }
    }
#undef EXPAND
}

void cvt_rgb2yuv_8u_neon(const Mat8u& src, Mat8u& dst) {
    const int yuv_shift = 14;
    const int coeffs[] = {1868, 9617, 4899, 8061, 14369};
    const int delta = 128 << yuv_shift;

    const int C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2], C3 = coeffs[3],
              C4 = coeffs[4];

    int16x4_t v_c0, v_c1, v_c2;
    int32x4_t v_c3, v_c4, v_delta, v_delta2;
    v_c0 = vdup_n_s16(coeffs[0]);
    v_c1 = vdup_n_s16(coeffs[1]);
    v_c2 = vdup_n_s16(coeffs[2]);
    v_c3 = vdupq_n_s32(coeffs[3]);
    v_c4 = vdupq_n_s32(coeffs[4]);
    v_delta = vdupq_n_s32(128 << yuv_shift);
    v_delta2 = vdupq_n_s32(1 << (yuv_shift - 1));

    for (size_t r = 0; r < src.rows(); ++r) {
        const uchar* psrc = src.ptr(r);
        uchar* pdst = dst.ptr(r);
        const uchar* const pend = psrc + src.cols() * 3;

        // pack 8 pixels (24 uchar)
        for (; psrc <= pend - 8 * 3; psrc += 8 * 3, pdst += 8 * 3) {
            uint8x8x3_t v_dst;
            int16x8x3_t v_src16;

            uint8x8x3_t v_src = vld3_u8(psrc);
            v_src16.val[0] = vreinterpretq_s16_u16(vmovl_u8(v_src.val[0]));
            v_src16.val[1] = vreinterpretq_s16_u16(vmovl_u8(v_src.val[1]));
            v_src16.val[2] = vreinterpretq_s16_u16(vmovl_u8(v_src.val[2]));

            int16x4x3_t v_src0;
            v_src0.val[0] = vget_low_s16(v_src16.val[0]);
            v_src0.val[1] = vget_low_s16(v_src16.val[1]);
            v_src0.val[2] = vget_low_s16(v_src16.val[2]);

            int32x4_t v_Y0 = vmlal_s16(vmlal_s16(vmull_s16(v_src0.val[0], v_c0),
                                                 v_src0.val[1], v_c1),
                                       v_src0.val[2], v_c2);
            v_Y0 = vshrq_n_s32(vaddq_s32(v_Y0, v_delta2), yuv_shift);
            int32x4_t v_Cr0 = vmlaq_s32(
                    v_delta, vsubq_s32(vmovl_s16(v_src0.val[0]), v_Y0), v_c3);
            v_Cr0 = vshrq_n_s32(vaddq_s32(v_Cr0, v_delta2), yuv_shift);
            int32x4_t v_Cb0 = vmlaq_s32(
                    v_delta, vsubq_s32(vmovl_s16(v_src0.val[2]), v_Y0), v_c4);
            v_Cb0 = vshrq_n_s32(vaddq_s32(v_Cb0, v_delta2), yuv_shift);

            v_src0.val[0] = vget_high_s16(v_src16.val[0]);
            v_src0.val[1] = vget_high_s16(v_src16.val[1]);
            v_src0.val[2] = vget_high_s16(v_src16.val[2]);

            int32x4_t v_Y1 = vmlal_s16(vmlal_s16(vmull_s16(v_src0.val[0], v_c0),
                                                 v_src0.val[1], v_c1),
                                       v_src0.val[2], v_c2);
            v_Y1 = vshrq_n_s32(vaddq_s32(v_Y1, v_delta2), yuv_shift);
            int32x4_t v_Cr1 = vmlaq_s32(
                    v_delta, vsubq_s32(vmovl_s16(v_src0.val[0]), v_Y1), v_c3);
            v_Cr1 = vshrq_n_s32(vaddq_s32(v_Cr1, v_delta2), yuv_shift);
            int32x4_t v_Cb1 = vmlaq_s32(
                    v_delta, vsubq_s32(vmovl_s16(v_src0.val[2]), v_Y1), v_c4);
            v_Cb1 = vshrq_n_s32(vaddq_s32(v_Cb1, v_delta2), yuv_shift);

            v_dst.val[0] = vqmovun_s16(
                    vcombine_s16(vqmovn_s32(v_Y0), vqmovn_s32(v_Y1)));
            v_dst.val[1] = vqmovun_s16(
                    vcombine_s16(vqmovn_s32(v_Cr0), vqmovn_s32(v_Cr1)));
            v_dst.val[2] = vqmovun_s16(
                    vcombine_s16(vqmovn_s32(v_Cb0), vqmovn_s32(v_Cb1)));

            vst3_u8(pdst, v_dst);
        }
        for (; psrc < pend; psrc += 3, pdst += 3) {
            int Y = descale(psrc[0] * C0 + psrc[1] * C1 + psrc[2] * C2,
                            yuv_shift);
            int Cr = descale((psrc[0] - Y) * C3 + delta, yuv_shift);
            int Cb = descale((psrc[2] - Y) * C4 + delta, yuv_shift);
            pdst[0] = saturate_cast<uchar>(Y);
            pdst[1] = saturate_cast<uchar>(Cr);
            pdst[2] = saturate_cast<uchar>(Cb);
        }
    }
}

void cvt_rgb2yuv_32f_neon(const Mat32f& src, Mat32f& dst) {
    const float coeffs[] = {0.114f, 0.587f, 0.299f, 0.492f, 0.877f};
    float32x4_t v_c0, v_c1, v_c2, v_c3, v_c4, v_delta;
    const float C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2], C3 = coeffs[3],
                C4 = coeffs[4];
    const float delta = 0.5f;
    v_c0 = vdupq_n_f32(coeffs[0]);
    v_c1 = vdupq_n_f32(coeffs[1]);
    v_c2 = vdupq_n_f32(coeffs[2]);
    v_c3 = vdupq_n_f32(coeffs[3]);
    v_c4 = vdupq_n_f32(coeffs[4]);
    v_delta = vdupq_n_f32(0.5f);

    for (size_t r = 0; r < src.rows(); ++r) {
        const float* psrc = src.ptr(r);
        float* pdst = dst.ptr(r);
        const float* const pend = psrc + src.cols() * 3;

        for (; psrc <= pend - 4 * 3; psrc += 4 * 3, pdst += 4 * 3) {
            float32x4x3_t v_src = vld3q_f32(psrc), v_dst;
            v_dst.val[0] = vmlaq_f32(vmlaq_f32(vmulq_f32(v_src.val[0], v_c0),
                                               v_src.val[1], v_c1),
                                     v_src.val[2], v_c2);
            v_dst.val[1] = vmlaq_f32(
                    v_delta, vsubq_f32(v_src.val[0], v_dst.val[0]), v_c3);
            v_dst.val[2] = vmlaq_f32(
                    v_delta, vsubq_f32(v_src.val[2], v_dst.val[0]), v_c4);

            vst3q_f32(pdst, v_dst);
        }
        for (; psrc < pend; psrc += 3, pdst += 3) {
            float Y = psrc[0] * C0 + psrc[1] * C1 + psrc[2] * C2;
            float Cr = (psrc[0] - Y) * C3 + delta;
            float Cb = (psrc[2] - Y) * C4 + delta;
            pdst[0] = Y;
            pdst[1] = Cr;
            pdst[2] = Cb;
        }
    }
}

void cvt_yuv2rgb_8u_neon(const Mat8u& src, Mat8u& dst) {
    static const int coeffs[] = {33292, -6472, -9519, 18678};
    const int C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2], C3 = coeffs[3];
    const int yuv_shift = 14;
    const int delta = 128;

    int32x4_t v_c0, v_c1, v_c2, v_c3, v_delta2;
    int16x4_t v_delta;
    v_c0 = vdupq_n_s32(coeffs[0]);
    v_c1 = vdupq_n_s32(coeffs[1]);
    v_c2 = vdupq_n_s32(coeffs[2]);
    v_c3 = vdupq_n_s32(coeffs[3]);
    v_delta = vdup_n_s16(128);
    v_delta2 = vdupq_n_s32(1 << (yuv_shift - 1));
    for (size_t r = 0; r < src.rows(); ++r) {
        const uchar* psrc = src.ptr(r);
        uchar* pdst = dst.ptr(r);
        const uchar* const pend = psrc + src.cols() * 3;
        for (; psrc <= pend - 8 * 3; psrc += 8 * 3, pdst += 8 * 3) {
            uint8x8x3_t v_src = vld3_u8(psrc);
            int16x8x3_t v_src16;
            v_src16.val[0] = vreinterpretq_s16_u16(vmovl_u8(v_src.val[0]));
            v_src16.val[1] = vreinterpretq_s16_u16(vmovl_u8(v_src.val[1]));
            v_src16.val[2] = vreinterpretq_s16_u16(vmovl_u8(v_src.val[2]));

            int16x4_t v_Y = vget_low_s16(v_src16.val[0]),
                      v_Cr = vget_low_s16(v_src16.val[1]),
                      v_Cb = vget_low_s16(v_src16.val[2]);

            int32x4_t v_b0 = vmulq_s32(v_c3, vsubl_s16(v_Cb, v_delta));
            v_b0 = vaddw_s16(vshrq_n_s32(vaddq_s32(v_b0, v_delta2), yuv_shift),
                             v_Y);
            int32x4_t v_g0 =
                    vmlaq_s32(vmulq_s32(vsubl_s16(v_Cr, v_delta), v_c1),
                              vsubl_s16(v_Cb, v_delta), v_c2);
            v_g0 = vaddw_s16(vshrq_n_s32(vaddq_s32(v_g0, v_delta2), yuv_shift),
                             v_Y);
            int32x4_t v_r0 = vmulq_s32(v_c0, vsubl_s16(v_Cr, v_delta));
            v_r0 = vaddw_s16(vshrq_n_s32(vaddq_s32(v_r0, v_delta2), yuv_shift),
                             v_Y);

            v_Y = vget_high_s16(v_src16.val[0]);
            v_Cr = vget_high_s16(v_src16.val[1]);
            v_Cb = vget_high_s16(v_src16.val[2]);

            int32x4_t v_b1 = vmulq_s32(v_c3, vsubl_s16(v_Cb, v_delta));
            v_b1 = vaddw_s16(vshrq_n_s32(vaddq_s32(v_b1, v_delta2), yuv_shift),
                             v_Y);
            int32x4_t v_g1 =
                    vmlaq_s32(vmulq_s32(vsubl_s16(v_Cr, v_delta), v_c1),
                              vsubl_s16(v_Cb, v_delta), v_c2);
            v_g1 = vaddw_s16(vshrq_n_s32(vaddq_s32(v_g1, v_delta2), yuv_shift),
                             v_Y);
            int32x4_t v_r1 = vmulq_s32(v_c0, vsubl_s16(v_Cr, v_delta));
            v_r1 = vaddw_s16(vshrq_n_s32(vaddq_s32(v_r1, v_delta2), yuv_shift),
                             v_Y);

            uint8x8_t v_b =
                    vqmovun_s16(vcombine_s16(vmovn_s32(v_b0), vmovn_s32(v_b1)));
            uint8x8_t v_g =
                    vqmovun_s16(vcombine_s16(vmovn_s32(v_g0), vmovn_s32(v_g1)));
            uint8x8_t v_r =
                    vqmovun_s16(vcombine_s16(vmovn_s32(v_r0), vmovn_s32(v_r1)));

            uint8x8x3_t v_dst;
            v_dst.val[0] = v_r;
            v_dst.val[1] = v_g;
            v_dst.val[2] = v_b;
            vst3_u8(pdst, v_dst);
        }
        for (; psrc < pend; psrc += 3, pdst += 3) {
            uchar Y = psrc[0];
            uchar Cr = psrc[1];
            uchar Cb = psrc[2];

            int b = Y + descale((Cb - delta) * C3, yuv_shift);
            int g = Y +
                    descale((Cb - delta) * C2 + (Cr - delta) * C1, yuv_shift);
            int r = Y + descale((Cr - delta) * C0, yuv_shift);

            pdst[0] = saturate_cast<uchar>(r);
            pdst[1] = saturate_cast<uchar>(g);
            pdst[2] = saturate_cast<uchar>(b);
        }
    }
}

void cvt_yuv2rgb_32f_neon(const Mat32f& src, Mat32f& dst) {
    static const float coeffs[] = {2.032f, -0.395f, -0.581f, 1.140f};
    const float delta = 0.5f;
    const float C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2], C3 = coeffs[3];

    float32x4_t v_c0, v_c1, v_c2, v_c3, v_delta;
    v_c0 = vdupq_n_f32(coeffs[0]);
    v_c1 = vdupq_n_f32(coeffs[1]);
    v_c2 = vdupq_n_f32(coeffs[2]);
    v_c3 = vdupq_n_f32(coeffs[3]);
    v_delta = vdupq_n_f32(0.5f);

    for (size_t r = 0; r < src.rows(); ++r) {
        const float* psrc = src.ptr(r);
        float* pdst = dst.ptr(r);
        const float* const pend = psrc + src.cols() * 3;
        for (; psrc <= pend - 4 * 3; psrc += 4 * 3, pdst += 4 * 3) {
            float32x4x3_t v_src = vld3q_f32(psrc), v_dst;
            float32x4_t v_Y = v_src.val[0], v_Cr = v_src.val[1],
                        v_Cb = v_src.val[2];

            v_dst.val[0] = vmlaq_f32(v_Y, vsubq_f32(v_Cr, v_delta), v_c0);
            v_dst.val[1] = vaddq_f32(
                    vmlaq_f32(vmulq_f32(vsubq_f32(v_Cb, v_delta), v_c2),
                              vsubq_f32(v_Cr, v_delta), v_c1),
                    v_Y);
            v_dst.val[2] = vmlaq_f32(v_Y, vsubq_f32(v_Cb, v_delta), v_c3);

            vst3q_f32(pdst, v_dst);
        }

        for (; psrc < pend; psrc += 3, pdst += 3) {
            float Y = psrc[0], Cr = psrc[1], Cb = psrc[2];

            float b = Y + (Cb - delta) * C3;
            float g = Y + (Cb - delta) * C2 + (Cr - delta) * C1;
            float r = Y + (Cr - delta) * C0;

            pdst[0] = r;
            pdst[1] = g;
            pdst[2] = b;
        }
    }
}

void cvt_rgba2rgb_8u_neon(const Mat8u& src, Mat8u& dst) {
    for (size_t r = 0; r < src.rows(); ++r) {
        const uchar* psrc = src.ptr(r);
        uchar* pdst = dst.ptr(r);
        const uchar* const pend = psrc + src.cols() * 4;

        for (; psrc <= pend - 64; pdst += 48, psrc += 64) {
            uint8x16x4_t v_src = vld4q_u8(psrc);
            uint8x16x3_t v_dst;
            v_dst.val[0] = v_src.val[0];
            v_dst.val[1] = v_src.val[1];
            v_dst.val[2] = v_src.val[2];
            vst3q_u8(pdst, v_dst);
        }
        for (; psrc <= pend - 32; pdst += 24, psrc += 32) {
            uint8x8x4_t v_src = vld4_u8(psrc);
            uint8x8x3_t v_dst;
            v_dst.val[0] = v_src.val[0];
            v_dst.val[1] = v_src.val[1];
            v_dst.val[2] = v_src.val[2];
            vst3_u8(pdst, v_dst);
        }
        for (; psrc < pend; pdst += 3, psrc += 4) {
            uchar t0 = psrc[0], t1 = psrc[1], t2 = psrc[2];
            pdst[0] = t0;
            pdst[1] = t1;
            pdst[2] = t2;
        }
    }
}

void cvt_rgba2bgr_8u_neon(const Mat8u& src, Mat8u& dst) {
    for (size_t r = 0; r < src.rows(); ++r) {
        const uchar* psrc = src.ptr(r);
        uchar* pdst = dst.ptr(r);
        const uchar* const pend = psrc + src.cols() * 4;

        for (; psrc <= pend - 64; pdst += 48, psrc += 64) {
            uint8x16x4_t v_src = vld4q_u8(psrc);
            uint8x16x3_t v_dst;
            v_dst.val[0] = v_src.val[2];
            v_dst.val[1] = v_src.val[1];
            v_dst.val[2] = v_src.val[0];
            vst3q_u8(pdst, v_dst);
        }
        for (; psrc <= pend - 32; pdst += 24, psrc += 32) {
            uint8x8x4_t v_src = vld4_u8(psrc);
            uint8x8x3_t v_dst;
            v_dst.val[0] = v_src.val[2];
            v_dst.val[1] = v_src.val[1];
            v_dst.val[2] = v_src.val[0];
            vst3_u8(pdst, v_dst);
        }
        for (; psrc < pend; pdst += 3, psrc += 4) {
            uchar t0 = psrc[0], t1 = psrc[1], t2 = psrc[2];
            pdst[0] = t2;
            pdst[1] = t1;
            pdst[2] = t0;
        }
    }
}

void cvt_rgb2bgr_8u_neon(const Mat8u& src, Mat8u& dst) {
    for (size_t r = 0; r < src.rows(); ++r) {
        const uchar* psrc = src.ptr(r);
        uchar* pdst = dst.ptr(r);
        const uchar* const pend = psrc + src.cols() * 3;

        for (; psrc <= pend - 48; pdst += 48, psrc += 48) {
            uint8x16x3_t v_src = vld3q_u8(psrc), v_dst;
            v_dst.val[0] = v_src.val[2];
            v_dst.val[1] = v_src.val[1];
            v_dst.val[2] = v_src.val[0];
            vst3q_u8(pdst, v_dst);
        }
        for (; psrc <= pend - 24; pdst += 24, psrc += 24) {
            uint8x8x3_t v_src = vld3_u8(psrc), v_dst;
            v_dst.val[0] = v_src.val[2];
            v_dst.val[1] = v_src.val[1];
            v_dst.val[2] = v_src.val[0];
            vst3_u8(pdst, v_dst);
        }
        for (; psrc < pend; pdst += 3, psrc += 3) {
            uchar t0 = psrc[0], t1 = psrc[1], t2 = psrc[2];
            pdst[0] = t2;
            pdst[1] = t1;
            pdst[2] = t0;
        }
    }
}

template <>
void cvt_rgb2gray<uchar>(const Mat8u& src, Mat8u& dst) {
    megdnn_assert(src.rows() == dst.rows());
    megdnn_assert(src.cols() == dst.cols());
    megdnn_assert(src.channels() == 3);
    megdnn_assert(dst.channels() == 1);

    const int yuv_shift = 14, R2Y = 4899, G2Y = 9617, B2Y = 1868;

    const uchar* _src = src.ptr();
    uchar* _dst = dst.ptr();
    size_t rows = src.rows();
    size_t cols = src.cols();
    size_t src_step = src.step();
    size_t dst_step = dst.step();
    for (size_t r = 0; r < rows; ++r, _src += src_step, _dst += dst_step) {
        const uchar* temp_src = _src;
        uchar* temp_dst = _dst;
        for (size_t c = 0; c < cols; ++c, temp_src += 3, temp_dst += 1) {
            uchar x0 = temp_src[0];
            uchar x1 = temp_src[1];
            uchar x2 = temp_src[2];
            temp_dst[0] =
                    (x0 * R2Y + x1 * G2Y + x2 * B2Y + (1 << (yuv_shift - 1))) >>
                    yuv_shift;
        }
    }
}

template <>
void cvt_rgb2gray<float>(const Mat32f& src, Mat32f& dst) {
    megdnn_assert(src.channels() == 3);
    megdnn_assert(dst.channels() == 1);
    megdnn_assert(src.rows() == dst.rows());
    megdnn_assert(src.cols() == dst.cols());

    return cvt_rgb2gray_32f_neon(src, dst);
}

// gray2rgb
template <>
void cvt_gray2rgb<uchar>(const Mat8u& src, Mat8u& dst) {
    megdnn_assert(src.rows() == dst.rows());
    megdnn_assert(src.cols() == dst.cols());
    megdnn_assert(src.channels() == 1);
    megdnn_assert(dst.channels() == 3);

    for (size_t r = 0; r < src.rows(); ++r) {
        const uchar* psrc = src.ptr(r);
        uchar* pdst = dst.ptr(r);
        const uchar* const pend = psrc + src.cols() * 1;
        for (; psrc < pend; psrc += 1, pdst += 3) {
            pdst[0] = pdst[1] = pdst[2] = psrc[0];
        }
    }
}
template <>
void cvt_gray2rgb<float>(const Mat32f& src, Mat32f& dst) {
    megdnn_assert(src.rows() == dst.rows());
    megdnn_assert(src.cols() == dst.cols());
    megdnn_assert(src.channels() == 1);
    megdnn_assert(dst.channels() == 3);

    for (size_t r = 0; r < src.rows(); ++r) {
        const float* psrc = src.ptr(r);
        float* pdst = dst.ptr(r);
        const float* const pend = psrc + src.cols() * 1;
        for (; psrc < pend; psrc += 1, pdst += 3) {
            pdst[0] = pdst[1] = pdst[2] = psrc[0];
        }
    }
}

// rgb2yuv
template <>
void cvt_rgb2yuv<uchar>(const Mat8u& src, Mat8u& dst) {
    megdnn_assert(src.channels() == 3);
    megdnn_assert(dst.channels() == 3);
    megdnn_assert(src.rows() == dst.rows());
    megdnn_assert(src.cols() == dst.cols());

    return cvt_rgb2yuv_8u_neon(src, dst);
}
template <>
void cvt_rgb2yuv<float>(const Mat32f& src, Mat32f& dst) {
    megdnn_assert(src.channels() == 3);
    megdnn_assert(dst.channels() == 3);
    megdnn_assert(src.rows() == dst.rows());
    megdnn_assert(src.cols() == dst.cols());

    return cvt_rgb2yuv_32f_neon(src, dst);
}

// yuv2rgb
template <>
void cvt_yuv2rgb<float>(const Mat32f& src, Mat32f& dst) {
    megdnn_assert(src.channels() == 3);
    megdnn_assert(dst.channels() == 3);
    megdnn_assert(src.rows() == dst.rows());
    megdnn_assert(src.cols() == dst.cols());

    // turn on neon optimization wont improve
    // return cvt_yuv2rgb_32f_neon(src, dst);

    const float coef[] = {2.032f, -0.395f, -0.581f, 1.140f};
    const float delta = 0.5f;
    for (size_t r = 0; r < src.rows(); ++r) {
        for (size_t c = 0; c < src.cols(); ++c) {
            const float* v = &src.at(r, c, 0);
            float Y = v[0];
            float Cr = v[1];
            float Cb = v[2];

            float R = Y + (Cr - delta) * coef[0];
            float G = Y + (Cb - delta) * coef[2] + (Cr - delta) * coef[1];
            float B = Y + (Cb - delta) * coef[3];

            float* target = &dst.at(r, c, 0);
            target[0] = R;
            target[1] = G;
            target[2] = B;
        }
    }
}

template <>
void cvt_yuv2rgb<uchar>(const Mat8u& src, Mat8u& dst) {
    megdnn_assert(src.channels() == 3);
    megdnn_assert(dst.channels() == 3);
    megdnn_assert(src.rows() == dst.rows());
    megdnn_assert(src.cols() == dst.cols());

    return cvt_yuv2rgb_8u_neon(src, dst);
}

template <>
void cvt_rgba2rgb<uchar>(const Mat8u& src, Mat8u& dst) {
    megdnn_assert(src.channels() == 4);
    megdnn_assert(dst.channels() == 3);
    megdnn_assert(src.rows() == dst.rows());
    megdnn_assert(src.cols() == dst.cols());

    return cvt_rgba2rgb_8u_neon(src, dst);
}

template <>
void cvt_rgba2bgr<uchar>(const Mat8u& src, Mat8u& dst) {
    megdnn_assert(src.channels() == 4);
    megdnn_assert(dst.channels() == 3);
    megdnn_assert(src.rows() == dst.rows());
    megdnn_assert(src.cols() == dst.cols());

    return cvt_rgba2bgr_8u_neon(src, dst);
}

template <>
void cvt_rgba2gray<uchar>(const Mat8u& src, Mat8u& dst) {
    megdnn_assert(src.channels() == 4);
    megdnn_assert(dst.channels() == 1);
    megdnn_assert(src.rows() == dst.rows());
    megdnn_assert(src.cols() == dst.cols());

    const int yuv_shift = 14, R2Y = 4899, G2Y = 9617, B2Y = 1868;

    const uchar* _src = src.ptr();
    uchar* _dst = dst.ptr();
    size_t rows = src.rows();
    size_t cols = src.cols();
    size_t src_step = src.step();
    size_t dst_step = dst.step();
    for (size_t r = 0; r < rows; ++r, _src += src_step, _dst += dst_step) {
        const uchar* temp_src = _src;
        uchar* temp_dst = _dst;
        for (size_t c = 0; c < cols; ++c, temp_src += 4, temp_dst += 1) {
            uchar x0 = temp_src[0];
            uchar x1 = temp_src[1];
            uchar x2 = temp_src[2];
            temp_dst[0] =
                    (x0 * R2Y + x1 * G2Y + x2 * B2Y + (1 << (yuv_shift - 1))) >>
                    yuv_shift;
        }
    }
}

template <>
void cvt_rgb2bgr<uchar>(const Mat8u& src, Mat8u& dst) {
    megdnn_assert(src.channels() == 3);
    megdnn_assert(dst.channels() == 3);
    megdnn_assert(src.rows() == dst.rows());
    megdnn_assert(src.cols() == dst.cols());

    return cvt_rgb2bgr_8u_neon(src, dst);
}

template <>
void cvt_bgr2gray<uchar>(const Mat8u& src, Mat8u& dst) {
    megdnn_assert(src.channels() == 3);
    megdnn_assert(dst.channels() == 1);
    megdnn_assert(src.rows() == dst.rows());
    megdnn_assert(src.cols() == dst.cols());

    const int yuv_shift = 14, R2Y = 4899, G2Y = 9617, B2Y = 1868;
    int tab[256 * 3];

    int b = 0, g = 0, r = (1 << (yuv_shift - 1));
    for (int i = 0; i < 256; ++i, r += R2Y, g += G2Y, b += B2Y) {
        tab[i] = r;
        tab[i + 256] = g;
        tab[i + 512] = b;
    }

    const uchar* _src = src.ptr();
    uchar* _dst = dst.ptr();
    size_t rows = src.rows();
    size_t cols = src.cols();
    size_t src_step = src.step();
    size_t dst_step = dst.step();
    for (size_t r = 0; r < rows; ++r, _src += src_step, _dst += dst_step) {
        const uchar* temp_src = _src;
        uchar* temp_dst = _dst;
        for (size_t c = 0; c < cols; ++c, temp_src += 3, temp_dst += 1) {
            uchar x0 = temp_src[0];
            uchar x1 = temp_src[1];
            uchar x2 = temp_src[2];
            temp_dst[0] =
                    (tab[x2] + tab[x1 + 256] + tab[x0 + 512]) >> yuv_shift;
        }
    }
}

template <>
void cvt_bgr2rgb<uchar>(const Mat8u& src, Mat8u& dst) {
    return cvt_rgb2bgr<uchar>(src, dst);
}

template <>
void cvt_yuv2gray_nv21<uchar>(const Mat8u& src, Mat8u& dst) {
    const uchar* _src = src.ptr();
    uchar* _dst = dst.ptr();
    size_t rows = dst.rows();
    size_t cols = dst.cols();
    size_t src_step = src.step();
    size_t dst_step = dst.step();
    for (size_t r = 0; r < rows; ++r, _src += src_step, _dst += dst_step) {
        const uchar* temp_src = _src;
        uchar* temp_dst = _dst;
        for (size_t c = 0; c < cols; ++c, temp_src += 1, temp_dst += 1) {
            temp_dst[0] = temp_src[0];
        }
    }
}

template <>
void cvt_yuv2rgb_nv21<uchar>(const Mat8u& src, Mat8u& dst) {
    return cvt_yuv_transform<true, false, false>(src, dst);
}

template <>
void cvt_yuv2bgr_nv21<uchar>(const Mat8u& src, Mat8u& dst) {
    return cvt_yuv_transform<false, false, false>(src, dst);
}

template <>
void cvt_yuv2rgb_nv12<uchar>(const Mat8u& src, Mat8u& dst) {
    return cvt_yuv_transform<true, false, true>(src, dst);
}

template <>
void cvt_yuv2bgr_nv12<uchar>(const Mat8u& src, Mat8u& dst) {
    return cvt_yuv_transform<false, false, true>(src, dst);
}

template <>
void cvt_yuv2rgb_yv12<uchar>(const Mat8u& src, Mat8u& dst) {
    return cvt_yuv_transform<true, true, false>(src, dst);
}

template <>
void cvt_yuv2bgr_yv12<uchar>(const Mat8u& src, Mat8u& dst) {
    return cvt_yuv_transform<false, true, false>(src, dst);
}

template <>
void cvt_yuv2rgb_yu12<uchar>(const Mat8u& src, Mat8u& dst) {
    return cvt_yuv_transform<true, true, true>(src, dst);
}

template <>
void cvt_yuv2bgr_yu12<uchar>(const Mat8u& src, Mat8u& dst) {
    return cvt_yuv_transform<false, true, true>(src, dst);
}

template <typename T>
void cvt_bt601_yuv(const megcv::Mat<T>& src, megcv::Mat<T>& dst,
                   param::CvtColor::Mode mode) {
    MEGDNN_MARK_USED_VAR(src);
    MEGDNN_MARK_USED_VAR(dst);
    MEGDNN_MARK_USED_VAR(mode);
    megdnn_throw("Unsupport dtype for real yuv");
}

template <>
void cvt_bt601_yuv<uchar>(const megcv::Mat<uchar>& src, megcv::Mat<uchar>& dst,
                          param::CvtColor::Mode mode) {
    using Mode = param::CvtColor::Mode;
    switch (mode) {
        case Mode::BT601_YUV2RGB_NV21:
            MIDOUT_BEGIN(megdnn_arm_cvt_bt601_yuv, midout_iv(0)) {
                return cvt_BT601_yuv_transform<true, false, false>(src, dst);
            }
            MIDOUT_END();
        case Mode::BT601_YUV2BGR_NV21:
            MIDOUT_BEGIN(megdnn_arm_cvt_bt601_yuv, midout_iv(1)) {
                return cvt_BT601_yuv_transform<false, false, false>(src, dst);
            }
            MIDOUT_END();
        case Mode::BT601_YUV2RGB_NV12:
            MIDOUT_BEGIN(megdnn_arm_cvt_bt601_yuv, midout_iv(2)) {
                return cvt_BT601_yuv_transform<true, false, true>(src, dst);
            }
            MIDOUT_END();
        case Mode::BT601_YUV2BGR_NV12:
            MIDOUT_BEGIN(megdnn_arm_cvt_bt601_yuv, midout_iv(3)) {
                return cvt_BT601_yuv_transform<false, false, true>(src, dst);
            }
            MIDOUT_END();
        case Mode::BT601_YUV2RGB_YV12:
            MIDOUT_BEGIN(megdnn_arm_cvt_bt601_yuv, midout_iv(4)) {
                return cvt_BT601_yuv_transform<true, true, false>(src, dst);
            }
            MIDOUT_END();
        case Mode::BT601_YUV2BGR_YV12:
            MIDOUT_BEGIN(megdnn_arm_cvt_bt601_yuv, midout_iv(5)) {
                return cvt_BT601_yuv_transform<false, true, false>(src, dst);
            }
            MIDOUT_END();
        case Mode::BT601_YUV2RGB_YU12:
            MIDOUT_BEGIN(megdnn_arm_cvt_bt601_yuv, midout_iv(6)) {
                return cvt_BT601_yuv_transform<true, true, true>(src, dst);
            }
            MIDOUT_END();
        case Mode::BT601_YUV2BGR_YU12:
            MIDOUT_BEGIN(megdnn_arm_cvt_bt601_yuv, midout_iv(7)) {
                return cvt_BT601_yuv_transform<false, true, true>(src, dst);
            }
            MIDOUT_END();
        default:
            megdnn_throw("unknown mode for real yuv.");
    }
}

template <typename T>
void CvtColorImpl::cvt_color_exec(const TensorND& src_tensor,
                                  const TensorND& dst_tensor) {
    auto mode = param().mode;
    for (size_t i = 0; i < src_tensor.layout.shape[0]; ++i) {
        Mat<T> src = TensorND2Mat<T>(src_tensor, i);
        Mat<T> dst = TensorND2Mat<T>(dst_tensor, i);
        switch (mode) {
            case Param::Mode::RGB2GRAY:
                MIDOUT_BEGIN(megdnn_arm_cvtcolor_cases, midout_iv(0)) {
                    cvt_rgb2gray<T>(src, dst);
                }
                MIDOUT_END();
                break;
            case Param::Mode::RGB2YUV:
                MIDOUT_BEGIN(megdnn_arm_cvtcolor_cases, midout_iv(1)) {
                    cvt_rgb2yuv<T>(src, dst);
                }
                MIDOUT_END();
                break;
            case Param::Mode::YUV2RGB:
                MIDOUT_BEGIN(megdnn_arm_cvtcolor_cases, midout_iv(2)) {
                    cvt_yuv2rgb<T>(src, dst);
                }
                MIDOUT_END();
                break;
            case Param::Mode::GRAY2RGB:
                MIDOUT_BEGIN(megdnn_arm_cvtcolor_cases, midout_iv(3)) {
                    cvt_gray2rgb<T>(src, dst);
                }
                MIDOUT_END();
                break;
            case Param::Mode::RGBA2RGB:
                MIDOUT_BEGIN(megdnn_arm_cvtcolor_cases, midout_iv(4)) {
                    cvt_rgba2rgb<T>(src, dst);
                }
                MIDOUT_END();
                break;
            case Param::Mode::RGBA2BGR:
                MIDOUT_BEGIN(megdnn_arm_cvtcolor_cases, midout_iv(5)) {
                    cvt_rgba2bgr<T>(src, dst);
                }
                MIDOUT_END();
                break;
            case Param::Mode::RGBA2GRAY:
                MIDOUT_BEGIN(megdnn_arm_cvtcolor_cases, midout_iv(6)) {
                    cvt_rgba2gray<T>(src, dst);
                }
                MIDOUT_END();
                break;
            case Param::Mode::RGB2BGR:
                MIDOUT_BEGIN(megdnn_arm_cvtcolor_cases, midout_iv(7)) {
                    cvt_rgb2bgr<T>(src, dst);
                }
                MIDOUT_END();
                break;
            case Param::Mode::BGR2GRAY:
                MIDOUT_BEGIN(megdnn_arm_cvtcolor_cases, midout_iv(8)) {
                    cvt_bgr2gray<T>(src, dst);
                }
                MIDOUT_END();
                break;
            case Param::Mode::BGR2RGB:
                MIDOUT_BEGIN(megdnn_arm_cvtcolor_cases, midout_iv(9)) {
                    cvt_bgr2rgb<T>(src, dst);
                }
                MIDOUT_END();
                break;
            case Param::Mode::YUV2GRAY_NV21:
            case Param::Mode::YUV2GRAY_NV12:
            case Param::Mode::YUV2GRAY_YV12:
            case Param::Mode::YUV2GRAY_YU12:
                cvt_yuv2gray_nv21<T>(src, dst);
                break;
            case Param::Mode::YUV2RGB_NV21:
            case Param::Mode::YCrCb2RGB:
                MIDOUT_BEGIN(megdnn_arm_cvtcolor_cases, midout_iv(10)) {
                    cvt_yuv2rgb_nv21<T>(src, dst);
                }
                MIDOUT_END();
                break;
            case Param::Mode::YUV2BGR_NV21:
            case Param::Mode::YCrCb2BGR:
                MIDOUT_BEGIN(megdnn_arm_cvtcolor_cases, midout_iv(11)) {
                    cvt_yuv2bgr_nv21<T>(src, dst);
                }
                MIDOUT_END();
                break;
            case Param::Mode::YUV2RGB_NV12:
                MIDOUT_BEGIN(megdnn_arm_cvtcolor_cases, midout_iv(12)) {
                    cvt_yuv2rgb_nv12<T>(src, dst);
                }
                MIDOUT_END();
                break;
            case Param::Mode::YUV2BGR_NV12:
                MIDOUT_BEGIN(megdnn_arm_cvtcolor_cases, midout_iv(13)) {
                    cvt_yuv2bgr_nv12<T>(src, dst);
                }
                MIDOUT_END();
                break;
            case Param::Mode::YUV2RGB_YV12:
                MIDOUT_BEGIN(megdnn_arm_cvtcolor_cases, midout_iv(14)) {
                    cvt_yuv2rgb_yv12<T>(src, dst);
                }
                MIDOUT_END();
                break;
            case Param::Mode::YUV2BGR_YV12:
                MIDOUT_BEGIN(megdnn_arm_cvtcolor_cases, midout_iv(15)) {
                    cvt_yuv2bgr_yv12<T>(src, dst);
                }
                MIDOUT_END();
                break;
            case Param::Mode::YUV2RGB_YU12:
                MIDOUT_BEGIN(megdnn_arm_cvtcolor_cases, midout_iv(16)) {
                    cvt_yuv2rgb_yu12<T>(src, dst);
                }
                MIDOUT_END();
                break;
            case Param::Mode::YUV2BGR_YU12:
                MIDOUT_BEGIN(megdnn_arm_cvtcolor_cases, midout_iv(17)) {
                    cvt_yuv2bgr_yu12<T>(src, dst);
                }
                MIDOUT_END();
                break;
            case Param::Mode::BT601_YUV2BGR_NV12:
            case Param::Mode::BT601_YUV2RGB_NV12:
            case Param::Mode::BT601_YUV2BGR_NV21:
            case Param::Mode::BT601_YUV2RGB_NV21:
            case Param::Mode::BT601_YUV2RGB_YU12:
            case Param::Mode::BT601_YUV2BGR_YU12:
            case Param::Mode::BT601_YUV2RGB_YV12:
            case Param::Mode::BT601_YUV2BGR_YV12:
                MIDOUT_BEGIN(megdnn_arm_cvtcolor_cases, midout_iv(18)) {
                    cvt_bt601_yuv<T>(src, dst, mode);
                }
                MIDOUT_END();
                break;

            default:
                megdnn_throw("Can not find property cvt_color operator.");
        }
    }
}
void CvtColorImpl::exec(_megdnn_tensor_in src, _megdnn_tensor_in dst,
                        _megdnn_workspace workspace) {
    using namespace megcv;
    check_exec(src.layout, dst.layout, workspace.size);
    if (dst.layout.dtype == dtype::Float32()) {
        MIDOUT_BEGIN(megdnn_arm_cvtcolor MEGDNN_COMMA midout_iv(0)) {
        MEGDNN_DISPATCH_CPU_KERN_OPR(cvt_color_exec<float>(src, dst));
        } MIDOUT_END();
    } else if (dst.layout.dtype == dtype::Uint8()) {
        MIDOUT_BEGIN(megdnn_arm_cvtcolor MEGDNN_COMMA midout_iv(1)) {
            MEGDNN_DISPATCH_CPU_KERN_OPR(cvt_color_exec<uchar>(src, dst));
        } MIDOUT_END();
    } else { megdnn_throw("Unsupported datatype of CvtColor optr."); };
}

}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
