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
 * \file dnn/src/x86/cvt_color/opr_impl.cpp
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

#include "src/x86/cvt_color/opr_impl.h"
#include "src/x86/utils.h"
#include "src/common/cv/common.h"
#include "src/common/cv/cvt_color.h"
#include "src/common/cv/helper.h"
#include "src/common/utils.h"
#include "src/naive/handle.h"

#include <cstring>

#include <pmmintrin.h>
#include <smmintrin.h>
#include <tmmintrin.h>
#include <xmmintrin.h>

namespace megdnn {
namespace x86 {

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
MEGDNN_ATTRIBUTE_TARGET("sse4.2")
void cvt_yuv_transform(const Mat8u& src, Mat8u& dst) {
    __m128i out0, out1, out2;
    __m128i Y0, Y1, VU, V0, V1, V3, U0, U1, U3;
    __m128i Y00, Y01, Y02, Y03;
    __m128i RV0, GUV0, BU0;
    __m128i RV1, GUV1, BU1;
    __m128i RV2, GUV2, BU2;
    __m128i RV3, GUV3, BU3;
    __m128i R0, G0, B0;
    __m128i R1, G1, B1;
    __m128i R2, G2, B2;
    __m128i R3, G3, B3;
    __m128i RG0, RG1;
    __m128i BG0, BG1;
    __m128i out_temp0, out_temp1;

    __m128i v128 = _mm_set1_epi16(128);
    __m128i v359 = _mm_set1_epi32(359);
    __m128i v88 = _mm_set1_epi32(88);
    __m128i v183 = _mm_set1_epi32(183);
    __m128i v454 = _mm_set1_epi32(454);

    __m128i _shuff_0 =
            _mm_set_epi8(15, 13, 11, 9, 7, 5, 3, 1, 14, 12, 10, 8, 6, 4, 2, 0);

    __m128i _shuff_1 =
            _mm_set_epi8(10, 0, 9, 8, 0, 7, 6, 0, 5, 4, 0, 3, 2, 0, 1, 0);
    __m128i _shuff_3 =
            _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 14, 0, 13, 12, 0, 11);
    __m128i _shuff_4 =
            _mm_set_epi8(5, 4, 0, 3, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    __m128i _shuff_6 =
            _mm_set_epi8(0, 15, 14, 0, 13, 12, 0, 11, 10, 0, 9, 8, 0, 7, 6, 0);

    __m128i _shuff_2 =
            _mm_set_epi8(0, 4, 0, 0, 3, 0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0);
    __m128i _shuff_5 =
            _mm_set_epi8(0, 0, 9, 0, 0, 8, 0, 0, 7, 0, 0, 6, 0, 0, 5, 0);
    __m128i _shuff_7 =
            _mm_set_epi8(15, 0, 0, 14, 0, 0, 13, 0, 0, 12, 0, 0, 11, 0, 0, 10);

    __m128i _blend_12 = _mm_set_epi8(0, -128, 0, 0, -128, 0, 0, -128, 0, 0,
                                     -128, 0, 0, -128, 0, 0);
    __m128i _blend_34 = _mm_set_epi8(-128, -128, -128, -128, -128, -128, -128,
                                     -128, 0, 0, 0, 0, 0, 0, 0, 0);
    __m128i _blend_345 = _mm_set_epi8(0, 0, -128, 0, 0, -128, 0, 0, -128, 0, 0,
                                      -128, 0, 0, -128, 0);
    __m128i _blend_67 = _mm_set_epi8(-128, 0, 0, -128, 0, 0, -128, 0, 0, -128,
                                     0, 0, -128, 0, 0, -128);

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

        for (; c <= (int)(width - 16); c += 16) {
            Y0 = _mm_lddqu_si128((__m128i*)(pY + c));
            Y1 = _mm_lddqu_si128((__m128i*)(pY + src_step + c));
            if (is_planar) {
                V0 = _mm_lddqu_si128((__m128i*)(pV + c / 2));
                V0 = _mm_cvtepu8_epi16(V0);
                U0 = _mm_lddqu_si128((__m128i*)(pU + c / 2));
                U0 = _mm_cvtepu8_epi16(U0);
            } else {
                if (is_uv) {
                    VU = _mm_lddqu_si128((__m128i*)(pU + c));
                    VU = _mm_shuffle_epi8(VU, _shuff_0);
                    U0 = _mm_cvtepu8_epi16(VU);
                    VU = _mm_shuffle_epi32(VU, 14);
                    V0 = _mm_cvtepu8_epi16(VU);
                } else {
                    VU = _mm_lddqu_si128((__m128i*)(pV + c));
                    VU = _mm_shuffle_epi8(VU, _shuff_0);
                    V0 = _mm_cvtepu8_epi16(VU);
                    VU = _mm_shuffle_epi32(VU, 14);
                    U0 = _mm_cvtepu8_epi16(VU);
                }
            }

            V0 = _mm_sub_epi16(V0, v128);
            U0 = _mm_sub_epi16(U0, v128);

            V1 = _mm_cvtepi16_epi32(V0);
            V0 = _mm_shuffle_epi32(V0, 14);
            V3 = _mm_cvtepi16_epi32(V0);

            U1 = _mm_cvtepi16_epi32(U0);
            U0 = _mm_shuffle_epi32(U0, 14);
            U3 = _mm_cvtepi16_epi32(U0);

            RV1 = _mm_srai_epi32(_mm_mullo_epi32(V1, v359), 8);
            RV3 = _mm_srai_epi32(_mm_mullo_epi32(V3, v359), 8);

            RV0 = _mm_shuffle_epi32(RV1, 80);
            RV1 = _mm_shuffle_epi32(RV1, 250);
            RV2 = _mm_shuffle_epi32(RV3, 80);
            RV3 = _mm_shuffle_epi32(RV3, 250);

            GUV1 = _mm_srai_epi32(_mm_add_epi32(_mm_mullo_epi32(U1, v88),
                                                _mm_mullo_epi32(V1, v183)),
                                  8);
            GUV3 = _mm_srai_epi32(_mm_add_epi32(_mm_mullo_epi32(U3, v88),
                                                _mm_mullo_epi32(V3, v183)),
                                  8);
            GUV0 = _mm_shuffle_epi32(GUV1, 80);
            GUV1 = _mm_shuffle_epi32(GUV1, 250);
            GUV2 = _mm_shuffle_epi32(GUV3, 80);
            GUV3 = _mm_shuffle_epi32(GUV3, 250);

            BU1 = _mm_srai_epi32(_mm_mullo_epi32(U1, v454), 8);
            BU3 = _mm_srai_epi32(_mm_mullo_epi32(U3, v454), 8);
            BU0 = _mm_shuffle_epi32(BU1, 80);
            BU1 = _mm_shuffle_epi32(BU1, 250);
            BU2 = _mm_shuffle_epi32(BU3, 80);
            BU3 = _mm_shuffle_epi32(BU3, 250);

            Y01 = _mm_cvtepu8_epi16(Y0);
            Y0 = _mm_shuffle_epi32(Y0, 14);
            Y03 = _mm_cvtepu8_epi16(Y0);

            Y00 = _mm_cvtepu16_epi32(Y01);
            Y01 = _mm_shuffle_epi32(Y01, 14);
            Y01 = _mm_cvtepu16_epi32(Y01);

            Y02 = _mm_cvtepu16_epi32(Y03);
            Y03 = _mm_shuffle_epi32(Y03, 14);
            Y03 = _mm_cvtepu16_epi32(Y03);

            R0 = _mm_add_epi32(Y00, RV0);
            R1 = _mm_add_epi32(Y01, RV1);
            R2 = _mm_add_epi32(Y02, RV2);
            R3 = _mm_add_epi32(Y03, RV3);
            G0 = _mm_sub_epi32(Y00, GUV0);
            G1 = _mm_sub_epi32(Y01, GUV1);
            G2 = _mm_sub_epi32(Y02, GUV2);
            G3 = _mm_sub_epi32(Y03, GUV3);
            B0 = _mm_add_epi32(Y00, BU0);
            B1 = _mm_add_epi32(Y01, BU1);
            B2 = _mm_add_epi32(Y02, BU2);
            B3 = _mm_add_epi32(Y03, BU3);

            R0 = _mm_packs_epi32(R0, R1);
            R2 = _mm_packs_epi32(R2, R3);
            R0 = _mm_packus_epi16(R0, R2);
            G0 = _mm_packs_epi32(G0, G1);
            G2 = _mm_packs_epi32(G2, G3);
            G0 = _mm_packus_epi16(G0, G2);
            B0 = _mm_packs_epi32(B0, B1);
            B2 = _mm_packs_epi32(B2, B3);
            B0 = _mm_packus_epi16(B0, B2);

            if (rgb) {
                RG0 = _mm_unpacklo_epi8(R0, G0);
                RG1 = _mm_unpackhi_epi8(R0, G0);

                out_temp0 = _mm_shuffle_epi8(RG0, _shuff_1);
                out_temp1 = _mm_shuffle_epi8(B0, _shuff_2);
                out0 = _mm_blendv_epi8(out_temp0, out_temp1, _blend_12);

                out_temp0 = _mm_shuffle_epi8(RG0, _shuff_3);
                out_temp1 = _mm_shuffle_epi8(RG1, _shuff_4);
                out_temp0 = _mm_blendv_epi8(out_temp0, out_temp1, _blend_34);
                out_temp1 = _mm_shuffle_epi8(B0, _shuff_5);
                out1 = _mm_blendv_epi8(out_temp0, out_temp1, _blend_345);

                out_temp0 = _mm_shuffle_epi8(RG1, _shuff_6);
                out_temp1 = _mm_shuffle_epi8(B0, _shuff_7);
                out2 = _mm_blendv_epi8(out_temp0, out_temp1, _blend_67);
            } else {
                BG0 = _mm_unpacklo_epi8(B0, G0);
                BG1 = _mm_unpackhi_epi8(B0, G0);

                out_temp0 = _mm_shuffle_epi8(BG0, _shuff_1);
                out_temp1 = _mm_shuffle_epi8(R0, _shuff_2);
                out0 = _mm_blendv_epi8(out_temp0, out_temp1, _blend_12);

                out_temp0 = _mm_shuffle_epi8(BG0, _shuff_3);
                out_temp1 = _mm_shuffle_epi8(BG1, _shuff_4);
                out_temp0 = _mm_blendv_epi8(out_temp0, out_temp1, _blend_34);
                out_temp1 = _mm_shuffle_epi8(R0, _shuff_5);
                out1 = _mm_blendv_epi8(out_temp0, out_temp1, _blend_345);

                out_temp0 = _mm_shuffle_epi8(BG1, _shuff_6);
                out_temp1 = _mm_shuffle_epi8(R0, _shuff_7);
                out2 = _mm_blendv_epi8(out_temp0, out_temp1, _blend_67);
            }

            _mm_storeu_si128((__m128i*)(dst0 + index0), out0);
            index0 += 16;
            _mm_storeu_si128((__m128i*)(dst0 + index0), out1);
            index0 += 16;
            _mm_storeu_si128((__m128i*)(dst0 + index0), out2);
            index0 += 16;

            Y01 = _mm_cvtepu8_epi16(Y1);
            Y1 = _mm_shuffle_epi32(Y1, 14);
            Y03 = _mm_cvtepu8_epi16(Y1);

            Y00 = _mm_cvtepu16_epi32(Y01);
            Y01 = _mm_shuffle_epi32(Y01, 14);
            Y01 = _mm_cvtepu16_epi32(Y01);

            Y02 = _mm_cvtepu16_epi32(Y03);
            Y03 = _mm_shuffle_epi32(Y03, 14);
            Y03 = _mm_cvtepu16_epi32(Y03);

            R0 = _mm_add_epi32(Y00, RV0);
            R1 = _mm_add_epi32(Y01, RV1);
            R2 = _mm_add_epi32(Y02, RV2);
            R3 = _mm_add_epi32(Y03, RV3);
            G0 = _mm_sub_epi32(Y00, GUV0);
            G1 = _mm_sub_epi32(Y01, GUV1);
            G2 = _mm_sub_epi32(Y02, GUV2);
            G3 = _mm_sub_epi32(Y03, GUV3);
            B0 = _mm_add_epi32(Y00, BU0);
            B1 = _mm_add_epi32(Y01, BU1);
            B2 = _mm_add_epi32(Y02, BU2);
            B3 = _mm_add_epi32(Y03, BU3);

            R0 = _mm_packs_epi32(R0, R1);
            R2 = _mm_packs_epi32(R2, R3);
            R0 = _mm_packus_epi16(R0, R2);
            G0 = _mm_packs_epi32(G0, G1);
            G2 = _mm_packs_epi32(G2, G3);
            G0 = _mm_packus_epi16(G0, G2);
            B0 = _mm_packs_epi32(B0, B1);
            B2 = _mm_packs_epi32(B2, B3);
            B0 = _mm_packus_epi16(B0, B2);

            if (rgb) {
                RG0 = _mm_unpacklo_epi8(R0, G0);
                RG1 = _mm_unpackhi_epi8(R0, G0);

                out_temp0 = _mm_shuffle_epi8(RG0, _shuff_1);
                out_temp1 = _mm_shuffle_epi8(B0, _shuff_2);
                out0 = _mm_blendv_epi8(out_temp0, out_temp1, _blend_12);

                out_temp0 = _mm_shuffle_epi8(RG0, _shuff_3);
                out_temp1 = _mm_shuffle_epi8(RG1, _shuff_4);
                out_temp0 = _mm_blendv_epi8(out_temp0, out_temp1, _blend_34);
                out_temp1 = _mm_shuffle_epi8(B0, _shuff_5);
                out1 = _mm_blendv_epi8(out_temp0, out_temp1, _blend_345);

                out_temp0 = _mm_shuffle_epi8(RG1, _shuff_6);
                out_temp1 = _mm_shuffle_epi8(B0, _shuff_7);
                out2 = _mm_blendv_epi8(out_temp0, out_temp1, _blend_67);
            } else {
                BG0 = _mm_unpacklo_epi8(B0, G0);
                BG1 = _mm_unpackhi_epi8(B0, G0);

                out_temp0 = _mm_shuffle_epi8(BG0, _shuff_1);
                out_temp1 = _mm_shuffle_epi8(R0, _shuff_2);
                out0 = _mm_blendv_epi8(out_temp0, out_temp1, _blend_12);

                out_temp0 = _mm_shuffle_epi8(BG0, _shuff_3);
                out_temp1 = _mm_shuffle_epi8(BG1, _shuff_4);
                out_temp0 = _mm_blendv_epi8(out_temp0, out_temp1, _blend_34);
                out_temp1 = _mm_shuffle_epi8(R0, _shuff_5);
                out1 = _mm_blendv_epi8(out_temp0, out_temp1, _blend_345);

                out_temp0 = _mm_shuffle_epi8(BG1, _shuff_6);
                out_temp1 = _mm_shuffle_epi8(R0, _shuff_7);
                out2 = _mm_blendv_epi8(out_temp0, out_temp1, _blend_67);
            }

            _mm_storeu_si128((__m128i*)(dst1 + index1), out0);
            index1 += 16;
            _mm_storeu_si128((__m128i*)(dst1 + index1), out1);
            index1 += 16;
            _mm_storeu_si128((__m128i*)(dst1 + index1), out2);
            index1 += 16;
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
 * \brief x86 intrinsic implementation of real yuv to rgb or bgr.
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
MEGDNN_ATTRIBUTE_TARGET("sse4.2")
void cvt_BT601_yuv_transform(const Mat8u& src, Mat8u& dst) {
    typedef unsigned char uint8;

    size_t height = dst.rows();
    size_t width = dst.cols();
    size_t src_step = src.step();
    const uint8* pY = src.ptr();
    const uint8* pU;
    const uint8* pV;

    if (is_uv) {
        pU = src.ptr(height);
        pV = src.ptr(height + height / 4);
    } else {
        pV = src.ptr(height);
        pU = src.ptr(height + height / 4);
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

    __m128i v32_YG257 = _mm_set1_epi32(0x0101 * YG),
            v32_UB = _mm_set1_epi32(UB), v32_UG = _mm_set1_epi32(UG),
            v32_VG = _mm_set1_epi32(VG), v32_BB = _mm_set1_epi32(BB),
            v32_BG = _mm_set1_epi32(BG), v32_BR = _mm_set1_epi32(BR),
            v32_VR = _mm_set1_epi32(VR);
    __m128i R0, G0, B0, R1, G1, B1, R2, G2, B2, R3, G3, B3;

    __m128i _shuff_0 =
            _mm_set_epi8(15, 13, 11, 9, 7, 5, 3, 1, 14, 12, 10, 8, 6, 4, 2, 0);
    __m128i _shuff_1 =
            _mm_set_epi8(10, 0, 9, 8, 0, 7, 6, 0, 5, 4, 0, 3, 2, 0, 1, 0);
    __m128i _shuff_3 =
            _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 14, 0, 13, 12, 0, 11);
    __m128i _shuff_4 =
            _mm_set_epi8(5, 4, 0, 3, 2, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    __m128i _shuff_6 =
            _mm_set_epi8(0, 15, 14, 0, 13, 12, 0, 11, 10, 0, 9, 8, 0, 7, 6, 0);

    __m128i _shuff_2 =
            _mm_set_epi8(0, 4, 0, 0, 3, 0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0);
    __m128i _shuff_5 =
            _mm_set_epi8(0, 0, 9, 0, 0, 8, 0, 0, 7, 0, 0, 6, 0, 0, 5, 0);
    __m128i _shuff_7 =
            _mm_set_epi8(15, 0, 0, 14, 0, 0, 13, 0, 0, 12, 0, 0, 11, 0, 0, 10);

    __m128i _blend_12 = _mm_set_epi8(0, -128, 0, 0, -128, 0, 0, -128, 0, 0,
                                     -128, 0, 0, -128, 0, 0);
    __m128i _blend_34 = _mm_set_epi8(-128, -128, -128, -128, -128, -128, -128,
                                     -128, 0, 0, 0, 0, 0, 0, 0, 0);
    __m128i _blend_345 = _mm_set_epi8(0, 0, -128, 0, 0, -128, 0, 0, -128, 0, 0,
                                      -128, 0, 0, -128, 0);
    __m128i _blend_67 = _mm_set_epi8(-128, 0, 0, -128, 0, 0, -128, 0, 0, -128,
                                     0, 0, -128, 0, 0, -128);

    __m128i Y0, Y1, U0, V0;
    __m128i VU;
    __m128i Y00, Y01, Y02, Y03;
    __m128i V1, V3, U1, U3;

    __m128i RG0, RG1;
    __m128i BG0, BG1;
    __m128i out_temp0, out_temp1;
    __m128i out0, out1, out2;
    __m128i RV0, GUV0, BU0;
    __m128i RV1, GUV1, BU1;
    __m128i RV2, GUV2, BU2;
    __m128i RV3, GUV3, BU3;

    for (size_t r = 0; r < height; r += 2, pY += (src_step << 1)) {
        unsigned char* dst0 = dst.ptr(r);
        unsigned char* dst1 = dst.ptr(r + 1);
        size_t index0 = 0;
        size_t index1 = 0;

        size_t c = 0;
        for (; c + 16 <= width; c += 16) {
            Y0 = _mm_lddqu_si128((__m128i*)(pY + c));
            Y1 = _mm_lddqu_si128((__m128i*)(pY + src_step + c));
            if (is_planar) {
                V0 = _mm_lddqu_si128((__m128i*)(pV + c / 2));
                V0 = _mm_cvtepu8_epi16(V0);
                U0 = _mm_lddqu_si128((__m128i*)(pU + c / 2));
                U0 = _mm_cvtepu8_epi16(U0);
            } else {
                if (is_uv) {
                    VU = _mm_lddqu_si128((__m128i*)(pU + c));
                    VU = _mm_shuffle_epi8(VU, _shuff_0);
                    U0 = _mm_cvtepu8_epi16(VU);
                    VU = _mm_shuffle_epi32(VU, 14);
                    V0 = _mm_cvtepu8_epi16(VU);
                } else {
                    VU = _mm_lddqu_si128((__m128i*)(pV + c));
                    VU = _mm_shuffle_epi8(VU, _shuff_0);
                    V0 = _mm_cvtepu8_epi16(VU);
                    VU = _mm_shuffle_epi32(VU, 14);
                    U0 = _mm_cvtepu8_epi16(VU);
                }
            }

            // read 8Y 8Y
            //      8Y 8Y
            //      8U 8V
            V1 = _mm_cvtepi16_epi32(V0);
            V0 = _mm_shuffle_epi32(V0, 14);
            V3 = _mm_cvtepi16_epi32(V0);

            U1 = _mm_cvtepi16_epi32(U0);
            U0 = _mm_shuffle_epi32(U0, 14);
            U3 = _mm_cvtepi16_epi32(U0);

            BU1 = _mm_sub_epi32(v32_BB, _mm_mullo_epi32(U1, v32_UB));
            BU3 = _mm_sub_epi32(v32_BB, _mm_mullo_epi32(U3, v32_UB));
            BU0 = _mm_shuffle_epi32(BU1, 80);
            BU1 = _mm_shuffle_epi32(BU1, 250);
            BU2 = _mm_shuffle_epi32(BU3, 80);
            BU3 = _mm_shuffle_epi32(BU3, 250);

            GUV1 = _mm_sub_epi32(v32_BG,
                                 _mm_add_epi32(_mm_mullo_epi32(U1, v32_UG),
                                               _mm_mullo_epi32(V1, v32_VG)));
            GUV3 = _mm_sub_epi32(v32_BG,
                                 _mm_add_epi32(_mm_mullo_epi32(U3, v32_UG),
                                               _mm_mullo_epi32(V3, v32_VG)));
            GUV0 = _mm_shuffle_epi32(GUV1, 80);
            GUV1 = _mm_shuffle_epi32(GUV1, 250);
            GUV2 = _mm_shuffle_epi32(GUV3, 80);
            GUV3 = _mm_shuffle_epi32(GUV3, 250);

            RV1 = _mm_sub_epi32(v32_BR, _mm_mullo_epi32(V1, v32_VR));
            RV3 = _mm_sub_epi32(v32_BR, _mm_mullo_epi32(V3, v32_VR));
            RV0 = _mm_shuffle_epi32(RV1, 80);
            RV1 = _mm_shuffle_epi32(RV1, 250);
            RV2 = _mm_shuffle_epi32(RV3, 80);
            RV3 = _mm_shuffle_epi32(RV3, 250);

            Y01 = _mm_cvtepu8_epi16(Y0);
            Y0 = _mm_shuffle_epi32(Y0, 14);
            Y03 = _mm_cvtepu8_epi16(Y0);

            Y00 = _mm_cvtepi16_epi32(Y01);
            Y01 = _mm_shuffle_epi32(Y01, 14);
            Y01 = _mm_cvtepi16_epi32(Y01);

            Y02 = _mm_cvtepu16_epi32(Y03);
            Y03 = _mm_shuffle_epi32(Y03, 14);
            Y03 = _mm_cvtepu16_epi32(Y03);

            Y00 = _mm_srai_epi32(_mm_mullo_epi32(Y00, v32_YG257), 16);
            Y01 = _mm_srai_epi32(_mm_mullo_epi32(Y01, v32_YG257), 16);
            Y02 = _mm_srai_epi32(_mm_mullo_epi32(Y02, v32_YG257), 16);
            Y03 = _mm_srai_epi32(_mm_mullo_epi32(Y03, v32_YG257), 16);

            // line 0, 0:3
            B0 = _mm_srai_epi32(_mm_add_epi32(Y00, BU0), 6);
            G0 = _mm_srai_epi32(_mm_add_epi32(Y00, GUV0), 6);
            R0 = _mm_srai_epi32(_mm_add_epi32(Y00, RV0), 6);

            // line 0, 4:7
            B1 = _mm_srai_epi32(_mm_add_epi32(Y01, BU1), 6);
            G1 = _mm_srai_epi32(_mm_add_epi32(Y01, GUV1), 6);
            R1 = _mm_srai_epi32(_mm_add_epi32(Y01, RV1), 6);

            // line 0, 8:11
            B2 = _mm_srai_epi32(_mm_add_epi32(Y02, BU2), 6);
            G2 = _mm_srai_epi32(_mm_add_epi32(Y02, GUV2), 6);
            R2 = _mm_srai_epi32(_mm_add_epi32(Y02, RV2), 6);

            // line 0, 12:15
            B3 = _mm_srai_epi32(_mm_add_epi32(Y03, BU3), 6);
            G3 = _mm_srai_epi32(_mm_add_epi32(Y03, GUV3), 6);
            R3 = _mm_srai_epi32(_mm_add_epi32(Y03, RV3), 6);

            R0 = _mm_packs_epi32(R0, R1);
            R2 = _mm_packs_epi32(R2, R3);
            R0 = _mm_packus_epi16(R0, R2);
            G0 = _mm_packs_epi32(G0, G1);
            G2 = _mm_packs_epi32(G2, G3);
            G0 = _mm_packus_epi16(G0, G2);
            B0 = _mm_packs_epi32(B0, B1);
            B2 = _mm_packs_epi32(B2, B3);
            B0 = _mm_packus_epi16(B0, B2);

            if (rgb) {
                RG0 = _mm_unpacklo_epi8(R0, G0);
                RG1 = _mm_unpackhi_epi8(R0, G0);

                out_temp0 = _mm_shuffle_epi8(RG0, _shuff_1);
                out_temp1 = _mm_shuffle_epi8(B0, _shuff_2);
                out0 = _mm_blendv_epi8(out_temp0, out_temp1, _blend_12);

                out_temp0 = _mm_shuffle_epi8(RG0, _shuff_3);
                out_temp1 = _mm_shuffle_epi8(RG1, _shuff_4);
                out_temp0 = _mm_blendv_epi8(out_temp0, out_temp1, _blend_34);
                out_temp1 = _mm_shuffle_epi8(B0, _shuff_5);
                out1 = _mm_blendv_epi8(out_temp0, out_temp1, _blend_345);

                out_temp0 = _mm_shuffle_epi8(RG1, _shuff_6);
                out_temp1 = _mm_shuffle_epi8(B0, _shuff_7);
                out2 = _mm_blendv_epi8(out_temp0, out_temp1, _blend_67);
            } else {
                BG0 = _mm_unpacklo_epi8(B0, G0);
                BG1 = _mm_unpackhi_epi8(B0, G0);

                out_temp0 = _mm_shuffle_epi8(BG0, _shuff_1);
                out_temp1 = _mm_shuffle_epi8(R0, _shuff_2);
                out0 = _mm_blendv_epi8(out_temp0, out_temp1, _blend_12);

                out_temp0 = _mm_shuffle_epi8(BG0, _shuff_3);
                out_temp1 = _mm_shuffle_epi8(BG1, _shuff_4);
                out_temp0 = _mm_blendv_epi8(out_temp0, out_temp1, _blend_34);
                out_temp1 = _mm_shuffle_epi8(R0, _shuff_5);
                out1 = _mm_blendv_epi8(out_temp0, out_temp1, _blend_345);

                out_temp0 = _mm_shuffle_epi8(BG1, _shuff_6);
                out_temp1 = _mm_shuffle_epi8(R0, _shuff_7);
                out2 = _mm_blendv_epi8(out_temp0, out_temp1, _blend_67);
            }

            _mm_storeu_si128((__m128i*)(dst0 + index0), out0);
            index0 += 16;
            _mm_storeu_si128((__m128i*)(dst0 + index0), out1);
            index0 += 16;
            _mm_storeu_si128((__m128i*)(dst0 + index0), out2);
            index0 += 16;

            Y01 = _mm_cvtepu8_epi16(Y1);
            Y1 = _mm_shuffle_epi32(Y1, 14);
            Y03 = _mm_cvtepu8_epi16(Y1);

            Y00 = _mm_cvtepi16_epi32(Y01);
            Y01 = _mm_shuffle_epi32(Y01, 14);
            Y01 = _mm_cvtepi16_epi32(Y01);

            Y02 = _mm_cvtepu16_epi32(Y03);
            Y03 = _mm_shuffle_epi32(Y03, 14);
            Y03 = _mm_cvtepu16_epi32(Y03);

            Y00 = _mm_srai_epi32(_mm_mullo_epi32(Y00, v32_YG257), 16);
            Y01 = _mm_srai_epi32(_mm_mullo_epi32(Y01, v32_YG257), 16);
            Y02 = _mm_srai_epi32(_mm_mullo_epi32(Y02, v32_YG257), 16);
            Y03 = _mm_srai_epi32(_mm_mullo_epi32(Y03, v32_YG257), 16);

            // line 1, 0:3
            B0 = _mm_srai_epi32(_mm_add_epi32(Y00, BU0), 6);
            G0 = _mm_srai_epi32(_mm_add_epi32(Y00, GUV0), 6);
            R0 = _mm_srai_epi32(_mm_add_epi32(Y00, RV0), 6);

            // line 1, 4:7
            B1 = _mm_srai_epi32(_mm_add_epi32(Y01, BU1), 6);
            G1 = _mm_srai_epi32(_mm_add_epi32(Y01, GUV1), 6);
            R1 = _mm_srai_epi32(_mm_add_epi32(Y01, RV1), 6);

            // line 1, 8:11
            B2 = _mm_srai_epi32(_mm_add_epi32(Y02, BU2), 6);
            G2 = _mm_srai_epi32(_mm_add_epi32(Y02, GUV2), 6);
            R2 = _mm_srai_epi32(_mm_add_epi32(Y02, RV2), 6);

            // line 1, 12:15
            B3 = _mm_srai_epi32(_mm_add_epi32(Y03, BU3), 6);
            G3 = _mm_srai_epi32(_mm_add_epi32(Y03, GUV3), 6);
            R3 = _mm_srai_epi32(_mm_add_epi32(Y03, RV3), 6);

            R0 = _mm_packs_epi32(R0, R1);
            R2 = _mm_packs_epi32(R2, R3);
            R0 = _mm_packus_epi16(R0, R2);
            G0 = _mm_packs_epi32(G0, G1);
            G2 = _mm_packs_epi32(G2, G3);
            G0 = _mm_packus_epi16(G0, G2);
            B0 = _mm_packs_epi32(B0, B1);
            B2 = _mm_packs_epi32(B2, B3);
            B0 = _mm_packus_epi16(B0, B2);

            if (rgb) {
                RG0 = _mm_unpacklo_epi8(R0, G0);
                RG1 = _mm_unpackhi_epi8(R0, G0);

                out_temp0 = _mm_shuffle_epi8(RG0, _shuff_1);
                out_temp1 = _mm_shuffle_epi8(B0, _shuff_2);
                out0 = _mm_blendv_epi8(out_temp0, out_temp1, _blend_12);

                out_temp0 = _mm_shuffle_epi8(RG0, _shuff_3);
                out_temp1 = _mm_shuffle_epi8(RG1, _shuff_4);
                out_temp0 = _mm_blendv_epi8(out_temp0, out_temp1, _blend_34);
                out_temp1 = _mm_shuffle_epi8(B0, _shuff_5);
                out1 = _mm_blendv_epi8(out_temp0, out_temp1, _blend_345);

                out_temp0 = _mm_shuffle_epi8(RG1, _shuff_6);
                out_temp1 = _mm_shuffle_epi8(B0, _shuff_7);
                out2 = _mm_blendv_epi8(out_temp0, out_temp1, _blend_67);
            } else {
                BG0 = _mm_unpacklo_epi8(B0, G0);
                BG1 = _mm_unpackhi_epi8(B0, G0);

                out_temp0 = _mm_shuffle_epi8(BG0, _shuff_1);
                out_temp1 = _mm_shuffle_epi8(R0, _shuff_2);
                out0 = _mm_blendv_epi8(out_temp0, out_temp1, _blend_12);

                out_temp0 = _mm_shuffle_epi8(BG0, _shuff_3);
                out_temp1 = _mm_shuffle_epi8(BG1, _shuff_4);
                out_temp0 = _mm_blendv_epi8(out_temp0, out_temp1, _blend_34);
                out_temp1 = _mm_shuffle_epi8(R0, _shuff_5);
                out1 = _mm_blendv_epi8(out_temp0, out_temp1, _blend_345);

                out_temp0 = _mm_shuffle_epi8(BG1, _shuff_6);
                out_temp1 = _mm_shuffle_epi8(R0, _shuff_7);
                out2 = _mm_blendv_epi8(out_temp0, out_temp1, _blend_67);
            }

            _mm_storeu_si128((__m128i*)(dst1 + index1), out0);
            index1 += 16;
            _mm_storeu_si128((__m128i*)(dst1 + index1), out1);
            index1 += 16;
            _mm_storeu_si128((__m128i*)(dst1 + index1), out2);
            index1 += 16;
        }

        for (; c < width; c += 2) {
            int U = 0, V = 0, s_Y0 = 0;
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

            s_Y0 = *((pY) + c);
            uint32_t s_Y1 = static_cast<uint32_t>(s_Y0 * 0x0101 * YG) >> 16;
            uint8_t B = saturate_cast<unsigned char>(
                    static_cast<int32_t>(-(U * UB) + s_Y1 + BB) >> 6);
            uint8_t G = saturate_cast<unsigned char>(
                    static_cast<int32_t>(-(U * UG + V * VG) + s_Y1 + BG) >> 6);
            uint8_t R = saturate_cast<unsigned char>(
                    static_cast<int32_t>(-(V * VR) + s_Y1 + BR) >> 6);
            SET_COLOR(dst0, index0)

            s_Y0 = *((pY) + c + 1);
            s_Y1 = static_cast<uint32_t>(s_Y0 * 0x0101 * YG) >> 16;
            B = saturate_cast<unsigned char>(
                    static_cast<int32_t>(-(U * UB) + s_Y1 + BB) >> 6);
            G = saturate_cast<unsigned char>(
                    static_cast<int32_t>(-(U * UG + V * VG) + s_Y1 + BG) >> 6);
            R = saturate_cast<unsigned char>(
                    static_cast<int32_t>(-(V * VR) + s_Y1 + BR) >> 6);
            SET_COLOR(dst0, index0)

            s_Y0 = *((pY) + src_step + c);
            s_Y1 = static_cast<uint32_t>(s_Y0 * 0x0101 * YG) >> 16;
            B = saturate_cast<unsigned char>(
                    static_cast<int32_t>(-(U * UB) + s_Y1 + BB) >> 6);
            G = saturate_cast<unsigned char>(
                    static_cast<int32_t>(-(U * UG + V * VG) + s_Y1 + BG) >> 6);
            R = saturate_cast<unsigned char>(
                    static_cast<int32_t>(-(V * VR) + s_Y1 + BR) >> 6);
            SET_COLOR(dst1, index1)

            s_Y0 = *((pY) + src_step + c + 1);
            s_Y1 = static_cast<uint32_t>(s_Y0 * 0x0101 * YG) >> 16;
            B = saturate_cast<unsigned char>(
                    static_cast<int32_t>(-(U * UB) + s_Y1 + BB) >> 6);
            G = saturate_cast<unsigned char>(
                    static_cast<int32_t>(-(U * UG + V * VG) + s_Y1 + BG) >> 6);
            R = saturate_cast<unsigned char>(
                    static_cast<int32_t>(-(V * VR) + s_Y1 + BR) >> 6);
            SET_COLOR(dst1, index1)
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

MEGDNN_ATTRIBUTE_TARGET("sse4.2")
void cvt_rgb2yuv_8u_SSE_4_2(const Mat8u& src, Mat8u& dst) {
    const int yuv_shift = 14;
    const int coef[] = {1868, 9617, 4899, 8061, 14369};
    const int delta = 128 << yuv_shift;
    const int yuv = 1 << (yuv_shift - 1);

    __m128i v_src_r, v_src_g, v_src_b;
    __m128i v_dst_r, v_dst_g, v_dst_b;

    __m128i v_coef_0 = _mm_set1_epi32(coef[0]);
    __m128i v_coef_1 = _mm_set1_epi32(coef[1]);
    __m128i v_coef_2 = _mm_set1_epi32(coef[2]);
    __m128i v_coef_3 = _mm_set1_epi32(coef[3]);
    __m128i v_coef_4 = _mm_set1_epi32(coef[4]);

    __m128i v_delta = _mm_set1_epi32(delta);
    __m128i v_yuv = _mm_set1_epi32(yuv);

    for (size_t r = 0; r < src.rows(); ++r) {
        const uchar* psrc = src.ptr(r);
        uchar* pdst = dst.ptr(r);
        const uchar* const pend = psrc + src.cols() * 3;

        for (; psrc <= pend - 4 * 3; psrc += 4 * 3, pdst += 4 * 3) {
            v_src_r = _mm_set_epi32((int)psrc[0], (int)psrc[3], (int)psrc[6],
                                    (int)psrc[9]);
            v_src_g = _mm_set_epi32((int)psrc[1], (int)psrc[4], (int)psrc[7],
                                    (int)psrc[10]);
            v_src_b = _mm_set_epi32((int)psrc[2], (int)psrc[5], (int)psrc[8],
                                    (int)psrc[11]);

            v_dst_r = _mm_add_epi32(
                    _mm_mullo_epi32(v_src_b, v_coef_2),
                    _mm_add_epi32(_mm_mullo_epi32(v_src_r, v_coef_0),
                                  _mm_mullo_epi32(v_src_g, v_coef_1)));
            v_dst_r = _mm_srai_epi32(_mm_add_epi32(v_dst_r, v_yuv), yuv_shift);

            v_dst_g = _mm_add_epi32(
                    v_delta,
                    _mm_mullo_epi32(v_coef_3, _mm_sub_epi32(v_src_r, v_dst_r)));
            v_dst_g = _mm_srai_epi32(_mm_add_epi32(v_dst_g, v_yuv), yuv_shift);

            v_dst_b = _mm_add_epi32(
                    v_delta,
                    _mm_mullo_epi32(v_coef_4, _mm_sub_epi32(v_src_b, v_dst_r)));
            v_dst_b = _mm_srai_epi32(_mm_add_epi32(v_dst_b, v_yuv), yuv_shift);

            pdst[0] = saturate_cast<uchar>(_mm_extract_epi32(v_dst_r, 3));
            pdst[1] = saturate_cast<uchar>(_mm_extract_epi32(v_dst_g, 3));
            pdst[2] = saturate_cast<uchar>(_mm_extract_epi32(v_dst_b, 3));

            pdst[3] = saturate_cast<uchar>(_mm_extract_epi32(v_dst_r, 2));
            pdst[4] = saturate_cast<uchar>(_mm_extract_epi32(v_dst_g, 2));
            pdst[5] = saturate_cast<uchar>(_mm_extract_epi32(v_dst_b, 2));

            pdst[6] = saturate_cast<uchar>(_mm_extract_epi32(v_dst_r, 1));
            pdst[7] = saturate_cast<uchar>(_mm_extract_epi32(v_dst_g, 1));
            pdst[8] = saturate_cast<uchar>(_mm_extract_epi32(v_dst_b, 1));

            pdst[9] = saturate_cast<uchar>(_mm_extract_epi32(v_dst_r, 0));
            pdst[10] = saturate_cast<uchar>(_mm_extract_epi32(v_dst_g, 0));
            pdst[11] = saturate_cast<uchar>(_mm_extract_epi32(v_dst_b, 0));
        }

        for (; psrc < pend; psrc += 3, pdst += 3) {
            int Y = descale(
                    psrc[0] * coef[0] + psrc[1] * coef[1] + psrc[2] * coef[2],
                    yuv_shift);
            int Cr = descale((psrc[0] - Y) * coef[3] + delta, yuv_shift);
            int Cb = descale((psrc[2] - Y) * coef[4] + delta, yuv_shift);
            pdst[0] = saturate_cast<uchar>(Y);
            pdst[1] = saturate_cast<uchar>(Cr);
            pdst[2] = saturate_cast<uchar>(Cb);
        }
    }
}

MEGDNN_ATTRIBUTE_TARGET("sse4.2")
void cvt_rgb2yuv_32f_SSE_4_2(const Mat32f& src, Mat32f& dst) {
    const float coef[] = {0.114f, 0.587f, 0.299f, 0.492f, 0.877f};
    const float delta = 0.5f;

    __m128 v_src_r, v_src_g, v_src_b;
    __m128 v_dst_r, v_dst_g, v_dst_b;

    __m128 v_coef_0 = _mm_set1_ps(coef[0]);
    __m128 v_coef_1 = _mm_set1_ps(coef[1]);
    __m128 v_coef_2 = _mm_set1_ps(coef[2]);
    __m128 v_coef_3 = _mm_set1_ps(coef[3]);
    __m128 v_coef_4 = _mm_set1_ps(coef[4]);

    __m128 v_delta = _mm_set1_ps(delta);

    for (size_t r = 0; r < src.rows(); ++r) {
        const float* psrc = src.ptr(r);
        float* pdst = dst.ptr(r);
        const float* const pend = psrc + src.cols() * 3;

        for (; psrc <= pend - 4 * 3; psrc += 4 * 3, pdst += 4 * 3) {
            v_src_r = _mm_set_ps(psrc[9], psrc[6], psrc[3], psrc[0]);
            v_src_g = _mm_set_ps(psrc[10], psrc[7], psrc[4], psrc[1]);
            v_src_b = _mm_set_ps(psrc[11], psrc[8], psrc[5], psrc[2]);

            v_dst_r = _mm_add_ps(_mm_mul_ps(v_src_b, v_coef_2),
                                 _mm_add_ps(_mm_mul_ps(v_src_r, v_coef_0),
                                            _mm_mul_ps(v_src_g, v_coef_1)));

            v_dst_g = _mm_add_ps(
                    v_delta,
                    _mm_mul_ps(v_coef_3, _mm_sub_ps(v_src_r, v_dst_r)));
            v_dst_b = _mm_add_ps(
                    v_delta,
                    _mm_mul_ps(v_coef_4, _mm_sub_ps(v_src_b, v_dst_r)));

            float* r = (float*)(&v_dst_r);
            float* g = (float*)(&v_dst_g);
            float* b = (float*)(&v_dst_b);

            pdst[0] = r[0];
            pdst[1] = g[0];
            pdst[2] = b[0];
            pdst[3] = r[1];
            pdst[4] = g[1];
            pdst[5] = b[1];
            pdst[6] = r[2];
            pdst[7] = g[2];
            pdst[8] = b[2];
            pdst[9] = r[3];
            pdst[10] = g[3];
            pdst[11] = b[3];
        }
        for (; psrc < pend; psrc += 3, pdst += 3) {
            float Y = psrc[0] * coef[0] + psrc[1] * coef[1] + psrc[2] * coef[2];
            float Cr = (psrc[0] - Y) * coef[3] + delta;
            float Cb = (psrc[2] - Y) * coef[4] + delta;

            pdst[0] = Y;
            pdst[1] = Cr;
            pdst[2] = Cb;
        }
    }
}

MEGDNN_ATTRIBUTE_TARGET("sse4.2")
void cvt_yuv2rgb_8u_SSE_4_2(const Mat8u& src, Mat8u& dst) {
    const int yuv_shift = 14;
    const int coef[] = {33292, -6472, -9519, 18678};
    const int delta = 128;
    const int yuv = 1 << (yuv_shift - 1);

    __m128i v_src_y, v_src_u, v_src_v;
    __m128i v_dst_r, v_dst_g, v_dst_b;

    __m128i v_coef_0 = _mm_set1_epi32(coef[0]);
    __m128i v_coef_1 = _mm_set1_epi32(coef[1]);
    __m128i v_coef_2 = _mm_set1_epi32(coef[2]);
    __m128i v_coef_3 = _mm_set1_epi32(coef[3]);

    __m128i v_delta = _mm_set1_epi32(delta);
    __m128i v_yuv = _mm_set1_epi32(yuv);

    for (size_t r = 0; r < src.rows(); ++r) {
        const uchar* psrc = src.ptr(r);
        uchar* pdst = dst.ptr(r);
        const uchar* const pend = psrc + src.cols() * 3;

        for (; psrc <= pend - 4 * 3; psrc += 4 * 3, pdst += 4 * 3) {
            v_src_y = _mm_set_epi32((int)psrc[0], (int)psrc[3], (int)psrc[6],
                                    (int)psrc[9]);
            v_src_u = _mm_set_epi32((int)psrc[1], (int)psrc[4], (int)psrc[7],
                                    (int)psrc[10]);
            v_src_v = _mm_set_epi32((int)psrc[2], (int)psrc[5], (int)psrc[8],
                                    (int)psrc[11]);

            __m128i v_u_delta = _mm_sub_epi32(v_src_u, v_delta);
            __m128i v_v_delta = _mm_sub_epi32(v_src_v, v_delta);

            __m128i v_u_delta_0 = _mm_mullo_epi32(v_coef_0, v_u_delta);
            __m128i v_u_delta_1 = _mm_mullo_epi32(v_coef_1, v_u_delta);
            __m128i v_v_delta_2 = _mm_mullo_epi32(v_coef_2, v_v_delta);
            __m128i v_v_delta_3 = _mm_mullo_epi32(v_coef_3, v_v_delta);

            __m128i v_v_delta_2_1 = _mm_add_epi32(v_v_delta_2, v_u_delta_1);

            __m128i v_u_delta_0_yuv = _mm_add_epi32(v_yuv, v_u_delta_0);
            __m128i v_v_delta_3_yuv = _mm_add_epi32(v_yuv, v_v_delta_3);
            __m128i v_v_delta_2_1_yuv = _mm_add_epi32(v_yuv, v_v_delta_2_1);

            __m128i v_dst_r_shift = _mm_srai_epi32(v_u_delta_0_yuv, yuv_shift);
            __m128i v_dst_g_shift =
                    _mm_srai_epi32(v_v_delta_2_1_yuv, yuv_shift);
            __m128i v_dst_b_shift = _mm_srai_epi32(v_v_delta_3_yuv, yuv_shift);

            v_dst_r = _mm_add_epi32(v_src_y, v_dst_r_shift);
            v_dst_g = _mm_add_epi32(v_src_y, v_dst_g_shift);
            v_dst_b = _mm_add_epi32(v_src_y, v_dst_b_shift);

            int* r = (int*)(&v_dst_r);
            int* g = (int*)(&v_dst_g);
            int* b = (int*)(&v_dst_b);

            pdst[0] = saturate_cast<uchar>(r[3]);
            pdst[1] = saturate_cast<uchar>(g[3]);
            pdst[2] = saturate_cast<uchar>(b[3]);

            pdst[3] = saturate_cast<uchar>(r[2]);
            pdst[4] = saturate_cast<uchar>(g[2]);
            pdst[5] = saturate_cast<uchar>(b[2]);

            pdst[6] = saturate_cast<uchar>(r[1]);
            pdst[7] = saturate_cast<uchar>(g[1]);
            pdst[8] = saturate_cast<uchar>(b[1]);

            pdst[9] = saturate_cast<uchar>(r[0]);
            pdst[10] = saturate_cast<uchar>(g[0]);
            pdst[11] = saturate_cast<uchar>(b[0]);
        }

        for (; psrc < pend; psrc += 3, pdst += 3) {
            uchar Y = psrc[0];
            uchar Cr = psrc[1];
            uchar Cb = psrc[2];

            int R = Y + descale((Cr - delta) * coef[0], yuv_shift);
            int G = Y + descale((Cb - delta) * coef[2] + (Cr - delta) * coef[1],
                                yuv_shift);
            int B = Y + descale((Cb - delta) * coef[3], yuv_shift);

            pdst[0] = saturate_cast<uchar>(R);
            pdst[1] = saturate_cast<uchar>(G);
            pdst[2] = saturate_cast<uchar>(B);
        }
    }
}

MEGDNN_ATTRIBUTE_TARGET("sse4.2")
void cvt_yuv2rgb_32f_SSE_4_2(const Mat32f& src, Mat32f& dst) {
    const float coef[] = {2.032f, -0.395f, -0.581f, 1.140f};
    __m128 v_coef_0 = _mm_set1_ps(coef[0]);
    __m128 v_coef_1 = _mm_set1_ps(coef[1]);
    __m128 v_coef_2 = _mm_set1_ps(coef[2]);
    __m128 v_coef_3 = _mm_set1_ps(coef[3]);

    __m128 v_src_y, v_src_u, v_src_v;
    __m128 v_dst_r, v_dst_g, v_dst_b;

    const float delta = 0.5f;
    __m128 v_delta = _mm_set1_ps(delta);

    for (size_t r = 0; r < src.rows(); ++r) {
        const float* psrc = src.ptr(r);
        float* pdst = dst.ptr(r);
        const float* const pend = psrc + src.cols() * 3;

        for (; psrc <= pend - 4 * 3; psrc += 4 * 3, pdst += 4 * 3) {
            v_src_y = _mm_set_ps(psrc[9], psrc[6], psrc[3], psrc[0]);
            v_src_u = _mm_set_ps(psrc[10], psrc[7], psrc[4], psrc[1]);
            v_src_v = _mm_set_ps(psrc[11], psrc[8], psrc[5], psrc[2]);

            __m128 temp1 = _mm_sub_ps(v_src_u, v_delta),
                   temp2 = _mm_sub_ps(v_src_v, v_delta);

            v_dst_r = _mm_add_ps(v_src_y, _mm_mul_ps(v_coef_0, temp1));
            v_dst_g = _mm_add_ps(v_src_y,
                                 _mm_add_ps(_mm_mul_ps(v_coef_2, temp2),
                                            _mm_mul_ps(v_coef_1, temp1)));
            v_dst_b = _mm_add_ps(v_src_y, _mm_mul_ps(v_coef_3, temp2));

            float* r = (float*)(&v_dst_r);
            float* g = (float*)(&v_dst_g);
            float* b = (float*)(&v_dst_b);

            pdst[0] = r[0];
            pdst[1] = g[0];
            pdst[2] = b[0];
            pdst[3] = r[1];
            pdst[4] = g[1];
            pdst[5] = b[1];
            pdst[6] = r[2];
            pdst[7] = g[2];
            pdst[8] = b[2];
            pdst[9] = r[3];
            pdst[10] = g[3];
            pdst[11] = b[3];
        }

        for (; psrc < pend; psrc += 3, pdst += 3) {
            float Y = psrc[0], Cr = psrc[1], Cb = psrc[2];

            float R = Y + (Cr - delta) * coef[0];
            float G = Y + (Cb - delta) * coef[2] + (Cr - delta) * coef[1];
            float B = Y + (Cb - delta) * coef[3];

            pdst[0] = R;
            pdst[1] = G;
            pdst[2] = B;
        }
    }
}

MEGDNN_ATTRIBUTE_TARGET("sse4.2")
void cvt_gray2rgb_8u_SSE_4_2(const Mat8u& src, Mat8u& dst) {
    __m128i src_data, dst_data;

    __m128i shuff_1 =
            _mm_set_epi8(5, 4, 4, 4, 3, 3, 3, 2, 2, 2, 1, 1, 1, 0, 0, 0);
    __m128i shuff_2 =
            _mm_set_epi8(10, 10, 9, 9, 9, 8, 8, 8, 7, 7, 7, 6, 6, 6, 5, 5);
    __m128i shuff_3 = _mm_set_epi8(15, 15, 15, 14, 14, 14, 13, 13, 13, 12, 12,
                                   12, 11, 11, 11, 10);

    for (size_t r = 0; r < src.rows(); ++r) {
        const uchar* psrc = src.ptr(r);
        uchar* pdst = dst.ptr(r);
        const uchar* const pend = psrc + src.cols() * 1;

        for (; psrc <= pend - 16; psrc += 16, pdst += 16) {
            src_data = _mm_lddqu_si128((__m128i*)psrc);

            dst_data = _mm_shuffle_epi8(src_data, shuff_1);
            _mm_storeu_si128((__m128i*)(pdst), dst_data);

            pdst += 16;
            dst_data = _mm_shuffle_epi8(src_data, shuff_2);
            _mm_storeu_si128((__m128i*)(pdst), dst_data);

            pdst += 16;
            dst_data = _mm_shuffle_epi8(src_data, shuff_3);
            _mm_storeu_si128((__m128i*)(pdst), dst_data);
        }

        for (; psrc <= pend - 4; psrc += 4, pdst += 4 * 3) {
            pdst[0] = pdst[1] = pdst[2] = psrc[0];
            pdst[3] = pdst[4] = pdst[5] = psrc[1];
            pdst[6] = pdst[7] = pdst[8] = psrc[2];
            pdst[9] = pdst[10] = pdst[11] = psrc[3];
        }

        for (; psrc < pend; psrc += 1, pdst += 3) {
            pdst[0] = pdst[1] = pdst[2] = psrc[0];
        }
    }
}

MEGDNN_ATTRIBUTE_TARGET("sse4.2")
void cvt_gray2rgb_32f_SSE_4_2(const Mat32f& src, Mat32f& dst) {
    __m128 dst_1, dst_2, dst_3;

    for (size_t r = 0; r < src.rows(); ++r) {
        const float* psrc = src.ptr(r);
        float* pdst = dst.ptr(r);
        const float* const pend = psrc + src.cols() * 1;
        for (; psrc <= pend - 4; psrc += 4, pdst += 4 * 3) {
            dst_1 = _mm_set_ps(psrc[1], psrc[0], psrc[0], psrc[0]);
            dst_2 = _mm_set_ps(psrc[2], psrc[2], psrc[1], psrc[1]);
            dst_3 = _mm_set_ps(psrc[3], psrc[3], psrc[3], psrc[2]);

            _mm_storeu_ps(pdst, dst_1);
            _mm_storeu_ps(pdst + 4, dst_2);
            _mm_storeu_ps(pdst + 8, dst_3);
        }

        for (; psrc < pend; psrc += 1, pdst += 3) {
            pdst[0] = pdst[1] = pdst[2] = psrc[0];
        }
    }
}

MEGDNN_ATTRIBUTE_TARGET("sse4.2")
void cvt_rgb2gray_32f_SSE_4_2(const Mat32f& src, Mat32f& dst) {
    const float coef_r = 0.299f, coef_g = 0.587f, coef_b = 0.114f;
    __m128 v_coef_r = _mm_set1_ps(coef_r);
    __m128 v_coef_g = _mm_set1_ps(coef_g);
    __m128 v_coef_b = _mm_set1_ps(coef_b);

    for (size_t r = 0; r < src.rows(); ++r) {
        const float* psrc = src.ptr(r);
        float* pdst = dst.ptr(r);
        const float* const pend = psrc + src.cols() * 3;
        __m128 v_r, v_g, v_b, ans;
        for (; psrc <= pend - 4 * 3; psrc += 4 * 3, pdst += 4) {
            v_r = _mm_set_ps(psrc[9], psrc[6], psrc[3], psrc[0]);
            v_r = _mm_mul_ps(v_r, v_coef_r);

            v_g = _mm_set_ps(psrc[10], psrc[7], psrc[4], psrc[1]);
            v_g = _mm_mul_ps(v_g, v_coef_g);

            v_b = _mm_set_ps(psrc[11], psrc[8], psrc[5], psrc[2]);
            v_b = _mm_mul_ps(v_b, v_coef_b);

            ans = _mm_add_ps(v_r, _mm_add_ps(v_g, v_b));

            _mm_storeu_ps(pdst, ans);
        }

        for (; psrc < pend; psrc += 3, pdst += 1) {
            pdst[0] = psrc[1] * coef_g + psrc[2] * coef_b + psrc[0] * coef_r;
        }
    }
}

MEGDNN_ATTRIBUTE_TARGET("sse4.2")
void cvt_rgba2rgb_8u_SSE_4_2(const Mat8u& src, Mat8u& dst) {
    __m128i dst_data0, dst_data1, dst_data2;
    __m128i src_data0, src_data1, src_data2, src_data3;

    __m128i shuff_ = _mm_set_epi8(15, 15, 15, 15, 14, 13, 12, 10, 9, 8, 6, 5, 4,
                                  2, 1, 0);

    for (size_t r = 0; r < src.rows(); ++r) {
        const uchar* psrc = src.ptr(r);
        uchar* pdst = dst.ptr(r);
        const uchar* const pend = psrc + src.cols() * 4;

        for (; psrc <= pend - 64;) {
            src_data0 = _mm_lddqu_si128((__m128i*)psrc);
            psrc += 16;
            src_data0 = _mm_shuffle_epi8(src_data0, shuff_);
            dst_data0 = src_data0;

            src_data1 = _mm_lddqu_si128((__m128i*)psrc);
            psrc += 16;
            src_data1 = _mm_shuffle_epi8(src_data1, shuff_);
            dst_data1 = _mm_shuffle_epi32(src_data1, 9);

            src_data2 = _mm_lddqu_si128((__m128i*)psrc);
            psrc += 16;
            src_data2 = _mm_shuffle_epi8(src_data2, shuff_);

            src_data3 = _mm_lddqu_si128((__m128i*)psrc);
            psrc += 16;
            src_data3 = _mm_shuffle_epi8(src_data3, shuff_);
            dst_data2 = _mm_shuffle_epi32(src_data3, 144);

            src_data1 = _mm_shuffle_epi32(src_data1, 0);
            dst_data0 = _mm_blend_epi16(dst_data0, src_data1, 192);

            src_data1 = _mm_shuffle_epi32(src_data2, 68);
            dst_data1 = _mm_blend_epi16(dst_data1, src_data1, 240);

            src_data1 = _mm_shuffle_epi32(src_data2, 170);
            dst_data2 = _mm_blend_epi16(dst_data2, src_data1, 3);

            _mm_storeu_si128((__m128i*)(pdst), dst_data0);
            pdst += 16;
            _mm_storeu_si128((__m128i*)(pdst), dst_data1);
            pdst += 16;
            _mm_storeu_si128((__m128i*)(pdst), dst_data2);
            pdst += 16;
        }

        for (; psrc < pend; psrc += 4, pdst += 3) {
            uchar x0 = psrc[0];
            uchar x1 = psrc[1];
            uchar x2 = psrc[2];
            pdst[0] = x0;
            pdst[1] = x1;
            pdst[2] = x2;
        }
    }
}

MEGDNN_ATTRIBUTE_TARGET("sse4.2")
void cvt_rgba2bgr_8u_SSE_4_2(const Mat8u& src, Mat8u& dst) {
    __m128i dst_data0, dst_data1, dst_data2;
    __m128i src_data0, src_data1, src_data2, src_data3;

    __m128i shuff_ = _mm_set_epi8(15, 15, 15, 15, 12, 13, 14, 8, 9, 10, 4, 5, 6,
                                  0, 1, 2);

    for (size_t r = 0; r < src.rows(); ++r) {
        const uchar* psrc = src.ptr(r);
        uchar* pdst = dst.ptr(r);
        const uchar* const pend = psrc + src.cols() * 4;

        for (; psrc <= pend - 64;) {
            src_data0 = _mm_lddqu_si128((__m128i*)psrc);
            psrc += 16;
            src_data0 = _mm_shuffle_epi8(src_data0, shuff_);
            dst_data0 = src_data0;

            src_data1 = _mm_lddqu_si128((__m128i*)psrc);
            psrc += 16;
            src_data1 = _mm_shuffle_epi8(src_data1, shuff_);
            dst_data1 = _mm_shuffle_epi32(src_data1, 9);

            src_data2 = _mm_lddqu_si128((__m128i*)psrc);
            psrc += 16;
            src_data2 = _mm_shuffle_epi8(src_data2, shuff_);

            src_data3 = _mm_lddqu_si128((__m128i*)psrc);
            psrc += 16;
            src_data3 = _mm_shuffle_epi8(src_data3, shuff_);
            dst_data2 = _mm_shuffle_epi32(src_data3, 144);

            src_data1 = _mm_shuffle_epi32(src_data1, 0);
            dst_data0 = _mm_blend_epi16(dst_data0, src_data1, 192);

            src_data1 = _mm_shuffle_epi32(src_data2, 68);
            dst_data1 = _mm_blend_epi16(dst_data1, src_data1, 240);

            src_data1 = _mm_shuffle_epi32(src_data2, 170);
            dst_data2 = _mm_blend_epi16(dst_data2, src_data1, 3);

            _mm_storeu_si128((__m128i*)(pdst), dst_data0);
            pdst += 16;
            _mm_storeu_si128((__m128i*)(pdst), dst_data1);
            pdst += 16;
            _mm_storeu_si128((__m128i*)(pdst), dst_data2);
            pdst += 16;
        }

        for (; psrc < pend; psrc += 4, pdst += 3) {
            uchar x0 = psrc[0];
            uchar x1 = psrc[1];
            uchar x2 = psrc[2];
            pdst[0] = x2;
            pdst[1] = x1;
            pdst[2] = x0;
        }
    }
}

MEGDNN_ATTRIBUTE_TARGET("sse4.2")
void cvt_rgb2bgr_8u_SSE_4_2(const Mat8u& src, Mat8u& dst) {
    __m128i dst_data0, dst_data1, dst_data2;
    __m128i src_data0, src_data1, src_data2, src_data_temp;

    __m128i shuff_0 =
            _mm_set_epi8(15, 12, 13, 14, 9, 10, 11, 6, 7, 8, 3, 4, 5, 0, 1, 2);
    __m128i shuff_1 =
            _mm_set_epi8(15, 14, 11, 12, 13, 8, 9, 10, 5, 6, 7, 2, 3, 4, 1, 0);
    __m128i shuff_2 =
            _mm_set_epi8(13, 14, 15, 10, 11, 12, 7, 8, 9, 4, 5, 6, 1, 2, 3, 0);

    __m128i _blend_shuff_0 =
            _mm_set_epi8(1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 14);
    __m128i _blend_shuff_1 = _mm_set_epi8(15, 15, 15, 15, 15, 15, 15, 15, 15,
                                          15, 15, 15, 15, 15, 15, 15);
    __m128i _blend_shuff_2 =
            _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

    __m128i blend_0 =
            _mm_set_epi8(-128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    __m128i blend_1 =
            _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -128, 0);
    __m128i blend_2 =
            _mm_set_epi8(0, -128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    __m128i blend_3 =
            _mm_set_epi8(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -128);

    for (size_t r = 0; r < src.rows(); ++r) {
        const uchar* psrc = src.ptr(r);
        uchar* pdst = dst.ptr(r);
        const uchar* const pend = psrc + src.cols() * 3;

        for (; psrc <= pend - 48;) {
            src_data0 = _mm_lddqu_si128((__m128i*)psrc);
            psrc += 16;
            dst_data0 = _mm_shuffle_epi8(src_data0, shuff_0);
            src_data1 = _mm_lddqu_si128((__m128i*)psrc);
            psrc += 16;

            src_data_temp = _mm_shuffle_epi8(src_data1, _blend_shuff_0);
            dst_data0 = _mm_blendv_epi8(dst_data0, src_data_temp, blend_0);

            dst_data1 = _mm_shuffle_epi8(src_data1, shuff_1);
            _mm_storeu_si128((__m128i*)(pdst), dst_data0);
            pdst += 16;

            src_data0 = _mm_shuffle_epi8(src_data0, _blend_shuff_1);
            dst_data1 = _mm_blendv_epi8(dst_data1, src_data0, blend_1);

            src_data2 = _mm_lddqu_si128((__m128i*)psrc);
            psrc += 16;
            dst_data0 = _mm_shuffle_epi8(src_data2, _blend_shuff_2);
            dst_data1 = _mm_blendv_epi8(dst_data1, dst_data0, blend_2);

            _mm_storeu_si128((__m128i*)(pdst), dst_data1);
            pdst += 16;

            dst_data2 = _mm_shuffle_epi8(src_data2, shuff_2);
            dst_data2 = _mm_blendv_epi8(dst_data2, src_data_temp, blend_3);

            _mm_storeu_si128((__m128i*)(pdst), dst_data2);
            pdst += 16;
        }

        for (; psrc < pend; psrc += 3, pdst += 3) {
            uchar x0 = psrc[0];
            uchar x1 = psrc[1];
            uchar x2 = psrc[2];
            pdst[0] = x2;
            pdst[1] = x1;
            pdst[2] = x0;
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

    int tab[256 * 3];

    int b = 0, g = 0, r = (1 << (yuv_shift - 1));
    for (int i = 0; i < 256; ++i, r += R2Y, g += G2Y, b += B2Y) {
        tab[i] = r;
        tab[i + 256] = g;
        tab[i + 512] = b;
    }
    for (size_t r = 0; r < src.rows(); ++r) {
        const uchar* psrc = src.ptr(r);
        const uchar* pend = psrc + src.cols() * src.channels();
        uchar* pdst = dst.ptr(r);
        for (; psrc < pend; psrc += 3, pdst += 1) {
            pdst[0] =
                    (tab[psrc[0]] + tab[psrc[1] + 256] + tab[psrc[2] + 512]) >>
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

    return cvt_rgb2gray_32f_SSE_4_2(src, dst);
}

// gray2rgb
template <>
void cvt_gray2rgb<uchar>(const Mat8u& src, Mat8u& dst) {
    megdnn_assert(src.rows() == dst.rows());
    megdnn_assert(src.cols() == dst.cols());
    megdnn_assert(src.channels() == 1);
    megdnn_assert(dst.channels() == 3);

    return cvt_gray2rgb_8u_SSE_4_2(src, dst);
}
template <>
void cvt_gray2rgb<float>(const Mat32f& src, Mat32f& dst) {
    megdnn_assert(src.rows() == dst.rows());
    megdnn_assert(src.cols() == dst.cols());
    megdnn_assert(src.channels() == 1);
    megdnn_assert(dst.channels() == 3);

    return cvt_gray2rgb_32f_SSE_4_2(src, dst);
}

// rgb2yuv
template <>
void cvt_rgb2yuv<uchar>(const Mat8u& src, Mat8u& dst) {
    megdnn_assert(src.channels() == 3);
    megdnn_assert(dst.channels() == 3);
    megdnn_assert(src.rows() == dst.rows());
    megdnn_assert(src.cols() == dst.cols());

    return cvt_rgb2yuv_8u_SSE_4_2(src, dst);
}
template <>
void cvt_rgb2yuv<float>(const Mat32f& src, Mat32f& dst) {
    megdnn_assert(src.channels() == 3);
    megdnn_assert(dst.channels() == 3);
    megdnn_assert(src.rows() == dst.rows());
    megdnn_assert(src.cols() == dst.cols());

    return cvt_rgb2yuv_32f_SSE_4_2(src, dst);
}

// yuv2rgb
template <>
void cvt_yuv2rgb<float>(const Mat32f& src, Mat32f& dst) {
    megdnn_assert(src.channels() == 3);
    megdnn_assert(dst.channels() == 3);
    megdnn_assert(src.rows() == dst.rows());
    megdnn_assert(src.cols() == dst.cols());

    return cvt_yuv2rgb_32f_SSE_4_2(src, dst);
}

template <>
void cvt_yuv2rgb<uchar>(const Mat8u& src, Mat8u& dst) {
    megdnn_assert(src.channels() == 3);
    megdnn_assert(dst.channels() == 3);
    megdnn_assert(src.rows() == dst.rows());
    megdnn_assert(src.cols() == dst.cols());

    return cvt_yuv2rgb_8u_SSE_4_2(src, dst);
}

template <>
void cvt_rgba2rgb<uchar>(const Mat8u& src, Mat8u& dst) {
    megdnn_assert(src.channels() == 4);
    megdnn_assert(dst.channels() == 3);
    megdnn_assert(src.rows() == dst.rows());
    megdnn_assert(src.cols() == dst.cols());

    return cvt_rgba2rgb_8u_SSE_4_2(src, dst);
}

template <>
void cvt_rgba2bgr<uchar>(const Mat8u& src, Mat8u& dst) {
    megdnn_assert(src.channels() == 4);
    megdnn_assert(dst.channels() == 3);
    megdnn_assert(src.rows() == dst.rows());
    megdnn_assert(src.cols() == dst.cols());

    return cvt_rgba2bgr_8u_SSE_4_2(src, dst);
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

    return cvt_rgb2bgr_8u_SSE_4_2(src, dst);
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
            return cvt_BT601_yuv_transform<true, false, false>(src, dst);
        case Mode::BT601_YUV2BGR_NV21:
            return cvt_BT601_yuv_transform<false, false, false>(src, dst);
        case Mode::BT601_YUV2RGB_NV12:
            return cvt_BT601_yuv_transform<true, false, true>(src, dst);
        case Mode::BT601_YUV2BGR_NV12:
            return cvt_BT601_yuv_transform<false, false, true>(src, dst);
        case Mode::BT601_YUV2RGB_YV12:
            return cvt_BT601_yuv_transform<true, true, false>(src, dst);
        case Mode::BT601_YUV2BGR_YV12:
            return cvt_BT601_yuv_transform<false, true, false>(src, dst);
        case Mode::BT601_YUV2RGB_YU12:
            return cvt_BT601_yuv_transform<true, true, true>(src, dst);
        case Mode::BT601_YUV2BGR_YU12:
            return cvt_BT601_yuv_transform<false, true, true>(src, dst);
        default:
            megdnn_throw("unknown mode for real yuv.");
    }
}

template <typename T>
void CvtColorImpl::cvt_color_exec(_megdnn_tensor_in src_tensor,
                                  _megdnn_tensor_out dst_tensor) {
    auto mode = param().mode;
    for (size_t i = 0; i < src_tensor.layout.shape[0]; ++i) {
        Mat<T> src = TensorND2Mat<T>(src_tensor, i);
        Mat<T> dst = TensorND2Mat<T>(dst_tensor, i);
        switch (param().mode) {
            case Param::Mode::RGB2GRAY:
                cvt_rgb2gray<T>(src, dst);
                break;
            case Param::Mode::RGB2YUV:
                cvt_rgb2yuv<T>(src, dst);
                break;
            case Param::Mode::YUV2RGB:
                cvt_yuv2rgb<T>(src, dst);
                break;
            case Param::Mode::GRAY2RGB:
                cvt_gray2rgb<T>(src, dst);
                break;
            case Param::Mode::RGBA2RGB:
                cvt_rgba2rgb<T>(src, dst);
                break;
            case Param::Mode::RGBA2BGR:
                cvt_rgba2bgr<T>(src, dst);
                break;
            case Param::Mode::RGBA2GRAY:
                cvt_rgba2gray<T>(src, dst);
                break;
            case Param::Mode::RGB2BGR:
                cvt_rgb2bgr<T>(src, dst);
                break;
            case Param::Mode::BGR2GRAY:
                cvt_bgr2gray<T>(src, dst);
                break;
            case Param::Mode::BGR2RGB:
                cvt_bgr2rgb<T>(src, dst);
                break;
            case Param::Mode::YUV2GRAY_NV21:
            case Param::Mode::YUV2GRAY_NV12:
            case Param::Mode::YUV2GRAY_YV12:
            case Param::Mode::YUV2GRAY_YU12:
                cvt_yuv2gray_nv21<T>(src, dst);
                break;
            case Param::Mode::YUV2RGB_NV21:
            case Param::Mode::YCrCb2RGB:
                cvt_yuv2rgb_nv21<T>(src, dst);
                break;
            case Param::Mode::YUV2BGR_NV21:
            case Param::Mode::YCrCb2BGR:
                cvt_yuv2bgr_nv21<T>(src, dst);
                break;
            case Param::Mode::YUV2RGB_NV12:
                cvt_yuv2rgb_nv12<T>(src, dst);
                break;
            case Param::Mode::YUV2BGR_NV12:
                cvt_yuv2bgr_nv12<T>(src, dst);
                break;
            case Param::Mode::YUV2RGB_YV12:
                cvt_yuv2rgb_yv12<T>(src, dst);
                break;
            case Param::Mode::YUV2BGR_YV12:
                cvt_yuv2bgr_yv12<T>(src, dst);
                break;
            case Param::Mode::YUV2RGB_YU12:
                cvt_yuv2rgb_yu12<T>(src, dst);
                break;
            case Param::Mode::YUV2BGR_YU12:
                cvt_yuv2bgr_yu12<T>(src, dst);
                break;
            case Param::Mode::BT601_YUV2BGR_NV12:
            case Param::Mode::BT601_YUV2RGB_NV12:
            case Param::Mode::BT601_YUV2BGR_NV21:
            case Param::Mode::BT601_YUV2RGB_NV21:
            case Param::Mode::BT601_YUV2RGB_YU12:
            case Param::Mode::BT601_YUV2BGR_YU12:
            case Param::Mode::BT601_YUV2RGB_YV12:
            case Param::Mode::BT601_YUV2BGR_YV12:
                cvt_bt601_yuv<T>(src, dst, mode);
                break;
            default:
                megdnn_throw("Can not find property cvt_color operator.");
        }
    }
}

void CvtColorImpl::exec(_megdnn_tensor_in src, _megdnn_tensor_out dst,
                        _megdnn_workspace workspace) {
    using namespace megcv;
    check_exec(src.layout, dst.layout, workspace.size);
    // x86 cvt_color implementation need sse4.2
    if (!is_supported(SIMDType::SSE4_2)) {
        naive::CvtColorImpl::exec(src, dst, workspace);
        return;
    }
    MEGDNN_DISPATCH_CPU_KERN_OPR(if (dst.layout.dtype == dtype::Float32()) {
        cvt_color_exec<float>(src, dst);
    } else if (dst.layout.dtype == dtype::Uint8()) {
        cvt_color_exec<uchar>(src, dst);
    } else { megdnn_throw("Unsupported datatype of CvtColor optr."); });
}

}  // namespace x86
}  // namespace megdnn

// vim: syntax=cpp.doxygen
