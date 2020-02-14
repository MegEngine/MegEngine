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
 * \file dnn/src/naive/cvt_color/opr_impl.cpp
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
#include "src/naive/cvt_color/opr_impl.h"

#include "midout.h"
#include "src/common/cv/common.h"
#include "src/common/cv/cvt_color.h"
#include "src/common/cv/helper.h"
#include "src/naive/handle.h"

MIDOUT_DECL(megdnn_naive_cvtcolor)

namespace megdnn {
namespace naive {

using namespace megcv;

GENERATE_CVT_OPR_DECL_FOREACH(GENERATE_CVT_OPR_DECL)
GENERATE_UNSUPPORT_CVT_OPR_FOR_FLOAT(GENERATE_UNSUPPORT_CVT_OPR)

namespace {

/**
 * \brief jpeg yuv(YCrCb) to rgb or bgr.
 *
 * \tparam rgb, is convert to rgb or bgr
 * \tparam is_planar, if true, the layout is YYYYUUVV or YYYYVVUU, otherwise
 *     YYYYYUVUV or YYYYYVUVU
 * \tparam is_uv, if true, U is before V, otherwise V is before U
 */
template <bool rgb = true, bool is_planar = true, bool is_uv = true>
void cvt_yuv_transform(const Mat8u& src, Mat8u& dst) {
    typedef unsigned char uint8;
    const uint8* pY;
    const uint8* pU;
    const uint8* pV;
    int Y00, Y01, U, V;
    int Y10, Y11;
    int i, j;
    int ruv, guv, buv;
    int R, G, B;

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
    for (i = 0; i < height; i += 2, pY += src_step * 2) {
        size_t index = 0;
        size_t index1 = 0;
        uint8* out = dst.ptr(i);
        uint8* out1 = dst.ptr(i + 1);
        int jV = 0;
        for (j = 0; j < width; j += 2) {
            Y00 = *((pY) + j);
            Y01 = *((pY) + j + 1);
            Y10 = *((pY) + src_step + j);
            Y11 = *((pY) + src_step + j + 1);
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

            ruv = ((359 * (V - 128)) >> 8);
            guv = -1 * ((88 * (U - 128) + 183 * (V - 128)) >> 8);
            buv = ((454 * (U - 128)) >> 8);

            R = Y00 + ruv;
            G = Y00 + guv;
            B = Y00 + buv;
            R = (R > 255) ? 255 : ((R < 0) ? 0 : R);
            G = (G > 255) ? 255 : ((G < 0) ? 0 : G);
            B = (B > 255) ? 255 : ((B < 0) ? 0 : B);

            SET_COLOR(out, index)

            R = Y01 + ruv;
            G = Y01 + guv;
            B = Y01 + buv;
            R = (R > 255) ? 255 : ((R < 0) ? 0 : R);
            G = (G > 255) ? 255 : ((G < 0) ? 0 : G);
            B = (B > 255) ? 255 : ((B < 0) ? 0 : B);

            SET_COLOR(out, index)

            ruv = ((359 * (V - 128)) >> 8);
            guv = -1 * ((88 * (U - 128) + 183 * (V - 128)) >> 8);
            buv = ((454 * (U - 128)) >> 8);
            R = Y10 + ruv;
            G = Y10 + guv;
            B = Y10 + buv;
            R = (R > 255) ? 255 : ((R < 0) ? 0 : R);
            G = (G > 255) ? 255 : ((G < 0) ? 0 : G);
            B = (B > 255) ? 255 : ((B < 0) ? 0 : B);

            SET_COLOR(out1, index1)

            R = Y11 + ruv;
            G = Y11 + guv;
            B = Y11 + buv;
            R = (R > 255) ? 255 : ((R < 0) ? 0 : R);
            G = (G > 255) ? 255 : ((G < 0) ? 0 : G);
            B = (B > 255) ? 255 : ((B < 0) ? 0 : B);

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
}

/**
 * \brief bt601 yuv to rgb or bgr.
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
        int jV = 0;
        for (int j = 0; j < width; j += 2) {
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

    const float coef_r = 0.299f, coef_g = 0.587f, coef_b = 0.114f;
    for (size_t r = 0; r < src.rows(); ++r) {
        for (size_t c = 0; c < src.cols(); ++c) {
            float R = src.at(r, c, 0);
            float G = src.at(r, c, 1);
            float B = src.at(r, c, 2);
            float& Y = dst.at(r, c, 0);
            Y = R * coef_r + G * coef_g + B * coef_b;
        }
    }
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

    const int yuv_shift = 14;
    const int coef[] = {1868, 9617, 4899, 8061, 14369};
    const int delta = 128 << yuv_shift;
    for (size_t r = 0; r < src.rows(); ++r) {
        for (size_t c = 0; c < src.cols(); ++c) {
            const uchar* v = &src.at(r, c, 0);
            int Y = descale(v[0] * coef[0] + v[1] * coef[1] + v[2] * coef[2],
                            yuv_shift);
            int Cr = descale((v[0] - Y) * coef[3] + delta, yuv_shift);
            int Cb = descale((v[2] - Y) * coef[4] + delta, yuv_shift);
            uchar* target = &dst.at(r, c, 0);
            target[0] = megcv::saturate_cast<uchar>(Y);
            target[1] = megcv::saturate_cast<uchar>(Cr);
            target[2] = megcv::saturate_cast<uchar>(Cb);
        }
    }
}
template <>
void cvt_rgb2yuv<float>(const Mat32f& src, Mat32f& dst) {
    megdnn_assert(src.channels() == 3);
    megdnn_assert(dst.channels() == 3);
    megdnn_assert(src.rows() == dst.rows());
    megdnn_assert(src.cols() == dst.cols());

    const float coef[] = {0.114f, 0.587f, 0.299f, 0.492f, 0.877f};
    const float delta = 0.5f;
    for (size_t r = 0; r < src.rows(); ++r) {
        for (size_t c = 0; c < src.cols(); ++c) {
            const float* v = &src.at(r, c, 0);
            float Y = v[0] * coef[0] + v[1] * coef[1] + v[2] * coef[2];
            float Cr = (v[0] - Y) * coef[3] + delta;
            float Cb = (v[2] - Y) * coef[4] + delta;
            float* target = &dst.at(r, c, 0);
            target[0] = Y;
            target[1] = Cr;
            target[2] = Cb;
        }
    }
}

// yuv2rgb
template <>
void cvt_yuv2rgb<float>(const Mat32f& src, Mat32f& dst) {
    megdnn_assert(src.channels() == 3);
    megdnn_assert(dst.channels() == 3);
    megdnn_assert(src.rows() == dst.rows());
    megdnn_assert(src.cols() == dst.cols());

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

    const int yuv_shift = 14;
    const int coef[] = {33292, -6472, -9519, 18678};
    const int delta = 128;
    for (size_t r = 0; r < src.rows(); ++r) {
        for (size_t c = 0; c < src.cols(); ++c) {
            const uchar* v = &src.at(r, c, 0);
            uchar Y = v[0];
            uchar Cr = v[1];
            uchar Cb = v[2];

            int R = Y + descale((Cr - delta) * coef[0], yuv_shift);
            int G = Y + descale((Cb - delta) * coef[2] + (Cr - delta) * coef[1],
                                yuv_shift);
            int B = Y + descale((Cb - delta) * coef[3], yuv_shift);

            uchar* target = &dst.at(r, c, 0);
            target[0] = megcv::saturate_cast<uchar>(R);
            target[1] = megcv::saturate_cast<uchar>(G);
            target[2] = megcv::saturate_cast<uchar>(B);
        }
    }
}

template <>
void cvt_rgba2rgb<uchar>(const Mat8u& src, Mat8u& dst) {
    megdnn_assert(src.channels() == 4);
    megdnn_assert(dst.channels() == 3);
    megdnn_assert(src.rows() == dst.rows());
    megdnn_assert(src.cols() == dst.cols());

    const uchar* _src = src.ptr();
    uchar* _dst = dst.ptr();
    size_t rows = src.rows();
    size_t cols = src.cols();
    size_t src_step = src.step();
    size_t dst_step = dst.step();
    for (size_t r = 0; r < rows; ++r, _src += src_step, _dst += dst_step) {
        const uchar* temp_src = _src;
        uchar* temp_dst = _dst;
        for (size_t c = 0; c < cols; ++c, temp_src += 4, temp_dst += 3) {
            uchar x0 = temp_src[0];
            uchar x1 = temp_src[1];
            uchar x2 = temp_src[2];
            temp_dst[0] = x0;
            temp_dst[1] = x1;
            temp_dst[2] = x2;
        }
    }
}

template <>
void cvt_rgba2bgr<uchar>(const Mat8u& src, Mat8u& dst) {
    megdnn_assert(src.channels() == 4);
    megdnn_assert(dst.channels() == 3);
    megdnn_assert(src.rows() == dst.rows());
    megdnn_assert(src.cols() == dst.cols());
    const uchar* _src = src.ptr();
    uchar* _dst = dst.ptr();
    size_t rows = src.rows();
    size_t cols = src.cols();
    size_t src_step = src.step();
    size_t dst_step = dst.step();
    for (size_t r = 0; r < rows; ++r, _src += src_step, _dst += dst_step) {
        const uchar* temp_src = _src;
        uchar* temp_dst = _dst;
        for (size_t c = 0; c < cols; ++c, temp_src += 4, temp_dst += 3) {
            uchar x0 = temp_src[0];
            uchar x1 = temp_src[1];
            uchar x2 = temp_src[2];
            temp_dst[0] = x2;
            temp_dst[1] = x1;
            temp_dst[2] = x0;
        }
    }
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

    const uchar* _src = src.ptr();
    uchar* _dst = dst.ptr();
    size_t rows = src.rows();
    size_t cols = src.cols();
    size_t src_step = src.step();
    size_t dst_step = dst.step();
    for (size_t r = 0; r < rows; ++r, _src += src_step, _dst += dst_step) {
        const uchar* temp_src = _src;
        uchar* temp_dst = _dst;
        for (size_t c = 0; c < cols; ++c, temp_src += 3, temp_dst += 3) {
            uchar x0 = temp_src[0];
            uchar x1 = temp_src[1];
            uchar x2 = temp_src[2];
            temp_dst[0] = x2;
            temp_dst[1] = x1;
            temp_dst[2] = x0;
        }
    }
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
    megdnn_assert(src.channels() == 1);
    megdnn_assert(dst.channels() == 1);
    megdnn_assert(src.rows() % 3 == 0);
    megdnn_assert(src.rows() / 3 * 2 == dst.rows());
    megdnn_assert(src.cols() == dst.cols());

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
            MIDOUT_BEGIN(megdnn_naive_cvtcolor, midout_iv(0)) {
                return cvt_BT601_yuv_transform<true, false, false>(src, dst);
            }
            MIDOUT_END();
        case Mode::BT601_YUV2BGR_NV21:
            MIDOUT_BEGIN(megdnn_naive_cvtcolor, midout_iv(1)) {
                return cvt_BT601_yuv_transform<false, false, false>(src, dst);
            }
            MIDOUT_END();
        case Mode::BT601_YUV2RGB_NV12:
            MIDOUT_BEGIN(megdnn_naive_cvtcolor, midout_iv(2)) {
                return cvt_BT601_yuv_transform<true, false, true>(src, dst);
            }
            MIDOUT_END();
        case Mode::BT601_YUV2BGR_NV12:
            MIDOUT_BEGIN(megdnn_naive_cvtcolor, midout_iv(3)) {
                return cvt_BT601_yuv_transform<false, false, true>(src, dst);
            }
            MIDOUT_END();
        case Mode::BT601_YUV2RGB_YV12:
            MIDOUT_BEGIN(megdnn_naive_cvtcolor, midout_iv(4)) {
                return cvt_BT601_yuv_transform<true, true, false>(src, dst);
            }
            MIDOUT_END();
        case Mode::BT601_YUV2BGR_YV12:
            MIDOUT_BEGIN(megdnn_naive_cvtcolor, midout_iv(5)) {
                return cvt_BT601_yuv_transform<false, true, false>(src, dst);
            }
            MIDOUT_END();
        case Mode::BT601_YUV2RGB_YU12:
            MIDOUT_BEGIN(megdnn_naive_cvtcolor, midout_iv(6)) {
                return cvt_BT601_yuv_transform<true, true, true>(src, dst);
            }
            MIDOUT_END();
        case Mode::BT601_YUV2BGR_YU12:
            MIDOUT_BEGIN(megdnn_naive_cvtcolor, midout_iv(7)) {
                return cvt_BT601_yuv_transform<false, true, true>(src, dst);
            }
            MIDOUT_END();
        default:
            megdnn_throw("unknown mode for real yuv.");
    }
}

template <typename T>
void CvtColorImpl::cvt_color_exec(_megdnn_tensor_in src_tensor,
                                  _megdnn_tensor_in dst_tensor) {
    auto mode = param().mode;
    for (size_t i = 0; i < src_tensor.layout.shape[0]; ++i) {
        Mat<T> src = TensorND2Mat<T>(src_tensor, i);
        Mat<T> dst = TensorND2Mat<T>(dst_tensor, i);
        switch (mode) {
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

void CvtColorImpl::exec(_megdnn_tensor_in src, _megdnn_tensor_in dst,
                        _megdnn_workspace workspace) {
    using namespace megcv;
    check_exec(src.layout, dst.layout, workspace.size);
    MEGDNN_DISPATCH_CPU_KERN_OPR(if (dst.layout.dtype == dtype::Float32()) {
        cvt_color_exec<float>(src, dst);
    } else if (dst.layout.dtype == dtype::Uint8()) {
        cvt_color_exec<uchar>(src, dst);
    } else { megdnn_throw("Unsupported datatype of CvtColor optr."); });
}

}  // namespace naive
}  // namespace megdnn

// vim: syntax=cpp.doxygen
