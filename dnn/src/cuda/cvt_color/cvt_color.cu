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
 * \file dnn/src/cuda/cvt_color/cvt_color.cu
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

#include "src/common/opr_param_defs_enumv.cuh"
#include "src/cuda/cv/kernel_common.cuh"
#include "src/cuda/cvt_color/cvt_color.cuh"
#include "src/cuda/utils.cuh"

#include <cassert>
#include <cfloat>
#include <cstdio>

namespace megdnn {
namespace cuda {
namespace cvt_color {

using namespace megcv;

#define THREADS_X 256
#define THREADS_Y 1

#define U8_PROCESS_PER_THREADS_X 4
#define F32_PROCESS_PER_THREADS_X 1

__global__ void cvt_rgb2gray_8u_kernel(const uchar* src, uchar* dst,
                                       const size_t rows, const size_t cols,
                                       const size_t src_step,
                                       const size_t dst_step) {
    size_t t = blockIdx.x * blockDim.x + threadIdx.x;

    if (t < (rows * cols) / U8_PROCESS_PER_THREADS_X) {
        size_t offset = t * U8_PROCESS_PER_THREADS_X;
        src += 3 * offset;
        dst += 1 * offset;

        uchar temp_des[4];
        uchar temp_src[12];
        *((uint3*)temp_src) = *((uint3*)src);

        temp_des[0] = (temp_src[0] * 4899 + temp_src[1] * 9617 +
                       temp_src[2] * 1868 + (1 << 13)) >>
                      14;
        temp_des[1] = (temp_src[3] * 4899 + temp_src[4] * 9617 +
                       temp_src[5] * 1868 + (1 << 13)) >>
                      14;
        temp_des[2] = (temp_src[6] * 4899 + temp_src[7] * 9617 +
                       temp_src[8] * 1868 + (1 << 13)) >>
                      14;
        temp_des[3] = (temp_src[9] * 4899 + temp_src[10] * 9617 +
                       temp_src[11] * 1868 + (1 << 13)) >>
                      14;

        *((uint32_t*)dst) = *((uint32_t*)temp_des);
    } else if (t == (rows * cols) / U8_PROCESS_PER_THREADS_X) {
        size_t rest = (rows * cols) % U8_PROCESS_PER_THREADS_X;
        if (rest != 0) {
            size_t offset = t * U8_PROCESS_PER_THREADS_X;
            src += 3 * offset;
            dst += 1 * offset;

            for (int i = 0; i < rest; i++, src += 3, dst += 1)
                dst[0] = (src[0] * 4899 + src[1] * 9617 + src[2] * 1868 +
                          (1 << 13)) >>
                         14;
        }
    }
}

__global__ void cvt_rgb2gray_32f_kernel(const float* src, float* dst,
                                        const size_t rows, const size_t cols,
                                        const size_t src_step,
                                        const size_t dst_step) {
    size_t t = blockIdx.x * blockDim.x + threadIdx.x;

    if (t < rows * cols) {
        size_t offset = t;
        src += offset * 3;
        dst += offset * 1;

        float temp_src[3], temp_dst;
        *((float3*)temp_src) = *((float3*)src);

        temp_dst = temp_src[0] * 0.299f + temp_src[1] * 0.587f +
                   temp_src[2] * 0.114f;

        dst[0] = temp_dst;
    }
}

__global__ void cvt_gray2rgb_8u_kernel(const uchar* src, uchar* dst,
                                       const size_t rows, const size_t cols,
                                       const size_t src_step,
                                       const size_t dst_step) {
    size_t t = blockIdx.x * blockDim.x + threadIdx.x;

    if (t < (rows * cols) / U8_PROCESS_PER_THREADS_X) {
        size_t offset = t * U8_PROCESS_PER_THREADS_X;
        src += 1 * offset;
        dst += 3 * offset;

        uchar temp_src[4], temp_des[12];
        *((uint32_t*)temp_src) = *((uint32_t*)src);

        temp_des[0] = temp_src[0];
        temp_des[1] = temp_src[0];
        temp_des[2] = temp_src[0];
        temp_des[3] = temp_src[1];
        temp_des[4] = temp_src[1];
        temp_des[5] = temp_src[1];
        temp_des[6] = temp_src[2];
        temp_des[7] = temp_src[2];
        temp_des[8] = temp_src[2];
        temp_des[9] = temp_src[3];
        temp_des[10] = temp_src[3];
        temp_des[11] = temp_src[3];

        *((uint3*)dst) = *((uint3*)temp_des);
    } else if (t == (rows * cols) / U8_PROCESS_PER_THREADS_X) {
        size_t rest = (rows * cols) % U8_PROCESS_PER_THREADS_X;
        if (rest != 0) {
            size_t offset = t * U8_PROCESS_PER_THREADS_X;
            src += 1 * offset;
            dst += 3 * offset;

            for (int i = 0; i < rest; i++, src += 1, dst += 3) {
                uchar temp_src = src[0];

                dst[0] = temp_src;
                dst[1] = temp_src;
                dst[2] = temp_src;
            }
        }
    }
}

__global__ void cvt_gray2rgb_32f_kernel(const float* src, float* dst,
                                        const size_t rows, const size_t cols,
                                        const size_t src_step,
                                        const size_t dst_step) {
    size_t t = blockIdx.x * blockDim.x + threadIdx.x;

    if (t < rows * cols) {
        src += t * 1;
        dst += t * 3;

        float temp_src, temp_dst[3];
        temp_src = src[0];

        temp_dst[0] = temp_src;
        temp_dst[1] = temp_src;
        temp_dst[2] = temp_src;

        *((float3*)dst) = *((float3*)temp_dst);
    }
}

#define descale(x, n) (((x) + (1 << ((n)-1))) >> (n))

__global__ void cvt_rgb2yuv_8u_kernel(const uchar* src, uchar* dst,
                                      const size_t rows, const size_t cols,
                                      const size_t src_step,
                                      const size_t dst_step) {
    size_t t = blockIdx.x * blockDim.x + threadIdx.x;

    const int yuv_shift = 14;
    const int coef[] = {1868, 9617, 4899, 8061, 14369};
    const int delta = 128 << yuv_shift;

    if (t < (rows * cols) / U8_PROCESS_PER_THREADS_X) {
        size_t offset_uchar = 3 * t * U8_PROCESS_PER_THREADS_X;
        src += offset_uchar;
        dst += offset_uchar;

        uchar temp_src[12], temp_dst[12];
        *((uint3*)temp_src) = *((uint3*)src);

        int p = 0;
        int y = descale(temp_src[0 + p] * coef[0] + temp_src[1 + p] * coef[1] +
                                temp_src[2 + p] * coef[2],
                        yuv_shift);
        int cr = descale((temp_src[0 + p] - y) * coef[3] + delta, yuv_shift);
        int cb = descale((temp_src[2 + p] - y) * coef[4] + delta, yuv_shift);
        temp_dst[0 + p] = saturate(y, 0, 255);
        temp_dst[1 + p] = saturate(cr, 0, 255);
        temp_dst[2 + p] = saturate(cb, 0, 255);

        p += 3;
        y = descale(temp_src[0 + p] * coef[0] + temp_src[1 + p] * coef[1] +
                            temp_src[2 + p] * coef[2],
                    yuv_shift);
        cr = descale((temp_src[0 + p] - y) * coef[3] + delta, yuv_shift);
        cb = descale((temp_src[2 + p] - y) * coef[4] + delta, yuv_shift);
        temp_dst[0 + p] = saturate(y, 0, 255);
        temp_dst[1 + p] = saturate(cr, 0, 255);
        temp_dst[2 + p] = saturate(cb, 0, 255);

        p += 3;
        y = descale(temp_src[0 + p] * coef[0] + temp_src[1 + p] * coef[1] +
                            temp_src[2 + p] * coef[2],
                    yuv_shift);
        cr = descale((temp_src[0 + p] - y) * coef[3] + delta, yuv_shift);
        cb = descale((temp_src[2 + p] - y) * coef[4] + delta, yuv_shift);
        temp_dst[0 + p] = saturate(y, 0, 255);
        temp_dst[1 + p] = saturate(cr, 0, 255);
        temp_dst[2 + p] = saturate(cb, 0, 255);

        p += 3;
        y = descale(temp_src[0 + p] * coef[0] + temp_src[1 + p] * coef[1] +
                            temp_src[2 + p] * coef[2],
                    yuv_shift);
        cr = descale((temp_src[0 + p] - y) * coef[3] + delta, yuv_shift);
        cb = descale((temp_src[2 + p] - y) * coef[4] + delta, yuv_shift);
        temp_dst[0 + p] = saturate(y, 0, 255);
        temp_dst[1 + p] = saturate(cr, 0, 255);
        temp_dst[2 + p] = saturate(cb, 0, 255);

        *((uint3*)dst) = *((uint3*)temp_dst);
    } else if (t == (rows * cols) / U8_PROCESS_PER_THREADS_X) {
        size_t rest = (rows * cols) % U8_PROCESS_PER_THREADS_X;
        if (rest != 0) {
            size_t offset_uchar = 3 * t * U8_PROCESS_PER_THREADS_X;
            src += offset_uchar;
            dst += offset_uchar;

            for (int i = 0; i < rest; i++, src += 3, dst += 3) {
                uchar temp_src[3], temp_dst[3];
                *((uchar3*)temp_src) = *((uchar3*)src);

                int Y = descale(temp_src[0] * coef[0] + temp_src[1] * coef[1] +
                                        temp_src[2] * coef[2],
                                yuv_shift);
                int Cr =
                        descale((temp_src[0] - Y) * coef[3] + delta, yuv_shift);
                int Cb =
                        descale((temp_src[2] - Y) * coef[4] + delta, yuv_shift);

                temp_dst[0] = saturate(Y, 0, 255);
                temp_dst[1] = saturate(Cr, 0, 255);
                temp_dst[2] = saturate(Cb, 0, 255);

                *((uchar3*)dst) = *((uchar3*)temp_dst);
            }
        }
    }
}

__global__ void cvt_rgb2yuv_32f_kernel(const float* src, float* dst,
                                       const size_t rows, const size_t cols,
                                       const size_t src_step,
                                       const size_t dst_step) {
    size_t t = blockIdx.x * blockDim.x + threadIdx.x;

    const float coef[] = {0.114f, 0.587f, 0.299f, 0.492f, 0.877f};
    const float delta = 0.5f;

    if (t < rows * cols) {
        size_t offset_float = t * 3;
        src += offset_float;
        dst += offset_float;

        float temp_src[3], temp_dst[3];
        *((float3*)temp_src) = *((float3*)src);

        float Y = temp_src[0] * coef[0] + temp_src[1] * coef[1] +
                  temp_src[2] * coef[2];
        temp_dst[0] = Y;
        temp_dst[1] = (temp_src[0] - Y) * coef[3] + delta;
        temp_dst[2] = (temp_src[2] - Y) * coef[4] + delta;

        *((float3*)dst) = *((float3*)temp_dst);
    }
}

__global__ void cvt_yuv2rgb_8u_kernel(const uchar* src, uchar* dst,
                                      const size_t rows, const size_t cols,
                                      const size_t src_step,
                                      const size_t dst_step) {
    size_t t = blockIdx.x * blockDim.x + threadIdx.x;

    const int yuv_shift = 14;
    const int coef[] = {33292, -6472, -9519, 18678};
    const int delta = 128;

    if (t < (rows * cols) / U8_PROCESS_PER_THREADS_X) {
        size_t offset_uchar = 3 * t * U8_PROCESS_PER_THREADS_X;
        src += offset_uchar;
        dst += offset_uchar;

        uchar temp_src[12], temp_dst[12];
        *((uint3*)temp_src) = *((uint3*)src);

        int p = 0;
        int R = temp_src[0 + p] +
                descale((temp_src[1 + p] - delta) * coef[0], yuv_shift);
        int G = temp_src[0 + p] +
                descale((temp_src[2 + p] - delta) * coef[2] +
                                (temp_src[1 + p] - delta) * coef[1],
                        yuv_shift);
        int B = temp_src[0 + p] +
                descale((temp_src[2 + p] - delta) * coef[3], yuv_shift);

        temp_dst[0 + p] = saturate(R, 0, 255);
        temp_dst[1 + p] = saturate(G, 0, 255);
        temp_dst[2 + p] = saturate(B, 0, 255);

        p += 3;
        R = temp_src[0 + p] +
            descale((temp_src[1 + p] - delta) * coef[0], yuv_shift);
        G = temp_src[0 + p] +
            descale((temp_src[2 + p] - delta) * coef[2] +
                            (temp_src[1 + p] - delta) * coef[1],
                    yuv_shift);
        B = temp_src[0 + p] +
            descale((temp_src[2 + p] - delta) * coef[3], yuv_shift);

        temp_dst[0 + p] = saturate(R, 0, 255);
        temp_dst[1 + p] = saturate(G, 0, 255);
        temp_dst[2 + p] = saturate(B, 0, 255);

        p += 3;
        R = temp_src[0 + p] +
            descale((temp_src[1 + p] - delta) * coef[0], yuv_shift);
        G = temp_src[0 + p] +
            descale((temp_src[2 + p] - delta) * coef[2] +
                            (temp_src[1 + p] - delta) * coef[1],
                    yuv_shift);
        B = temp_src[0 + p] +
            descale((temp_src[2 + p] - delta) * coef[3], yuv_shift);

        temp_dst[0 + p] = saturate(R, 0, 255);
        temp_dst[1 + p] = saturate(G, 0, 255);
        temp_dst[2 + p] = saturate(B, 0, 255);

        p += 3;
        R = temp_src[0 + p] +
            descale((temp_src[1 + p] - delta) * coef[0], yuv_shift);
        G = temp_src[0 + p] +
            descale((temp_src[2 + p] - delta) * coef[2] +
                            (temp_src[1 + p] - delta) * coef[1],
                    yuv_shift);
        B = temp_src[0 + p] +
            descale((temp_src[2 + p] - delta) * coef[3], yuv_shift);

        temp_dst[0 + p] = saturate(R, 0, 255);
        temp_dst[1 + p] = saturate(G, 0, 255);
        temp_dst[2 + p] = saturate(B, 0, 255);

        *((uint3*)dst) = *((uint3*)temp_dst);
    } else if (t == (rows * cols) / U8_PROCESS_PER_THREADS_X) {
        size_t rest = (rows * cols) % U8_PROCESS_PER_THREADS_X;
        if (rest != 0) {
            size_t offset_uchar = 3 * t * U8_PROCESS_PER_THREADS_X;
            src += offset_uchar;
            dst += offset_uchar;

            for (int i = 0; i < rest; i++, src += 3, dst += 3) {
                uchar Y = src[0], Cr = src[1], Cb = src[2];

                int R = Y + descale((Cr - delta) * coef[0], yuv_shift);
                int G = Y +
                        descale((Cb - delta) * coef[2] + (Cr - delta) * coef[1],
                                yuv_shift);
                int B = Y + descale((Cb - delta) * coef[3], yuv_shift);

                dst[0] = saturate(R, 0, 255);
                dst[1] = saturate(G, 0, 255);
                dst[2] = saturate(B, 0, 255);
            }
        }
    }
}

__global__ void cvt_yuv2rgb_32f_kernel(const float* src, float* dst,
                                       const size_t rows, const size_t cols,
                                       const size_t src_step,
                                       const size_t dst_step) {
    size_t t = blockIdx.x * blockDim.x + threadIdx.x;

    const float coef[] = {2.032f, -0.395f, -0.581f, 1.140f};
    const float delta = 0.5f;

    if (t < rows * cols) {
        size_t offset_float = t * 3;
        src += offset_float;
        dst += offset_float;

        float Y = src[0];
        float Cr = src[1];
        float Cb = src[2];

        float R = Y + (Cr - delta) * coef[0];
        float G = Y + (Cb - delta) * coef[2] + (Cr - delta) * coef[1];
        float B = Y + (Cb - delta) * coef[3];

        dst[0] = R;
        dst[1] = G;
        dst[2] = B;
    }
}

// convert planar or semi-planar YUV to gray. data type: uint8
__global__ void cvt_yuv2gray_psp_8u_kernel(const uchar* src, uchar* dst,
                                           const size_t dst_rows,
                                           const size_t dst_cols,
                                           const size_t src_step,
                                           const size_t dst_step) {
    int c = (blockIdx.x * blockDim.x + threadIdx.x) * U8_PROCESS_PER_THREADS_X;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    src += r * src_step + c;
    dst += r * dst_step + c;
    int remain = dst_cols - c;
    if (remain > U8_PROCESS_PER_THREADS_X)
        remain = U8_PROCESS_PER_THREADS_X;
    for (int i = 0; i < remain; ++i)
        *(dst++) = *(src++);
}

// convert semi-planar YUV to RGB or BGR. data type: uint8
// is_rgb: convert to RGB if true, otherwise convert to BGR
// is_nv12: decode src as YUV_NV12 if true, YUV_NV21 otherwise
template <bool is_rgb, bool is_nv12>
__global__ void cvt_yuv2rgbbgr_sp_8u_kernel(const uchar* src, uchar* dst,
                                            const size_t dst_rows,
                                            const size_t dst_cols,
                                            const size_t src_step,
                                            const size_t dst_step) {
    int c = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    int r = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
    if (c >= dst_cols || r >= dst_rows)
        return;

    dst += r * dst_step + c * 3;

    const uchar* pY = src + r * src_step + c;
    int Y00 = *pY;
    int Y01 = *(pY + 1);
    int Y10 = *(pY + src_step);
    int Y11 = *(pY + src_step + 1);

    const uchar* pUV = src + (dst_rows + r / 2) * src_step + c;
    int U, V;
    if (is_nv12) {
        U = *pUV;
        V = *(pUV + 1);
    } else {
        V = *pUV;
        U = *(pUV + 1);
    }

    int ruv = ((359 * (V - 128)) >> 8);
    int guv = -1 * ((88 * (U - 128) + 183 * (V - 128)) >> 8);
    int buv = ((454 * (U - 128)) >> 8);

#define SET_COLOR                     \
    if (is_rgb) {                     \
        dst[0] = saturate(R, 0, 255); \
        dst[1] = saturate(G, 0, 255); \
        dst[2] = saturate(B, 0, 255); \
    } else {                          \
        dst[0] = saturate(B, 0, 255); \
        dst[1] = saturate(G, 0, 255); \
        dst[2] = saturate(R, 0, 255); \
    }

    int R = Y00 + ruv;
    int G = Y00 + guv;
    int B = Y00 + buv;
    SET_COLOR
    dst += 3;

    R = Y01 + ruv;
    G = Y01 + guv;
    B = Y01 + buv;
    SET_COLOR
    dst += dst_step - 3;

    R = Y10 + ruv;
    G = Y10 + guv;
    B = Y10 + buv;
    SET_COLOR
    dst += 3;

    R = Y11 + ruv;
    G = Y11 + guv;
    B = Y11 + buv;
    SET_COLOR

#undef SET_COLOR
}

// convert planar YUV to RGB or BGR. data type: uint8
// is_rgb: convert to RGB if true, otherwise convert to BGR
// is_nv12: decode src as YUV_NV12 if true, YUV_NV21 otherwise
template <bool is_rgb, bool is_yu12>
__global__ void cvt_yuv2rgbbgr_p_8u_kernel(const uchar* src, uchar* dst,
                                           const size_t dst_rows,
                                           const size_t dst_cols,
                                           const size_t src_step,
                                           const size_t dst_step) {
    int c = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    int r = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
    if (c >= dst_cols || r >= dst_rows)
        return;

    dst += r * dst_step + c * 3;

    const uchar* pY = src + r * src_step + c;
    int Y00 = *pY;
    int Y01 = *(pY + 1);
    int Y10 = *(pY + src_step);
    int Y11 = *(pY + src_step + 1);

    size_t u_offset, v_offset;
    if (is_yu12) {
        u_offset = dst_rows * src_step + (r / 2) * (src_step / 2) + c / 2;
        v_offset = u_offset + (dst_rows / 4) * src_step;
    } else {
        v_offset = dst_rows * src_step + (r / 2) * (src_step / 2) + c / 2;
        u_offset = v_offset + (dst_rows / 4) * src_step;
    }
    int U = src[u_offset], V = src[v_offset];

    int ruv = ((359 * (V - 128)) >> 8);
    int guv = -1 * ((88 * (U - 128) + 183 * (V - 128)) >> 8);
    int buv = ((454 * (U - 128)) >> 8);

#define SET_COLOR                     \
    if (is_rgb) {                     \
        dst[0] = saturate(R, 0, 255); \
        dst[1] = saturate(G, 0, 255); \
        dst[2] = saturate(B, 0, 255); \
    } else {                          \
        dst[0] = saturate(B, 0, 255); \
        dst[1] = saturate(G, 0, 255); \
        dst[2] = saturate(R, 0, 255); \
    }

    int R = Y00 + ruv;
    int G = Y00 + guv;
    int B = Y00 + buv;
    SET_COLOR
    dst += 3;

    R = Y01 + ruv;
    G = Y01 + guv;
    B = Y01 + buv;
    SET_COLOR
    dst += dst_step - 3;

    R = Y10 + ruv;
    G = Y10 + guv;
    B = Y10 + buv;
    SET_COLOR
    dst += 3;

    R = Y11 + ruv;
    G = Y11 + guv;
    B = Y11 + buv;
    SET_COLOR

#undef SET_COLOR
}

#define CALL_CVT_OPR_8U_KERNEL(_func)                              \
    {                                                              \
        dim3 THREADS(THREADS_X);                                   \
        dim3 BLOCKS(DIVUP(src_cols* src_rows,                      \
                          THREADS_X* U8_PROCESS_PER_THREADS_X));   \
        cvt_##_func##_8u_kernel<<<BLOCKS, THREADS, 0, stream>>>(   \
                src, dst, src_rows, src_cols, src_step, dst_step); \
    }

#define CALL_CVT_OPR_32F_KERNEL(_func)                             \
    {                                                              \
        dim3 THREADS(THREADS_X);                                   \
        dim3 BLOCKS(DIVUP(src_cols* src_rows, THREADS_X));         \
        cvt_##_func##_32f_kernel<<<BLOCKS, THREADS, 0, stream>>>(  \
                src, dst, src_rows, src_cols, src_step, dst_step); \
    }

// convert planar or semi-planar YUV to gray, data tyoe: uint8
#define CALL_CVT_YUV2GRAY_PSP_OPR_8U_KERNEL                               \
    {                                                                     \
        dim3 THREADS(THREADS_X, 1);                                       \
        dim3 BLOCKS(DIVUP(dst_cols, THREADS_X* U8_PROCESS_PER_THREADS_X), \
                    dst_rows);                                            \
        cvt_yuv2gray_psp_8u_kernel<<<BLOCKS, THREADS, 0, stream>>>(       \
                src, dst, dst_rows, dst_cols, src_step, dst_step);        \
    }

// convert semi-planar YUV to RGB or BGR. data type: uint8
// is_rgb: convert to RGB if true, otherwise convert to BGR
// is_nv12: decode src as YUV_NV12 if true, YUV_NV21 otherwise
#define CALL_CVT_YUV2RGBBGR_SP_OPR_8U_KERNEL(is_rgb, is_nv12)                  \
    {                                                                          \
        dim3 THREADS(THREADS_X, THREADS_Y);                                    \
        dim3 BLOCKS(DIVUP(dst_cols / 2, THREADS_X),                            \
                    DIVUP(dst_rows / 2, THREADS_Y));                           \
        cvt_yuv2rgbbgr_sp_8u_kernel<is_rgb, is_nv12>                           \
                <<<BLOCKS, THREADS, 0, stream>>>(src, dst, dst_rows, dst_cols, \
                                                 src_step, dst_step);          \
    }

// convert planar YUV to RGB or BGR. data type: uint8
// is_rgb: convert to RGB if true, otherwise convert to BGR
// is_yu12: decode src as YUV_YU12 if true, YUV_YV12 otherwise
#define CALL_CVT_YUV2RGBBGR_P_OPR_8U_KERNEL(is_rgb, is_yu12)                   \
    {                                                                          \
        dim3 THREADS(THREADS_X, THREADS_Y);                                    \
        dim3 BLOCKS(DIVUP(dst_cols / 2, THREADS_X),                            \
                    DIVUP(dst_rows / 2, THREADS_Y));                           \
        cvt_yuv2rgbbgr_p_8u_kernel<is_rgb, is_yu12>                            \
                <<<BLOCKS, THREADS, 0, stream>>>(src, dst, dst_rows, dst_cols, \
                                                 src_step, dst_step);          \
    }

using namespace param_enumv;

void cvt_color_8u_proxy(const uchar* src, uchar* dst, const size_t src_rows,
                        const size_t src_cols, const size_t src_step,
                        const size_t dst_rows, const size_t dst_cols,
                        const size_t dst_step, const uint32_t mode,
                        cudaStream_t stream) {
    switch (mode) {
        case CvtColor::Mode::RGB2GRAY:
            CALL_CVT_OPR_8U_KERNEL(rgb2gray)
            break;
        case CvtColor::Mode::RGB2YUV:
            CALL_CVT_OPR_8U_KERNEL(rgb2yuv)
            break;
        case CvtColor::Mode::YUV2RGB:
            CALL_CVT_OPR_8U_KERNEL(yuv2rgb)
            break;
        case CvtColor::Mode::GRAY2RGB:
            CALL_CVT_OPR_8U_KERNEL(gray2rgb)
            break;
        case CvtColor::Mode::YUV2GRAY_NV12:
        case CvtColor::Mode::YUV2GRAY_NV21:
        case CvtColor::Mode::YUV2GRAY_YU12:
        case CvtColor::Mode::YUV2GRAY_YV12:
            CALL_CVT_YUV2GRAY_PSP_OPR_8U_KERNEL
            break;
        case CvtColor::Mode::YUV2RGB_NV12:
            CALL_CVT_YUV2RGBBGR_SP_OPR_8U_KERNEL(true, true)
            break;
        case CvtColor::Mode::YUV2RGB_NV21:
            CALL_CVT_YUV2RGBBGR_SP_OPR_8U_KERNEL(true, false)
            break;
        case CvtColor::Mode::YUV2BGR_NV12:
            CALL_CVT_YUV2RGBBGR_SP_OPR_8U_KERNEL(false, true)
            break;
        case CvtColor::Mode::YUV2BGR_NV21:
            CALL_CVT_YUV2RGBBGR_SP_OPR_8U_KERNEL(false, false)
            break;
        case CvtColor::Mode::YUV2RGB_YU12:
            CALL_CVT_YUV2RGBBGR_P_OPR_8U_KERNEL(true, true);
            break;
        case CvtColor::Mode::YUV2RGB_YV12:
            CALL_CVT_YUV2RGBBGR_P_OPR_8U_KERNEL(true, false);
            break;
        case CvtColor::Mode::YUV2BGR_YU12:
            CALL_CVT_YUV2RGBBGR_P_OPR_8U_KERNEL(false, true);
            break;
        case CvtColor::Mode::YUV2BGR_YV12:
            CALL_CVT_YUV2RGBBGR_P_OPR_8U_KERNEL(false, false);
            break;
        default:
            megdnn_throw("unsupported cvt_color mode for cuda");
            break;
    }
}

void cvt_color_32f_proxy(const float* src, float* dst, const size_t src_rows,
                         const size_t src_cols, const size_t src_step,
                         const size_t dst_rows, const size_t dst_cols,
                         const size_t dst_step, const uint32_t mode,
                         cudaStream_t stream) {
    MEGDNN_MARK_USED_VAR(dst_rows);
    MEGDNN_MARK_USED_VAR(dst_cols);
    switch (mode) {
        case CvtColor::Mode::RGB2GRAY:
            CALL_CVT_OPR_32F_KERNEL(rgb2gray)
            break;
        case CvtColor::Mode::RGB2YUV:
            CALL_CVT_OPR_32F_KERNEL(rgb2yuv)
            break;
        case CvtColor::Mode::YUV2RGB:
            CALL_CVT_OPR_32F_KERNEL(yuv2rgb)
            break;
        case CvtColor::Mode::GRAY2RGB:
            CALL_CVT_OPR_32F_KERNEL(gray2rgb)
            break;
        default:
            megdnn_throw("unsupported cvt_color mode for cuda");
            break;
    }
}

#undef CALL_CVT_OPR_8U_KERNEL
#undef CALL_CVT_OPR_32F_KERNEL
#undef CALL_CVT_YUV2GRAY_PSP_OPR_8U_KERNEL
#undef CALL_CVT_YUV2RGBBGR_SP_OPR_8U_KERNEL
#undef CALL_CVT_YUV2RGBBGR_P_OPR_8U_KERNEL

}  // namespace cvt_color
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
