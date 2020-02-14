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
 * \file dnn/src/cuda/warp_perspective/warp_perspective_cv.cu
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

#include "./warp_perspective_cv.cuh"
#include "src/cuda/cv/kernel_common.cuh"

#define at(A, r, c, ch) A[(r) * A##_step + (c) * CH + (ch)]
#define AB_BITS 10
#define AB_SCALE (1 << AB_BITS)
#define INTER_BITS 5
#define INTER_TAB_SIZE (1 << INTER_BITS)
#define INTER_REMAP_COEF_BITS 15
#define INTER_REMAP_COEF_SCALE (1 << INTER_REMAP_COEF_BITS)
#define ROUND_DELTA (1 << (AB_BITS - INTER_BITS - 1))
#define rep(i, n) for (int i = 0; i < (n); ++i)


#define BLOCK_THREADS_X0 64
#define BLOCK_THREADS_Y0 8
#define BLOCK_THREADS_X1 32
#define BLOCK_THREADS_Y1 8
#define PROCESS_PER_THREADS 8

namespace megdnn {
namespace cuda {
namespace warp_perspective {

//! transform matrix
__constant__ double M[9];
//! border_val
__constant__ float border_val;

using namespace megcv;

__global__ void preprocess_trans(double* trans, const float* src) {
    //! The size is 9
#pragma unroll
    for (size_t i = 0; i < 9; i++)
        trans[i] = src[i];
}

template <typename T, size_t CH, BorderMode bmode>
__global__ void warp_perspective_cv_kernel_LAN_cacheToLandVECTOR(
        const T * __restrict__ src, T *dst,
        const size_t src_rows, const size_t src_cols,
        const size_t dst_rows, const size_t dst_cols,
        const size_t src_step, const size_t dst_step)
{
    int dc = threadIdx.x + blockIdx.x * blockDim.x;
    int dr = threadIdx.y + blockIdx.y * (blockDim.y * PROCESS_PER_THREADS);

    __shared__ double cols_data[BLOCK_THREADS_X1][3];
    __shared__ double rows_data[BLOCK_THREADS_Y1*PROCESS_PER_THREADS][3];

    if (dr < dst_rows && dc < dst_cols) {

        if(threadIdx.y == 0)
        {
            cols_data[threadIdx.x][0] = M[0]*dc;
            cols_data[threadIdx.x][1] = M[3]*dc;
            cols_data[threadIdx.x][2] = M[6]*dc;
        }
        if(threadIdx.x == 0)
        {
            for(int i = 0; i < blockDim.y * PROCESS_PER_THREADS; i += blockDim.y)
            {
                rows_data[threadIdx.y + i][0] = M[1]*(dr+i)+M[2];
                rows_data[threadIdx.y + i][1] = M[4]*(dr+i)+M[5];
                rows_data[threadIdx.y + i][2] = M[7]*(dr+i)+M[8];
            }
        }
    }

    __syncthreads();

    if (dr < dst_rows && dc < dst_cols) {

        for(int i=0; i<blockDim.y*PROCESS_PER_THREADS; i+=blockDim.y)
        {
            double w = cols_data[threadIdx.x][2] + rows_data[threadIdx.y+i][2];
            w = (w == 0.000000) ? 0 : INTER_TAB_SIZE/w;
            double fsc = (cols_data[threadIdx.x][0] + rows_data[threadIdx.y+i][0])*w;
            double fsr = (cols_data[threadIdx.x][1] + rows_data[threadIdx.y+i][1])*w;
            fsc = fsc < (double)INT_MAX ? fsc : (double)INT_MAX;
            fsc = fsc > (double)INT_MIN ? fsc : (double)INT_MIN;
            fsr = fsr < (double)INT_MAX ? fsr : (double)INT_MAX;
            fsr = fsr > (double)INT_MIN ? fsr : (double)INT_MIN;
            int sc = (int)lrint(fsc);
            int sr = (int)lrint(fsr);
            int fc = sc & (INTER_TAB_SIZE - 1);
            int fr = sr & (INTER_TAB_SIZE - 1);
            sc = sc >> INTER_BITS;
            sr = sr >> INTER_BITS;
            sc = sc < -32768 ? -32768 : ( sc > 32767 ? 32767 : sc);
            sr = sr < -32768 ? -32768 : ( sr > 32767 ? 32767 : sr);
            const int ksize = IModeTrait<INTER_LANCZOS4>::ksize;
            float coefr[ksize], coefc[ksize];
            int x[ksize], y[ksize];
            if (bmode == BORDER_TRANSPARENT &&
                    ((unsigned)sr >= (unsigned)src_rows ||
                     (unsigned)sc >= (unsigned)src_cols
                    )) {
                continue;
            }
            interpolate_coefs<INTER_LANCZOS4>((float)fr/INTER_TAB_SIZE, coefr);
            interpolate_coefs<INTER_LANCZOS4>((float)fc/INTER_TAB_SIZE, coefc);
            const BorderMode bmode1 = BModeTrait<bmode>::bmode1;
            {
#pragma unroll
                rep(k, ksize) {
                    x[k] = border_interpolate<bmode1>(sr+k-(ksize/2)+1, src_rows);
                }
#pragma unroll
                rep(k, ksize) {
                    y[k] = border_interpolate<bmode1>(sc+k-(ksize/2)+1, src_cols);
                }
            }
            float sum[CH] = {0};
            rep(kr, ksize) {
                if (x[kr] < 0) {
#pragma unroll
                    rep(ch, CH) sum[ch] += coefr[kr]*border_val;
                    continue;
                }
#pragma unroll
                rep(kc, ksize) {
                    if (y[kc] < 0) {
#pragma unroll
                        rep(ch, CH) {
                            sum[ch] += coefr[kr]*coefc[kc]*border_val;
                        }
                    } else {
#pragma unroll
                        rep(ch, CH) {
                            sum[ch] += coefr[kr]*coefc[kc]*at(src, x[kr], y[kc], ch);
                        }
                    }
                }
            }
#pragma unroll
            rep(ch, CH) {
                typedef typename TypeTrait<T>::WorkType WorkType;
                if(dr+i < dst_rows)
                {
                    if (TypeTrait<T>::need_saturate) {
                        at(dst, dr+i, dc, ch) = saturate<WorkType>(
                                sum[ch],
                                TypeTrait<T>::min(),
                                TypeTrait<T>::max());
                    } else {
                        at(dst, dr+i, dc, ch) = sum[ch];
                    }
                }
            }
        }
    }
}

    template <typename T, size_t CH, BorderMode bmode>
__global__ void warp_perspective_cv_kernel_CUBIC_cacheToLAndVECTOR(
        const T * __restrict__ src, T *dst,
        const size_t src_rows, const size_t src_cols,
        const size_t dst_rows, const size_t dst_cols,
        const size_t src_step, const size_t dst_step)
{
    int dc = threadIdx.x + blockIdx.x * blockDim.x;
    int dr = threadIdx.y + blockIdx.y * (blockDim.y * PROCESS_PER_THREADS);

    __shared__ double cols_data[BLOCK_THREADS_X1][3];
    __shared__ double rows_data[BLOCK_THREADS_Y1*PROCESS_PER_THREADS][3];

    if (dr < dst_rows && dc < dst_cols) {

        if(threadIdx.y == 0)
        {
            cols_data[threadIdx.x][0] = M[0]*dc;
            cols_data[threadIdx.x][1] = M[3]*dc;
            cols_data[threadIdx.x][2] = M[6]*dc;
        }
        if(threadIdx.x == 0)
        {
            for(int i = 0; i < blockDim.y * PROCESS_PER_THREADS; i += blockDim.y)
            {
                rows_data[threadIdx.y + i][0] = M[1]*(dr+i)+M[2];
                rows_data[threadIdx.y + i][1] = M[4]*(dr+i)+M[5];
                rows_data[threadIdx.y + i][2] = M[7]*(dr+i)+M[8];
            }
        }

    }

    __syncthreads();

    if (dr < dst_rows && dc < dst_cols) {

        for(int i=0; i<blockDim.y*PROCESS_PER_THREADS; i+=blockDim.y)
        {
            double w = cols_data[threadIdx.x][2] + rows_data[threadIdx.y+i][2];
            w = (w == 0.000000) ? 0 : INTER_TAB_SIZE/w;
            double fsc = (cols_data[threadIdx.x][0] + rows_data[threadIdx.y+i][0])*w;
            double fsr = (cols_data[threadIdx.x][1] + rows_data[threadIdx.y+i][1])*w;
            fsc = fsc < (double)INT_MAX ? fsc : (double)INT_MAX;
            fsc = fsc > (double)INT_MIN ? fsc : (double)INT_MIN;
            fsr = fsr < (double)INT_MAX ? fsr : (double)INT_MAX;
            fsr = fsr > (double)INT_MIN ? fsr : (double)INT_MIN;
            int sc = (int)lrint(fsc);
            int sr = (int)lrint(fsr);
            int fc = sc & (INTER_TAB_SIZE - 1);
            int fr = sr & (INTER_TAB_SIZE - 1);
            sc = sc >> INTER_BITS;
            sr = sr >> INTER_BITS;
            sc = sc < -32768 ? -32768 : ( sc > 32767 ? 32767 : sc);
            sr = sr < -32768 ? -32768 : ( sr > 32767 ? 32767 : sr);

            const int ksize = IModeTrait<INTER_CUBIC>::ksize;
            float coefr[ksize], coefc[ksize];
            int x[ksize], y[ksize];

            if (bmode == BORDER_TRANSPARENT &&
                    ((unsigned)sr >= (unsigned)src_rows ||
                     (unsigned)sc >= (unsigned)src_cols
                    )) {
                continue;
            }

            interpolate_coefs<INTER_CUBIC>((float)fr/INTER_TAB_SIZE, coefr);
            interpolate_coefs<INTER_CUBIC>((float)fc/INTER_TAB_SIZE, coefc);

            const BorderMode bmode1 = BModeTrait<bmode>::bmode1;
            {
#pragma unroll
                rep(k, ksize) {
                    x[k] = border_interpolate<bmode1>(sr+k-(ksize/2)+1, src_rows);
                }
#pragma unroll
                rep(k, ksize) {
                    y[k] = border_interpolate<bmode1>(sc+k-(ksize/2)+1, src_cols);
                }
            }
            float sum[CH] = {0};
            rep(kr, ksize) {
                if (x[kr] < 0) {
#pragma unroll
                    rep(ch, CH) sum[ch] += coefr[kr]*border_val;
                    continue;
                }
#pragma unroll
                rep(kc, ksize) {
                    if (y[kc] < 0) {
#pragma unroll
                        rep(ch, CH) {
                            sum[ch] += coefr[kr]*coefc[kc]*border_val;
                        }
                    } else {
#pragma unroll
                        rep(ch, CH) {
                            sum[ch] += coefr[kr]*coefc[kc]*at(src, x[kr], y[kc], ch);
                        }
                    }
                }
            }
#pragma unroll
            rep(ch, CH) {
                typedef typename TypeTrait<T>::WorkType WorkType;
                if(dr+i < dst_rows)
                {
                    if (TypeTrait<T>::need_saturate) {
                        at(dst, dr+i, dc, ch) = saturate<WorkType>(
                                sum[ch],
                                TypeTrait<T>::min(),
                                TypeTrait<T>::max());
                    } else {
                        at(dst, dr+i, dc, ch) = sum[ch];
                    }
                }
            }
        }
    }
}

    template <typename T, size_t CH, BorderMode bmode>
__global__ void warp_perspective_cv_kernel_LINEAR_cacheToLAndVECTOR(
        const T * __restrict__ src, T *dst,
        const size_t src_rows, const size_t src_cols,
        const size_t dst_rows, const size_t dst_cols,
        const size_t src_step, const size_t dst_step)
{
    int dc = threadIdx.x + blockIdx.x * blockDim.x;
    int dr = threadIdx.y + blockIdx.y * (blockDim.y * PROCESS_PER_THREADS);

    __shared__ double cols_data[BLOCK_THREADS_X1][3];
    __shared__ double rows_data[BLOCK_THREADS_Y1*PROCESS_PER_THREADS][3];

    if (dr < dst_rows && dc < dst_cols) {
        if(threadIdx.y == 0)
        {
            cols_data[threadIdx.x][0] = M[0]*dc;
            cols_data[threadIdx.x][1] = M[3]*dc;
            cols_data[threadIdx.x][2] = M[6]*dc;
        }
        if(threadIdx.x == 0)
        {
            for(int i = 0; i < blockDim.y * PROCESS_PER_THREADS; i += blockDim.y)
            {
                rows_data[threadIdx.y + i][0] = M[1]*(dr+i)+M[2];
                rows_data[threadIdx.y + i][1] = M[4]*(dr+i)+M[5];
                rows_data[threadIdx.y + i][2] = M[7]*(dr+i)+M[8];
            }
        }

    }

    __syncthreads();

    if (dr < dst_rows && dc < dst_cols) {

        for(int i=0; i<blockDim.y*PROCESS_PER_THREADS; i+=blockDim.y)
        {
            double w = cols_data[threadIdx.x][2] + rows_data[threadIdx.y+i][2];
            w = (w == 0.000000) ? 0 : INTER_TAB_SIZE/w;
            double fsc = (cols_data[threadIdx.x][0] + rows_data[threadIdx.y+i][0])*w;
            double fsr = (cols_data[threadIdx.x][1] + rows_data[threadIdx.y+i][1])*w;
            fsc = fsc < (double)INT_MAX ? fsc : (double)INT_MAX;
            fsc = fsc > (double)INT_MIN ? fsc : (double)INT_MIN;
            fsr = fsr < (double)INT_MAX ? fsr : (double)INT_MAX;
            fsr = fsr > (double)INT_MIN ? fsr : (double)INT_MIN;
            int sc = (int)lrint(fsc);
            int sr = (int)lrint(fsr);
            int fc = sc & (INTER_TAB_SIZE - 1);
            int fr = sr & (INTER_TAB_SIZE - 1);
            sc = sc >> INTER_BITS;
            sr = sr >> INTER_BITS;
            sc = sc < -32768 ? -32768 : ( sc > 32767 ? 32767 : sc);
            sr = sr < -32768 ? -32768 : ( sr > 32767 ? 32767 : sr);

            const int ksize = IModeTrait<INTER_LINEAR>::ksize;
            float coefr[ksize], coefc[ksize];
            int x[ksize], y[ksize];

            if (bmode == BORDER_TRANSPARENT &&
                    ((unsigned)(sr+1) >= (unsigned)src_rows ||
                     (unsigned)(sc+1) >= (unsigned)src_cols
                    )) {
                continue;
            }

            interpolate_coefs<INTER_LINEAR>((float)fr/INTER_TAB_SIZE, coefr);
            interpolate_coefs<INTER_LINEAR>((float)fc/INTER_TAB_SIZE, coefc);

            const BorderMode bmode1 = BModeTrait<bmode>::bmode1;
            {
#pragma unroll
                rep(k, ksize) {
                    x[k] = border_interpolate<bmode1>(sr+k-(ksize/2)+1, src_rows);
                }
#pragma unroll
                rep(k, ksize) {
                    y[k] = border_interpolate<bmode1>(sc+k-(ksize/2)+1, src_cols);
                }
            }
            float sum[CH] = {0};
            rep(kr, ksize) {
                if (x[kr] < 0) {
#pragma unroll
                    rep(ch, CH) sum[ch] += coefr[kr]*border_val;
                    continue;
                }
#pragma unroll
                rep(kc, ksize) {
                    if (y[kc] < 0) {
#pragma unroll
                        rep(ch, CH) {
                            sum[ch] += coefr[kr]*coefc[kc]*border_val;
                        }
                    } else {
#pragma unroll
                        rep(ch, CH) {
                            sum[ch] += coefr[kr]*coefc[kc]*at(src, x[kr], y[kc], ch);
                        }
                    }
                }
            }
#pragma unroll
            rep(ch, CH) {
                typedef typename TypeTrait<T>::WorkType WorkType;
                if(dr+i < dst_rows)
                {
                    if (TypeTrait<T>::need_saturate) {
                        at(dst, dr+i, dc, ch) = saturate<WorkType>(
                                sum[ch],
                                TypeTrait<T>::min(),
                                TypeTrait<T>::max());
                    } else {
                        at(dst, dr+i, dc, ch) = sum[ch];
                    }
                }
            }
        }
    }
}


    template <typename T, size_t CH, BorderMode bmode>
__global__ void warp_perspective_cv_kernel_cacheToL_NEAREST(
        const T * __restrict__ src, T *dst,
        const size_t src_rows, const size_t src_cols,
        const size_t dst_rows, const size_t dst_cols,
        const size_t src_step, const size_t dst_step)
{
#define SET_DST_CH_VALUE \
    if (CH == 1) { \
        dst[dst_address_increase] = src[src_address_increase]; \
    } else { \
        dst[dst_address_increase] = src[src_address_increase]; \
        dst[dst_address_increase+1] = src[src_address_increase+1]; \
        dst[dst_address_increase+2] = src[src_address_increase+2]; \
    } \

    int dc = threadIdx.x + blockIdx.x * blockDim.x;
    int dr = threadIdx.y + blockIdx.y * blockDim.y;

    __shared__ double cols_data[BLOCK_THREADS_X1][3];
    __shared__ double rows_data[BLOCK_THREADS_Y1][3];

    if (dr < dst_rows && dc < dst_cols) {
        if(threadIdx.y == 0)
        {
            cols_data[threadIdx.x][0] = M[0]*dc;
            cols_data[threadIdx.x][1] = M[3]*dc;
            cols_data[threadIdx.x][2] = M[6]*dc;
        }
        if(threadIdx.x == 0)
        {
            rows_data[threadIdx.y][0] = M[1]*dr+M[2];
            rows_data[threadIdx.y][1] = M[4]*dr+M[5];
            rows_data[threadIdx.y][2] = M[7]*dr+M[8];
        }

    }

    __syncthreads();

    if (dr < dst_rows && dc < dst_cols) {

        double w = cols_data[threadIdx.x][2] + rows_data[threadIdx.y][2];
        w = (w == 0) ? 0 : 1/w;
        double fsc = (cols_data[threadIdx.x][0] + rows_data[threadIdx.y][0])*w;
        double fsr = (cols_data[threadIdx.x][1] + rows_data[threadIdx.y][1])*w;
        int sc = saturate_cast_short(fsc);
        int sr = saturate_cast_short(fsr);

        size_t dst_address_increase = dr*dst_step + dc*CH;
        if ((size_t)sc < src_cols && (size_t)sr < src_rows) {
            size_t src_address_increase = sr*src_step + sc*CH;
            SET_DST_CH_VALUE
            return;
        }


        if (bmode == BORDER_REPLICATE) {
            sr = saturate(sr, 0, (int)src_rows-1);
            sc = saturate(sc, 0, (int)src_cols-1);

            size_t src_address_increase = sr*src_step + sc*CH;
            SET_DST_CH_VALUE
        } else if (bmode == BORDER_CONSTANT) {
            if (CH == 1) {
                dst[dst_address_increase] = border_val;
            } else {
                dst[dst_address_increase + 0] = border_val;
                dst[dst_address_increase + 1] = border_val;
                dst[dst_address_increase + 2] = border_val;
            }
        } else if (bmode != BORDER_TRANSPARENT) {
            sr = border_interpolate<bmode>(sr, src_rows);
            sc = border_interpolate<bmode>(sc, src_cols);

            size_t src_address_increase = sr*src_step + sc*CH;
            src_address_increase = sr*src_step + sc*CH;
            SET_DST_CH_VALUE
        }

    }
#undef SET_DST_CH_VALUE
}


    template <typename T, size_t CH, BorderMode bmode>
__global__ void warp_perspective_cv_kernel_NEAREST_VECTOR(const T * __restrict__ src, T *dst,
        const size_t src_rows, const size_t src_cols,
        const size_t dst_rows, const size_t dst_cols,
        const size_t src_step, const size_t dst_step)
{
    int dc = threadIdx.x + blockIdx.x * blockDim.x;
    int dr = threadIdx.y + blockIdx.y * (blockDim.y * PROCESS_PER_THREADS);

#define SET_DST_CH_VALUE \
    if (CH == 1) { \
        dst[dst_address_increase] = src[src_address_increase]; \
    } else { \
        dst[dst_address_increase] = src[src_address_increase]; \
        dst[dst_address_increase+1] = src[src_address_increase+1]; \
        dst[dst_address_increase+2] = src[src_address_increase+2]; \
    }

    if (dr < dst_rows && dc < dst_cols) {
        double w0 = M[6]*dc + M[7]*dr + M[8];
        double fc0 = M[0]*dc + M[1]*dr + M[2];
        double fr0 = M[3]*dc + M[4]*dr + M[5];
        for(int i=0; i < blockDim.y*PROCESS_PER_THREADS; i+=blockDim.y)
        {
            if(dr + i >= dst_rows)
                return ;

            //! To make the result equal to the naive version
            double w = w0 + M[7] * i;
            w = w ? 1./w : 0;
            double fsc = (fc0 +  M[1] * i) * w;
            double fsr = (fr0 + M[4] * i) * w;

            fsc = fsc < (double)INT_MAX ? fsc : (double)INT_MAX;
            fsc = fsc > (double)INT_MIN ? fsc : (double)INT_MIN;
            fsr = fsr < (double)INT_MAX ? fsr : (double)INT_MAX;
            fsr = fsr > (double)INT_MIN ? fsr : (double)INT_MIN;

            int sc = saturate_cast_short(fsc);
            int sr = saturate_cast_short(fsr);

            size_t dst_address_increase = (dr+i)*dst_step + dc*CH;
            if ((size_t)sc < src_cols && (size_t)sr < src_rows) {
                size_t src_address_increase = sr*src_step + sc*CH;
                SET_DST_CH_VALUE
                continue;
            }


            if (bmode == BORDER_REPLICATE) {
                sr = saturate(sr, 0, (int)src_rows-1);
                sc = saturate(sc, 0, (int)src_cols-1);

                size_t src_address_increase = sr*src_step + sc*CH;
                SET_DST_CH_VALUE
            } else if (bmode == BORDER_CONSTANT) {
                if (CH == 1) {
                    dst[dst_address_increase] = border_val;
                } else {
                    dst[dst_address_increase + 0] = border_val;
                    dst[dst_address_increase + 1] = border_val;
                    dst[dst_address_increase + 2] = border_val;
                }
            } else if (bmode != BORDER_TRANSPARENT) {
                sr = border_interpolate<bmode>(sr, src_rows);
                sc = border_interpolate<bmode>(sc, src_cols);

                size_t src_address_increase = sr*src_step + sc*CH;
                SET_DST_CH_VALUE
            }

        }
    }
#undef SET_DST_CH_VALUE
}


template <typename T, size_t CH>
void warp_perspective_cv_proxy(const T *src, T *dst,
        const size_t src_rows, const size_t src_cols,
        const size_t dst_rows, const size_t dst_cols,
        const size_t src_step, const size_t dst_step,
        BorderMode bmode, InterpolationMode imode,
        const float * trans,
        const T bval,
        double* workspace,
        cudaStream_t stream
        )
{
    preprocess_trans<<<1, 1, 0, stream>>>(workspace, trans);
    cuda_check(cudaStreamSynchronize(stream));
    //! Copy trans to const memory
    cuda_check(cudaMemcpyToSymbol(M, workspace, sizeof(double) * 9, 0, cudaMemcpyHostToDevice));
    //! Copy bval to const memory
    cuda_check(cudaMemcpyToSymbol(border_val, &bval, sizeof(float), 0, cudaMemcpyHostToDevice));

    dim3 THREADS, BLOCKS;
    dim3 THREADS_VECTOR, BLOCKS_VECTOR;
    switch (imode){
        case INTER_NEAREST:

            if(CH == 3 && sizeof(T) == sizeof(float)){

                THREADS.x = BLOCK_THREADS_X1;
                THREADS.y = BLOCK_THREADS_Y1;
                BLOCKS.x = DIVUP(dst_cols, THREADS.x);
                BLOCKS.y = DIVUP(dst_rows, THREADS.y);

                switch (bmode) {
                    case BORDER_REPLICATE:
                        warp_perspective_cv_kernel_cacheToL_NEAREST <T, CH, BORDER_REPLICATE><<<BLOCKS, THREADS, 0, stream>>>(src, dst, src_rows, src_cols, dst_rows, dst_cols, src_step, dst_step);
                        break;
                    case BORDER_REFLECT:
                        warp_perspective_cv_kernel_cacheToL_NEAREST <T, CH, BORDER_REFLECT><<<BLOCKS, THREADS, 0, stream>>>(src, dst, src_rows, src_cols, dst_rows, dst_cols, src_step, dst_step);
                        break;
                    case BORDER_REFLECT_101:
                        warp_perspective_cv_kernel_cacheToL_NEAREST <T, CH, BORDER_REFLECT_101><<<BLOCKS, THREADS, 0, stream>>>(src, dst, src_rows, src_cols, dst_rows, dst_cols, src_step, dst_step);
                        break;
                    case BORDER_WRAP:
                        warp_perspective_cv_kernel_cacheToL_NEAREST <T, CH, BORDER_WRAP><<<BLOCKS, THREADS, 0, stream>>>(src, dst, src_rows, src_cols, dst_rows, dst_cols, src_step, dst_step);
                        break;
                    case BORDER_CONSTANT:
                        warp_perspective_cv_kernel_cacheToL_NEAREST <T, CH, BORDER_CONSTANT><<<BLOCKS, THREADS, 0, stream>>>(src, dst, src_rows, src_cols, dst_rows, dst_cols, src_step, dst_step);
                        break;
                    case BORDER_TRANSPARENT:
                        warp_perspective_cv_kernel_cacheToL_NEAREST <T, CH, BORDER_TRANSPARENT><<<BLOCKS, THREADS, 0, stream>>>(src, dst, src_rows, src_cols, dst_rows, dst_cols, src_step, dst_step);
                        break;
                    default:
                        break;
                }
            }
            else{

                THREADS_VECTOR.x = BLOCK_THREADS_X1;
                THREADS_VECTOR.y = BLOCK_THREADS_Y1;
                BLOCKS_VECTOR.x = DIVUP(dst_cols, THREADS_VECTOR.x);
                BLOCKS_VECTOR.y = DIVUP(dst_rows, THREADS_VECTOR.y*PROCESS_PER_THREADS);

                cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

                switch (bmode) {
                    case BORDER_REPLICATE:
                        warp_perspective_cv_kernel_NEAREST_VECTOR<T, CH, BORDER_REPLICATE><<<BLOCKS_VECTOR, THREADS_VECTOR, 0, stream>>>(src, dst, src_rows, src_cols, dst_rows, dst_cols, src_step, dst_step);
                        break;
                    case BORDER_REFLECT:
                        warp_perspective_cv_kernel_NEAREST_VECTOR<T, CH, BORDER_REFLECT><<<BLOCKS_VECTOR, THREADS_VECTOR, 0, stream>>>(src, dst, src_rows, src_cols, dst_rows, dst_cols, src_step, dst_step);
                        break;
                    case BORDER_REFLECT_101:
                        warp_perspective_cv_kernel_NEAREST_VECTOR<T, CH, BORDER_REFLECT_101><<<BLOCKS_VECTOR, THREADS_VECTOR, 0, stream>>>(src, dst, src_rows, src_cols, dst_rows, dst_cols, src_step, dst_step);
                        break;
                    case BORDER_WRAP:
                        warp_perspective_cv_kernel_NEAREST_VECTOR<T, CH, BORDER_WRAP><<<BLOCKS_VECTOR, THREADS_VECTOR, 0, stream>>>(src, dst, src_rows, src_cols, dst_rows, dst_cols, src_step, dst_step);
                        break;
                    case BORDER_CONSTANT:
                        warp_perspective_cv_kernel_NEAREST_VECTOR<T, CH, BORDER_CONSTANT><<<BLOCKS_VECTOR, THREADS_VECTOR, 0, stream>>>(src, dst, src_rows, src_cols, dst_rows, dst_cols, src_step, dst_step);
                        break;
                    case BORDER_TRANSPARENT:
                        warp_perspective_cv_kernel_NEAREST_VECTOR<T, CH, BORDER_TRANSPARENT><<<BLOCKS_VECTOR, THREADS_VECTOR, 0, stream>>>(src, dst, src_rows, src_cols, dst_rows, dst_cols, src_step, dst_step);
                        break;
                    default:
                        break;
                }
            }

            break;

        case INTER_LINEAR:

            {
                {
                    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

                    THREADS_VECTOR.x = BLOCK_THREADS_X1;
                    THREADS_VECTOR.y = BLOCK_THREADS_Y1;
                    BLOCKS_VECTOR.x = DIVUP(dst_cols, THREADS_VECTOR.x);
                    BLOCKS_VECTOR.y = DIVUP(dst_rows, THREADS_VECTOR.y*PROCESS_PER_THREADS);

                    switch (bmode){

                        case BORDER_REPLICATE:
                            warp_perspective_cv_kernel_LINEAR_cacheToLAndVECTOR<T, CH, BORDER_REPLICATE><<<BLOCKS_VECTOR, THREADS_VECTOR, 0, stream>>>(src, dst, src_rows, src_cols, dst_rows, dst_cols, src_step, dst_step);
                            break;
                        case BORDER_REFLECT:
                            warp_perspective_cv_kernel_LINEAR_cacheToLAndVECTOR<T, CH, BORDER_REFLECT><<<BLOCKS_VECTOR, THREADS_VECTOR, 0, stream>>>(src, dst, src_rows, src_cols, dst_rows, dst_cols, src_step, dst_step);
                            break;
                        case BORDER_REFLECT_101:
                            warp_perspective_cv_kernel_LINEAR_cacheToLAndVECTOR<T, CH, BORDER_REFLECT_101><<<BLOCKS_VECTOR, THREADS_VECTOR, 0, stream>>>(src, dst, src_rows, src_cols, dst_rows, dst_cols, src_step, dst_step);
                            break;
                        case BORDER_WRAP:
                            warp_perspective_cv_kernel_LINEAR_cacheToLAndVECTOR<T, CH, BORDER_WRAP><<<BLOCKS_VECTOR, THREADS_VECTOR, 0, stream>>>(src, dst, src_rows, src_cols, dst_rows, dst_cols, src_step, dst_step);
                            break;
                        case BORDER_CONSTANT:
                            warp_perspective_cv_kernel_LINEAR_cacheToLAndVECTOR<T, CH, BORDER_CONSTANT><<<BLOCKS_VECTOR, THREADS_VECTOR, 0, stream>>>(src, dst, src_rows, src_cols, dst_rows, dst_cols, src_step, dst_step);
                            break;
                        case BORDER_TRANSPARENT:
                            if (CH == 3)
                                warp_perspective_cv_kernel_LINEAR_cacheToLAndVECTOR<T, CH, BORDER_TRANSPARENT><<<BLOCKS_VECTOR, THREADS_VECTOR, 0, stream>>>(src, dst, src_rows, src_cols, dst_rows, dst_cols, src_step, dst_step);
                            break;
                        default:
                            break;
                    }
                }
            }

            break;

        case INTER_CUBIC:

            THREADS_VECTOR.x = BLOCK_THREADS_X1;
            THREADS_VECTOR.y = BLOCK_THREADS_Y1;
            BLOCKS_VECTOR.x = DIVUP(dst_cols, THREADS_VECTOR.x);
            BLOCKS_VECTOR.y = DIVUP(dst_rows, THREADS_VECTOR.y*PROCESS_PER_THREADS);

            switch (bmode){

                case BORDER_REPLICATE:
                    warp_perspective_cv_kernel_CUBIC_cacheToLAndVECTOR<T, CH, BORDER_REPLICATE><<<BLOCKS_VECTOR, THREADS_VECTOR, 0, stream>>>(src, dst, src_rows, src_cols, dst_rows, dst_cols, src_step, dst_step);
                    break;
                case BORDER_REFLECT:
                    warp_perspective_cv_kernel_CUBIC_cacheToLAndVECTOR<T, CH, BORDER_REFLECT><<<BLOCKS_VECTOR, THREADS_VECTOR, 0, stream>>>(src, dst, src_rows, src_cols, dst_rows, dst_cols, src_step, dst_step);
                    break;
                case BORDER_REFLECT_101:
                    warp_perspective_cv_kernel_CUBIC_cacheToLAndVECTOR<T, CH, BORDER_REFLECT_101><<<BLOCKS_VECTOR, THREADS_VECTOR, 0, stream>>>(src, dst, src_rows, src_cols, dst_rows, dst_cols, src_step, dst_step);
                    break;
                case BORDER_WRAP:
                    warp_perspective_cv_kernel_CUBIC_cacheToLAndVECTOR<T, CH, BORDER_WRAP><<<BLOCKS_VECTOR, THREADS_VECTOR, 0, stream>>>(src, dst, src_rows, src_cols, dst_rows, dst_cols, src_step, dst_step);
                    break;
                case BORDER_CONSTANT:
                    warp_perspective_cv_kernel_CUBIC_cacheToLAndVECTOR<T, CH, BORDER_CONSTANT><<<BLOCKS_VECTOR, THREADS_VECTOR, 0, stream>>>(src, dst, src_rows, src_cols, dst_rows, dst_cols, src_step, dst_step);
                    break;
                case BORDER_TRANSPARENT:
                    warp_perspective_cv_kernel_CUBIC_cacheToLAndVECTOR<T, CH, BORDER_TRANSPARENT><<<BLOCKS_VECTOR, THREADS_VECTOR, 0, stream>>>(src, dst, src_rows, src_cols, dst_rows, dst_cols, src_step, dst_step);
                    break;
                default:
                    break;
            }
            break;

        case INTER_LANCZOS4:

            {
                THREADS_VECTOR.x = BLOCK_THREADS_X1;
                THREADS_VECTOR.y = BLOCK_THREADS_Y1;
                BLOCKS_VECTOR.x = DIVUP(dst_cols, THREADS_VECTOR.x);
                BLOCKS_VECTOR.y = DIVUP(dst_rows, THREADS_VECTOR.y*PROCESS_PER_THREADS);

                switch (bmode){

                    case BORDER_REPLICATE:
                        warp_perspective_cv_kernel_LAN_cacheToLandVECTOR<T, CH, BORDER_REPLICATE><<<BLOCKS_VECTOR, THREADS_VECTOR, 0, stream>>>(src, dst, src_rows, src_cols, dst_rows, dst_cols, src_step, dst_step);
                        break;
                    case BORDER_REFLECT:
                        warp_perspective_cv_kernel_LAN_cacheToLandVECTOR<T, CH, BORDER_REFLECT><<<BLOCKS_VECTOR, THREADS_VECTOR, 0, stream>>>(src, dst, src_rows, src_cols, dst_rows, dst_cols, src_step, dst_step);
                        break;
                    case BORDER_REFLECT_101:
                        warp_perspective_cv_kernel_LAN_cacheToLandVECTOR<T, CH, BORDER_REFLECT_101><<<BLOCKS_VECTOR, THREADS_VECTOR, 0, stream>>>(src, dst, src_rows, src_cols, dst_rows, dst_cols, src_step, dst_step);
                        break;
                    case BORDER_WRAP:
                        warp_perspective_cv_kernel_LAN_cacheToLandVECTOR<T, CH, BORDER_WRAP><<<BLOCKS_VECTOR, THREADS_VECTOR, 0, stream>>>(src, dst, src_rows, src_cols, dst_rows, dst_cols, src_step, dst_step);
                        break;
                    case BORDER_CONSTANT:
                        warp_perspective_cv_kernel_LAN_cacheToLandVECTOR<T, CH, BORDER_CONSTANT><<<BLOCKS_VECTOR, THREADS_VECTOR, 0, stream>>>(src, dst, src_rows, src_cols, dst_rows, dst_cols, src_step, dst_step);
                        break;
                    case BORDER_TRANSPARENT:
                        warp_perspective_cv_kernel_LAN_cacheToLandVECTOR<T, CH, BORDER_TRANSPARENT><<<BLOCKS_VECTOR, THREADS_VECTOR, 0, stream>>>(src, dst, src_rows, src_cols, dst_rows, dst_cols, src_step, dst_step);
                        break;
                    default:
                        break;
                }
            }

            break;

        default:
            break;

    }

}

template void warp_perspective_cv_proxy<float, 1>(const float *src, float *dst,
        const size_t src_rows, const size_t src_cols,
        const size_t dst_rows, const size_t dst_cols,
        const size_t src_step, const size_t dst_step,
        BorderMode bmode, InterpolationMode imode,
        const float * trans,
        const float border_val,
        double* workspace,
        cudaStream_t stream);

template void warp_perspective_cv_proxy<uchar, 1>(const uchar *src, uchar *dst,
        const size_t src_rows, const size_t src_cols,
        const size_t dst_rows, const size_t dst_cols,
        const size_t src_step, const size_t dst_step,
        BorderMode bmode, InterpolationMode imode,
        const float * trans,
        const uchar border_val,
        double* workspace,
        cudaStream_t stream);

template void warp_perspective_cv_proxy<float, 3>(const float *src, float *dst,
        const size_t src_rows, const size_t src_cols,
        const size_t dst_rows, const size_t dst_cols,
        const size_t src_step, const size_t dst_step,
        BorderMode bmode, InterpolationMode imode,
        const float * trans,
        const float border_val,
        double* workspace,
        cudaStream_t stream);

template void warp_perspective_cv_proxy<uchar, 3>(const uchar *src, uchar *dst,
        const size_t src_rows, const size_t src_cols,
        const size_t dst_rows, const size_t dst_cols,
        const size_t src_step, const size_t dst_step,
        BorderMode bmode, InterpolationMode imode,
        const float * trans,
        const uchar border_val,
        double* workspace,
        cudaStream_t stream);

} // warp_perspective
} // cuda
} // megdnn

// vim: syntax=cpp.doxygen
