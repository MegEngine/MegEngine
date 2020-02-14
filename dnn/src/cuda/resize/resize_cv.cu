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
 * \file dnn/src/cuda/resize/resize_cv.cu
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
#include "src/cuda/cv/kernel_common.cuh"
#include "src/cuda/resize/resize_cv.cuh"
#include "src/cuda/utils.cuh"

using namespace megdnn;
using namespace cuda;
using namespace megcv;

namespace {

#define SCALE 11
#define at(A, r, c, ch) A[(r)*A##_step + (c)*CH + (ch)]
#define ONE (1 << SCALE)

#define ELEMENTS_PER_THREADS 8
#define THREADS_X 32
#define THREADS_Y 16

__global__ void precompute_lanczos4_coef_f32(float* dst, float scale,
                                             size_t size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size)
        return;

    float fr = (tid + 0.5) * scale - 0.5;
    int* sr = (int*)(dst + size * 8);
    sr[tid] = (int)(floorf(fr));

    fr -= sr[tid];
    float coef[8];
    interpolate_lanczos4_coefs(fr, coef);
#pragma unroll
    for (int j = 0, index = 0; j < 8; j++, index += size) {
        dst[tid + index] = coef[j];
    }
}

__global__ void precompute_lanczos4_coef_u8(short* dst, float scale,
                                            size_t size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size)
        return;

    float fr = (tid + 0.5) * scale - 0.5;
    int* sr = (int*)(dst + size * 8);
    sr[tid] = (int)(floorf(fr));

    fr -= sr[tid];
    float coef[8];
    interpolate_lanczos4_coefs(fr, coef);
#pragma unroll
    for (int j = 0, index = 0; j < 8; j++, index += size) {
        dst[tid + index] = (short)(coef[j] * ONE);
    }
}

__global__ void precompute_cubic_coef_f32(float* dst, float scale,
                                          size_t size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size)
        return;

    float fr = (tid + 0.5) * scale - 0.5;
    int* sr = (int*)(dst + size * 4);
    sr[tid] = (int)(floorf(fr));

    fr -= sr[tid];
    float coef[4];
    interpolate_cubic_coefs(fr, coef);
#pragma unroll
    for (int j = 0, index = 0; j < 4; j++, index += size) {
        dst[tid + index] = coef[j];
    }
}

__global__ void precompute_cubic_coef_u8(short* dst, float scale, size_t size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= size)
        return;

    float fr = (tid + 0.5) * scale - 0.5;
    int* sr = (int*)(dst + size * 4);
    sr[tid] = (int)(floorf(fr));

    fr -= sr[tid];
    float coef[4];
    interpolate_cubic_coefs(fr, coef);
#pragma unroll
    for (int j = 0, index = 0; j < 4; j++, index += size) {
        dst[tid + index] = (short)(coef[j] * ONE);
    }
}

template <typename T, size_t CH>
__global__ void resize_nearest_vector_kernel(
        const T* src, T* dst, const size_t dst_rows, const size_t dst_cols,
        const size_t src_step, const size_t dst_step, const float row_scale,
        const float col_scale) {
    size_t dc = blockIdx.x * blockDim.x + threadIdx.x;
    size_t dr = blockIdx.y * blockDim.y * ELEMENTS_PER_THREADS + threadIdx.y;

    if (dr < dst_rows && dc < dst_cols) {
        int dst_address_incress = dr * dst_step + dc * CH;
        size_t sc = dc * col_scale;
        src += sc * CH;

        for (int i = 0; i < ELEMENTS_PER_THREADS; i++) {
            if (dr >= dst_rows)
                return;

            size_t sr = dr * row_scale;
            int src_address_incress = sr * src_step;
            for (size_t ch = 0; ch < CH; ch++)
                dst[dst_address_incress + ch] = src[src_address_incress + ch];

            dr += blockDim.y;
            dst_address_incress += blockDim.y * dst_step;
        }
    }
}

template <typename T, size_t CH>
__global__ void resize_nearest_kernel(
        const T* __restrict__ src, T* dst, const size_t dst_rows,
        const size_t dst_cols, const size_t src_step, const size_t dst_step,
        const float row_scale, const float col_scale) {
    size_t dc = blockIdx.x * blockDim.x + threadIdx.x;
    size_t dr = blockIdx.y * blockDim.y + threadIdx.y;
    if (dr < dst_rows && dc < dst_cols) {
        size_t sr = dr * row_scale;
        size_t sc = dc * col_scale;
        src += sr * src_step + sc * CH;
        dst += dr * dst_step + dc * CH;
#pragma unroll
        for (size_t ch = 0; ch < CH; ++ch)
            dst[ch] = src[ch];
    }
}

template <typename T, size_t CH>
void resize_nearest_proxy(const T* src, T* dst, const size_t src_rows,
                          const size_t src_cols, const size_t dst_rows,
                          const size_t dst_cols, const size_t src_step,
                          const size_t dst_step, void* workspace,
                          cudaStream_t stream) {
    MEGDNN_MARK_USED_VAR(workspace);
    float row_scale = (float)src_rows / dst_rows;
    float col_scale = (float)src_cols / dst_cols;

    if (CH == 3 && sizeof(T) == 4 &&
        (dst_cols * dst_rows <= src_cols * src_rows)) {
        dim3 THREADS(32, 8, 1);
        dim3 BLOCKS(DIVUP(dst_cols, THREADS.x), DIVUP(dst_rows, THREADS.y));

        cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

        resize_nearest_kernel<T, CH><<<BLOCKS, THREADS, 0, stream>>>(
                src, dst, dst_rows, dst_cols, src_step, dst_step, row_scale,
                col_scale);

    } else {
        dim3 THREADS(32, 8, 1);
        dim3 BLOCKS(DIVUP(dst_cols, THREADS.x),
                    DIVUP(dst_rows, THREADS.y * ELEMENTS_PER_THREADS));

        if (CH == 3 && sizeof(T) == 1)
            cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

        resize_nearest_vector_kernel<T, CH><<<BLOCKS, THREADS, 0, stream>>>(
                src, dst, dst_rows, dst_cols, src_step, dst_step, row_scale,
                col_scale);
    }
}

template <typename T, size_t CH>
__global__ void resize_linear_Restric_kernel(
        const T* __restrict__ src, T* dst, const size_t src_rows,
        const size_t src_cols, const size_t dst_rows, const size_t dst_cols,
        const size_t src_step, const size_t dst_step, const float row_scale,
        const float col_scale, const float inverse_row_scale,
        const float inverse_col_scale) {
    size_t dc = blockIdx.x * blockDim.x + threadIdx.x;
    size_t dr = blockIdx.y * blockDim.y + threadIdx.y;

    if (dr < dst_rows && dc < dst_cols) {
        float fc = (dc + 0.5f) * inverse_col_scale - 0.5f;
        float fr = (dr + 0.5f) * inverse_row_scale - 0.5f;
        int sc = __float2int_rd(fc);
        int sr = __float2int_rd(fr);

        fc -= sc;
        fr -= sr;

        if (sc < 0) {
            sc = 0;
            fc = 0;
        }
        if (sr < 0) {
            sr = 0;
            fr = 0;
        }

        if (sc + 1 >= src_cols) {
            sc = src_cols - 2;
            fc = 1;
        }

        if (sr + 1 >= src_rows) {
            sr = src_rows - 2;
            fr = 1;
        }

        int src_address = sr * src_step + sc * CH;

        // if the type is uchar, use sr and sc to donate fx * (1 << SCALE)
        float dst_data[CH] = {0};
#pragma unroll
        for (int ch = 0; ch < CH; ch++) {
            float pcrsc00 = src[src_address + ch];
            float pcrsc01 = src[src_address + CH + ch];
            float pcrsc10 = src[src_address + src_step + ch];
            float pcrsc11 = src[src_address + src_step + CH + ch];
            dst_data[ch] = fr * (pcrsc11 * fc + pcrsc10 * (1 - fc)) +
                           (1 - fr) * (pcrsc01 * fc + pcrsc00 * (1 - fc));
        }
        int dst_address = dr * dst_step + dc * CH;
#pragma unroll
        for (int ch = 0; ch < CH; ch++)
            dst[dst_address++] = (T)(dst_data[ch]);
    }
}

template <typename T, size_t CH>
__global__ void resize_linear_vector_kernel(
        const T* src, T* dst, const size_t src_rows, const size_t src_cols,
        const size_t dst_rows, const size_t dst_cols, const size_t src_step,
        const size_t dst_step, const float row_scale, const float col_scale,
        const float inverse_row_scale, const float inverse_col_scale) {
    size_t dc = blockIdx.x * blockDim.x + threadIdx.x;
    size_t dr = blockIdx.y * blockDim.y * ELEMENTS_PER_THREADS + threadIdx.y;

    if (dr < dst_rows && dc < dst_cols) {
        float fc = (dc + 0.5f) * inverse_col_scale - 0.5f;
        int sc = __float2int_rd(fc);
        fc -= sc;
        if (sc < 0) {
            sc = 0;
            fc = 0;
        }

        if (sc + 1 >= src_cols) {
            sc = src_cols - 2;
            fc = 1;
        }
        for (int i = 0; i < ELEMENTS_PER_THREADS; i++) {
            if (dr >= dst_rows)
                return;

            float fr = (dr + 0.5f) * inverse_row_scale - 0.5f;
            int sr = __float2int_rd(fr);
            fr -= sr;

            if (sr < 0) {
                sr = 0;
                fr = 0;
            }
            if (sr + 1 >= src_rows) {
                sr = src_rows - 2;
                fr = 1;
            }
            int src_address = sr * src_step + sc * CH;
            float dst_data[CH] = {0};
#pragma unroll
            for (int ch = 0; ch < CH; ch++) {
                float pcrsc00 = src[src_address + ch];
                float pcrsc01 = src[src_address + CH + ch];
                float pcrsc10 = src[src_address + src_step + ch];
                float pcrsc11 = src[src_address + src_step + CH + ch];
                dst_data[ch] = fr * (pcrsc11 * fc + pcrsc10 * (1 - fc)) +
                               (1 - fr) * (pcrsc01 * fc + pcrsc00 * (1 - fc));
            }

            int dst_address = dr * dst_step + dc * CH;
#pragma unroll
            for (int ch = 0; ch < CH; ch++)
                dst[dst_address++] = (T)(dst_data[ch]);

            dr += blockDim.y;
        }
    }
}

template <typename T, size_t CH>
void resize_area_proxy(const T*, T*, size_t, size_t, size_t, size_t, size_t,
                       size_t, void*, cudaStream_t);

template <typename T, size_t CH>
void resize_linear_proxy(const T* src, T* dst, const size_t src_rows,
                         const size_t src_cols, const size_t dst_rows,
                         const size_t dst_cols, const size_t src_step,
                         const size_t dst_step, void* workspace,
                         cudaStream_t stream) {
    if (src_rows == dst_rows * 2 && src_cols == dst_cols * 2) {
        resize_area_proxy<T, CH>(src, dst, src_rows, src_cols, dst_rows,
                                 dst_cols, src_step, dst_step, workspace,
                                 stream);
        return;
    }

    dim3 THREADS(32, 8, 1);

    float row_scale = (float)dst_rows / src_rows;
    float col_scale = (float)dst_cols / src_cols;

    if (CH == 3 && (dst_rows < src_rows && dst_cols < src_cols)) {
        dim3 BLOCKS(DIVUP(dst_cols, THREADS.x), DIVUP(dst_rows, THREADS.y));

        cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

        resize_linear_Restric_kernel<T, CH><<<BLOCKS, THREADS, 0, stream>>>(
                src, dst, src_rows, src_cols, dst_rows, dst_cols, src_step,
                dst_step, row_scale, col_scale, 1 / row_scale, 1 / col_scale);

    } else {
        dim3 BLOCKS(DIVUP(dst_cols, THREADS.x),
                    DIVUP(dst_rows, THREADS.y * ELEMENTS_PER_THREADS));

        cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

        resize_linear_vector_kernel<T, CH><<<BLOCKS, THREADS, 0, stream>>>(
                src, dst, src_rows, src_cols, dst_rows, dst_cols, src_step,
                dst_step, row_scale, col_scale, 1 / row_scale, 1 / col_scale);
    }
}

template <size_t CH>
__global__ void resize_cubic_32f_kernel_vector(
        const float* __restrict__ src, float* dst, const size_t src_rows,
        const size_t src_cols, const size_t dst_rows, const size_t dst_cols,
        const size_t src_step, const size_t dst_step, const float row_scale,
        const float col_scale) {
    size_t dc = blockIdx.x * blockDim.x + threadIdx.x;
    size_t dr = blockIdx.y * blockDim.y * ELEMENTS_PER_THREADS + threadIdx.y;
    if (dr < dst_rows && dc < dst_cols) {
        float fc = ((float)dc + 0.5) * col_scale - 0.5;
        int sc = floor(fc);
        fc -= sc;
        float coef_col[4];
        interpolate_cubic_coefs(fc, coef_col);

        for (int i = 0; i < ELEMENTS_PER_THREADS; i++) {
            if (dr >= dst_rows)
                return;
            float fr = ((float)dr + 0.5) * row_scale - 0.5;
            int sr = floor(fr);
            fr -= sr;
            float coef_row[4];
            interpolate_cubic_coefs(fr, coef_row);
            float dst_data[CH] = {0};
#pragma unroll
            for (int offset_r = 0; offset_r < 4; ++offset_r) {
                int tr_step =
                        saturate(sr + offset_r - 1, 0, (int)src_rows - 1) *
                        src_step;
#pragma unroll
                for (int offset_c = 0; offset_c < 4; ++offset_c) {
                    int tc_step =
                            saturate(sc + offset_c - 1, 0, (int)src_cols - 1) *
                            CH;
                    int src_address = tr_step + tc_step;
#pragma unroll
                    for (size_t ch = 0; ch < CH; ++ch) {
                        dst_data[ch] += coef_row[offset_r] *
                                        coef_col[offset_c] * src[src_address++];
                    }
                }
            }
            int dst_address = dr * dst_step + dc * CH;
#pragma unroll
            for (int i = 0; i < CH; i++)
                dst[dst_address++] = dst_data[i];
            dr += blockDim.y;
        }
    }
}

template <size_t CH>
__global__ void resize_cubic_8u_kernel_vector(
        const uchar* __restrict__ src, uchar* dst, const size_t src_rows,
        const size_t src_cols, const size_t dst_rows, const size_t dst_cols,
        const size_t src_step, const size_t dst_step, const float row_scale,
        const float col_scale) {
    size_t dc = blockIdx.x * blockDim.x + threadIdx.x;
    size_t dr = blockIdx.y * blockDim.y * ELEMENTS_PER_THREADS + threadIdx.y;
    if (dr < dst_rows && dc < dst_cols) {
        float fc = ((float)dc + 0.5) * col_scale - 0.5;
        int sc = __float2int_rd(fc);
        fc -= sc;
        short icoef_col[4] = {0};

        float coef_col[4];
        interpolate_cubic_coefs(fc, coef_col);
#pragma unroll
        for (int i = 0; i < 4; i++) {
            icoef_col[i] = (short)(coef_col[i] * ONE);
        }

        for (int i = 0; i < ELEMENTS_PER_THREADS; i++) {
            if (dr >= dst_rows)
                return;
            float fr = ((float)dr + 0.5) * row_scale - 0.5;
            int sr = __float2int_rd(fr);
            fr -= sr;
            short icoef_row[4];
            float coef_row[4];
            interpolate_cubic_coefs(fr, coef_row);
#pragma unroll
            for (int i = 0; i < 4; i++) {
                icoef_row[i] = (short)(coef_row[i] * ONE);
            }

            int dst_data[CH] = {0};
#pragma unroll
            for (int offset_r = 0; offset_r < 4; ++offset_r) {
                int tr_step =
                        saturate(sr + offset_r - 1, 0, (int)src_rows - 1) *
                        src_step;
#pragma unroll
                for (int offset_c = 0; offset_c < 4; ++offset_c) {
                    int tc_step =
                            saturate(sc + offset_c - 1, 0, (int)src_cols - 1) *
                            CH;
                    int src_address = tr_step + tc_step;
#pragma unroll
                    for (size_t ch = 0; ch < CH; ++ch) {
                        dst_data[ch] += icoef_row[offset_r] *
                                        icoef_col[offset_c] *
                                        src[src_address++];
                    }
                }
            }
            int dst_address = dr * dst_step + dc * CH;
#pragma unroll
            for (int i = 0; i < CH; i++)
                dst[dst_address++] =
                        saturate(dst_data[i] >> (SCALE + SCALE), 0, 255);
            dr += blockDim.y;
        }
    }
}

template <size_t CH>
__global__ void resize_cubic_32f_kernel_cacheToGlobal(
        const float* src, float* dst, const size_t src_rows,
        const size_t src_cols, const size_t dst_rows, const size_t dst_cols,
        const size_t src_step, const size_t dst_step, const float* gl_coef_row,
        const float* gl_coef_col, const int* gl_sr, const int* gl_sc) {
    size_t dc = blockIdx.x * blockDim.x + threadIdx.x;
    size_t dr = blockIdx.y * blockDim.y * ELEMENTS_PER_THREADS + threadIdx.y;

    if (dr < dst_rows && dc < dst_cols) {
        int sc = gl_sc[dc];
        float coef_col[4];
#pragma unroll
        for (int i = 0, index = dc; i < 4; i++, index += dst_cols)
            coef_col[i] = gl_coef_col[index];

        for (int i = 0; i < ELEMENTS_PER_THREADS; i++) {
            if (dr >= dst_rows)
                return;
            int sr = gl_sr[dr];
            float coef_row[4];
#pragma unroll
            for (int i = 0, index = dr; i < 4; i++, index += dst_rows)
                coef_row[i] = gl_coef_row[index];

            float dst_data[CH] = {0};
#pragma unroll
            for (int offset_r = 0; offset_r < 4; ++offset_r) {
                int tr_step =
                        saturate(sr + offset_r - 1, 0, (int)src_rows - 1) *
                        src_step;
#pragma unroll
                for (int offset_c = 0; offset_c < 4; ++offset_c) {
                    int tc_step =
                            saturate(sc + offset_c - 1, 0, (int)src_cols - 1) *
                            CH;
                    int src_address = tr_step + tc_step;
#pragma unroll
                    for (size_t ch = 0; ch < CH; ++ch) {
                        dst_data[ch] += coef_row[offset_r] *
                                        coef_col[offset_c] * src[src_address++];
                    }
                }
            }
            int dst_address = dr * dst_step + dc * CH;
#pragma unroll
            for (int i = 0; i < CH; i++)
                dst[dst_address++] = dst_data[i];

            dr += blockDim.y;
        }
    }
}

template <size_t CH>
__global__ void resize_cubic_8u_kernel_cacheToGlobal(
        const uchar* src, uchar* dst, const size_t src_rows,
        const size_t src_cols, const size_t dst_rows, const size_t dst_cols,
        const size_t src_step, const size_t dst_step, const short* gl_icoef_row,
        const short* gl_icoef_col, const int* gl_sr, const int* gl_sc) {
    size_t dc = blockIdx.x * blockDim.x + threadIdx.x;
    size_t dr = blockIdx.y * blockDim.y * ELEMENTS_PER_THREADS + threadIdx.y;

    if (dr < dst_rows && dc < dst_cols) {
        int sc = gl_sc[dc];
        short icoef_col[4];
#pragma unroll
        for (int i = 0, index = dc; i < 4; i++, index += dst_cols)
            icoef_col[i] = gl_icoef_col[index];

        for (int i = 0; i < ELEMENTS_PER_THREADS; i++) {
            if (dr >= dst_rows)
                return;
            int sr = gl_sr[dr];
            short icoef_row[4];
#pragma unroll
            for (int i = 0, index = dr; i < 4; i++, index += dst_rows)
                icoef_row[i] = gl_icoef_row[index];

            int dst_data[CH] = {0};
#pragma unroll
            for (int offset_r = 0; offset_r < 4; ++offset_r) {
                int tr_step =
                        saturate(sr + offset_r - 1, 0, (int)src_rows - 1) *
                        src_step;
#pragma unroll
                for (int offset_c = 0; offset_c < 4; ++offset_c) {
                    int tc_step =
                            saturate(sc + offset_c - 1, 0, (int)src_cols - 1) *
                            CH;
                    int src_address = tr_step + tc_step;
#pragma unroll
                    for (size_t ch = 0; ch < CH; ++ch) {
                        dst_data[ch] += icoef_row[offset_r] *
                                        icoef_col[offset_c] *
                                        src[src_address++];
                    }
                }
            }
            int dst_address = dr * dst_step + dc * CH;
#pragma unroll
            for (int i = 0; i < CH; i++)
                dst[dst_address++] =
                        saturate(dst_data[i] >> (SCALE + SCALE), 0, 255);

            dr += blockDim.y;
        }
    }
}

template <typename T, size_t CH>
void resize_cubic_proxy(const T* src, T* dst, const size_t src_rows,
                        const size_t src_cols, const size_t dst_rows,
                        const size_t dst_cols, const size_t src_step,
                        const size_t dst_step, void* workspace,
                        cudaStream_t stream) {
    dim3 THREADS(32, 8, 1);
    float row_scale = (float)src_rows / dst_rows;
    float col_scale = (float)src_cols / dst_cols;

    size_t dst_area_size = dst_rows * dst_cols;
    size_t src_area_size = src_rows * src_cols;

    bool enlarge = dst_area_size > src_area_size;
    bool shrink = dst_area_size <= src_area_size;
    bool U8 = sizeof(T) == sizeof(uchar);
    bool F32_1 = sizeof(T) == sizeof(float) && CH == 1;
    bool F32_3 = sizeof(T) == sizeof(float) && CH == 3;

    bool use_vector = (enlarge && (dst_area_size <= 500 * 500)) ||
                      (shrink && (F32_3 || (U8 && dst_area_size <= 500 * 500) ||
                                  (F32_1 && dst_area_size <= 1000 * 1000)));

    if (use_vector) {
        dim3 BLOCKS(DIVUP(dst_cols, THREADS.x),
                    DIVUP(dst_rows, THREADS.y * ELEMENTS_PER_THREADS));

        if (sizeof(T) == sizeof(float)) {
            resize_cubic_32f_kernel_vector<CH><<<BLOCKS, THREADS, 0, stream>>>(
                    (const float*)src, (float*)dst, src_rows, src_cols,
                    dst_rows, dst_cols, src_step, dst_step, row_scale,
                    col_scale);
        } else {
            resize_cubic_8u_kernel_vector<CH><<<BLOCKS, THREADS, 0, stream>>>(
                    (const uchar*)src, (uchar*)dst, src_rows, src_cols,
                    dst_rows, dst_cols, src_step, dst_step, row_scale,
                    col_scale);
        }

    } else {
        dim3 BLOCKS(DIVUP(dst_cols, THREADS.x),
                    DIVUP(dst_rows, THREADS.y * ELEMENTS_PER_THREADS));

        cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

        if (sizeof(T) == sizeof(float)) {
            float* dev_coef_row = static_cast<float*>(workspace);
            int* dev_sr = reinterpret_cast<int*>(dev_coef_row + dst_rows * 4);
            float* dev_coef_col = reinterpret_cast<float*>(dev_sr + dst_rows);
            int* dev_sc = reinterpret_cast<int*>(dev_coef_col + dst_cols * 4);

            precompute_cubic_coef_f32<<<DIVUP(dst_rows, 128), 128, 0, stream>>>(
                    dev_coef_row, row_scale, dst_rows);
            precompute_cubic_coef_f32<<<DIVUP(dst_cols, 128), 128, 0, stream>>>(
                    dev_coef_col, col_scale, dst_cols);

            resize_cubic_32f_kernel_cacheToGlobal<CH>
                    <<<BLOCKS, THREADS, 0, stream>>>(
                            (const float*)src, (float*)dst, src_rows, src_cols,
                            dst_rows, dst_cols, src_step, dst_step,
                            dev_coef_row, dev_coef_col, dev_sr, dev_sc);

        } else {
            short* dev_coef_row = static_cast<short*>(workspace);
            int* dev_sr = reinterpret_cast<int*>(dev_coef_row + dst_rows * 4);
            short* dev_coef_col = reinterpret_cast<short*>(dev_sr + dst_rows);
            int* dev_sc = reinterpret_cast<int*>(dev_coef_col + dst_cols * 4);

            precompute_cubic_coef_u8<<<DIVUP(dst_rows, 128), 128, 0, stream>>>(
                    dev_coef_row, row_scale, dst_rows);
            precompute_cubic_coef_u8<<<DIVUP(dst_cols, 128), 128, 0, stream>>>(
                    dev_coef_col, col_scale, dst_cols);

            resize_cubic_8u_kernel_cacheToGlobal<CH>
                    <<<BLOCKS, THREADS, 0, stream>>>(
                            (const uchar*)src, (uchar*)dst, src_rows, src_cols,
                            dst_rows, dst_cols, src_step, dst_step,
                            dev_coef_row, dev_coef_col, dev_sr, dev_sc);
        }
    }
}

template <size_t CH>
__global__ void resize_lanczos4_32f_kernel_vector(
        const float* __restrict__ src, float* dst, const size_t src_rows,
        const size_t src_cols, const size_t dst_rows, const size_t dst_cols,
        const size_t src_step, const size_t dst_step, const float row_scale,
        const float col_scale) {
    size_t dc = blockIdx.x * blockDim.x + threadIdx.x;
    size_t dr = blockIdx.y * blockDim.y * ELEMENTS_PER_THREADS + threadIdx.y;
    if (dr < dst_rows && dc < dst_cols) {
        float fc = ((float)dc + 0.5) * col_scale - 0.5;
        int sc = floor(fc);
        fc -= sc;
        float coef_col[8];
        interpolate_lanczos4_coefs(fc, coef_col);

        for (int i = 0; i < ELEMENTS_PER_THREADS; i++) {
            if (dr >= dst_rows)
                return;
            float fr = ((float)dr + 0.5) * row_scale - 0.5;
            int sr = floor(fr);
            fr -= sr;
            float coef_row[8];
            interpolate_lanczos4_coefs(fr, coef_row);
            float dst_data[CH] = {0};
#pragma unroll
            for (int offset_r = 0; offset_r < 8; ++offset_r) {
                int tr_step =
                        saturate(sr + offset_r - 3, 0, (int)src_rows - 1) *
                        src_step;
#pragma unroll
                for (int offset_c = 0; offset_c < 8; ++offset_c) {
                    int tc_step =
                            saturate(sc + offset_c - 3, 0, (int)src_cols - 1) *
                            CH;
                    int src_address = tr_step + tc_step;
#pragma unroll
                    for (size_t ch = 0; ch < CH; ++ch) {
                        dst_data[ch] += coef_row[offset_r] *
                                        coef_col[offset_c] * src[src_address++];
                    }
                }
            }
            int dst_address = dr * dst_step + dc * CH;
#pragma unroll
            for (int i = 0; i < CH; i++)
                dst[dst_address++] = dst_data[i];
            dr += blockDim.y;
        }
    }
}

template <size_t CH>
__global__ void resize_lanczos4_8u_kernel_vector(
        const uchar* __restrict__ src, uchar* dst, const size_t src_rows,
        const size_t src_cols, const size_t dst_rows, const size_t dst_cols,
        const size_t src_step, const size_t dst_step, const float row_scale,
        const float col_scale) {
    size_t dc = blockIdx.x * blockDim.x + threadIdx.x;
    size_t dr = blockIdx.y * blockDim.y * ELEMENTS_PER_THREADS + threadIdx.y;
    if (dr < dst_rows && dc < dst_cols) {
        float fc = ((float)dc + 0.5) * col_scale - 0.5;
        int sc = floor(fc);
        fc -= sc;
        short icoef_col[8] = {0};
        const float s45 = 0.70710678118654752440084436210485;
        const float cs[][2] = {{1, 0},  {-s45, -s45}, {0, 1},  {s45, -s45},
                               {-1, 0}, {s45, s45},   {0, -1}, {-s45, s45}};
        const float MEGCV_PI = 3.1415926536;

        {
            if (fc < FLT_EPSILON)
                icoef_col[3] = ONE;
            else {
                float coef_col[8];
                float sum = 0;
                float y0 = -(fc + 3) * MEGCV_PI * 0.25, s0 = sin(y0),
                      c0 = cos(y0);
#pragma unroll
                for (int i = 0; i < 8; i++) {
                    float y = -(fc + 3 - i) * MEGCV_PI * 0.25;
                    coef_col[i] =
                            (float)((cs[i][0] * s0 + cs[i][1] * c0) / (y * y));
                    sum += coef_col[i];
                }

                sum = 1.f / sum;
#pragma unroll
                for (int i = 0; i < 8; i++) {
                    coef_col[i] *= sum;
                    icoef_col[i] = (short)(coef_col[i] * ONE);
                }
            }
        }

        for (int i = 0; i < ELEMENTS_PER_THREADS; i++) {
            if (dr >= dst_rows)
                return;
            float fr = ((float)dr + 0.5) * row_scale - 0.5;
            int sr = floor(fr);
            fr -= sr;
            short icoef_row[8] = {0};
            {
                if (fr < FLT_EPSILON)
                    icoef_row[3] = ONE;
                else {
                    float coef_row[8];
                    float sum = 0;
                    float y0 = -(fr + 3) * MEGCV_PI * 0.25, s0 = sin(y0),
                          c0 = cos(y0);
#pragma unroll
                    for (int i = 0; i < 8; i++) {
                        float y = -(fr + 3 - i) * MEGCV_PI * 0.25;
                        coef_row[i] = (float)((cs[i][0] * s0 + cs[i][1] * c0) /
                                              (y * y));
                        sum += coef_row[i];
                    }

                    sum = 1.f / sum;
#pragma unroll
                    for (int i = 0; i < 8; i++) {
                        coef_row[i] *= sum;
                        icoef_row[i] = (short)(coef_row[i] * ONE);
                    }
                }
            }

            int dst_data[CH] = {0};
#pragma unroll
            for (int offset_r = 0; offset_r < 8; ++offset_r) {
                int tr_step =
                        saturate(sr + offset_r - 3, 0, (int)src_rows - 1) *
                        src_step;
#pragma unroll
                for (int offset_c = 0; offset_c < 8; ++offset_c) {
                    int tc_step =
                            saturate(sc + offset_c - 3, 0, (int)src_cols - 1) *
                            CH;
                    int src_address = tr_step + tc_step;
#pragma unroll
                    for (size_t ch = 0; ch < CH; ++ch) {
                        dst_data[ch] += icoef_row[offset_r] *
                                        icoef_col[offset_c] *
                                        src[src_address++];
                    }
                }
            }

            int dst_address = dr * dst_step + dc * CH;
            for (int ch = 0; ch < CH; ch++)
                dst[dst_address++] =
                        saturate(dst_data[ch] >> (SCALE + SCALE), 0, 255);
            dr += blockDim.y;
        }
    }
}

template <size_t CH>
__global__ void resize_lanczos4_32f_kernel_cacheToGlobal(
        const float* src, float* dst, const size_t src_rows,
        const size_t src_cols, const size_t dst_rows, const size_t dst_cols,
        const size_t src_step, const size_t dst_step, const float* gl_coef_row,
        const float* gl_coef_col, const int* gl_sr, const int* gl_sc) {
    size_t dc = blockIdx.x * blockDim.x + threadIdx.x;
    size_t dr = blockIdx.y * blockDim.y * ELEMENTS_PER_THREADS + threadIdx.y;

    if (dr < dst_rows && dc < dst_cols) {
        int sc = gl_sc[dc];
        float coef_col[8];
#pragma unroll
        for (int i = 0, index = dc; i < 8; i++, index += dst_cols)
            coef_col[i] = gl_coef_col[index];

        for (int i = 0; i < ELEMENTS_PER_THREADS; i++) {
            if (dr >= dst_rows)
                return;
            int sr = gl_sr[dr];
            float coef_row[8];
#pragma unroll
            for (int i = 0, index = dr; i < 8; i++, index += dst_rows)
                coef_row[i] = gl_coef_row[index];

            float dst_data[CH] = {0};
#pragma unroll
            for (int offset_r = 0; offset_r < 8; ++offset_r) {
                int tr_step =
                        saturate(sr + offset_r - 3, 0, (int)src_rows - 1) *
                        src_step;
#pragma unroll
                for (int offset_c = 0; offset_c < 8; ++offset_c) {
                    int tc_step =
                            saturate(sc + offset_c - 3, 0, (int)src_cols - 1) *
                            CH;
                    int src_address = tr_step + tc_step;
#pragma unroll
                    for (size_t ch = 0; ch < CH; ++ch) {
                        dst_data[ch] += coef_row[offset_r] *
                                        coef_col[offset_c] * src[src_address++];
                    }
                }
            }
            int dst_address = dr * dst_step + dc * CH;
#pragma unroll
            for (int i = 0; i < CH; i++)
                dst[dst_address++] = dst_data[i];

            dr += blockDim.y;
        }
    }
}

template <size_t CH>
__global__ void resize_lanczos4_8u_kernel_cacheToGlobal(
        const uchar* src, uchar* dst, const size_t src_rows,
        const size_t src_cols, const size_t dst_rows, const size_t dst_cols,
        const size_t src_step, const size_t dst_step, const short* gl_icoef_row,
        const short* gl_icoef_col, const int* gl_sr, const int* gl_sc) {
    size_t dc = blockIdx.x * blockDim.x + threadIdx.x;
    size_t dr = blockIdx.y * blockDim.y * ELEMENTS_PER_THREADS + threadIdx.y;

    if (dr < dst_rows && dc < dst_cols) {
        int sc = gl_sc[dc];
        short icoef_col[8];
#pragma unroll
        for (int i = 0, index = dc; i < 8; i++, index += dst_cols)
            icoef_col[i] = gl_icoef_col[index];

        for (int i = 0; i < ELEMENTS_PER_THREADS; i++) {
            if (dr >= dst_rows)
                return;
            int sr = gl_sr[dr];
            short icoef_row[8];
#pragma unroll
            for (int i = 0, index = dr; i < 8; i++, index += dst_rows)
                icoef_row[i] = gl_icoef_row[index];

            int dst_data[CH] = {0};
#pragma unroll
            for (int offset_r = 0; offset_r < 8; ++offset_r) {
                int tr_step =
                        saturate(sr + offset_r - 3, 0, (int)src_rows - 1) *
                        src_step;
#pragma unroll
                for (int offset_c = 0; offset_c < 8; ++offset_c) {
                    int tc_step =
                            saturate(sc + offset_c - 3, 0, (int)src_cols - 1) *
                            CH;
                    int src_address = tr_step + tc_step;
#pragma unroll
                    for (size_t ch = 0; ch < CH; ++ch) {
                        dst_data[ch] += icoef_row[offset_r] *
                                        icoef_col[offset_c] *
                                        src[src_address++];
                    }
                }
            }
            int dst_address = dr * dst_step + dc * CH;
#pragma unroll
            for (int i = 0; i < CH; i++)
                dst[dst_address++] =
                        saturate(dst_data[i] >> (SCALE + SCALE), 0, 255);

            dr += blockDim.y;
        }
    }
}

template <typename T, size_t CH>
void resize_lanczos4_proxy(const T* src, T* dst, const size_t src_rows,
                           const size_t src_cols, const size_t dst_rows,
                           const size_t dst_cols, const size_t src_step,
                           const size_t dst_step, void* workspace,
                           cudaStream_t stream) {
    dim3 THREADS(16, 16, 1);

    float row_scale = (float)src_rows / dst_rows;
    float col_scale = (float)src_cols / dst_cols;

    size_t dst_area_size = dst_rows * dst_cols;
    size_t src_area_size = src_rows * src_cols;

    bool enlarge = dst_area_size > src_area_size;
    bool shrink = dst_area_size <= src_area_size;
    bool U8 = sizeof(T) == sizeof(uchar);
    bool F32_1 = sizeof(T) == sizeof(float) && CH == 1;
    bool F32_3 = sizeof(T) == sizeof(float) && CH == 3;

    bool use_vector = (enlarge && (dst_area_size <= 500 * 500)) ||
                      (shrink && (F32_3 || (U8 && dst_area_size <= 500 * 500) ||
                                  (F32_1 && dst_area_size <= 1000 * 1000)));

    if (use_vector) {
        dim3 BLOCKS(DIVUP(dst_cols, THREADS.x),
                    DIVUP(dst_rows, THREADS.y * ELEMENTS_PER_THREADS));

        if (sizeof(T) == sizeof(float)) {
            resize_lanczos4_32f_kernel_vector<CH>
                    <<<BLOCKS, THREADS, 0, stream>>>(
                            (const float*)src, (float*)dst, src_rows, src_cols,
                            dst_rows, dst_cols, src_step, dst_step, row_scale,
                            col_scale);
        } else {
            resize_lanczos4_8u_kernel_vector<CH>
                    <<<BLOCKS, THREADS, 0, stream>>>(
                            (const uchar*)src, (uchar*)dst, src_rows, src_cols,
                            dst_rows, dst_cols, src_step, dst_step, row_scale,
                            col_scale);
        }

    } else {
        dim3 BLOCKS(DIVUP(dst_cols, THREADS.x),
                    DIVUP(dst_rows, THREADS.y * ELEMENTS_PER_THREADS));

        cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

        if (sizeof(T) == sizeof(float)) {
            float* dev_coef_row = static_cast<float*>(workspace);
            int* dev_sr = reinterpret_cast<int*>(dev_coef_row + dst_rows * 8);
            float* dev_coef_col = reinterpret_cast<float*>(dev_sr + dst_rows);
            int* dev_sc = reinterpret_cast<int*>(dev_coef_col + dst_cols * 8);

            precompute_lanczos4_coef_f32<<<DIVUP(dst_rows, 128), 128, 0,
                                           stream>>>(dev_coef_row, row_scale,
                                                     dst_rows);
            precompute_lanczos4_coef_f32<<<DIVUP(dst_cols, 128), 128, 0,
                                           stream>>>(dev_coef_col, col_scale,
                                                     dst_cols);
            resize_lanczos4_32f_kernel_cacheToGlobal<CH>
                    <<<BLOCKS, THREADS, 0, stream>>>(
                            (const float*)src, (float*)dst, src_rows, src_cols,
                            dst_rows, dst_cols, src_step, dst_step,
                            dev_coef_row, dev_coef_col, dev_sr, dev_sc);

        } else {
            short* dev_coef_row = static_cast<short*>(workspace);
            int* dev_sr = reinterpret_cast<int*>(dev_coef_row + dst_rows * 8);
            short* dev_coef_col = reinterpret_cast<short*>(dev_sr + dst_rows);
            int* dev_sc = reinterpret_cast<int*>(dev_coef_col + dst_cols * 8);

            precompute_lanczos4_coef_u8<<<DIVUP(dst_rows, 128), 128, 0,
                                          stream>>>(dev_coef_row, row_scale,
                                                    dst_rows);
            precompute_lanczos4_coef_u8<<<DIVUP(dst_cols, 128), 128, 0,
                                          stream>>>(dev_coef_col, col_scale,
                                                    dst_cols);

            resize_lanczos4_8u_kernel_cacheToGlobal<CH>
                    <<<BLOCKS, THREADS, 0, stream>>>(
                            (const uchar*)src, (uchar*)dst, src_rows, src_cols,
                            dst_rows, dst_cols, src_step, dst_step,
                            dev_coef_row, dev_coef_col, dev_sr, dev_sc);
        }
    }
}

template <size_t CH>
__global__ void resize_area_version1_shrink_32f_kernel(
        const float* src, float* dst, const size_t src_rows,
        const size_t src_cols, const size_t dst_rows, const size_t dst_cols,
        const size_t src_step, const size_t dst_step, const float row_scale,
        const float col_scale, const float _row_scale, const float _col_scale) {
    size_t dc = blockIdx.x * blockDim.x + threadIdx.x;
    size_t dr = blockIdx.y * blockDim.y + threadIdx.y;
    if (dr < dst_rows && dc < dst_cols) {
        float fsr1 = (float)dr * row_scale;
        float fsr2 = (float)(dr + 1) * row_scale;
        int sr1 = floor(fsr1);
        int sr2 = ceil(fsr2);

        float fsc1 = (float)dc * col_scale;
        float fsc2 = (float)(dc + 1) * col_scale;
        int sc1 = floor(fsc1);
        int sc2 = ceil(fsc2);

        float dst_data[CH] = {0};

        {
            float coefr = (float)(sr1 + 1 - fsr1) * _row_scale;
            {
                float coefc = (float)(sc1 + 1 - fsc1) * _col_scale;
                for (size_t ch = 0; ch < CH; ++ch) {
                    dst_data[ch] += coefr * coefc * at(src, sr1, sc1, ch);
                }
            }
            for (int sc = sc1 + 1; sc < sc2 - 1; ++sc) {
                float coefc = _col_scale;
                for (size_t ch = 0; ch < CH; ++ch) {
                    dst_data[ch] += coefr * coefc * at(src, sr1, sc, ch);
                }
            }
            {
                float coefc = (float)(fsc2 - (sc2 - 1)) * _col_scale;
                for (size_t ch = 0; ch < CH; ++ch) {
                    dst_data[ch] += coefr * coefc * at(src, sr1, sc2 - 1, ch);
                }
            }
        }

        for (int sr = sr1 + 1; sr < sr2 - 1; ++sr) {
            float coefr = 1.0f * _row_scale;
            {
                float coefc = (float)(sc1 + 1 - fsc1) * _col_scale;
                for (size_t ch = 0; ch < CH; ++ch) {
                    dst_data[ch] += coefr * coefc * at(src, sr, sc1, ch);
                }
            }
            for (int sc = sc1 + 1; sc < sc2 - 1; ++sc) {
                float coefc = _col_scale;
                for (size_t ch = 0; ch < CH; ++ch) {
                    dst_data[ch] += coefr * coefc * at(src, sr, sc, ch);
                }
            }
            {
                float coefc = (float)(fsc2 - (sc2 - 1)) * _col_scale;
                for (size_t ch = 0; ch < CH; ++ch) {
                    dst_data[ch] += coefr * coefc * at(src, sr, sc2 - 1, ch);
                }
            }
        }

        {
            float coefr = (float)(fsr2 - (sr2 - 1)) * _row_scale;
            {
                float coefc = (float)(sc1 + 1 - fsc1) * _col_scale;
                for (size_t ch = 0; ch < CH; ++ch) {
                    dst_data[ch] += coefr * coefc * at(src, sr2 - 1, sc1, ch);
                }
            }
            for (int sc = sc1 + 1; sc < sc2 - 1; ++sc) {
                float coefc = _col_scale;
                for (size_t ch = 0; ch < CH; ++ch) {
                    dst_data[ch] += coefr * coefc * at(src, sr2 - 1, sc, ch);
                }
            }
            {
                float coefc = (float)(fsc2 - (sc2 - 1)) * _col_scale;
                for (size_t ch = 0; ch < CH; ++ch) {
                    dst_data[ch] +=
                            coefr * coefc * at(src, sr2 - 1, sc2 - 1, ch);
                }
            }
        }

        for (size_t ch = 0; ch < CH; ++ch)
            at(dst, dr, dc, ch) = dst_data[ch];
    }
}

template <size_t CH>
__global__ void resize_area_version1_shrink_8u_kernel(
        const uchar* src, uchar* dst, const size_t src_rows,
        const size_t src_cols, const size_t dst_rows, const size_t dst_cols,
        const size_t src_step, const size_t dst_step, const float row_scale,
        const float col_scale, const float _row_scale, const float _col_scale) {
    size_t dc = blockIdx.x * blockDim.x + threadIdx.x;
    size_t dr = blockIdx.y * blockDim.y + threadIdx.y;
    if (dr < dst_rows && dc < dst_cols) {
        float fsr1 = (float)dr * row_scale;
        float fsr2 = (float)(dr + 1) * row_scale;
        int sr1 = floor(fsr1);
        int sr2 = ceil(fsr2);

        float fsc1 = (float)dc * col_scale;
        float fsc2 = (float)(dc + 1) * col_scale;
        int sc1 = floor(fsc1);
        int sc2 = ceil(fsc2);
        float dst_data[CH] = {0};

        {
            float coefr = (float)(sr1 + 1 - fsr1) * _row_scale;
            {
                float coefc = (float)(sc1 + 1 - fsc1) * _col_scale;
                for (size_t ch = 0; ch < CH; ++ch) {
                    dst_data[ch] += coefr * coefc * at(src, sr1, sc1, ch);
                }
            }
            for (int sc = sc1 + 1; sc < sc2 - 1; ++sc) {
                float coefc = _col_scale;
                for (size_t ch = 0; ch < CH; ++ch) {
                    dst_data[ch] += coefr * coefc * at(src, sr1, sc, ch);
                }
            }
            {
                float coefc = (float)(fsc2 - (sc2 - 1)) * _col_scale;
                for (size_t ch = 0; ch < CH; ++ch) {
                    dst_data[ch] += coefr * coefc * at(src, sr1, sc2 - 1, ch);
                }
            }
        }
        for (int sr = sr1 + 1; sr < sr2 - 1; ++sr) {
            float coefr = 1.0f * _row_scale;
            {
                float coefc = (float)(sc1 + 1 - fsc1) * _col_scale;
                for (size_t ch = 0; ch < CH; ++ch) {
                    dst_data[ch] += coefr * coefc * at(src, sr, sc1, ch);
                }
            }
            for (int sc = sc1 + 1; sc < sc2 - 1; ++sc) {
                float coefc = _col_scale;
                for (size_t ch = 0; ch < CH; ++ch) {
                    dst_data[ch] += coefr * coefc * at(src, sr, sc, ch);
                }
            }
            {
                float coefc = (float)(fsc2 - (sc2 - 1)) * _col_scale;
                for (size_t ch = 0; ch < CH; ++ch) {
                    dst_data[ch] += coefr * coefc * at(src, sr, sc2 - 1, ch);
                }
            }
        }

        {
            float coefr = (float)(fsr2 - (sr2 - 1)) * _row_scale;
            {
                float coefc = (float)(sc1 + 1 - fsc1) * _col_scale;
                for (size_t ch = 0; ch < CH; ++ch) {
                    dst_data[ch] += coefr * coefc * at(src, sr2 - 1, sc1, ch);
                }
            }
            for (int sc = sc1 + 1; sc < sc2 - 1; ++sc) {
                float coefc = _col_scale;
                for (size_t ch = 0; ch < CH; ++ch) {
                    dst_data[ch] += coefr * coefc * at(src, sr2 - 1, sc, ch);
                }
            }
            {
                float coefc = (float)(fsc2 - (sc2 - 1)) * _col_scale;
                for (size_t ch = 0; ch < CH; ++ch) {
                    dst_data[ch] +=
                            coefr * coefc * at(src, sr2 - 1, sc2 - 1, ch);
                }
            }
        }

        for (size_t ch = 0; ch < CH; ++ch)
            at(dst, dr, dc, ch) = saturate((int)dst_data[ch], 0, 255);
    }
}

template <size_t CH>
__global__ void resize_area_version2_shrink_32f_kernel(
        const float* src, float* dst, const size_t src_rows,
        const size_t src_cols, const size_t dst_rows, const size_t dst_cols,
        const size_t src_step, const size_t dst_step, const float row_scale,
        const float col_scale, const float _row_scale, const float _col_scale) {
    size_t dc0 = blockIdx.x * blockDim.x;
    size_t dr = blockIdx.y * blockDim.y + threadIdx.y;
    if (dr < dst_rows && dc0 < dst_cols) {
        __shared__ float lc_dst_data[THREADS_Y][THREADS_X * CH];

        size_t dc = dc0 + threadIdx.x;

        float fsr1 = (float)dr * row_scale;
        float fsr2 = (float)(dr + 1) * row_scale;
        int sr1 = floor(fsr1);
        int sr2 = ceil(fsr2);

        float fsc1 = (float)dc0 * col_scale;
        float fsc2 = (float)(dc0 + blockDim.x) * col_scale;
        int sc1 = floor(fsc1);
        int sc2 = ceil(fsc2);

        for (size_t ch = 0; ch < CH; ch++)
            lc_dst_data[threadIdx.y][threadIdx.x * CH + ch] = 0;

        __syncthreads();

        size_t min_col_edge = min((int)src_cols, sc2) * CH;
        for (int sc_address = sc1 * CH + threadIdx.x; sc_address < min_col_edge;
             sc_address += blockDim.x) {
            float sum = 0;
            {
                float coefr = (float)(sr1 + 1 - fsr1) * _row_scale;
                sum += coefr * src[sr1 * src_step + sc_address];
            }
            float coefr = _row_scale;
            for (int sr = sr1 + 1; sr < sr2 - 1; ++sr) {
                sum += coefr * src[sr * src_step + sc_address];
            }
            {
                float coefr = (float)(fsr2 - (sr2 - 1)) * _row_scale;
                sum += coefr * src[(sr2 - 1) * src_step + sc_address];
            }

            size_t multi = floor(((sc_address / CH) + 1) * _col_scale);
            float x = ((sc_address / CH) + 1) - multi * col_scale;
            if (x >= 1) {
                atomicAdd(&(lc_dst_data[threadIdx.y]
                                       [(multi - dc0) * CH + sc_address % CH]),
                          sum * _col_scale);
            } else {
                if (multi < dc0 + blockDim.x)
                    atomicAdd(&(lc_dst_data[threadIdx.y][(multi - dc0) * CH +
                                                         sc_address % CH]),
                              sum * (x * _col_scale));
                if (multi - 1 >= dc0)
                    atomicAdd(
                            &(lc_dst_data[threadIdx.y][(multi - 1 - dc0) * CH +
                                                       sc_address % CH]),
                            sum * ((1 - x) * _col_scale));
            }
        }

        __syncthreads();

        if (dc < dst_cols) {
            for (size_t ch = 0; ch < CH; ++ch)
                at(dst, dr, dc, ch) =
                        lc_dst_data[threadIdx.y][(threadIdx.x) * CH + ch];
        }
    }
}

template <size_t CH>
__global__ void resize_area_version2_shrink_8u_kernel(
        const uchar* src, uchar* dst, const size_t src_rows,
        const size_t src_cols, const size_t dst_rows, const size_t dst_cols,
        const size_t src_step, const size_t dst_step, const float row_scale,
        const float col_scale, const float _row_scale, const float _col_scale) {
    size_t dc0 = blockIdx.x * blockDim.x;
    size_t dr = blockIdx.y * blockDim.y + threadIdx.y;
    if (dr < dst_rows && dc0 < dst_cols) {
        __shared__ float lc_dst_data[THREADS_Y][THREADS_X * CH];

        size_t dc = dc0 + threadIdx.x;

        float fsr1 = (float)dr * row_scale;
        float fsr2 = (float)(dr + 1) * row_scale;
        int sr1 = floor(fsr1);
        int sr2 = ceil(fsr2);

        float fsc1 = (float)dc0 * col_scale;
        float fsc2 = (float)(dc0 + blockDim.x) * col_scale;
        int sc1 = floor(fsc1);
        int sc2 = ceil(fsc2);

        for (size_t ch = 0; ch < CH; ch++)
            lc_dst_data[threadIdx.y][threadIdx.x * CH + ch] = 0;

        __syncthreads();

        size_t min_col_edge = min((int)src_cols, sc2) * CH;
        for (int sc_address = sc1 * CH + threadIdx.x; sc_address < min_col_edge;
             sc_address += blockDim.x) {
            float sum = 0;
            {
                float coefr = (float)(sr1 + 1 - fsr1) * _row_scale;
                sum += coefr * src[sr1 * src_step + sc_address];
            }
            float coefr = _row_scale;
            for (int sr = sr1 + 1; sr < sr2 - 1; ++sr) {
                sum += coefr * src[sr * src_step + sc_address];
            }
            {
                float coefr = (float)(fsr2 - (sr2 - 1)) * _row_scale;
                sum += coefr * src[(sr2 - 1) * src_step + sc_address];
            }

            size_t multi = floor(((sc_address / CH) + 1) * _col_scale);
            float x = ((sc_address / CH) + 1) - multi * col_scale;
            if (x >= 1) {
                atomicAdd(&(lc_dst_data[threadIdx.y]
                                       [(multi - dc0) * CH + sc_address % CH]),
                          sum * _col_scale);
            } else {
                if (multi < dc0 + blockDim.x)
                    atomicAdd(&(lc_dst_data[threadIdx.y][(multi - dc0) * CH +
                                                         sc_address % CH]),
                              sum * (x * _col_scale));
                if (multi - 1 >= dc0)
                    atomicAdd(
                            &(lc_dst_data[threadIdx.y][(multi - 1 - dc0) * CH +
                                                       sc_address % CH]),
                            sum * ((1 - x) * _col_scale));
            }
        }

        __syncthreads();

        if (dc < dst_cols) {
            for (size_t ch = 0; ch < CH; ++ch)
                at(dst, dr, dc, ch) = saturate(
                        (int)lc_dst_data[threadIdx.y][(threadIdx.x) * CH + ch],
                        0, 255);
        }
    }
}

template <size_t CH>
__global__ void resize_area_version1_shrink_fast_32f_kernel(
        const float* __restrict__ src, float* dst, const size_t src_rows,
        const size_t src_cols, const size_t dst_rows, const size_t dst_cols,
        const size_t src_step, const size_t dst_step, const size_t cell_rows,
        const size_t cell_cols, const float _cell_rows,
        const float _cell_cols) {
    size_t dc = blockIdx.x * blockDim.x + threadIdx.x;
    size_t dr = blockIdx.y * blockDim.y + threadIdx.y;
    if (dr < dst_rows && dc < dst_cols) {
        int sr0 = dr * cell_rows;
        int sc0 = dc * cell_cols;
        float dst_data[CH] = {0};
        for (int sr = sr0; sr < cell_rows + sr0; ++sr) {
            for (int sc = sc0; sc < cell_cols + sc0; ++sc) {
                for (size_t ch = 0; ch < CH; ++ch) {
                    dst_data[ch] += at(src, sr, sc, ch);
                }
            }
        }

        for (size_t ch = 0; ch < CH; ++ch)
            at(dst, dr, dc, ch) = dst_data[ch] * _cell_rows * _cell_cols;
    }
}

template <size_t CH>
__global__ void resize_area_version1_shrink_fast_8u_kernel(
        const uchar* __restrict__ src, uchar* dst, const size_t src_rows,
        const size_t src_cols, const size_t dst_rows, const size_t dst_cols,
        const size_t src_step, const size_t dst_step, const size_t cell_rows,
        const size_t cell_cols, const float _cell_rows,
        const float _cell_cols) {
    size_t dc = blockIdx.x * blockDim.x + threadIdx.x;
    size_t dr = blockIdx.y * blockDim.y + threadIdx.y;
    if (dr < dst_rows && dc < dst_cols) {
        int sr0 = dr * cell_rows;
        int sc0 = dc * cell_cols;
        int dst_data[CH] = {0};
        for (int sr = sr0; sr < cell_rows + sr0; ++sr) {
            for (int sc = sc0; sc < cell_cols + sc0; ++sc) {
                for (size_t ch = 0; ch < CH; ++ch) {
                    dst_data[ch] += at(src, sr, sc, ch);
                }
            }
        }

        for (size_t ch = 0; ch < CH; ++ch) {
            at(dst, dr, dc, ch) =
                    (uchar)(dst_data[ch] * _cell_rows * _cell_cols);
        }
    }
}

template <size_t CH>
__global__ void resize_area_version2_shrink_fast_32f_kernel(
        const float* __restrict__ src, float* dst, const size_t src_rows,
        const size_t src_cols, const size_t dst_rows, const size_t dst_cols,
        const size_t src_step, const size_t dst_step, const size_t cell_rows,
        const size_t cell_cols, const float _cell_rows,
        const float _cell_cols) {
    size_t dc0 = blockIdx.x * blockDim.x;
    size_t dr = blockIdx.y * blockDim.y + threadIdx.y;
    if (dr < dst_rows && dc0 < dst_cols) {
        __shared__ float lc_dst_data[THREADS_Y][THREADS_X * CH];
        int sc0 = dc0 * cell_cols * CH;
        int sr0 = dr * cell_rows;

        for (size_t ch = 0; ch < CH; ch++)
            lc_dst_data[threadIdx.y][threadIdx.x * CH + ch] = 0;

        __syncthreads();

        size_t block_cell_width = cell_cols * CH * blockDim.x;
        for (int i = threadIdx.x, sc = sc0 + threadIdx.x;
             i < block_cell_width && sc < src_cols * CH;
             i += blockDim.x, sc += blockDim.x) {
            float sum = 0;
            for (int j = 0, sr = sr0 * src_step; j < cell_rows;
                 j++, sr += src_step)
                sum += src[sr + sc];
            atomicAdd(&(lc_dst_data[threadIdx.y]
                                   [(i / (cell_cols * CH)) * CH + i % CH]),
                      sum);
        }

        __syncthreads();

        size_t dc = dc0 + threadIdx.x;
        if (dc < dst_cols) {
            for (size_t ch = 0; ch < CH; ++ch)
                at(dst, dr, dc, ch) =
                        lc_dst_data[threadIdx.y][threadIdx.x * CH + ch] *
                        _cell_rows * _cell_cols;
        }
    }
}

template <size_t CH>
__global__ void resize_area_version2_shrink_fast_8u_kernel(
        const uchar* __restrict__ src, uchar* dst, const size_t src_rows,
        const size_t src_cols, const size_t dst_rows, const size_t dst_cols,
        const size_t src_step, const size_t dst_step, const size_t cell_rows,
        const size_t cell_cols, const float _cell_rows,
        const float _cell_cols) {
    size_t dc0 = blockIdx.x * blockDim.x;
    size_t dr = blockIdx.y * blockDim.y + threadIdx.y;
    if (dr < dst_rows && dc0 < dst_cols) {
        __shared__ int lc_dst_data[THREADS_Y][THREADS_X * CH];
        int sc0 = dc0 * cell_cols * CH;
        int sr0 = dr * cell_rows;

        for (size_t ch = 0; ch < CH; ch++)
            lc_dst_data[threadIdx.y][threadIdx.x * CH + ch] = 0;

        __syncthreads();

        size_t block_cell_width = cell_cols * CH * blockDim.x;
        for (int i = threadIdx.x, sc = sc0 + threadIdx.x;
             i < block_cell_width && sc < src_cols * CH;
             i += blockDim.x, sc += blockDim.x) {
            int sum = 0;
            for (int j = 0, sr = sr0 * src_step; j < cell_rows;
                 j++, sr += src_step)
                sum += src[sr + sc];
            atomicAdd(&(lc_dst_data[threadIdx.y]
                                   [(i / (cell_cols * CH)) * CH + i % CH]),
                      sum);
        }

        __syncthreads();

        size_t dc = dc0 + threadIdx.x;
        if (dc < dst_cols) {
            for (size_t ch = 0; ch < CH; ++ch)
                at(dst, dr, dc, ch) = (uchar)(
                        lc_dst_data[threadIdx.y][threadIdx.x * CH + ch] *
                        _cell_rows * _cell_cols);
        }
    }
}

template <typename T, size_t CH>
void resize_area_proxy(const T* src, T* dst, const size_t src_rows,
                       const size_t src_cols, const size_t dst_rows,
                       const size_t dst_cols, const size_t src_step,
                       const size_t dst_step, void* workspace,
                       cudaStream_t stream) {
    dim3 THREADS(THREADS_X, THREADS_Y, 1);

    float row_scale = (float)src_rows / dst_rows;
    float col_scale = (float)src_cols / dst_cols;

    if (src_rows > dst_rows && src_cols > dst_cols) {
        if (src_rows % dst_rows == 0 && src_cols % dst_cols == 0) {
            dim3 BLOCKS(DIVUP(dst_cols, THREADS.x), DIVUP(dst_rows, THREADS.y));

            if (sizeof(T) == sizeof(float)) {
                if ((CH == 1 && (sizeof(T) * CH * col_scale <= 24)) ||
                    (CH == 3 && (sizeof(T) * CH * col_scale <= 36))) {
                    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
                    resize_area_version1_shrink_fast_32f_kernel<CH>
                            <<<BLOCKS, THREADS, 0, stream>>>(
                                    (const float*)src, (float*)dst, src_rows,
                                    src_cols, dst_rows, dst_cols, src_step,
                                    dst_step, (size_t)row_scale,
                                    (size_t)col_scale, (float)1 / row_scale,
                                    (float)1 / col_scale);
                } else {
                    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
                    resize_area_version2_shrink_fast_32f_kernel<CH>
                            <<<BLOCKS, THREADS, 0, stream>>>(
                                    (const float*)src, (float*)dst, src_rows,
                                    src_cols, dst_rows, dst_cols, src_step,
                                    dst_step, (size_t)row_scale,
                                    (size_t)col_scale, (float)1 / row_scale,
                                    (float)1 / col_scale);
                }

            } else {
                if (sizeof(T) * CH * col_scale <= 24) {
                    cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
                    resize_area_version1_shrink_fast_8u_kernel<CH>
                            <<<BLOCKS, THREADS, 0, stream>>>(
                                    (const uchar*)src, (uchar*)dst, src_rows,
                                    src_cols, dst_rows, dst_cols, src_step,
                                    dst_step, (size_t)row_scale,
                                    (size_t)col_scale, (float)1 / row_scale,
                                    (float)1 / col_scale);
                } else {
                    cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
                    resize_area_version2_shrink_fast_8u_kernel<CH>
                            <<<BLOCKS, THREADS, 0, stream>>>(
                                    (const uchar*)src, (uchar*)dst, src_rows,
                                    src_cols, dst_rows, dst_cols, src_step,
                                    dst_step, (size_t)row_scale,
                                    (size_t)col_scale, (float)1 / row_scale,
                                    (float)1 / col_scale);
                }
            }

        } else {
            size_t access_step = (int)(sizeof(T) * CH * col_scale);
            if (access_step <= 24) {
                dim3 BLOCKS(DIVUP(dst_cols, THREADS.x),
                            DIVUP(dst_rows, THREADS.y));

                cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

                if (sizeof(T) == sizeof(float)) {
                    resize_area_version1_shrink_32f_kernel<CH>
                            <<<BLOCKS, THREADS, 0, stream>>>(
                                    (const float*)src, (float*)dst, src_rows,
                                    src_cols, dst_rows, dst_cols, src_step,
                                    dst_step, row_scale, col_scale,
                                    (float)1 / row_scale, (float)1 / col_scale);
                } else {
                    resize_area_version1_shrink_8u_kernel<CH>
                            <<<BLOCKS, THREADS, 0, stream>>>(
                                    (const uchar*)src, (uchar*)dst, src_rows,
                                    src_cols, dst_rows, dst_cols, src_step,
                                    dst_step, row_scale, col_scale,
                                    (float)1 / row_scale, (float)1 / col_scale);
                }

            } else if (access_step > 24) {
                dim3 BLOCKS(DIVUP(dst_cols, THREADS.x),
                            DIVUP(dst_rows, THREADS.y));

                cudaDeviceSetCacheConfig(cudaFuncCachePreferNone);

                if (sizeof(T) == sizeof(float)) {
                    resize_area_version2_shrink_32f_kernel<CH>
                            <<<BLOCKS, THREADS, 0, stream>>>(
                                    (const float*)src, (float*)dst, src_rows,
                                    src_cols, dst_rows, dst_cols, src_step,
                                    dst_step, row_scale, col_scale,
                                    (float)1 / row_scale, (float)1 / col_scale);
                } else {
                    resize_area_version2_shrink_8u_kernel<CH>
                            <<<BLOCKS, THREADS, 0, stream>>>(
                                    (const uchar*)src, (uchar*)dst, src_rows,
                                    src_cols, dst_rows, dst_cols, src_step,
                                    dst_step, row_scale, col_scale,
                                    (float)1 / row_scale, (float)1 / col_scale);
                }
            }
        }
    } else {
        resize_linear_proxy<T, CH>(src, dst, src_rows, src_cols, dst_rows,
                                   dst_cols, src_step, dst_step, workspace,
                                   stream);
    }
}

}  // anonymous namespace

template <typename T>
void megdnn::cuda::resize::resize_cv(
        const T* src, T* dst, const size_t src_rows, const size_t src_cols,
        const size_t dst_rows, const size_t dst_cols, const size_t src_step,
        const size_t dst_step, size_t ch, InterpolationMode imode,
        void* workspace, cudaStream_t stream) {
    megdnn_assert(ch == 1 || ch == 3);
#define cb(_mode, _MODE)                                               \
    case INTER_##_MODE: {                                              \
        if (ch == 1) {                                                 \
            resize_##_mode##_proxy<T, 1>(src, dst, src_rows, src_cols, \
                                         dst_rows, dst_cols, src_step, \
                                         dst_step, workspace, stream); \
        } else {                                                       \
            resize_##_mode##_proxy<T, 3>(src, dst, src_rows, src_cols, \
                                         dst_rows, dst_cols, src_step, \
                                         dst_step, workspace, stream); \
        }                                                              \
        break;                                                         \
    }

    switch (imode) {
        cb(nearest, NEAREST);
        cb(linear, LINEAR);
        cb(cubic, CUBIC);
        cb(lanczos4, LANCZOS4);
        cb(area, AREA);
        default:
            megdnn_throw("unsupported interpolation mode");
            break;
    }
#undef cb
}

#define INST(_type)                                                    \
    template void megdnn::cuda::resize::resize_cv<_type>(              \
            const _type* src, _type* dst, const size_t src_rows,       \
            const size_t src_cols, const size_t dst_rows,              \
            const size_t dst_cols, const size_t src_step,              \
            const size_t dst_step, size_t ch, InterpolationMode imode, \
            void* workspace, cudaStream_t stream);

INST(float);
INST(uchar);

#undef cb

// vim: syntax=cpp.doxygen
