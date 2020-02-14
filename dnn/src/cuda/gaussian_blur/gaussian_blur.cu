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
 * \file dnn/src/cuda/gaussian_blur/gaussian_blur.cu
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
#include "./gaussian_blur.cuh"

#include "megdnn/dtype.h"
#include "src/cuda/cv/kernel_common.cuh"
#include "src/cuda/utils.cuh"

namespace megdnn {
namespace cuda {

namespace {

static const uint8_t BITS = 8;

#define rep(i, n) for (size_t i = 0; i < (n); ++i)

template <typename T>
__global__ void prepare_kernel(uint8_t* kernel_ptr, size_t kernel_height,
                               size_t kernel_width, double sigma_x,
                               double sigma_y);

template <>
__global__ void prepare_kernel<float>(uint8_t* _kernel_ptr,
                                      size_t kernel_height, size_t kernel_width,
                                      double sigma_x, double sigma_y) {
    float* kernel_ptr = reinterpret_cast<float*>(_kernel_ptr);
    const int kSmallGaussianSize = 7;
    const float small_gaussian_table[4][kSmallGaussianSize] = {
            {1.f},
            {0.25f, 0.5f, 0.25f},
            {0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f},
            {0.03125f, 0.109375f, 0.21875f, 0.28125f, 0.21875f, 0.109375f,
             0.03125f}};

    const float* fixed_kernel_x =
            (kernel_width % 2 == 1 && kernel_width <= kSmallGaussianSize &&
             sigma_x <= 0)
                    ? small_gaussian_table[kernel_width >> 1]
                    : NULL;
    const float* fixed_kernel_y =
            (kernel_height % 2 == 1 && kernel_height <= kSmallGaussianSize &&
             sigma_y <= 0)
                    ? small_gaussian_table[kernel_height >> 1]
                    : NULL;
    sigma_x =
            sigma_x > 0 ? sigma_x : ((kernel_width - 1) * 0.5 - 1) * 0.3 + 0.8;
    double scale_2x = -0.5 / (sigma_x * sigma_x);
    sigma_y =
            sigma_y > 0 ? sigma_y : ((kernel_height - 1) * 0.5 - 1) * 0.3 + 0.8;
    double scale_2y = -0.5 / (sigma_y * sigma_y);

    //! calc gaussian kernel
    double sum = 0;
    rep(iy, kernel_height) {
        double y = iy - (kernel_height - 1) * 0.5;
        double ky = fixed_kernel_y ? static_cast<double>(fixed_kernel_y[iy])
                                   : std::exp(scale_2y * y * y);
        rep(ix, kernel_width) {
            double x = ix - (kernel_width - 1) * 0.5;
            double kx = fixed_kernel_x ? static_cast<double>(fixed_kernel_x[ix])
                                       : std::exp(scale_2x * x * x);

            float kxy = static_cast<float>(kx * ky);
            kernel_ptr[iy * kernel_width + ix] = kxy;
            sum += kxy;
        }
    }

    //! normalize
    sum = 1. / sum;
    rep(i, kernel_width * kernel_height) {
        kernel_ptr[i] = static_cast<float>(sum * kernel_ptr[i]);
    }
}

template <>
__global__ void prepare_kernel<uint8_t>(uint8_t* _kernel_ptr,
                                        size_t kernel_height,
                                        size_t kernel_width, double sigma_x,
                                        double sigma_y) {
    int32_t* kernel_ptr = reinterpret_cast<int32_t*>(_kernel_ptr);
    const int kSmallGaussianSize = 7;
    const float small_gaussian_table[4][kSmallGaussianSize] = {
            {1.f},
            {0.25f, 0.5f, 0.25f},
            {0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f},
            {0.03125f, 0.109375f, 0.21875f, 0.28125f, 0.21875f, 0.109375f,
             0.03125f}};

    const float* fixed_kernel_x =
            (kernel_width % 2 == 1 && kernel_width <= kSmallGaussianSize &&
             sigma_x <= 0)
                    ? small_gaussian_table[kernel_width >> 1]
                    : NULL;
    const float* fixed_kernel_y =
            (kernel_height % 2 == 1 && kernel_height <= kSmallGaussianSize &&
             sigma_y <= 0)
                    ? small_gaussian_table[kernel_height >> 1]
                    : NULL;
    sigma_x =
            sigma_x > 0 ? sigma_x : ((kernel_width - 1) * 0.5 - 1) * 0.3 + 0.8;
    double scale_2x = -0.5 / (sigma_x * sigma_x);
    sigma_y =
            sigma_y > 0 ? sigma_y : ((kernel_height - 1) * 0.5 - 1) * 0.3 + 0.8;
    double scale_2y = -0.5 / (sigma_y * sigma_y);

    size_t kernel_size = kernel_width * kernel_height;

    //! calc the sum of horizontal kernel filter
    double sum_y = 0;
    float* ky_ptr = reinterpret_cast<float*>(kernel_ptr + kernel_size);
    rep(iy, kernel_height) {
        double y = iy - (kernel_height - 1) * 0.5;
        double ky = fixed_kernel_y ? static_cast<double>(fixed_kernel_y[iy])
                                   : std::exp(scale_2y * y * y);
        sum_y += ky;
        ky_ptr[iy] = static_cast<float>(ky);
    }
    sum_y = 1 / sum_y;

    //! calc the sum of vertical kernel filter
    double sum_x = 0;
    float* kx_ptr =
            reinterpret_cast<float*>(kernel_ptr + kernel_size) + kernel_height;
    rep(ix, kernel_width) {
        double x = ix - (kernel_width - 1) * 0.5;
        double kx = fixed_kernel_x ? static_cast<double>(fixed_kernel_x[ix])
                                   : std::exp(scale_2x * x * x);
        sum_x += kx;
        kx_ptr[ix] = static_cast<float>(kx);
    }
    sum_x = 1 / sum_x;

    rep(iy, kernel_height) {
        float ky = ky_ptr[iy];
        int ky_int = (ky * sum_y * (1 << BITS));
        rep(ix, kernel_width) {
            float kx = kx_ptr[ix];

            int kx_int = (kx * sum_x * (1 << BITS));
            kernel_ptr[iy * kernel_width + ix] = kx_int * ky_int;
        }
    }
}

template <typename T, size_t CH, BorderMode bmode>
__global__ void gaussian_blur_kern(const T* src, T* dst, size_t N, size_t H,
                                   size_t W, size_t stride0, size_t stride1,
                                   size_t stride2, size_t stride3,
                                   uint8_t* kernel_ptr, size_t kernel_height,
                                   size_t kernel_width) {
    int iw = blockIdx.x * blockDim.x + threadIdx.x;
    int ih = blockIdx.y * blockDim.y + threadIdx.y;
    if (iw < W && ih < H) {
#pragma unroll
        rep(c, CH) {
            double val = 0;
            rep(iy, kernel_height) {
                int y = megcv::border_interpolate<bmode>(
                        ih + iy - kernel_height / 2, H);
                rep(ix, kernel_width) {
                    int x = megcv::border_interpolate<bmode>(
                            iw + ix - kernel_width / 2, W);

                    //! BORDER_CONSTANT or BORDER_TRANSPARENT
                    if (x != -1 && y != -1) {
                        if (is_same<T, uint8_t>::value) {
                            val += static_cast<double>(reinterpret_cast<int*>(
                                           kernel_ptr)[iy * kernel_width +
                                                       ix]) *
                                   src[blockIdx.z * stride0 + y * stride1 +
                                       x * stride2 + c * stride3];
                        } else {
                            val += static_cast<double>(reinterpret_cast<float*>(
                                           kernel_ptr)[iy * kernel_width +
                                                       ix]) *
                                   src[blockIdx.z * stride0 + y * stride1 +
                                       x * stride2 + c * stride3];
                        }
                    }
                }
            }

            if (is_same<T, uint8_t>::value) {
                dst[blockIdx.z * stride0 + ih * stride1 + iw * stride2 +
                    c * stride3] =
                        static_cast<T>(static_cast<int>(val) >> (2 * BITS));
            } else {
                //! float32
                dst[blockIdx.z * stride0 + ih * stride1 + iw * stride2 +
                    c * stride3] = static_cast<T>(val);
            }
        }
    }
}

#undef rep
}  // namespace

namespace gaussian_blur {

template <typename T, size_t CH, BorderMode bmode>
void gaussian_blur(const T* src, T* dst, size_t N, size_t H, size_t W,
                   size_t stride0, size_t stride1, size_t stride2,
                   size_t stride3, uint8_t* kernel_ptr, size_t kernel_height,
                   size_t kernel_width, double sigma_x, double sigma_y,
                   cudaStream_t stream) {
    //! calc gaussian kernel
    prepare_kernel<T><<<1, 1, 0, stream>>>(kernel_ptr, kernel_height,
                                           kernel_width, sigma_x, sigma_y);
    cuda_check(cudaStreamSynchronize(stream));

    static const int BX = 16;
    static const int BY = 16;
    dim3 threads(BX, BY);
    dim3 blocks(DIVUP(W, BX), DIVUP(H, BY), N);
    gaussian_blur_kern<T, CH, bmode><<<blocks, threads, 0, stream>>>(
            src, dst, N, H, W, stride0, stride1, stride2, stride3, kernel_ptr,
            kernel_height, kernel_width);
    after_kernel_launch();
}

#define INST(T, CH, bmode)                                                  \
    template void gaussian_blur<T, CH, bmode>(                              \
            const T* src, T* dst, size_t N, size_t H, size_t W,             \
            size_t stride0, size_t stride1, size_t stride2, size_t stride3, \
            uint8_t*, size_t, size_t, double, double, cudaStream_t);

#define cb(DType)                                                  \
    INST(typename DTypeTrait<DType>::ctype, 1, BORDER_REPLICATE)   \
    INST(typename DTypeTrait<DType>::ctype, 3, BORDER_REPLICATE)   \
    INST(typename DTypeTrait<DType>::ctype, 1, BORDER_REFLECT)     \
    INST(typename DTypeTrait<DType>::ctype, 3, BORDER_REFLECT)     \
    INST(typename DTypeTrait<DType>::ctype, 1, BORDER_REFLECT_101) \
    INST(typename DTypeTrait<DType>::ctype, 3, BORDER_REFLECT_101) \
    INST(typename DTypeTrait<DType>::ctype, 1, BORDER_CONSTANT)    \
    INST(typename DTypeTrait<DType>::ctype, 3, BORDER_CONSTANT)

cb(dtype::Uint8);
cb(dtype::Float32);

#undef cb
#undef INST

}  // namespace gaussian_blur
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
