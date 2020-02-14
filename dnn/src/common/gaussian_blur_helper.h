/**
 * \file dnn/src/common/gaussian_blur_helper.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/common/cv/common.h"
#include "src/common/utils.h"

#pragma once

namespace megdnn {
namespace megcv {
namespace gaussian_blur {

template <typename T>
inline static Mat<T> getGaussianKernel(size_t n, double sigma) {
    const int SMALL_GAUSSIAN_SIZE = 7;
    static const float small_gaussian_tab[][SMALL_GAUSSIAN_SIZE] = {
            {1.f},
            {0.25f, 0.5f, 0.25f},
            {0.0625f, 0.25f, 0.375f, 0.25f, 0.0625f},
            {0.03125f, 0.109375f, 0.21875f, 0.28125f, 0.21875f, 0.109375f,
             0.03125f}};

    const float* fixed_kernel =
            n % 2 == 1 && n <= SMALL_GAUSSIAN_SIZE && sigma <= 0
                    ? small_gaussian_tab[n >> 1]
                    : 0;

    Mat<T> kernel(1, n, 1);

    T* c = kernel.ptr();

    double sigmaX = sigma > 0 ? sigma : ((n - 1) * 0.5 - 1) * 0.3 + 0.8;
    double scale2X = -0.5 / (sigmaX * sigmaX);
    double sum = 0;

    int i;
    for (i = 0; i < (int)n; i++) {
        double x = i - (n - 1) * 0.5;
        double t = fixed_kernel ? (double)fixed_kernel[i]
                                : std::exp(scale2X * x * x);
        {
            c[i] = (T)t;
            sum += c[i];
        }
    }

    sum = 1. / sum;
    for (i = 0; i < (int)n; i++)
        c[i] = (T)(c[i] * sum);

    return kernel;
}

template <typename T>
inline static void createGaussianKernels(Mat<T>& kx, Mat<T>& ky, Size ksize,
                                         double sigma1, double sigma2) {
    if (sigma2 <= 0)
        sigma2 = sigma1;

    if (ksize.cols() <= 0 && sigma1 > 0) {
        double num =
                sigma1 * (std::is_same<T, unsigned char>::value ? 3 : 4) * 2 +
                1;
        num = (int)(num + (num >= 0 ? 0.5 : -0.5));
        ksize.cols() = ((int)num) | 1;
    }
    if (ksize.rows() <= 0 && sigma2 > 0) {
        double num =
                sigma2 * (std::is_same<T, unsigned char>::value ? 3 : 4) * 2 +
                1;
        num = (int)(num + (num >= 0 ? 0.5 : -0.5));
        ksize.rows() = ((int)num) | 1;
    }

    megdnn_assert(ksize.cols() > 0 && ksize.cols() % 2 == 1 &&
                  ksize.rows() > 0 && ksize.rows() % 2 == 1);

    sigma1 = std::max(sigma1, 0.);
    sigma2 = std::max(sigma2, 0.);

    kx = getGaussianKernel<T>(ksize.cols(), sigma1);
    if (ksize.rows() == ksize.cols() && std::abs(sigma1 - sigma2) < DBL_EPSILON)
        ky = kx;
    else
        ky = getGaussianKernel<T>(ksize.rows(), sigma2);
}

}  // namespace gaussian_blur
}  // namespace megcv
}  // namespace megdnn

// vim: syntax=cpp.doxygen
