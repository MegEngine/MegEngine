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
 * \file dnn/src/fallback/gaussian_blur/opr_impl.cpp
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

#include "./opr_impl.h"
#include "./filter.h"

#include "src/common/cv/common.h"
#include "src/common/cv/helper.h"
#include "src/common/gaussian_blur_helper.h"
#include "src/fallback/handle.h"

namespace megdnn {
namespace fallback {
using namespace megcv;

using BorderMode = param::GaussianBlur::BorderMode;

template <typename T>
void GaussianBlurImpl::gaussian_blur_exec(const TensorND& src_tensor,
                                          const TensorND& dst_tensor) {
    Size ksize = Size(param().kernel_height, param().kernel_width);

    Mat<T> kernel_column(1, ksize.cols(), 1);
    Mat<T> kernel_row(1, ksize.rows(), 1);

    gaussian_blur::createGaussianKernels<T>(kernel_column, kernel_row, ksize,
                                            param().sigma_x, param().sigma_y);
    size_t src_channels = src_tensor.layout.shape[3];

    T border_value[4] = {0, 0, 0, 0};

    using namespace gaussian_blur;

    BaseRowFilter* row_filter = getLinearRowFilter<T, T>(kernel_column);
    BaseColumnFilter* column_filter =
            getLinearColumnFilter<T, T>(kernel_row, (int)0);

    FilterEngine<T, T> filter(row_filter, column_filter, src_channels,
                              border_value, param().border_mode);

    megdnn_assert(param().border_mode != BorderMode::BORDER_ISOLATED);
    for (size_t i = 0; i < src_tensor.layout.shape[0]; ++i) {
        Mat<T> src = TensorND2Mat<T>(src_tensor, i);
        Mat<T> dst = TensorND2Mat<T>(dst_tensor, i);

        filter.apply(src, dst);
    }
}

void GaussianBlurImpl::gaussian_blur_exec_8u(const TensorND& src_tensor,
                                             const TensorND& dst_tensor) {
    megdnn_assert(src_tensor.layout.dtype == dtype::Uint8());
    Size ksize = Size(param().kernel_height, param().kernel_width);

    Mat<float> kernel_column(1, ksize.cols(), 1);
    Mat<float> kernel_row(1, ksize.rows(), 1);

    gaussian_blur::createGaussianKernels<float>(
            kernel_column, kernel_row, ksize, param().sigma_x, param().sigma_y);
    size_t src_channels = src_tensor.layout.shape[3];

    const uint8_t bits = 8;
    //! Shift, make the elements of the kernel int
    Mat<int> kernel_column_int(1, kernel_column.cols(), 1);
    Mat<int> kernel_row_int(1, kernel_row.cols(), 1);
    for (size_t i = 0; i < kernel_row.cols(); i++) {
        kernel_row_int.at(0, i, 0) =
                static_cast<int>(kernel_row.at(0, i, 0) * (1 << bits));
    }
    for (size_t i = 0; i < kernel_column.cols(); i++) {
        kernel_column_int.at(0, i, 0) =
                static_cast<int>(kernel_column.at(0, i, 0) * (1 << bits));
    }

    uchar border_value[4] = {0, 0, 0, 0};

    using namespace gaussian_blur;
    BaseRowFilter* rowFilter =
            getLinearRowFilter<uchar, int>(kernel_column_int);
    BaseColumnFilter* columnFilter =
            getLinearColumnFilter<int, uchar>(kernel_row_int, bits * 2);

    FilterEngine<uchar, int> filter(rowFilter, columnFilter, src_channels,
                                    border_value, param().border_mode);

    megdnn_assert(param().border_mode != BorderMode::BORDER_ISOLATED);
    for (size_t i = 0; i < src_tensor.layout.shape[0]; ++i) {
        Mat<uchar> src = TensorND2Mat<uchar>(src_tensor, i);
        Mat<uchar> dst = TensorND2Mat<uchar>(dst_tensor, i);

        filter.apply(src, dst);
    }
}

void GaussianBlurImpl::exec(_megdnn_tensor_in src, _megdnn_tensor_in dst,
                            _megdnn_workspace workspace) {
    using namespace megcv;
    check_exec(src.layout, dst.layout, workspace.size);
    MEGDNN_DISPATCH_CPU_KERN_OPR(if (dst.layout.dtype == dtype::Float32()) {
        gaussian_blur_exec<float>(src, dst);
    } else if (dst.layout.dtype == dtype::Uint8()) {
        gaussian_blur_exec_8u(src, dst);
    } else { megdnn_throw("Unsupported datatype of GaussianBlur optr."); });
}

}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
