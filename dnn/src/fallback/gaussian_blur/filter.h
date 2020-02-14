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
 * \file dnn/src/fallback/gaussian_blur/filter.h
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
#include "src/common/cv/filter.h"

#include <cfloat>
#include <cmath>
#include <type_traits>

namespace megdnn {
namespace megcv {
namespace gaussian_blur {

using namespace filter_common;

/*!
 * \brief get the row filter.
 * \tparam ST The src image type.
 * \tparam FT The inner buffer type, used to store the product of src and
 * filter.
 */
template <typename ST, typename FT>
static BaseRowFilter* getLinearRowFilter(Mat<FT>& kernel) {
    int ksize = kernel.cols();
    int anchor = ksize / 2;

    uchar* kernel_str = static_cast<uchar*>(kernel.raw_ptr());

    if (ksize <= 5) {
        if (std::is_same<ST, uchar>::value && std::is_same<FT, int>::value)
            return new SymmRowSmallFilter<ST, FT, SymmRowSmallNoVec>(
                    kernel, anchor, SymmRowSmallNoVec(kernel_str, ksize));

        if (std::is_same<ST, float>::value && std::is_same<FT, float>::value)
            return new SymmRowSmallFilter<ST, FT, SymmRowSmallNoVec>(
                    kernel, anchor, SymmRowSmallNoVec(kernel_str, ksize));
    }

    if (std::is_same<ST, uchar>::value && std::is_same<FT, int>::value)
        return new RowFilter<ST, FT, RowNoVec>(kernel, anchor,
                                              RowNoVec(kernel_str, ksize));

    if (std::is_same<ST, float>::value && std::is_same<FT, float>::value)
        return new RowFilter<ST, FT, RowNoVec>(kernel, anchor,
                                              RowNoVec(kernel_str, ksize));

    MegCVException(
            "Unsupported combination of source format and buffer format\n");
}

/*!
 * \brief get the column filter.
 * \tparam FT The inner buffer type, used to store the product of src and
 * filter.
 * \tparam DT The dst image type.
 */
template <typename FT, typename DT>
static BaseColumnFilter* getLinearColumnFilter(Mat<FT>& kernel, int bits) {
    int ksize = kernel.cols();
    int anchor = ksize / 2;

    uchar* kernel_str = static_cast<uchar*>(kernel.raw_ptr());
    if (ksize == 3) {
        if (std::is_same<DT, uchar>::value && std::is_same<FT, int>::value)
            return new SymmColumnSmallFilter<FixedPtCastEx<FT, DT>,
                                             SymmColumnSmallNoVec>(
                    kernel, anchor, FixedPtCastEx<FT, DT>(bits),
                    SymmColumnSmallNoVec(kernel_str, ksize, bits));

        if (std::is_same<DT, float>::value && std::is_same<FT, float>::value)
            return new SymmColumnSmallFilter<FixedPtCastEx<FT, DT>,
                                             SymmColumnSmallNoVec>(
                    kernel, anchor, FixedPtCastEx<FT, DT>(0),
                    SymmColumnSmallNoVec(kernel_str, ksize, 0));
    }
    if (std::is_same<DT, uchar>::value && std::is_same<FT, int>::value)
        return new SymmColumnFilter<FixedPtCastEx<FT, DT>, ColumnNoVec>(
                kernel, anchor, FixedPtCastEx<FT, DT>(bits),
                ColumnNoVec(kernel_str, ksize, bits));

    if (std::is_same<DT, float>::value && std::is_same<FT, float>::value)
        return new SymmColumnFilter<FixedPtCastEx<FT, DT>, ColumnNoVec>(
                kernel, anchor, FixedPtCastEx<FT, DT>(),
                ColumnNoVec(kernel_str, ksize, 0));

    MegCVException(
            "Unsupported combination of source format and buffer format\n");
}

}  // namespace gaussian_blur
}  // namespace megcv
}  // namespace megdnn

// vim: syntax=cpp.doxygen
