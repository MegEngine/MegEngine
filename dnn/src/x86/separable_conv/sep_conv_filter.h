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
 * \file dnn/src/x86/separable_conv/sep_conv_filter.h
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

#include "./sep_conv_common.h"
#include "src/common/utils.h"
#pragma once
namespace megdnn {
namespace x86 {
namespace sep_conv {
//#define BorderMode param::SeparableConv::BorderMode
//#define BorderMode SeparableConv::Param::BorderMode
using BorderMode = SeparableConv::Param::BorderMode;
//using uchar = unsigned char;
//using ushort = unsigned short;

class BaseRowFilter {
public:
    //! the default constructor
    BaseRowFilter();
    //! the destructor
    virtual ~BaseRowFilter();
    //! the filtering operator. Must be overridden in the derived classes. The horizontal border interpolation is done outside of the class.
    virtual void operator()(const uchar* src, uchar* dst, uchar* kernel, int width, int cn) = 0;

    int ksize;
    int anchor;
};


class BaseColumnFilter {
public:
    //! the default constructor
    BaseColumnFilter();
    //! the destructor
    virtual ~BaseColumnFilter();
    //! the filtering operator. Must be overridden in the derived classes. The vertical border interpolation is done outside of the class.
    virtual void operator()(const uchar** src, uchar* dst, uchar* kernel, int dststep, int dstcount, int width) = 0;
    //! resets the internal buffers, if any
    virtual void reset();

    int ksize;
    int anchor;
};

class FilterEngine {
public:
    //FilterEngine();

    FilterEngine(const int &ih, const int &iw,
                const int &oh, const int &ow,
                const int &kh, const int &kw,
                const int &anchor_h, const int &anchor_w,
                BorderMode borderType = BorderMode::BORDER_CONSTANT,
                bool is_symm_kernel = true);

    virtual ~FilterEngine();

    void init(  const int &ih, const int &iw,
                const int &oh, const int &ow,
                const int &kh, const int &kw,
                const int &anchor_h, const int &anchor_w,
                BorderMode borderType,
                bool is_symm_kernel);

    void exec(  const TensorND & src,
                const TensorND & kernel_x,
                const TensorND & kernel_y,
                const TensorND & dst);

    BaseRowFilter* getSepRowFilter();
    BaseColumnFilter* getSepColFilter();

    inline int getBorderRowIdx1(int idx);


private:
    // kernel
    int ksize_x_,  ksize_y_;
    int anchor_x_, anchor_y_;               // anchors is useless in this version.
    int is_symm_kernel_;                     // are the kernels symmtric.

    //filter
    BaseRowFilter *rowFilter_;
    BaseColumnFilter *colFilter_;

    //buffer
    std::vector<float> srcRow_;              // a buffer of a single appended input row
    std::vector<uchar> ringBuf_;             // a buffer of middle results. size = maxBufferRow * (maxWidth + kernel_w - 1)
    std::vector<float*> row_ptr_;
    int rowBuffStride_;                      // aligned stride of a row in the buffer.
    int rowBufferOutputRow_;                 // each time the buffer is full, we can calculate 'rowBufferOutputRow' out rows at one time.
                                             // In this version rowBufferOutputRow_ = 1.
    int maxBufferRow_;                       // max_size_of buffer row. maxBufferRow_ = ksize_y + (rowBufferOutputRow_ - 1)
                                             // In this version maxBufferRow_ = ksize_y.

    //border
    BorderMode borderType_;
    int dx1_, dx2_, dy1_, dy2_;
    std::vector<int> borderTab_;             // src idx of border elements
    std::vector<uchar> constBorderValue_;    // template of append value (out of mat edge)
    std::vector<uchar> constBorderRow_;      // a row of srcRow full of border value ---rowFilter---> constBorderRow
};


} // namespace sep_conv
} // namespace x86
} // namespace megdnn

// vim: syntax=cpp.doxygen
