/**
 * \file dnn/src/arm_common/separable_conv/sep_conv_filter.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "./sep_conv_common.h"
#include "src/common/utils.h"
#pragma once
namespace megdnn {
namespace arm_common {
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
} // namespace arm_common
} // namespace megdnn

// vim: syntax=cpp.doxygen
