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
 * \file dnn/src/common/cv/common.h
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

#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>
#include "megdnn/basic_types.h"

// for x86, armv7, armv8, naive
#define MEGCV_ENABLE_UNROLLED 1

namespace megdnn {
namespace megcv {

class Size {
public:
    Size(size_t rows, size_t cols) : m_rows(rows), m_cols(cols) {}
    Size() : m_rows(0), m_cols(0) {}

    size_t rows() const { return m_rows; }
    size_t& rows() { return m_rows; }
    size_t cols() const { return m_cols; }
    size_t& cols() { return m_cols; }
    size_t height() const { return rows(); }
    size_t& height() { return rows(); }
    size_t width() const { return cols(); }
    size_t& width() { return cols(); }

    bool operator==(const Size& rhs) const {
        return rows() == rhs.rows() && cols() == rhs.cols();
    }

private:
    size_t m_rows, m_cols;
};

class MatShape : public Size {
public:
    MatShape(size_t rows, size_t cols, size_t channels)
            : Size(rows, cols), m_channels(channels) {}

    size_t channels() const { return m_channels; }

    bool operator==(const MatShape& rhs) const {
        return Size::operator==(rhs) && channels() == rhs.channels();
    }

private:
    size_t m_channels;
};

/*!
 * A row-major device matrix wrapper
 */
template <typename T>
class Mat {
private:
    size_t m_rows, m_cols;
    size_t m_channels;
    size_t m_step;

    std::shared_ptr<T> m_data;

    size_t m_offset;

public:
    void* raw_ptr() { return static_cast<void*>(m_data.get() + m_offset); }
    const void* raw_ptr() const {
        return static_cast<void*>(m_data.get() + m_offset);
    }

    Mat();
    Mat(size_t rows, size_t cols, size_t channels, size_t step);
    Mat(size_t rows, size_t cols, size_t channels);
    // do not try to manage data by shared_ptr
    Mat(size_t rows, size_t cols, size_t channels, T* data);
    Mat(size_t rows, size_t cols, size_t channels, size_t step, T* data);
    // shallow-copy constructor
    Mat(const Mat<T>& rhs);
    Mat(const Mat<T>& rhs, size_t row_offset, size_t row_count,
        size_t col_offset, size_t col_count);
    Mat<T>& operator=(const Mat<T>& rhs);

    T& at(size_t r, size_t c, size_t ch);
    const T& at(size_t r, size_t c, size_t ch) const;

    Mat<T> clone() const;

    // read data from src
    void read(const T* src);
    // write data to dst
    void write(T* dst) const;

    const T* ptr(size_t r = 0) const {
        return static_cast<const T*>(raw_ptr()) + r * m_step;
    }
    T* ptr(size_t r = 0) { return static_cast<T*>(raw_ptr()) + r * m_step; }
    size_t height() const { return rows(); }
    size_t width() const { return cols(); }
    size_t rows() const { return m_rows; }
    size_t cols() const { return m_cols; }
    size_t channels() const { return m_channels; }
    size_t step() const { return m_step; }
    size_t total_nr_elem() const { return rows() * cols() * channels(); }
    size_t total_span_elem() const { return rows() * step(); }
    bool equals(const Mat<T>& rhs) const;
    bool is_continuous() const;

    Size size() const { return {rows(), cols()}; }
    MatShape shape() const { return {rows(), cols(), channels()}; }
};

class Rect {
public:
    size_t y, x, height, width;
    Rect(size_t _y, size_t _x, size_t _height, size_t _width)
            : y(_y), x(_x), height(_height), width(_width) {}
    Rect() : y(0), x(0), height(0), width(0) {}
};

template <class scalar_t>
struct Point {
    scalar_t x, y;

    Point() {}
    Point(scalar_t x, scalar_t y) : x(x), y(y) {}

    Point operator+(const Point& rhs) const { return {x + rhs.x, y + rhs.y}; }
    Point operator-(const Point& rhs) const { return {x - rhs.x, y - rhs.y}; }
    Point operator*(scalar_t f) const { return {x * f, y * f}; }
    Point operator/(scalar_t f) const { return {x / f, y / f}; }
};

template <typename T>
Mat<T> TensorND2Mat(const TensorND& tensor, size_t batch);

// type aliases
using uchar = unsigned char;
using ushort = unsigned short;
using Mat8u = Mat<uchar>;
using Mat32f = Mat<float>;
using Mat64f = Mat<double>;

extern template class Mat<uchar>;
extern template class Mat<float>;
extern template class Mat<double>;
extern template class Mat<short>;
extern template class Mat<unsigned short>;
extern template class Mat<int>;

}  // namespace megcv
}  // namespace megdnn

// vim: syntax=cpp.doxygen
