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
 * \file dnn/src/common/cv/filter.cpp
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

#include "./filter.h"

namespace megdnn {
namespace megcv {
namespace filter_common {

#define VEC_ALIGN 16

template <typename ST, typename FT>
FilterEngine<ST, FT>::FilterEngine(BaseRowFilter* row_filter,
                                  BaseColumnFilter* column_filter, size_t ch,
                                  const ST* border_value, BorderMode bmode)
        : m_row_filter(row_filter),
          m_column_filter(column_filter),
          m_ch(ch),
          m_bmode(bmode) {
    megdnn_assert(m_row_filter && m_column_filter);
    megdnn_assert(m_bmode != BorderMode::BORDER_WRAP);

    m_ksize.cols() = m_row_filter->ksize;
    m_ksize.rows() = m_column_filter->ksize;
    m_anchor.x = m_row_filter->anchor;
    m_anchor.y = m_column_filter->anchor;
    m_buf_step = 0;

    //! the anchor must be in the kernerl
    megdnn_assert(0 <= m_anchor.x && m_anchor.x < m_ksize.cols() &&
                  0 <= m_anchor.y && m_anchor.y < m_ksize.rows());

    int src_elem_size = (int)sizeof(ST) * m_ch;
    m_border_elem_size = src_elem_size / ((sizeof(ST) >= 4) ? sizeof(int) : 1);
    int border_length = std::max<int>((int)(m_ksize.cols() - 1), (int)1);
    m_border_table.resize(border_length * m_border_elem_size);

    if (m_bmode == BorderMode::BORDER_CONSTANT) {
        //! store the border_value array to m_const_border_value, the type
        //! of buffer and image may be different, So use byte to store
        m_const_border_value.resize(m_ch * sizeof(ST) * border_length);
        for (int i = 0; i < src_elem_size * border_length; i += src_elem_size)
            for (int j = 0; j < src_elem_size; j++)
                m_const_border_value[i + j] = ((uchar*)(border_value))[j];
    }
    m_whole_size = Size(-1, -1);
}

template <typename ST, typename FT>
FilterEngine<ST, FT>::~FilterEngine() {
    delete m_row_filter;
    delete m_column_filter;
}

template <typename ST, typename FT>
void FilterEngine<ST, FT>::start(const Mat<ST>& src) {
    m_whole_size.cols() = src.cols();
    m_whole_size.rows() = src.rows();

    int element_size = (int)sizeof(ST) * m_ch;
    int buf_elem_size = (int)sizeof(FT) * m_ch;

    int cn = m_ch;
    m_src_row.resize(element_size * (m_whole_size.width() + m_ksize.width() - 1));
    if (m_bmode == BorderMode::BORDER_CONSTANT) {
        m_const_border_row.resize(
                buf_elem_size *
                (m_whole_size.width() + m_ksize.width() - 1 + VEC_ALIGN));
        uchar *dst = align_ptr(&m_const_border_row[0], VEC_ALIGN), *tdst;
        int n = (int)m_const_border_value.size(), N;
        N = (m_whole_size.width() + m_ksize.width() - 1) * element_size;
        tdst = &m_src_row[0];

        for (int i = 0; i < N; i += n) {
            n = std::min<int>((int)n, (int)(N - i));
            for (int j = 0; j < n; j++)
                tdst[i + j] = m_const_border_row[j];
        }

        (*m_row_filter)(&m_src_row[0], dst, m_whole_size.width(), cn);
    }


    m_buf_step = buf_elem_size *
              (int)align_size(m_whole_size.width() + m_ksize.width() - 1,
                              VEC_ALIGN);
    m_ring_buf.resize(m_buf_step * m_ksize.height() + VEC_ALIGN);
    m_left_width = m_anchor.x;
    m_right_width = m_ksize.width() - m_anchor.x - 1;

    //! init the row with border values
    if (m_left_width > 0 || m_right_width > 0) {
        //! calc the index of the border value, we will not calc it when process
        //! border each time
        if (m_bmode == BorderMode::BORDER_CONSTANT) {
            memcpy(m_src_row.data(), m_const_border_row.data(),
                   m_left_width * element_size);
            memcpy(m_src_row.data() +
                           (m_whole_size.width() + m_left_width) * element_size,
                   m_const_border_row.data(), m_right_width * element_size);
        } else {
            //! calc the index of the border value, we will not calc it when
            //! process border each time
            for (int i = 0; i < m_left_width; i++) {
                int p0 = gaussian_blur::border_interpolate(i - m_left_width,
                                            m_whole_size.width(), m_bmode) *
                         m_border_elem_size;
                for (int j = 0; j < m_border_elem_size; j++)
                    m_border_table[i * m_border_elem_size + j] = p0 + j;
            }

            for (int i = 0; i < m_right_width; i++) {
                int p0 = gaussian_blur::border_interpolate(m_whole_size.width() + i,
                                            m_whole_size.width(), m_bmode) *
                         m_border_elem_size;
                for (int j = 0; j < m_border_elem_size; j++)
                    m_border_table[(i + m_left_width) * m_border_elem_size +
                                   j] = p0 + j;
            }
        }
    }

    if (m_column_filter)
        m_column_filter->reset();
}

template <typename ST, typename FT>
int FilterEngine<ST, FT>::proceed(const uchar* src, int srcstep, int count,
                                  uchar* dst, int dststep) {
    const int* btab = &m_border_table[0];
    int src_elem_size = static_cast<int>(sizeof(ST) * m_ch);
    bool makeBorder = (m_left_width > 0 || m_right_width > 0) &&
                      m_bmode != BorderMode::BORDER_CONSTANT;
    int dy = 0, i = 0;

    int row_count = 0;
    int start_y = 0;
    std::vector<uchar*> buf_rows(m_ksize.rows(), nullptr);
    for (;; dst += dststep * i, dy += i) {
        int dcount = m_ksize.height() - m_anchor.y - start_y - row_count;
        dcount = dcount > 0 ? dcount : 1;
        dcount = std::min<int>(dcount, count);
        count -= dcount;
        for (; dcount-- > 0; src += srcstep) {
            int bi = (start_y + row_count) % m_ksize.height();
            uchar* brow =
                    align_ptr(&m_ring_buf[0], VEC_ALIGN) + bi * m_buf_step;
            uchar* row = &m_src_row[0];

            if (++row_count > static_cast<int>(m_ksize.height())) {
                --row_count;
                ++start_y;
            }

            memcpy(row + m_left_width * src_elem_size, src,
                   m_whole_size.width() * src_elem_size);

            if (makeBorder) {
                if (m_border_elem_size * static_cast<int>(sizeof(int)) ==
                    src_elem_size) {
                    const int* isrc = reinterpret_cast<const int*>(src);
                    int* irow = reinterpret_cast<int*>(row);

                    for (int i = 0; i < m_left_width * m_border_elem_size; i++)
                        irow[i] = isrc[btab[i]];
                    for (int i = 0; i < m_right_width * m_border_elem_size;
                         i++) {
                        irow[i + (m_whole_size.width() + m_left_width) *
                                         m_border_elem_size] =
                                isrc[btab[i +
                                          m_left_width * m_border_elem_size]];
                    }
                } else {
                    for (int i = 0; i < m_left_width * src_elem_size; i++)
                        row[i] = src[btab[i]];
                    for (int i = 0; i < m_right_width * src_elem_size; i++)
                        row[i + (m_whole_size.width() + m_left_width) *
                                        src_elem_size] =
                                src[btab[i + m_left_width * src_elem_size]];
                }
            }

            (*m_row_filter)(row, brow, m_whole_size.width(), m_ch);
        }

        int max_i = std::min<int>(
                m_ksize.height(),
                m_whole_size.height() - dy + (m_ksize.height() - 1));
        for (i = 0; i < max_i; i++) {
            int src_y = gaussian_blur::border_interpolate(dy + i - m_anchor.y,
                                           m_whole_size.rows(), m_bmode);
            if (src_y < 0)
                buf_rows[i] = align_ptr(&m_const_border_row[0], VEC_ALIGN);
            else {
                megdnn_assert(src_y >= start_y);
                if (src_y >= start_y + row_count) {
                    break;
                }
                int bi = src_y % m_ksize.height();
                buf_rows[i] =
                        align_ptr(&m_ring_buf[0], VEC_ALIGN) + bi * m_buf_step;
            }
        }
        if (i < static_cast<int>(m_ksize.height())) {
            break;
        }
        i -= m_ksize.height() - 1;
        (*m_column_filter)(const_cast<const uchar**>(&buf_rows[0]), dst,
                           dststep, i, m_whole_size.width() * m_ch);
    }

    return dy;
}

template <typename ST, typename FT>
void FilterEngine<ST, FT>::apply(const Mat<ST>& src, Mat<ST>& dst) {
    int src_step = src.step() * sizeof(ST);
    int dst_step = dst.step() * sizeof(ST);
    start(src);
    proceed(reinterpret_cast<const uchar*>(src.ptr()),
            static_cast<int>(src_step), m_whole_size.height(),
            reinterpret_cast<uchar*>(dst.ptr()), static_cast<int>(dst_step));
}

//! explicit instantiation template
template FilterEngine<uchar, int>::FilterEngine(
        BaseRowFilter* _rowFilter, BaseColumnFilter* _columnFilter, size_t _CH,
        const uchar* _borderValue, BorderMode _BorderType);
template FilterEngine<float, float>::FilterEngine(
        BaseRowFilter* _rowFilter, BaseColumnFilter* _columnFilter, size_t _CH,
        const float* _borderValue, BorderMode _BorderType);

template void FilterEngine<uchar, int>::apply(const Mat<uchar>& src,
                                              Mat<uchar>& dst);
template void FilterEngine<float, float>::apply(const Mat<float>& src,
                                              Mat<float>& dst);

template FilterEngine<unsigned char, int>::~FilterEngine();
template FilterEngine<float, float>::~FilterEngine();

}  // namespace filter_common
}  // namespace megcv
}  // namespace megdnn

// vim: syntax=cpp.doxygen
