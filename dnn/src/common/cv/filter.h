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
 * \file dnn/src/common/cv/filter.h
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

#include "src/common/cv/common.h"
#include "src/common/cv/helper.h"
#include "src/common/utils.h"

#include <type_traits>

namespace megdnn  {
namespace megcv {
namespace filter_common {

using BorderMode = param::WarpPerspective::BorderMode;

/* ============================ vecOp ============================== */

/*!
 * \struct RowNoVec
 * \brief Filter a row using the kernel.
 */
struct RowNoVec {
    RowNoVec() {}
    /*!
     * \param kernel The filter kernel
     * \param ksize The size of the kernel
     */
    RowNoVec(const uchar* /*kernel*/, int /*ksize*/) {}

    /*!
     * \param src The src data
     * \param dst The dst data
     * \param width The width of the src
     * \param cn The channel size
     */
    int operator()(const uchar* /*src*/, uchar* /*dst*/, int /*width*/,
                   int /*cn*/) const {
        return 0;
    }
};

/*!
 * \struct ColumnNoVec
 * \brief Filter a column using the kernel.
 */
struct ColumnNoVec {
    ColumnNoVec() {}
    /*!
     * \param kernel The filter kernel
     * \param ksize The size of the kernel
     * \param bits The bits shift, Used only if the type is \c uint8_t
     */
    ColumnNoVec(const uchar* /*kernel*/, int /*ksize*/, int /*bits*/) {}

    /*!
     * \param src The src data
     * \param dst The dst data
     * \param count The count of rows that this column kernel processed.
     * \param width The width of the src
     */
    int operator()(const uchar** /*src*/, uchar* /*dst*/, int& /*count*/,
                   int /*width*/) const {
        return 0;
    }
};

/*!
 * \struct SymmRowSmallFilter
 * \brief Filter a row using the kernel, used if the kernel is symmetry.
 */
struct SymmRowSmallNoVec {
    SymmRowSmallNoVec() {}
    SymmRowSmallNoVec(const uchar*, int) {}
    int operator()(const uchar*, uchar*, int, int) const { return 0; }
};

struct SymmColumnSmallNoVec {
    SymmColumnSmallNoVec() {}
    SymmColumnSmallNoVec(const uchar*, int, int) {}
    int operator()(const uchar**, uchar*, int&, int) const { return 0; }
};

/* ============================ Filters ============================== */

class BaseRowFilter {
public:
    BaseRowFilter() { ksize = anchor = -1; }
    virtual ~BaseRowFilter() {}

    //! the filtering operator. Must be overridden in the derived classes. The
    //! horizontal border interpolation is done outside of the class.
    virtual void operator()(const uchar* src, uchar* dst, int width,
                            int cn) = 0;

    //! The size of the kernel
    int ksize;
    //! The center of the filter, e.g. gaussian blur, anchor is ksize / 2
    int anchor;
};

class BaseColumnFilter {
public:
    BaseColumnFilter() { ksize = anchor = -1; }
    virtual ~BaseColumnFilter() {}

    //! the filtering operator. Must be overridden in the derived classes. The
    //! vertical border interpolation is done outside of the class.
    virtual void operator()(const uchar** src, uchar* dst, int dststep,
                            int dstcount, int width) = 0;
    //! resets the internal buffers, if any
    virtual void reset() {}

    //! The size of the kernel
    int ksize;
    //! The center of the filter, e.g. gaussian blur, anchor is ksize / 2
    int anchor;
};

/*!
 * \struct RowFilter
 * \brief The filter of the row
 * \tparam ST the type of src
 * \tparam DT the type of dst
 * \tparam VecOp process the element using vectorized operator.
 */
template <typename ST, typename DT, class VecOp>
struct RowFilter : public BaseRowFilter {
    RowFilter(const Mat<DT>& kernel_, int anchor_,
              const VecOp& vec_op_ = VecOp()) {
        anchor = anchor_;
        kernel = kernel_.clone();
        ksize = kernel.cols();
        vec_op = vec_op_;
    }

    void operator()(const uchar* src, uchar* dst, int width, int cn) {
        const DT* kx = kernel.ptr();
        const ST* S;
        DT* D = reinterpret_cast<DT*>(dst);
        int i, k;

        i = vec_op(src, dst, width, cn);
        width *= cn;
#if MEGCV_ENABLE_UNROLLED
        for (; i <= width - 4; i += 4) {
            S = reinterpret_cast<const ST*>(src) + i;
            DT f = kx[0];
            DT s0 = f * S[0], s1 = f * S[1], s2 = f * S[2], s3 = f * S[3];

            for (k = 1; k < ksize; k++) {
                S += cn;
                f = kx[k];
                s0 += f * S[0];
                s1 += f * S[1];
                s2 += f * S[2];
                s3 += f * S[3];
            }

            D[i] = s0;
            D[i + 1] = s1;
            D[i + 2] = s2;
            D[i + 3] = s3;
        }
#endif
        for (; i < width; i++) {
            S = reinterpret_cast<const ST*>(src) + i;
            DT s0 = kx[0] * S[0];
            for (k = 1; k < ksize; k++) {
                S += cn;
                s0 += kx[k] * S[0];
            }
            D[i] = s0;
        }
    }

    //! The kernel used in RowFilter
    Mat<DT> kernel;
    //! The vectorized operator used in RowFilter
    VecOp vec_op;
};

template <typename ST, typename DT, class VecOp>
struct SymmRowSmallFilter : public RowFilter<ST, DT, VecOp> {
    SymmRowSmallFilter(const Mat<DT>& kernel_, int anchor_,
                       const VecOp& vec_op_ = VecOp())
            : RowFilter<ST, DT, VecOp>(kernel_, anchor_, vec_op_) {}

    void operator()(const uchar* src, uchar* dst, int width, int cn) {
        int ksize2 = this->ksize / 2, ksize2n = ksize2 * cn;
        const DT* kx = this->kernel.ptr() + ksize2;
        DT* D = reinterpret_cast<DT*>(dst);
        int i = this->vec_op(src, dst, width, cn), j, k;

        //! The center
        const ST* S = reinterpret_cast<const ST*>(src) + i + ksize2n;
        width *= cn;

        if (this->ksize == 1 && kx[0] == 1) {
            for (; i <= width - 2; i += 2) {
                DT s0 = S[i], s1 = S[i + 1];
                D[i] = s0;
                D[i + 1] = s1;
            }
            S += i;
        } else if (this->ksize == 3) {
            DT k0 = kx[0], k1 = kx[1];
            for (; i <= width - 2; i += 2, S += 2) {
                DT s0 = S[0] * k0 + (S[-cn] + S[cn]) * k1,
                   s1 = S[1] * k0 + (S[1 - cn] + S[1 + cn]) * k1;
                D[i] = s0;
                D[i + 1] = s1;
            }
        } else if (this->ksize == 5) {
            DT k0 = kx[0], k1 = kx[1], k2 = kx[2];
            for (; i <= width - 2; i += 2, S += 2) {
                DT s0 = S[0] * k0 + (S[-cn] + S[cn]) * k1 +
                        (S[-cn * 2] + S[cn * 2]) * k2;
                DT s1 = S[1] * k0 + (S[1 - cn] + S[1 + cn]) * k1 +
                        (S[1 - cn * 2] + S[1 + cn * 2]) * k2;
                D[i] = s0;
                D[i + 1] = s1;
            }
        }

        for (; i < width; i++, S++) {
            DT s0 = kx[0] * S[0];
            for (k = 1, j = cn; k <= ksize2; k++, j += cn)
                s0 += kx[k] * (S[j] + S[-j]);
            D[i] = s0;
        }

    }
};

template <class CastOp, class VecOp>
struct ColumnFilter : public BaseColumnFilter {
    typedef typename CastOp::type1 ST;
    typedef typename CastOp::rtype DT;

    ColumnFilter(const Mat<ST>& kernel_, int anchor_,
                     const CastOp& cast_op_ = CastOp(),
                     const VecOp& vec_op_ = VecOp()) {
        kernel = kernel_.clone();
        anchor = anchor_;
        ksize = kernel.cols();
        cast_op = cast_op_;
        vec_op = vec_op_;
    }

    void operator()(const uchar** src, uchar* dst, int dststep, int count, int width)
    {
        const ST* ky = this->kernel.ptr();
        int i = 0, k;
        CastOp castOp = this->cast_op;
        {
            for( ; count > 0; count--, dst += dststep, src++ )
            {
                DT* D = (DT*)dst;
                i = (this->vec_op)(src, dst, count, width);
#if MEGCV_ENABLE_UNROLLED
                for( ; i <= width - 4; i += 4 )
                {
                    ST f = ky[0];
                    const ST* S = (const ST*)src[0] + i;
                    ST s0 = f*S[0], s1 = f*S[1],
                       s2 = f*S[2], s3 = f*S[3];

                    for( k = 1; k < ksize; k++ )
                    {
                        S = (const ST*)src[k] + i;
                        f = ky[k];
                        s0 += f*S[0];
                        s1 += f*S[1];
                        s2 += f*S[2];
                        s3 += f*S[3];
                    }

                    D[i] = castOp(s0); D[i+1] = castOp(s1);
                    D[i+2] = castOp(s2); D[i+3] = castOp(s3);
                }
#endif
                for( ; i < width; i++ )
                {
                    ST s0 = 0;
                    for( k = 0; k < ksize; k++ ) {
                        s0 += ky[k]* ((const ST*)src[k])[i];
                    }
                    D[i] = castOp(s0);
                }
            }
        }
    }

    Mat<ST> kernel;
    CastOp cast_op;
    VecOp vec_op;
};

template <class CastOp, class VecOp>
struct SymmColumnFilter : public ColumnFilter<CastOp, VecOp> {
    typedef typename CastOp::type1 ST;
    typedef typename CastOp::rtype DT;

    SymmColumnFilter(const Mat<ST>& kernel_, int anchor_,
                     const CastOp& cast_op_ = CastOp(),
                     const VecOp& vec_op_ = VecOp())
            : ColumnFilter<CastOp, VecOp>(kernel_, anchor_, cast_op_,
                                              vec_op_) {
    }

    void operator()(const uchar** src, uchar* dst, int dststep, int count,
                    int width) {
        int ksize2 = this->ksize / 2;
        const ST* ky = this->kernel.ptr() + ksize2;
        int i, k;
        src += ksize2;

        for (; count > 0; count--, dst += dststep, src++) {
            DT* D = (DT*)dst;
            i = (this->vec_op)(src, dst, count, width);
#if MEGCV_ENABLE_UNROLLED
            for (; i <= width - 4; i += 4) {
                ST f = ky[0];
                const ST *S = (const ST*)src[0] + i, *S2;
                ST s0 = f * S[0], s1 = f * S[1], s2 = f * S[2], s3 = f * S[3];

                for (k = 1; k <= ksize2; k++) {
                    S = (const ST*)src[k] + i;
                    S2 = (const ST*)src[-k] + i;
                    f = ky[k];
                    s0 += f * (S[0] + S2[0]);
                    s1 += f * (S[1] + S2[1]);
                    s2 += f * (S[2] + S2[2]);
                    s3 += f * (S[3] + S2[3]);
                }

                D[i] = this->cast_op(s0);
                D[i + 1] = this->cast_op(s1);
                D[i + 2] = this->cast_op(s2);
                D[i + 3] = this->cast_op(s3);
            }
#endif
            for (; i < width; i++) {
                ST s0 = ky[0] * ((const ST*)src[0])[i];
                for (k = 1; k <= ksize2; k++) {
                    s0 += ky[k] *
                          (((const ST*)src[k])[i] + ((const ST*)src[-k])[i]);
                }
                D[i] = this->cast_op(s0);
            }
        }
    }
};

template <class CastOp, class VecOp>
struct SymmColumnSmallFilter : public SymmColumnFilter<CastOp, VecOp> {
    typedef typename CastOp::type1 ST;
    typedef typename CastOp::rtype DT;

    SymmColumnSmallFilter(const Mat<ST>& kernel_, int anchor_,
                          const CastOp& cast_op_ = CastOp(),
                          const VecOp& vec_op_ = VecOp())
            : SymmColumnFilter<CastOp, VecOp>(kernel_, anchor_, cast_op_,
                                              vec_op_) {
        //! \warning Only process if the kernel size is 3
        megdnn_assert(this->ksize == 3);
    }

    void operator()(const uchar** src, uchar* dst, int dststep, int count,
                    int width) {
        int ksize2 = this->ksize / 2;
        const ST* ky = this->kernel.ptr() + ksize2;
        int i;
        ST f0 = ky[0], f1 = ky[1];
        src += ksize2;

        if (std::is_same<ST, int>::value && std::is_same<DT, uchar>::value) {
            (this->vec_op)(src, dst, count, width);
        }

        for (; count > 0; count--, dst += dststep, src++) {
            DT* D = (DT*)dst;
            i = (this->vec_op)(src, dst, count, width);
            if (count == 0)
                break;
            const ST* S0 = (const ST*)src[-1];
            const ST* S1 = (const ST*)src[0];
            const ST* S2 = (const ST*)src[1];

            {
#if MEGCV_ENABLE_UNROLLED
                for (; i <= width - 4; i += 4) {
                    ST s0 = (S0[i] + S2[i]) * f1 + S1[i] * f0;
                    ST s1 = (S0[i + 1] + S2[i + 1]) * f1 + S1[i + 1] * f0;
                    D[i] = this->cast_op(s0);
                    D[i + 1] = this->cast_op(s1);

                    s0 = (S0[i + 2] + S2[i + 2]) * f1 + S1[i + 2] * f0;
                    s1 = (S0[i + 3] + S2[i + 3]) * f1 + S1[i + 3] * f0;
                    D[i + 2] = this->cast_op(s0);
                    D[i + 3] = this->cast_op(s1);
                }
#endif
                for (; i < width; i++) {
                    ST s0 = (S0[i] + S2[i]) * f1 + S1[i] * f0;
                    D[i] = this->cast_op(s0);
                }
            }
        }
    }
};

/* ============================ Filter Engine ========================= */

/*!
 * \brief The common class for filtering the image. First filter the image using
 *     row filter and store in buffer data, and then using column filter.
 * \tparam ST The image data type
 * \tparam FT The inner buffer data type.
 *
 * \note As for uint8_t type, we may use int to store the buffer, which calc the
 *     product of the image and the filter kernel.
 */
template <typename ST, typename FT>
class FilterEngine {
public:
    FilterEngine() = default;
    /*!
     * \brief Init the filter and border.
     * \warning row_filter and column_filter must be non-null
     */
    FilterEngine(BaseRowFilter* row_filter, BaseColumnFilter* column_filter,
                 size_t ch, const ST* border_value, BorderMode bmode);

    //! the destructor
    ~FilterEngine();
    //! applies filter to the the whole image.
    void apply(const Mat<ST>& src, Mat<ST>& dst);

private:
    //! starts filtering of the src image.
    void start(const Mat<ST>& src);
    //! processes the next srcCount rows of the image.
    int proceed(const uchar* src, int srcStep, int srcCount, uchar* dst,
                        int dstStep);

    //! row filter filter
    BaseRowFilter* m_row_filter;
    //! column filter filter
    BaseColumnFilter* m_column_filter;
    //! the channel of the image
    size_t m_ch;
    BorderMode m_bmode;

    //! the size of the kernel
    Size m_ksize;

    //! the center of kernel, e.g GuassianBlur m_anchor is (kernel_row/2,
    //! kernel_column/2)
    Point<size_t> m_anchor;

    //! the whole size.
    Size m_whole_size;
    //! store the border value, if sizeof(src_type) >= 4,
    std::vector<int> m_border_table;
    //! nr of border value
    int m_border_elem_size;

    //! the step of the buffer data.
    int m_buf_step;

    //! store the border value, The size is ksize.cols - 1
    std::vector<uchar> m_const_border_value;
    //! store the total row if the border is BORDER_CONSTANT, the size is
    //! image_width + kernel_width - 1, which include the row and the border.
    std::vector<uchar> m_const_border_row;
    //! store the total row if the border is not BORDER_CONSTANT
    std::vector<uchar> m_src_row;

    //! store the kernel_height rows data.
    std::vector<uchar> m_ring_buf;

    //! the border left width, equal to m_anchor.x
    int m_left_width;
    //! equal to m_ksize.width() - m_left_width - 1
    int m_right_width;
};

}  // namespace filter_common
}  // namespace megcv
}  // namespace megdnn

// vim: filetype=cpp.doxygen
