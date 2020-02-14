/**
 * \file dnn/src/fallback/rotate/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include <cstring>

#include "src/fallback/handle.h"
#include "src/fallback/rotate/opr_impl.h"

#include "src/common/cv/common.h"
#include "src/common/cv/helper.h"
#include "src/common/utils.h"

#include "midout.h"

MIDOUT_DECL(megdnn_fb_rotate)

using namespace megdnn;
using namespace fallback;

namespace rotate_intl {
using namespace megcv;

template <typename T, size_t CH, bool clockwise>
static void rotate_fallback_tpl(const T* src, T* dst, size_t src_rows,
                                size_t src_cols, size_t src_step,
                                size_t dst_step) {
    size_t sr = 0;
    static const size_t BLOCK = 4;
    auto do_pixel = [&](size_t sr, size_t sc) {
        size_t dr, dc;
        size_t M = src_rows;
        size_t N = src_cols;
        if (clockwise) {
            dr = sc;
            dc = M - 1 - sr;
        } else {
            dr = N - 1 - sc;
            dc = sr;
        }
        for (size_t ch = 0; ch < CH; ++ch) {
            dst[dr * dst_step + dc * CH + ch] =
                    src[sr * src_step + sc * CH + ch];
        }
    };

    for (; sr + BLOCK <= src_rows; sr += BLOCK) {
        size_t sc = 0;
        for (; sc + BLOCK <= src_cols; sc += BLOCK) {
            // block
            for (size_t sr2 = sr; sr2 < sr + BLOCK; ++sr2)
                for (size_t sc2 = sc; sc2 < sc + BLOCK; ++sc2) {
                    do_pixel(sr2, sc2);
                }
        }
        for (; sc < src_cols; ++sc) {
            for (size_t sr2 = sr; sr2 < sr + BLOCK; ++sr2) {
                do_pixel(sr2, sc);
            }
        }
    }
    for (; sr < src_rows; ++sr) {
        for (size_t sc = 0; sc < src_cols; ++sc) {
            do_pixel(sr, sc);
        }
    }
}

template <typename T>
static void rotate_fallback(const Mat<T>& src, Mat<T>& dst, bool clockwise) {
    size_t CH = src.channels();
#define cb(_ch, _clockwise)                                                   \
    if (CH == _ch && clockwise == _clockwise) {                               \
        MIDOUT_BEGIN(megdnn_fb_rotate, T, midout_iv(_ch),                     \
                     midout_iv(_clockwise)) {                                 \
            return rotate_fallback_tpl<T, _ch, _clockwise>(                   \
                    src.ptr(), dst.ptr(), src.rows(), src.cols(), src.step(), \
                    dst.step());                                              \
        }                                                                     \
        MIDOUT_END();                                                         \
    }

    cb(1, true);
    cb(1, false);
    cb(3, true);
    cb(3, false);
#undef cb
    MegCVException("Unsupported channel size, only support 1 and 3");
}

template <typename T>
void rotate(const Mat<T>& src, Mat<T>& dst, bool clockwise) {
    megdnn_assert(src.rows() == dst.cols());
    megdnn_assert(src.cols() == dst.rows());
    megdnn_assert(src.channels() == dst.channels());
    megdnn_assert(src.channels() == 1 || src.channels() == 3);

    rotate_fallback<T>(src, dst, clockwise);
}

}  // namespace rotate_intl

void RotateImpl::exec(_megdnn_tensor_in src, _megdnn_tensor_in dst,
                      _megdnn_workspace workspace) {
    using namespace megcv;
    check_exec(src.layout, dst.layout, workspace.size);

    MEGDNN_DISPATCH_CPU_KERN_OPR(if (dst.layout.dtype == dtype::Float32()) {
        for (size_t i = 0; i < src.layout.shape[0]; ++i) {
            Mat<float> src_mat = TensorND2Mat<float>(src, i);
            Mat<float> dst_mat = TensorND2Mat<float>(dst, i);
            rotate_intl::rotate<float>(src_mat, dst_mat, param().clockwise);
        }
    } else if (dst.layout.dtype == dtype::Int32()) {
        for (size_t i = 0; i < src.layout.shape[0]; ++i) {
            Mat<int> src_mat = TensorND2Mat<int>(src, i);
            Mat<int> dst_mat = TensorND2Mat<int>(dst, i);
            rotate_intl::rotate<int>(src_mat, dst_mat, param().clockwise);
        }
    } else if (dst.layout.dtype == dtype::Uint8()) {
        for (size_t i = 0; i < src.layout.shape[0]; ++i) {
            Mat<uchar> src_mat = TensorND2Mat<uchar>(src, i);
            Mat<uchar> dst_mat = TensorND2Mat<uchar>(dst, i);
            rotate_intl::rotate<uchar>(src_mat, dst_mat, param().clockwise);
        }
    } else { megdnn_throw("Unsupported datatype of Rotate optr."); });
}

// vim: syntax=cpp.doxygen
