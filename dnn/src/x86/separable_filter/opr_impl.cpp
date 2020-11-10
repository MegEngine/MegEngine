/**
 * \file dnn/src/x86/separable_filter/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/x86/separable_filter/opr_impl.h"
#include "src/x86/separable_filter/filter.h"
#include "src/common/cv/common.h"
#include "src/common/cv/helper.h"
#include "src/common/utils.h"
#include "src/x86/utils.h"
#include "src/x86/handle.h"
#include <cstring>

namespace megdnn {
namespace x86 {
using namespace megcv;
using namespace sep_filter;
using BorderMode = param::SeparableFilter::BorderMode;

void SeparableFilterImpl::separable_filter_exec_8u(_megdnn_tensor_in src,
                                                   _megdnn_tensor_in filter_x,
                                                   _megdnn_tensor_in filter_y,
                                                   _megdnn_tensor_out dst) {
    megdnn_assert(src.layout.dtype == dtype::Uint8());

    Mat<float> kernel_column(1, filter_y.layout.shape[3], 1,
                             static_cast<float*>(filter_y.raw_ptr));
    Mat<float> kernel_row(1, filter_x.layout.shape[3], 1,
                          static_cast<float*>(filter_x.raw_ptr));

    size_t src_channels = src.layout.shape[3];

    constexpr uint8_t bits = 8;
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
    BaseRowFilter* rowFilter = nullptr;
    BaseColumnFilter* columnFilter = nullptr;
    if (param().is_symm_kernel) {
        rowFilter = getLinearRowFilter<uchar, int, true>(kernel_row_int);
        columnFilter = getLinearColumnFilter<int, uchar, true>(
                kernel_column_int, bits * 2);
    } else {
        rowFilter = getLinearRowFilter<uchar, int, false>(kernel_row_int);
        columnFilter = getLinearColumnFilter<int, uchar, false>(
                kernel_column_int, bits * 2);
    }

    FilterEngine<uchar, int> filter(rowFilter, columnFilter, src_channels,
                                    border_value, param().borderMode);

    megdnn_assert(param().borderMode != BorderMode::BORDER_ISOLATED);
    for (size_t i = 0; i < src.layout.shape[0]; ++i) {
        Mat<uchar> src_mat = TensorND2Mat<uchar>(src, i);
        Mat<uchar> dst_mat = TensorND2Mat<uchar>(dst, i);

        filter.apply(src_mat, dst_mat);
    }
}

template <typename T>
void SeparableFilterImpl::separable_filter_exec(_megdnn_tensor_in src,
                                                _megdnn_tensor_in filter_x,
                                                _megdnn_tensor_in filter_y,
                                                _megdnn_tensor_out dst) {
    Mat<T> kernel_column(1, filter_y.layout.shape[3], 1,
                         static_cast<T*>(filter_y.raw_ptr));
    Mat<T> kernel_row(1, filter_x.layout.shape[3], 1,
                      static_cast<T*>(filter_x.raw_ptr));
    size_t src_channels = src.layout.shape[3];

    T border_value[4] = {0, 0, 0, 0};

    BaseRowFilter* row_filter = nullptr;
    BaseColumnFilter* column_filter = nullptr;
    if (param().is_symm_kernel) {
        row_filter = getLinearRowFilter<T, T, true>(kernel_row);
        column_filter =
                getLinearColumnFilter<T, T, true>(kernel_column, (int)0);
    } else {
        row_filter = getLinearRowFilter<T, T, false>(kernel_row);
        column_filter =
                getLinearColumnFilter<T, T, false>(kernel_column, (int)0);
    }

    FilterEngine<T, T> filter(row_filter, column_filter, src_channels,
                              border_value, param().borderMode);

    megdnn_assert(param().borderMode != BorderMode::BORDER_ISOLATED);
    for (size_t i = 0; i < src.layout.shape[0]; ++i) {
        Mat<T> src_mat = TensorND2Mat<T>(src, i);
        Mat<T> dst_mat = TensorND2Mat<T>(dst, i);
        filter.apply(src_mat, dst_mat);
    }
}

void SeparableFilterImpl::exec(_megdnn_tensor_in src,
                               _megdnn_tensor_in filter_x,
                               _megdnn_tensor_in filter_y,
                               _megdnn_tensor_out dst,
                               _megdnn_workspace workspace) {
    check_exec(src.layout, filter_x.layout, filter_y.layout, dst.layout,
               workspace.size);
    if (dst.layout.dtype == dtype::Float32()) {
        MEGDNN_DISPATCH_CPU_KERN_OPR(
                separable_filter_exec<float>(src, filter_x, filter_y, dst));
    } else if (dst.layout.dtype == dtype::Uint8()) {
        MEGDNN_DISPATCH_CPU_KERN_OPR(
                separable_filter_exec_8u(src, filter_x, filter_y, dst));
    } else {
        megdnn_throw(
                megdnn_mangle("Unsupported datatype of SeparableFilter opr."));
    };
}

} // namespace x86
} // namespace megdnn

// vim: syntax=cpp.doxygen
