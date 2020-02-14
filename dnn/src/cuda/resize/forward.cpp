/**
 * \file dnn/src/cuda/resize/forward.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/common/cv/common.h"
#include "src/cuda/handle.h"
#include "src/cuda/resize/common.h"
#include "src/cuda/resize/helper.h"
#include "src/cuda/resize/opr_impl.h"
#include "src/cuda/resize/resize_cv.cuh"
#include "src/cuda/utils.h"

using namespace megdnn;
using namespace cuda;

namespace {

void resize_cv_proxy(_megdnn_tensor_in src, _megdnn_tensor_out dst,
                     InterpolationMode imode, void* workspace,
                     cudaStream_t stream) {
    using namespace megcv;
    for (size_t i = 0; i < src.layout.shape[0]; ++i) {
        if (dst.layout.dtype == dtype::Float32()) {
            Mat<float> src_mat = TensorND2Mat<float>(src, i);
            Mat<float> dst_mat = TensorND2Mat<float>(dst, i);
            resize::resize_cv<float>(
                    src_mat.ptr(), dst_mat.ptr(), src_mat.rows(),
                    src_mat.cols(), dst_mat.rows(), dst_mat.cols(),
                    src_mat.step(), dst_mat.step(), src_mat.channels(), imode,
                    workspace, stream);
        } else if (dst.layout.dtype == dtype::Uint8()) {
            Mat<uchar> src_mat = TensorND2Mat<uchar>(src, i);
            Mat<uchar> dst_mat = TensorND2Mat<uchar>(dst, i);
            resize::resize_cv<uchar>(
                    src_mat.ptr(), dst_mat.ptr(), src_mat.rows(),
                    src_mat.cols(), dst_mat.rows(), dst_mat.cols(),
                    src_mat.step(), dst_mat.step(), src_mat.channels(), imode,
                    workspace, stream);
        } else {
            megdnn_throw(
                    megdnn_mangle("Unsupported datatype of WarpAffine optr."));
        }
    }
}

}  // anonymous namespace

size_t ResizeImpl::get_workspace_in_bytes(const TensorLayout& src,
                                          const TensorLayout& dst) {
    InterpolationMode imode = param().imode;
    if (param().format == Param::Format::NCHW ||
        (imode != Param::InterpolationMode::CUBIC &&
         imode != Param::InterpolationMode::LANCZOS4)) {
        return 0;
    }

    size_t src_rows = src.shape[1];
    size_t dst_rows = dst.shape[1];
    size_t src_cols = src.shape[2];
    size_t dst_cols = dst.shape[2];
    size_t ch = src.shape[3];

    size_t dst_area_size = dst_rows * dst_cols;
    size_t src_area_size = src_rows * src_cols;

    bool enlarge = dst_area_size > src_area_size;
    bool shrink = dst_area_size <= src_area_size;
    bool U8 = src.dtype == dtype::Uint8();
    megdnn_assert(src.dtype == dtype::Uint8() || src.dtype == dtype::Float32());
    bool F32_1 = !U8 && ch == 1;
    bool F32_3 = !U8 && ch == 3;

    bool use_vector = (enlarge && (dst_area_size <= 500 * 500)) ||
                      (shrink && (F32_3 || (U8 && dst_area_size <= 500 * 500) ||
                                  (F32_1 && dst_area_size <= 1000 * 1000)));

    if (!use_vector) {
        int coef_size = 0;
        if (imode == Param::InterpolationMode::CUBIC) {
            coef_size = 4;
        } else {
            coef_size = 8;
            megdnn_assert(imode == Param::InterpolationMode::LANCZOS4);
        }
        if (U8) {
            return dst_rows * coef_size * sizeof(short) +  //! dev_coef_row
                   dst_rows * sizeof(int) +                //! dev_sr
                   dst_cols * coef_size * sizeof(short) +  //! dev_coef_col
                   dst_cols * sizeof(int);                 //! dev_sc
        } else {
            return dst_rows * coef_size * sizeof(float) +  //! dev_coef_row
                   dst_rows * sizeof(int) +                //! dev_sr
                   dst_cols * coef_size * sizeof(float) +  //! dev_coef_col
                   dst_cols * sizeof(int);                 //! dev_sc
        }
    }

    return 0;
}

void ResizeImpl::exec(_megdnn_tensor_in src, _megdnn_tensor_in dst,
                      _megdnn_workspace workspace) {
    check_exec(src.layout, dst.layout, workspace.size);
    auto stream = cuda_stream(this->handle());
    bool is_nhwc = param().format == param::Resize::Format::NHWC;
    size_t C, IH, IW, OH, OW;
    ptrdiff_t S_IN = 0, S_IC = 0, S_IH = 0, S_IW = 0;
    if (is_nhwc) {
        if (param().imode != Param::InterpolationMode::LINEAR &&
            is_nhwc_contig_wc(src.layout)) {
            resize_cv_proxy(src, dst, resize::get_imode(param().imode),
                            workspace.raw_ptr, stream);
            return;
        }
        C = src.layout.shape[3];
        IH = src.layout.shape[1];
        IW = src.layout.shape[2];
        OH = dst.layout.shape[1];
        OW = dst.layout.shape[2];
    } else if (param().format == param::Resize::Format::NCHW) {
        C = src.layout.shape[1];
        IH = src.layout.shape[2];
        IW = src.layout.shape[3];
        OH = dst.layout.shape[2];
        OW = dst.layout.shape[3];
        S_IN = src.layout.stride[0];
        S_IC = src.layout.stride[1];
        S_IH = src.layout.stride[2];
        S_IW = src.layout.stride[3];
    } else {
        megdnn_assert(param().format == param::Resize::Format::NCHW4,
                      "invalid resize format");
        megdnn_assert(src.layout.dtype.enumv() == DTypeEnum::QuantizedS8);
        C = src.layout.shape[1] * 4;
        IH = src.layout.shape[2];
        IW = src.layout.shape[3];
        OH = dst.layout.shape[2];
        OW = dst.layout.shape[3];
        resize::forward_proxy_nchw4(src.compatible_ptr<int8_t>(),
                                    dst.compatible_ptr<int8_t>(), src.layout[0],
                                    C, IH, IW, OH, OW, stream);
        return;
    }
    megdnn_assert(param().imode == Param::InterpolationMode::LINEAR,
                  "unsupported interpolation mode for NCHW format");

    if (src.layout.dtype == dtype::Float32{}) {
        resize::forward_proxy(is_nhwc, src.ptr<dt_float32>(),
                              dst.ptr<dt_float32>(), src.layout[0], C, IH, IW,
                              OH, OW, S_IN, S_IC, S_IH, S_IW, stream);
    } else if (src.layout.dtype == dtype::Uint8()) {
        resize::forward_proxy(is_nhwc, src.ptr<dt_uint8>(), dst.ptr<dt_uint8>(),
                              src.layout[0], C, IH, IW, OH, OW, S_IN, S_IC,
                              S_IH, S_IW, stream);
    } else if (src.layout.dtype == dtype::Int8()) {
        resize::forward_proxy(is_nhwc, src.ptr<dt_int8>(), dst.ptr<dt_int8>(),
                              src.layout[0], C, IH, IW, OH, OW, S_IN, S_IC,
                              S_IH, S_IW, stream);
    } else {
        megdnn_throw(
                ssprintf("unsupported dtype: %s", src.layout.dtype.name()));
    }
}

// vim: syntax=cpp.doxygen
