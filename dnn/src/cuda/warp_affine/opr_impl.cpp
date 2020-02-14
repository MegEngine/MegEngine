/**
 * \file dnn/src/cuda/warp_affine/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/cuda/warp_affine/opr_impl.h"
#include "src/cuda/warp_affine/warp_affine_cv.cuh"
#include "src/cuda/warp_affine/helper.h"
#include "src/cuda/warp_affine/common.h"
#include "src/cuda/handle.h"
#include "src/cuda/utils.h"

#include "src/common/cv/common.h"
#include "src/common/cv/helper.h"
#include "src/common/cv/enums.h"
#include "src/common/utils.h"

#include <cstring>

namespace megdnn {
namespace cuda {

namespace warp_affine {

void warp_affine_cv_exec(_megdnn_tensor_in src, _megdnn_tensor_in mat,
                         _megdnn_tensor_in dst, float border_val,
                         BorderMode bmode, InterpolationMode imode,
                         _megdnn_workspace workspace, cudaStream_t stream) {
    using namespace megcv;
    megdnn_assert(src.layout[3] == 1 || src.layout[3] == 3,
            "unsupported src channel");
    using namespace megcv;
    const float* trans_ptr = mat.ptr<dt_float32>();
    double *workspace_ptr = workspace.ptr<double>();
    for (size_t i = 0; i < src.layout.shape[0]; ++i) {
        if (dst.layout.dtype == dtype::Float32()) {
            Mat<float> src_mat = TensorND2Mat<float>(src, i);
            Mat<float> dst_mat = TensorND2Mat<float>(dst, i);
            if (src_mat.channels() == 1) {
                warp_affine_cv_proxy<float, 1>(
                    src_mat.ptr(), dst_mat.ptr(), src_mat.rows(),
                    src_mat.cols(), dst_mat.rows(), dst_mat.cols(),
                    src_mat.step(), dst_mat.step(), bmode, imode,
                    trans_ptr, border_val, workspace_ptr, stream);
            } else {
                warp_affine_cv_proxy<float, 3>(
                    src_mat.ptr(), dst_mat.ptr(), src_mat.rows(),
                    src_mat.cols(), dst_mat.rows(), dst_mat.cols(),
                    src_mat.step(), dst_mat.step(), bmode, imode,
                    trans_ptr, border_val, workspace_ptr, stream);
            }
        } else if (dst.layout.dtype == dtype::Uint8()) {
            Mat<uchar> src_mat = TensorND2Mat<uchar>(src, i);
            Mat<uchar> dst_mat = TensorND2Mat<uchar>(dst, i);
            if (src_mat.channels() == 1) {
                warp_affine_cv_proxy<uchar, 1>(
                    src_mat.ptr(), dst_mat.ptr(), src_mat.rows(),
                    src_mat.cols(), dst_mat.rows(), dst_mat.cols(),
                    src_mat.step(), dst_mat.step(), bmode, imode, trans_ptr,
                    static_cast<uchar>(border_val), workspace_ptr, stream);
            } else {
                warp_affine_cv_proxy<uchar, 3>(
                    src_mat.ptr(), dst_mat.ptr(), src_mat.rows(),
                    src_mat.cols(), dst_mat.rows(), dst_mat.cols(),
                    src_mat.step(), dst_mat.step(), bmode, imode, trans_ptr,
                    static_cast<uchar>(border_val), workspace_ptr, stream);
            }

        } else {
            megdnn_throw(
                megdnn_mangle("Unsupported datatype of Warpaffine optr."));
        }

        trans_ptr += 2 * 3;
        workspace_ptr += 2 * 3;
    }
}

} // warp_affine


void WarpAffineImpl::exec(_megdnn_tensor_in src, _megdnn_tensor_in mat,
                          _megdnn_tensor_out dst, _megdnn_workspace workspace) {
    using namespace megcv;
    check_exec(src.layout, mat.layout, dst.layout, workspace.size);
    auto stream = cuda_stream(this->handle());
    bool is_nhwc = param().format == param::WarpAffine::Format::NHWC;
    size_t C, IH, IW, OH, OW;
    if (is_nhwc) {
        if (param().imode != Param::InterpolationMode::LINEAR) {
            warp_affine::warp_affine_cv_exec(
                    src, mat, dst, param().border_val,
                    warp_affine::get_bmode(param().border_mode),
                    warp_affine::get_imode(param().imode), workspace, stream);
            return;
        }
        C = src.layout.shape[3];
        IH = src.layout.shape[1];
        IW = src.layout.shape[2];
        OH = dst.layout.shape[1];
        OW = dst.layout.shape[2];
    } else {
        megdnn_assert(param().format == param::WarpAffine::Format::NCHW,
                "invalid warp_affine format");
        C = src.layout.shape[1];
        IH = src.layout.shape[2];
        IW = src.layout.shape[3];
        OH = dst.layout.shape[2];
        OW = dst.layout.shape[3];
    }
    megdnn_assert(param().imode == Param::InterpolationMode::LINEAR,
            "unsupported interpolation mode for NCHW format");
    auto bval = param().border_val;
    auto bmode = warp_affine::get_bmode(param().border_mode);

    if (src.layout.dtype == dtype::Float32{}) {
        warp_affine::forward_proxy(is_nhwc, src.ptr<dt_float32>(),
                                   mat.ptr<dt_float32>(), dst.ptr<dt_float32>(),
                                   src.layout[0], C, IH, IW, OH, OW, bval,
                                   bmode, stream);
    } else if (src.layout.dtype == dtype::Uint8()) {
        warp_affine::forward_proxy<dt_uint8>(
                is_nhwc, src.ptr<dt_uint8>(), mat.ptr<dt_float32>(),
                dst.ptr<dt_uint8>(), src.layout[0], C, IH, IW, OH, OW, bval,
                bmode, stream);
    } else if (src.layout.dtype == dtype::Int8()) {
        megdnn_assert(!is_nhwc,
                      "WarpPerspective on CUDA does not support NHWC + Int8");
        warp_affine::forward_proxy<dt_int8>(
                is_nhwc, src.ptr<dt_int8>(), mat.ptr<dt_float32>(),
                dst.ptr<dt_int8>(), src.layout[0], C, IH, IW, OH, OW, bval,
                bmode, stream);

    } else {
        megdnn_throw(
                ssprintf("unsupported dtype: %s", src.layout.dtype.name()));
    }
}

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
