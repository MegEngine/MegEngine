/**
 * \file dnn/src/cuda/warp_perspective/forward.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/cuda/warp_perspective/opr_impl.h"
#include "src/cuda/warp_perspective/warp_perspective_cv.cuh"

#include "src/cuda/utils.h"
#include "src/cuda/warp_perspective/common.h"
#include "src/cuda/warp_perspective/helper.h"

#include "src/common/cv/common.h"
#include "src/common/warp_common.h"

namespace megdnn {
namespace cuda {

namespace warp_perspective {

void warp_perspective_cv_exec(_megdnn_tensor_in src, _megdnn_tensor_in mat,
                              _megdnn_tensor_in dst, float border_val,
                              BorderMode bmode, InterpolationMode imode,
                              _megdnn_workspace workspace,
                              cudaStream_t stream) {
    megdnn_assert(src.layout[3] == 1 || src.layout[3] == 3,
                  "unsupported src channel");
    megdnn_assert(src.layout.dtype != dtype::Float32() ||
                          src.layout.dtype != dtype::Uint8(),
                  "unsupported src dtype");
    if (imode == InterpolationMode::INTER_AREA) {
        imode = InterpolationMode::INTER_LINEAR;
    }
    using namespace megcv;
    const float* trans_ptr = mat.ptr<dt_float32>();
    double* workspace_ptr = workspace.ptr<double>();
    for (size_t i = 0; i < src.layout.shape[0]; ++i) {
        if (dst.layout.dtype == dtype::Float32()) {
            Mat<float> src_mat = TensorND2Mat<float>(src, i);
            Mat<float> dst_mat = TensorND2Mat<float>(dst, i);
            if (src_mat.channels() == 1) {
                warp_perspective_cv_proxy<float, 1>(
                        src_mat.ptr(), dst_mat.ptr(), src_mat.rows(),
                        src_mat.cols(), dst_mat.rows(), dst_mat.cols(),
                        src_mat.step(), dst_mat.step(), bmode, imode, trans_ptr,
                        border_val, workspace_ptr, stream);
            } else {
                warp_perspective_cv_proxy<float, 3>(
                        src_mat.ptr(), dst_mat.ptr(), src_mat.rows(),
                        src_mat.cols(), dst_mat.rows(), dst_mat.cols(),
                        src_mat.step(), dst_mat.step(), bmode, imode, trans_ptr,
                        border_val, workspace_ptr, stream);
            }
        } else if (dst.layout.dtype == dtype::Uint8()) {
            Mat<uchar> src_mat = TensorND2Mat<uchar>(src, i);
            Mat<uchar> dst_mat = TensorND2Mat<uchar>(dst, i);
            if (src_mat.channels() == 1) {
                warp_perspective_cv_proxy<uchar, 1>(
                        src_mat.ptr(), dst_mat.ptr(), src_mat.rows(),
                        src_mat.cols(), dst_mat.rows(), dst_mat.cols(),
                        src_mat.step(), dst_mat.step(), bmode, imode, trans_ptr,
                        static_cast<uchar>(border_val), workspace_ptr, stream);
            } else {
                warp_perspective_cv_proxy<uchar, 3>(
                        src_mat.ptr(), dst_mat.ptr(), src_mat.rows(),
                        src_mat.cols(), dst_mat.rows(), dst_mat.cols(),
                        src_mat.step(), dst_mat.step(), bmode, imode, trans_ptr,
                        static_cast<uchar>(border_val), workspace_ptr, stream);
            }

        } else {
            megdnn_throw(megdnn_mangle(
                    "Unsupported datatype of WarpPerspective optr."));
        }

        trans_ptr += 3 * 3;
        workspace_ptr += 3 * 3;
    }
}

}  // namespace warp_perspective

WorkspaceBundle WarpPerspectiveForwardImpl::get_workspace_bundle(
        void* ptr, const TensorLayout& src, const TensorLayout& mat,
        const TensorLayout& mat_idx, const TensorLayout& dst) const {
    MEGDNN_MARK_USED_VAR(mat_idx);
    SmallVector<size_t> sizes;
    TensorLayout fsrc = src;
    TensorLayout fmat = mat;
    TensorLayout fdst = dst;
    auto get_workspace = [&sizes](TensorLayout& layout) {
        if (layout.dtype == dtype::BFloat16()) {
            layout.dtype = dtype::Float32();
            sizes.push_back(layout.span().dist_byte());
        }
    };
    get_workspace(fsrc);
    get_workspace(fmat);
    get_workspace(fdst);
    if (param().format == param::WarpPerspective::Format::NHWC) {
        //! use double for the workspace dtype as float may cause
        //! accuracy problems
        sizes.push_back(mat.total_nr_elems() * sizeof(double));
    }

    return {ptr, std::move(sizes)};
}

void WarpPerspectiveForwardImpl::exec(_megdnn_tensor_in ssrc,
                                      _megdnn_tensor_in smat,
                                      _megdnn_tensor_in smat_idx,
                                      _megdnn_tensor_out sdst,
                                      _megdnn_workspace sworkspace) {
    check_exec_allow_nhwc_mat_idx(ssrc.layout, smat.layout, smat_idx.layout,
                                  sdst.layout, sworkspace.size);

    TensorND src = ssrc;
    TensorND mat = smat;
    TensorND mat_idx = smat_idx;
    TensorND dst = sdst;
    auto bundle =
            get_workspace_bundle(sworkspace.raw_ptr, ssrc.layout, smat.layout,
                                 smat_idx.layout, sdst.layout);
    auto ctypecvt = CompTypeCvter<dtype::BFloat16, dtype::Float32>(
            concrete_handle(this->handle()), &bundle);
    if (ssrc.layout.dtype.enumv() == DTypeTrait<dtype::BFloat16>::enumv) {
        ctypecvt.src_to_comp_type(ssrc, src)
                .src_to_comp_type(smat, mat)
                .src_to_comp_type(sdst, dst);
    }

    {
        auto stream = cuda_stream(this->handle());
        bool is_nhwc = param().format == param::WarpPerspective::Format::NHWC;

        if (is_nhwc && param().imode != Param::InterpolationMode::LINEAR) {
            // use opencv impl only for nhwc and non-linear interp
            megdnn_assert(!mat_idx.raw_ptr,
                          "mat_idx is not supported in NHWC case with "
                          "non-linear interpolation");
            warp_perspective::warp_perspective_cv_exec(
                    src, mat, dst, param().border_val,
                    warp_perspective::get_bmode(param().bmode),
                    warp_perspective::get_imode(param().imode),
                    ctypecvt.workspace(), stream);

        } else {
            megdnn_assert(warp::is_dnn_available(src.layout, mat.layout,
                                                 dst.layout, param().imode,
                                                 param().format));
            size_t C, IH, IW, OH, OW;
            if (is_nhwc) {
                C = src.layout.shape[3];
                IH = src.layout.shape[1];
                IW = src.layout.shape[2];
                OH = dst.layout.shape[1];
                OW = dst.layout.shape[2];
            } else if (param().format == Param::Format::NCHW4) {
                C = src.layout.shape[1] * 4;
                IH = src.layout.shape[2];
                IW = src.layout.shape[3];
                OH = dst.layout.shape[2];
                OW = dst.layout.shape[3];
            } else {
                megdnn_assert(
                        param().format == param::WarpPerspective::Format::NCHW,
                        "invalid warp_perspective format");
                C = src.layout.shape[1];
                IH = src.layout.shape[2];
                IW = src.layout.shape[3];
                OH = dst.layout.shape[2];
                OW = dst.layout.shape[3];
            }
            megdnn_assert(param().imode == Param::InterpolationMode::LINEAR,
                          "unsupported interpolation mode for NCHW format");
            auto bval = param().border_val;
            auto bmode = warp_perspective::get_bmode(param().bmode);

            if (src.layout.dtype == dtype::Float32{}) {
                warp_perspective::forward_proxy(
                        is_nhwc, src.ptr<dt_float32>(), mat.ptr<dt_float32>(),
                        mat_idx.raw_ptr ? mat_idx.ptr<int>() : nullptr,
                        dst.ptr<dt_float32>(), src.layout[0], mat.layout[0], C,
                        IH, IW, OH, OW, bval, bmode, async_error_info(handle()),
                        m_error_tracker, stream);
            } else if (MEGDNN_FLOAT16_SELECT(
                               src.layout.dtype == dtype::Float16(), false)) {
#ifndef MEGDNN_DISABLE_FLOAT16
                warp_perspective::forward_proxy(
                        is_nhwc, src.ptr<dt_float16>(), mat.ptr<dt_float32>(),
                        mat_idx.raw_ptr ? mat_idx.ptr<int>() : nullptr,
                        dst.ptr<dt_float16>(), src.layout[0], mat.layout[0], C,
                        IH, IW, OH, OW, static_cast<dt_float16>(bval), bmode,
                        async_error_info(handle()), m_error_tracker, stream);
#endif
            } else if (src.layout.dtype == dtype::Uint8()) {
                warp_perspective::forward_proxy<dt_uint8>(
                        is_nhwc, src.ptr<dt_uint8>(), mat.ptr<dt_float32>(),
                        mat_idx.raw_ptr ? mat_idx.ptr<int>() : nullptr,
                        dst.ptr<dt_uint8>(), src.layout[0], mat.layout[0], C,
                        IH, IW, OH, OW, bval, bmode, async_error_info(handle()),
                        m_error_tracker, stream);
            } else if (src.layout.dtype == dtype::Int8()) {
                megdnn_assert(
                        !is_nhwc,
                        "WarpPerspective on CUDA does not support NHWC + Int8");
                warp_perspective::forward_proxy<dt_int8>(
                        false, src.ptr<dt_int8>(), mat.ptr<dt_float32>(),
                        mat_idx.raw_ptr ? mat_idx.ptr<int>() : nullptr,
                        dst.ptr<dt_int8>(), src.layout[0], mat.layout[0], C, IH,
                        IW, OH, OW,
                        bval /* implicit float -> int8 conversion, should be
                                safe */
                        ,
                        bmode, async_error_info(handle()), m_error_tracker,
                        stream);
            } else if (src.layout.dtype.enumv() == DTypeEnum::QuantizedS8) {
                megdnn_assert(param().format == Param::Format::NCHW4,
                              "WarpPerspective on CUDA supports NCHW4 + "
                              "QuantizedS8 only");
                warp_perspective::forward_proxy_nchw4<dt_int8>(
                        src.compatible_ptr<dt_int8>(), mat.ptr<dt_float32>(),
                        mat_idx.raw_ptr ? mat_idx.ptr<int>() : nullptr,
                        dst.compatible_ptr<dt_int8>(), src.layout[0],
                        mat.layout[0], C, IH, IW, OH, OW, bval, bmode,
                        async_error_info(handle()), m_error_tracker, stream);
            } else {
                megdnn_throw(ssprintf("unsupported dtype: %s",
                                      src.layout.dtype.name()));
            }
        }
    }
    if (ssrc.layout.dtype.enumv() == DTypeTrait<dtype::BFloat16>::enumv) {
        ctypecvt.comp_to_dst_type(dst, sdst);
    }
}

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
