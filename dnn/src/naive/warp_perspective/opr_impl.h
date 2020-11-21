/**
 * \file dnn/src/naive/warp_perspective/opr_impl.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#pragma once
#include "megdnn/oprs.h"
#include "src/common/utils.h"

namespace megdnn {
namespace naive {

class WarpPerspectiveForwardImpl : public WarpPerspectiveForward {
protected:
    using Format = Param::Format;
    template <typename ctype, typename mtype>
    struct KernParam {
        Format format;
        BorderMode bmode;
        float border_val;
        size_t n_src, n_mat, c, ih, iw, oh, ow;
        ctype *sptr, *dptr;
        DType src_dtype, dst_dtype;
        mtype* mptr;
        int* midx_ptr;  //!< can be null
        Workspace workspace;

        static KernParam from_tensors(Format format, BorderMode bmode,
                                      float border_val, _megdnn_tensor_in src,
                                      _megdnn_tensor_in mat,
                                      _megdnn_tensor_in mat_idx,
                                      _megdnn_tensor_out dst,
                                      _megdnn_workspace workspace) {
            KernParam ret;
            ret.format = format;
            ret.bmode = bmode;
            ret.border_val = border_val;
            ret.n_src = src.layout.shape[0];
            ret.src_dtype = src.layout.dtype;
            ret.dst_dtype = dst.layout.dtype;
            if (mat_idx.raw_ptr) {
                megdnn_assert(mat_idx.layout.ndim == 1);
                ret.n_mat = mat_idx.layout.shape[0];
                ret.midx_ptr = mat_idx.ptr<int>();
            } else {
                megdnn_assert(mat_idx.layout.ndim == 0);
                ret.n_mat = ret.n_src;
                ret.midx_ptr = nullptr;
            }
            if (format == Format::NCHW ||
                format == Format::NCHW_NCHW4_IC_SMALL) {
                ret.c = src.layout.shape[1];
                ret.ih = src.layout.shape[2];
                ret.iw = src.layout.shape[3];
                ret.oh = dst.layout.shape[2];
                ret.ow = dst.layout.shape[3];
            } else if (format == Format::NHWC) {
                ret.c = src.layout.shape[3];
                ret.ih = src.layout.shape[1];
                ret.iw = src.layout.shape[2];
                ret.oh = dst.layout.shape[1];
                ret.ow = dst.layout.shape[2];
            } else if (format == Format::NHWC_NCHW ||
                       format == Format::NHWC_NCHW4_IC_SMALL) {
                ret.c = src.layout.shape[3];
                ret.ih = src.layout.shape[1];
                ret.iw = src.layout.shape[2];
                ret.oh = dst.layout.shape[2];
                ret.ow = dst.layout.shape[3];
            } else if (format == Format::NCHW4) {
                ret.c = src.layout.shape[1] * 4;
                ret.ih = src.layout.shape[2];
                ret.iw = src.layout.shape[3];
                ret.oh = dst.layout.shape[2];
                ret.ow = dst.layout.shape[3];
            } else {
                megdnn_assert(format == Format::NHWCD4);
                ret.c = src.layout.shape[2] * 4;
                ret.ih = src.layout.shape[1];
                ret.iw = src.layout.shape[3];
                ret.oh = dst.layout.shape[1];
                ret.ow = dst.layout.shape[3];
            }
            if ((src.layout.dtype.enumv() == DTypeEnum::Float32 ||
                 MEGDNN_FLOAT16_SELECT(
                         (src.layout.dtype.enumv() == DTypeEnum::Float16 ||
                          src.layout.dtype.enumv() == DTypeEnum::BFloat16),
                         false) ||
                 src.layout.dtype.enumv() == DTypeEnum::Int8 ||
                 src.layout.dtype.enumv() == DTypeEnum::Uint8 ||
                 src.layout.dtype.enumv() == DTypeEnum::QuantizedS8 ||
                 src.layout.dtype.enumv() == DTypeEnum::Quantized8Asymm) &&
                (src.layout.dtype == dst.layout.dtype)) {
                ret.sptr = src.compatible_ptr<ctype>();
                ret.mptr = mat.ptr<mtype>();
                ret.dptr = dst.compatible_ptr<ctype>();
            } else if (src.layout.dtype.enumv() == DTypeEnum::QuantizedS8) {
                ret.sptr = src.compatible_ptr<ctype>();
                ret.mptr = mat.ptr<mtype>();
                ret.dptr = dst.compatible_ptr<ctype>();
            } else if ((src.layout.dtype.enumv() == DTypeEnum::Uint8 ||
                        src.layout.dtype.enumv() ==
                                DTypeEnum::Quantized8Asymm) &&
                       src.layout.dtype.enumv() != dst.layout.dtype.enumv()) {
                ret.sptr = src.compatible_ptr<ctype>();
                ret.mptr = mat.ptr<mtype>();
                ret.dptr = reinterpret_cast<ctype*>(dst.raw_ptr);
            } else {
                ret.sptr = nullptr;
                ret.mptr = nullptr;
                ret.dptr = nullptr;
            }
            ret.workspace = workspace;
            return ret;
        }
    };

    // ctype: C type of input data type.
    // mtype: C type of transformation matrix data type.
    template <typename ctype, typename mtype>
    void kern_naive(const KernParam<ctype, mtype>& kern_param, size_t task_id);

public:
    using WarpPerspectiveForward::WarpPerspectiveForward;
    void exec(_megdnn_tensor_in src, _megdnn_tensor_in mat,
              _megdnn_tensor_in mat_idx, _megdnn_tensor_out dst,
              _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(const TensorLayout&, const TensorLayout&,
                                  const TensorLayout&,
                                  const TensorLayout&) override {
        return 0;
    }

private:
    template <typename ctype, typename mtype>
    void kern_naive_nhwcd4(const KernParam<ctype, mtype>& kern_param,
                           size_t task_id);
    template <typename ctype, typename dst_ctype, typename mtype>
    void kern_naive_dimshuffle_typecvt(
            const KernParam<ctype, mtype>& kern_param, size_t task_id);
};

class WarpPerspectiveBackwardDataImpl : public WarpPerspectiveBackwardData {
protected:
    template <typename ctype, typename mtype>
    struct KernParam {
        size_t n_src, n_mat, c, ih, iw, oh, ow;
        ctype *sptr, *hptr;
        mtype* mptr;
        int* midx_ptr;  //!< can be null

        static KernParam from_tensors(_megdnn_tensor_in mat,
                                      _megdnn_tensor_in mat_idx,
                                      _megdnn_tensor_in diff,
                                      _megdnn_tensor_out grad) {
            KernParam ret;
            ret.n_src = grad.layout.shape[0], ret.c = grad.layout.shape[1];
            ret.ih = grad.layout.shape[2], ret.iw = grad.layout.shape[3];
            ret.oh = diff.layout.shape[2], ret.ow = diff.layout.shape[3];
            ret.hptr = diff.ptr<ctype>();
            ret.mptr = mat.ptr<mtype>();
            ret.sptr = grad.ptr<ctype>();
            if (mat_idx.raw_ptr) {
                megdnn_assert(mat_idx.layout.ndim == 1);
                ret.n_mat = mat_idx.layout.shape[0];
                ret.midx_ptr = mat_idx.ptr<int>();
            } else {
                megdnn_assert(mat_idx.layout.ndim == 0);
                ret.n_mat = ret.n_src;
                ret.midx_ptr = nullptr;
            }
            return ret;
        }
    };

public:
    using WarpPerspectiveBackwardData::WarpPerspectiveBackwardData;
    void exec(_megdnn_tensor_in mat, _megdnn_tensor_in mat_idx,
              _megdnn_tensor_in diff, _megdnn_tensor_out grad,
              _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(const TensorLayout&, const TensorLayout&,
                                  const TensorLayout&,
                                  const TensorLayout&) override {
        return 0;
    }

private:
    template <typename ctype, typename mtype>
    void kern_naive(const KernParam<ctype, mtype>& kern_param);
};

class WarpPerspectiveBackwardMatImpl : public WarpPerspectiveBackwardMat {
protected:
    template <typename ctype, typename mtype>
    struct KernParam {
        size_t n_src, n_mat, c, ih, iw, oh, ow;
        ctype *sptr, *hptr;
        mtype *mptr, *res;
        int* midx_ptr;  //!< can be null
        float border_val;

        static KernParam from_tensors(float border_val_, _megdnn_tensor_in src,
                                      _megdnn_tensor_in mat,
                                      _megdnn_tensor_in mat_idx,
                                      _megdnn_tensor_in diff,
                                      _megdnn_tensor_out grad) {
            KernParam ret;
            ret.border_val = border_val_;
            ret.n_src = src.layout.shape[0], ret.c = src.layout.shape[1];
            ret.ih = src.layout.shape[2], ret.iw = src.layout.shape[3];
            ret.oh = diff.layout.shape[2], ret.ow = diff.layout.shape[3];
            ret.hptr = diff.ptr<ctype>();
            ret.mptr = mat.ptr<mtype>();
            ret.sptr = src.ptr<ctype>();
            ret.res = grad.ptr<mtype>();
            if (mat_idx.raw_ptr) {
                megdnn_assert(mat_idx.layout.ndim == 1);
                ret.n_mat = mat_idx.layout.shape[0];
                ret.midx_ptr = mat_idx.ptr<int>();
            } else {
                megdnn_assert(mat_idx.layout.ndim == 0);
                ret.n_mat = ret.n_src;
                ret.midx_ptr = nullptr;
            }
            return ret;
        }
    };

public:
    using WarpPerspectiveBackwardMat::WarpPerspectiveBackwardMat;
    void exec(_megdnn_tensor_in src, _megdnn_tensor_in mat,
              _megdnn_tensor_in mat_idx, _megdnn_tensor_in diff,
              _megdnn_tensor_out grad, _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(const TensorLayout&, const TensorLayout&,
                                  const TensorLayout&, const TensorLayout&,
                                  const TensorLayout&) override {
        return 0;
    }

private:
    template <typename ctype, typename mtype>
    void kern_naive(const KernParam<ctype, mtype>& kern_param);
};

#define UNPACK_WARP_PERSPECTIVE_FWD_KERN_PARAM(p)                         \
    auto N_SRC = p.n_src, N_MAT = p.n_mat, C = p.c, IH = p.ih, IW = p.iw, \
         OH = p.oh, OW = p.ow;                                            \
    ctype* __restrict sptr = p.sptr;                                      \
    mtype* __restrict mptr = p.mptr;                                      \
    ctype* __restrict dptr = p.dptr;                                      \
    int* __restrict midx_ptr = p.midx_ptr;                                \
    auto bmode = p.bmode;                                                 \
    float border_val = p.border_val

}  // namespace naive
}  // namespace megdnn

// vim: syntax=cpp.doxygen
