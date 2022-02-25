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
        DType src_dtype, dst_dtype;
        RefPtr src_ptr, mat_ptr, dst_ptr;
        RefPtr midx_ptr;  //!< can be null
        Workspace workspace;

        static KernParam from_tensors(
                Format format, BorderMode bmode, float border_val,
                _megdnn_tensor_in src, _megdnn_tensor_in mat, _megdnn_tensor_in mat_idx,
                _megdnn_tensor_out dst, _megdnn_workspace workspace) {
            KernParam ret;
            ret.format = format;
            ret.bmode = bmode;
            ret.border_val = border_val;
            ret.n_src = src.layout.shape[0];
            ret.src_dtype = src.layout.dtype;
            ret.dst_dtype = dst.layout.dtype;

            if (mat_idx.raw_ptr()) {
                megdnn_assert(mat_idx.layout.ndim == 1);
                ret.n_mat = mat_idx.layout.shape[0];
                ret.midx_ptr = mat_idx.get_ref_ptr();
            } else {
                megdnn_assert(mat_idx.layout.ndim == 0);
                ret.n_mat = ret.n_src;
                ret.midx_ptr = nullptr;
            }

            if (format == Format::NCHW || format == Format::NCHW_NCHW4_IC_SMALL) {
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
            } else if (
                    format == Format::NHWC_NCHW ||
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
            } else if (format == Format::NCHW64) {
                ret.c = src.layout.shape[1] * 64;
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
                 DNN_FLOAT16_SELECT(
                         (src.layout.dtype.enumv() == DTypeEnum::Float16 ||
                          src.layout.dtype.enumv() == DTypeEnum::BFloat16),
                         false) ||
                 src.layout.dtype.enumv() == DTypeEnum::Int8 ||
                 src.layout.dtype.enumv() == DTypeEnum::Uint8 ||
                 src.layout.dtype.enumv() == DTypeEnum::QuantizedS8 ||
                 src.layout.dtype.enumv() == DTypeEnum::Quantized8Asymm) &&
                (src.layout.dtype == dst.layout.dtype)) {
                ret.src_ptr = src.get_ref_ptr();
                ret.mat_ptr = mat.get_ref_ptr();
                ret.dst_ptr = dst.get_ref_ptr();
            } else if (
                    src.layout.dtype.enumv() == DTypeEnum::QuantizedS8 ||
                    src.layout.dtype.enumv() == DTypeEnum::QuantizedS4 ||
                    src.layout.dtype.enumv() == DTypeEnum::Quantized4Asymm) {
                ret.src_ptr = src.get_ref_ptr();
                ret.mat_ptr = mat.get_ref_ptr();
                ret.dst_ptr = dst.get_ref_ptr();
            } else if (
                    (src.layout.dtype.enumv() == DTypeEnum::Uint8 ||
                     src.layout.dtype.enumv() == DTypeEnum::Quantized8Asymm) &&
                    src.layout.dtype.enumv() != dst.layout.dtype.enumv()) {
                ret.src_ptr = src.get_ref_ptr();
                ret.mat_ptr = mat.get_ref_ptr();
                ret.dst_ptr = dst.get_ref_ptr();
            } else {
                ret.src_ptr = nullptr;
                ret.mat_ptr = nullptr;
                ret.dst_ptr = nullptr;
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
    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_in mat, _megdnn_tensor_in mat_idx,
            _megdnn_tensor_out dst, _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout&, const TensorLayout&, const TensorLayout&,
            const TensorLayout&) override {
        return 0;
    }

private:
    template <typename ctype, typename mtype>
    void kern_naive_nhwcd4(const KernParam<ctype, mtype>& kern_param, size_t task_id);
    template <typename ctype, typename mtype>
    void kern_naive_int4(const KernParam<ctype, mtype>& kern_param, size_t task_id);
    template <typename ctype, typename dst_ctype, typename mtype>
    void kern_naive_dimshuffle_typecvt(
            const KernParam<ctype, mtype>& kern_param, size_t task_id);
};

class WarpPerspectiveBackwardDataImpl : public WarpPerspectiveBackwardData {
protected:
    template <typename ctype, typename mtype>
    struct KernParam {
        size_t n_src, n_mat, c, ih, iw, oh, ow;
        RefPtr grad_ptr, diff_ptr;
        RefPtr mat_ptr;
        RefPtr midx_ptr;

        static KernParam from_tensors(
                _megdnn_tensor_in mat, _megdnn_tensor_in mat_idx,
                _megdnn_tensor_in diff, _megdnn_tensor_out grad) {
            KernParam ret;
            ret.n_src = grad.layout.shape[0], ret.c = grad.layout.shape[1];
            ret.ih = grad.layout.shape[2], ret.iw = grad.layout.shape[3];
            ret.oh = diff.layout.shape[2], ret.ow = diff.layout.shape[3];
            ret.diff_ptr = diff.get_ref_ptr();
            ret.mat_ptr = mat.get_ref_ptr();
            ret.grad_ptr = grad.get_ref_ptr();
            if (mat_idx.raw_ptr()) {
                megdnn_assert(mat_idx.layout.ndim == 1);
                ret.n_mat = mat_idx.layout.shape[0];
                ret.midx_ptr = mat_idx.get_ref_ptr();
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
    void exec(
            _megdnn_tensor_in mat, _megdnn_tensor_in mat_idx, _megdnn_tensor_in diff,
            _megdnn_tensor_out grad, _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout&, const TensorLayout&, const TensorLayout&,
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
        RefPtr src_ptr, grad_ptr, diff_ptr;
        RefPtr mat_ptr;
        RefPtr midx_ptr;
        float border_val;

        static KernParam from_tensors(
                float border_val_, _megdnn_tensor_in src, _megdnn_tensor_in mat,
                _megdnn_tensor_in mat_idx, _megdnn_tensor_in diff,
                _megdnn_tensor_out grad) {
            KernParam ret;
            ret.border_val = border_val_;
            ret.n_src = src.layout.shape[0], ret.c = src.layout.shape[1];
            ret.ih = src.layout.shape[2], ret.iw = src.layout.shape[3];
            ret.oh = diff.layout.shape[2], ret.ow = diff.layout.shape[3];
            ret.src_ptr = src.get_ref_ptr();
            ret.diff_ptr = diff.get_ref_ptr();
            ret.mat_ptr = mat.get_ref_ptr();
            ret.grad_ptr = grad.get_ref_ptr();
            if (mat_idx.raw_ptr()) {
                megdnn_assert(mat_idx.layout.ndim == 1);
                ret.n_mat = mat_idx.layout.shape[0];
                ret.midx_ptr = mat_idx.get_ref_ptr();
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
    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_in mat, _megdnn_tensor_in mat_idx,
            _megdnn_tensor_in diff, _megdnn_tensor_out grad,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout&, const TensorLayout&, const TensorLayout&,
            const TensorLayout&, const TensorLayout&) override {
        return 0;
    }

private:
    template <typename ctype, typename mtype>
    void kern_naive(const KernParam<ctype, mtype>& kern_param);
};

#define UNPACK_WARP_PERSPECTIVE_FWD_KERN_PARAM(p)                                    \
    auto N_SRC = p.n_src, N_MAT = p.n_mat, C = p.c, IH = p.ih, IW = p.iw, OH = p.oh, \
         OW = p.ow;                                                                  \
    auto sptr = static_cast<const ctype*>(p.src_ptr.get_ptr());                      \
    auto mptr = static_cast<const mtype*>(p.mat_ptr.get_ptr());                      \
    auto dptr = static_cast<ctype*>(p.dst_ptr.get_ptr());                            \
    auto midx_ptr = static_cast<int*>(p.midx_ptr.get_ptr());                         \
    auto bmode = p.bmode;                                                            \
    float border_val = p.border_val

}  // namespace naive
}  // namespace megdnn

// vim: syntax=cpp.doxygen
