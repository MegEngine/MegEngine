#pragma once

#include "megdnn/oprs.h"
#include "src/common/utils.h"

namespace megdnn {
namespace naive {

class ResizeImpl : public Resize {
public:
    using Format = Param::Format;
    using InterpolationMode = Param::InterpolationMode;
    template <typename ctype>
    struct KernParam {
        Format format;
        InterpolationMode imode;
        size_t n, c, ih, iw, oh, ow;
        ptrdiff_t s_in, s_ic, s_ih, s_iw;
        RefPtr sptr, dptr;
        Workspace workspace;

        static KernParam from_tensors(
                Format format, InterpolationMode imode, _megdnn_tensor_in src,
                _megdnn_tensor_out dst, _megdnn_workspace workspace);

        const ctype* src() const { return static_cast<const ctype*>(sptr.get_ptr()); }

        ctype* dst() const { return static_cast<ctype*>(dptr.get_ptr()); }
    };

    using Resize::Resize;

    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) override;

    size_t get_workspace_in_bytes(const TensorLayout&, const TensorLayout&) override {
        return 0;
    }

private:
    // ctype: C type of input data type.
    template <typename ctype>
    void kern_naive(const KernParam<ctype>& kern_param);

    template <typename ctype>
    void kern_nchw(const KernParam<ctype>& kern_param, InterpolationMode imode);

    template <typename ctype>
    void kern_naive_nhwc(const KernParam<ctype>& kern_param);

    template <typename ctype>
    void kern_naive_nhwcd4(const KernParam<ctype>& kern_param);

    template <typename ctype, size_t pack_size>
    void kern_naive_nchwx(const KernParam<ctype>& kern_param);

};  // class ResizeImpl

#define UNPACK_RESIZE_FWD_KERN_PARAM(p)                                \
    auto N = p.n, C = p.c, IH = p.ih, IW = p.iw, OH = p.oh, OW = p.ow; \
    ctype* __restrict sptr = static_cast<ctype*>(p.sptr.get_ptr());    \
    ctype* __restrict dptr = static_cast<ctype*>(p.dptr.get_ptr());

#define UNPACK_RESIZE_FWD_KERN_PARAM_WITH_STRIDE(p) \
    UNPACK_RESIZE_FWD_KERN_PARAM(p)                 \
    auto S_IN = p.s_in, S_IC = p.s_ic, S_IH = p.s_ih, S_IW = p.s_iw;

class ResizeBackwardImpl : public ResizeBackward {
public:
    using ResizeBackward::ResizeBackward;
    void exec(
            _megdnn_tensor_in diff, _megdnn_tensor_out grad,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(const TensorLayout&, const TensorLayout&) override {
        return 0;
    }

private:
    template <typename ctype>
    void kern_naive(
            bool is_nhwc, InterpolationMode imode, const ctype* diff, ctype* grad,
            int N, int C, int IH, int IW, int OH, int OW);
};

class Resize3DImpl final : public Resize3D {
public:
    using Resize3D::Resize3D;
    void exec(
            _megdnn_tensor_in src, _megdnn_tensor_out dst,
            _megdnn_workspace workspace) override;
    size_t get_workspace_in_bytes(
            const TensorLayout& src, const TensorLayout& dst) override;

private:
    template <typename ctype>
    void kern_naive(
            const float rdepth, const float rheight, const float rwidth,
            const bool align_corners, const ctype* iptr, ctype* optr, const int N,
            const int C, const int ID, const int IH, const int IW, const int OD,
            const int OH, const int OW);
};

}  // namespace naive
}  // namespace megdnn

// vim: syntax=cpp.doxygen
