#include "src/naive/resize/opr_impl.h"
#include "midout.h"
#include "src/common/cv/enums.h"
#include "src/common/resize.cuh"
#include "src/common/rounding_converter.cuh"
#include "src/common/utils.cuh"
#include "src/naive/handle.h"
#include "src/naive/resize/resize_cv.h"

MIDOUT_DECL(megdnn_naive_resize_layout)
MIDOUT_DECL(megdnn_naive_resize_nchw)

using namespace megdnn;
using namespace naive;
using namespace resize;

template <typename ctype>
ResizeImpl::KernParam<ctype> ResizeImpl::KernParam<ctype>::from_tensors(
        Format format, InterpolationMode imode, _megdnn_tensor_in src,
        _megdnn_tensor_out dst, _megdnn_workspace workspace) {
    KernParam<ctype> ret;
    ret.format = format;
    ret.imode = imode;
    ret.n = src.layout.shape[0];
    if (format == Format::NCHW) {
        ret.c = src.layout.shape[1];
        ret.ih = src.layout.shape[2];
        ret.iw = src.layout.shape[3];
        ret.oh = dst.layout.shape[2];
        ret.ow = dst.layout.shape[3];
        ret.s_in = src.layout.stride[0];
        ret.s_ic = src.layout.stride[1];
        ret.s_ih = src.layout.stride[2];
        ret.s_iw = src.layout.stride[3];
    } else if (format == Format::NHWC) {
        ret.c = src.layout.shape[3];
        ret.ih = src.layout.shape[1];
        ret.iw = src.layout.shape[2];
        ret.oh = dst.layout.shape[1];
        ret.ow = dst.layout.shape[2];
    } else if (format == Format::NCHW4) {
        ret.c = src.layout.shape[1] * 4;
        ret.ih = src.layout.shape[2];
        ret.iw = src.layout.shape[3];
        ret.oh = dst.layout.shape[2];
        ret.ow = dst.layout.shape[3];
    } else if (format == Format::NCHW44) {
        ret.c = src.layout.shape[1] * 4;
        ret.ih = src.layout.shape[2];
        ret.iw = src.layout.shape[3];
        ret.oh = dst.layout.shape[2];
        ret.ow = dst.layout.shape[3];
    } else if (format == Format::NCHW88) {
        ret.c = src.layout.shape[1] * 8;
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
    if (src.layout.dtype.enumv() == DTypeEnum::Float32 ||
        DNN_FLOAT16_SELECT(src.layout.dtype.enumv() == DTypeEnum::Float16, false) ||
        src.layout.dtype.enumv() == DTypeEnum::Int8 ||
        src.layout.dtype.enumv() == DTypeEnum::Uint8 ||
        src.layout.dtype.enumv() == DTypeEnum::QuantizedS8 ||
        src.layout.dtype.enumv() == DTypeEnum::Quantized8Asymm) {
        ret.sptr = src.get_ref_ptr();
        ret.dptr = dst.get_ref_ptr();
    } else {
        megdnn_assert(
                0, "current do not support dtype %s in resize",
                src.layout.dtype.name());
    }
    ret.workspace = workspace;
    return ret;
}

#define INST(_dtype) template struct ResizeImpl::KernParam<_dtype>;

INST(dt_float32);
#if !MEGDNN_DISABLE_FLOAT16
INST(dt_float16);
#endif
INST(dt_int8);
INST(dt_uint8);
INST(dt_qint8);
INST(dt_quint8);

#undef INST

template <typename ctype>
void ResizeImpl::kern_nchw(
        const KernParam<ctype>& kern_param, InterpolationMode imode) {
    megdnn_assert(kern_param.format == Format::NCHW);
    UNPACK_RESIZE_FWD_KERN_PARAM_WITH_STRIDE(kern_param);
    float scale_h = static_cast<float>(OH) / IH;
    float scale_w = static_cast<float>(OW) / IW;
    rounding::RoundingConverter<ctype> output_converter;

    rep(n, N) {
        rep(oh, OH) rep(ow, OW) {
            switch (imode) {
                case InterpolationMode::NEAREST: {
                    auto ih = get_nearest_src(scale_h, IH, oh);
                    auto iw = get_nearest_src(scale_w, IW, ow);

                    rep(c, static_cast<int>(C)) {
                        dptr[c * OH * OW + oh * OW + ow] =
                                sptr[c * S_IC + ih * S_IH + iw * S_IW];
                    }
                    break;
                }
                case InterpolationMode::INTER_LINEAR: {
                    int ih0, ih1, iw0, iw1;
                    float ah0, ah1, aw0, aw1;

                    std::tie(ah0, ih0, ah1, ih1) =
                            get_nearest_linear_coord(kern_param.imode, scale_h, IH, oh);
                    std::tie(aw0, iw0, aw1, iw1) =
                            get_nearest_linear_coord(kern_param.imode, scale_w, IW, ow);

                    rep(c, static_cast<int>(C)) {
                        dptr[c * OH * OW + oh * OW + ow] = output_converter(
                                sptr[c * S_IC + ih0 * S_IH + iw0 * S_IW] * ah0 * aw0 +
                                sptr[c * S_IC + ih0 * S_IH + iw1 * S_IW] * ah0 * aw1 +
                                sptr[c * S_IC + ih1 * S_IH + iw0 * S_IW] * ah1 * aw0 +
                                sptr[c * S_IC + ih1 * S_IH + iw1 * S_IW] * ah1 * aw1);
                    }
                    break;
                }
                case InterpolationMode::INTER_CUBIC: {
                    auto coord_h = get_cubic_coord(scale_h, oh);
                    auto coord_w = get_cubic_coord(scale_w, ow);

                    float alphah = coord_h.first;
                    float alphaw = coord_w.first;

                    int ih0 = coord_h.second - 1;
                    int iw0 = coord_w.second - 1;
                    float h_coeff[4], w_coeff[4];
                    interpolate_cubic(alphah, h_coeff);
                    interpolate_cubic(alphaw, w_coeff);

                    rep(c, static_cast<int>(C)) {
                        constexpr int ksize = 4;
                        float ret = 0;
                        rep(kh, ksize) {
                            int h = saturate<int, int>(ih0 + kh, 0, IH - 1);
                            rep(kw, ksize) {
                                int w = saturate<int, int>(iw0 + kw, 0, IW - 1);
                                ret += sptr[c * S_IC + h * S_IH + w * S_IW] *
                                       h_coeff[kh] * w_coeff[kw];
                            }
                        }
                        dptr[c * OH * OW + oh * OW + ow] = output_converter(ret);
                    }
                    break;
                }
                default:
                    megdnn_throw("unsupported mode in ResizeBackwardImpl");
                    break;
            }
        }
        sptr += S_IN;
        dptr += C * OH * OW;
    }
}
template <typename ctype>
void ResizeImpl::kern_naive(const KernParam<ctype>& kern_param) {
    if (kern_param.format == Format::NHWC) {
        MIDOUT_BEGIN(megdnn_naive_resize_layout, midout_iv(0)) {
            kern_naive_nhwc(kern_param);
        }
        MIDOUT_END();
        return;
    } else if (kern_param.format == Format::NHWCD4) {
        MIDOUT_BEGIN(megdnn_naive_resize_layout, midout_iv(1)) {
            kern_naive_nhwcd4(kern_param);
        }
        MIDOUT_END();
        return;
    } else if (kern_param.format == Format::NCHW4) {
        MIDOUT_BEGIN(megdnn_naive_resize_layout, midout_iv(2)) {
            kern_naive_nchwx<ctype, 4>(kern_param);
        }
        MIDOUT_END();
        return;
    } else if (kern_param.format == Format::NCHW44) {
        MIDOUT_BEGIN(megdnn_naive_resize_layout, midout_iv(3)) {
            kern_naive_nchwx<ctype, 4>(kern_param);
        }
        MIDOUT_END();
        return;
    } else if (kern_param.format == Format::NCHW88) {
        MIDOUT_BEGIN(megdnn_naive_resize_layout, midout_iv(4)) {
            kern_naive_nchwx<ctype, 8>(kern_param);
        }
        MIDOUT_END();
        return;
    }
}

template <typename ctype>
void ResizeImpl::kern_naive_nhwc(const KernParam<ctype>& kern_param) {
    UNPACK_RESIZE_FWD_KERN_PARAM(kern_param);
    rounding::RoundingConverter<ctype> output_converter;
    float scale_h = static_cast<float>(OH) / IH;
    float scale_w = static_cast<float>(OW) / IW;

    rep(n, N) {
        rep(oh, OH) rep(ow, OW) {
            int ih0, ih1, iw0, iw1;
            float ah0, ah1, aw0, aw1;

            std::tie(ah0, ih0, ah1, ih1) =
                    get_nearest_linear_coord(kern_param.imode, scale_h, IH, oh);
            std::tie(aw0, iw0, aw1, iw1) =
                    get_nearest_linear_coord(kern_param.imode, scale_w, IW, ow);

            rep(c, C) {
                dptr[(oh * OW + ow) * C + c] = output_converter(
                        sptr[(ih0 * IW + iw0) * C + c] * ah0 * aw0 +
                        sptr[(ih0 * IW + iw1) * C + c] * ah0 * aw1 +
                        sptr[(ih1 * IW + iw0) * C + c] * ah1 * aw0 +
                        sptr[(ih1 * IW + iw1) * C + c] * ah1 * aw1);
            }
        }
        sptr += C * IH * IW;
        dptr += C * OH * OW;
    }
}

template <typename ctype>
void ResizeImpl::kern_naive_nhwcd4(const KernParam<ctype>& kern_param) {
    UNPACK_RESIZE_FWD_KERN_PARAM(kern_param);
    rounding::RoundingConverter<ctype> output_converter;
    float scale_h = static_cast<float>(OH) / IH;
    float scale_w = static_cast<float>(OW) / IW;

    auto get_tensor_addr = [&](size_t h, size_t w, size_t c, size_t W,
                               size_t C) -> size_t {
        megdnn_assert((C & 0x3) == 0);
        size_t CBLK = (C >> 2);
        return (h * W * CBLK * 4 + (c >> 2) * W * 4 + w * 4 + (c & 0x3));
    };

    rep(n, N) {
        rep(oh, OH) rep(ow, OW) {
            int ih0, ih1, iw0, iw1;
            float ah0, ah1, aw0, aw1;

            std::tie(ah0, ih0, ah1, ih1) =
                    get_nearest_linear_coord(kern_param.imode, scale_h, IH, oh);
            std::tie(aw0, iw0, aw1, iw1) =
                    get_nearest_linear_coord(kern_param.imode, scale_w, IW, ow);

            rep(c, C) {
                dptr[get_tensor_addr(oh, ow, c, OW, C)] = output_converter(
                        sptr[get_tensor_addr(ih0, iw0, c, IW, C)] * ah0 * aw0 +
                        sptr[get_tensor_addr(ih0, iw1, c, IW, C)] * ah0 * aw1 +
                        sptr[get_tensor_addr(ih1, iw0, c, IW, C)] * ah1 * aw0 +
                        sptr[get_tensor_addr(ih1, iw1, c, IW, C)] * ah1 * aw1);
            }
        }
        sptr += IH * (C / 4) * IW * 4;
        dptr += OH * (C / 4) * OW * 4;
    }
}

template <typename ctype, size_t pack_size>
void ResizeImpl::kern_naive_nchwx(const KernParam<ctype>& kern_param) {
    UNPACK_RESIZE_FWD_KERN_PARAM(kern_param);
    rounding::RoundingConverter<ctype> output_converter;
    float scale_h = static_cast<float>(OH) / IH;
    float scale_w = static_cast<float>(OW) / IW;

    megdnn_assert(pack_size == 4 || pack_size == 8);
    size_t log_pack_size = 2;
    if (pack_size == 8) {
        log_pack_size = 3;
    }

    auto get_tensor_addr = [&](size_t h, size_t w, size_t c, size_t H, size_t W,
                               size_t C) -> size_t {
        megdnn_assert((C & (pack_size - 1)) == 0);
        return (((c >> log_pack_size) * H * W + h * W + w) << log_pack_size) +
               (c & (pack_size - 1));
    };

    rep(n, N) {
        rep(oh, OH) rep(ow, OW) {
            int ih0, ih1, iw0, iw1;
            float ah0, ah1, aw0, aw1;

            std::tie(ah0, ih0, ah1, ih1) =
                    get_nearest_linear_coord(kern_param.imode, scale_h, IH, oh);
            std::tie(aw0, iw0, aw1, iw1) =
                    get_nearest_linear_coord(kern_param.imode, scale_w, IW, ow);

            rep(c, C) {
                dptr[get_tensor_addr(oh, ow, c, OH, OW, C)] = output_converter(
                        sptr[get_tensor_addr(ih0, iw0, c, IH, IW, C)] * ah0 * aw0 +
                        sptr[get_tensor_addr(ih0, iw1, c, IH, IW, C)] * ah0 * aw1 +
                        sptr[get_tensor_addr(ih1, iw0, c, IH, IW, C)] * ah1 * aw0 +
                        sptr[get_tensor_addr(ih1, iw1, c, IH, IW, C)] * ah1 * aw1);
            }
        }
        sptr += IH * IW * C;
        dptr += OH * OW * C;
    }
}

void ResizeImpl::exec(
        _megdnn_tensor_in src, _megdnn_tensor_in dst, _megdnn_workspace workspace) {
#if !MGE_BUILD_WITHOUT_NAIVE_EXEC
    check_exec(src.layout, dst.layout, workspace.size);
    if (param().format == param::Resize::Format::NCHW) {
#define cb(dt, ct, _midout_iv)                                              \
    case DTypeTrait<dt>::enumv: {                                           \
        MIDOUT_BEGIN(megdnn_naive_resize_nchw, midout_iv(_midout_iv)) {     \
            auto kparam = KernParam<ct>::from_tensors(                      \
                    param().format, param().imode, src, dst, workspace);    \
            MEGDNN_DISPATCH_CPU_KERN_OPR(kern_nchw(kparam, param().imode)); \
        }                                                                   \
        MIDOUT_END();                                                       \
        return;                                                             \
    }

        switch (src.layout.dtype.enumv()) {
            cb(dtype::Float32, float, 0);
            DNN_INC_FLOAT16(cb(dtype::Float16, dt_float16, 1));
            cb(dtype::Int8, int8_t, 2);
            cb(dtype::QuantizedS8, int8_t, 3);
            cb(dtype::Uint8, uint8_t, 4);
            cb(dtype::Quantized8Asymm, uint8_t, 5);
            default:
                megdnn_throw(ssprintf(
                                     "Unsupported input DType in Resize "
                                     "NEAREST mode: %s",
                                     src.layout.dtype.name())
                                     .c_str());
                return;
        }

#undef cb
    }

    if (((src.layout[3] != 1 && src.layout[3] != 3) ||
         !is_nhwc_contig_wc(src.layout)) ||
        (param().imode == param::Resize::InterpolationMode::LINEAR)) {
#define cb(dt, ct, _midout_iv)                                            \
    case DTypeTrait<dt>::enumv: {                                         \
        MIDOUT_BEGIN(megdnn_naive_resize_layout, midout_iv(_midout_iv)) { \
            auto kparam = KernParam<ct>::from_tensors(                    \
                    param().format, param().imode, src, dst, workspace);  \
            MEGDNN_DISPATCH_CPU_KERN_OPR(kern_naive(kparam));             \
        }                                                                 \
        MIDOUT_END();                                                     \
        return;                                                           \
    }

        switch (src.layout.dtype.enumv()) {
            cb(dtype::Float32, float, 0);
            DNN_INC_FLOAT16(cb(dtype::Float16, dt_float16, 1));
            cb(dtype::Int8, int8_t, 2);
            cb(dtype::QuantizedS8, int8_t, 3);
            cb(dtype::Uint8, uint8_t, 4);
            cb(dtype::Quantized8Asymm, uint8_t, 5);
            default:
                megdnn_throw(ssprintf(
                                     "Unsupported input DType in Resize: %s",
                                     src.layout.dtype.name())
                                     .c_str());
                return;
        }

#undef cb
    } else {
        megdnn_assert(
                param().format == param::Resize::Format::NHWC, "invalid resize format");
        MEGDNN_DISPATCH_CPU_KERN_OPR(resize_cv_exec(src, dst, param().imode));
    }
#else
    __builtin_trap();
#endif
}

// ***************************Backward*************************** //
template <typename ctype>
void ResizeBackwardImpl::kern_naive(
        bool is_nhwc, InterpolationMode imode, const ctype* diff, ctype* grad, int N,
        int C, int IH, int IW, int OH, int OW) {
    float scale_h = static_cast<float>(OH) / IH;
    float scale_w = static_cast<float>(OW) / IW;
    rounding::RoundingConverter<ctype> output_converter;
    auto kern = [=]() {
        auto hptr = diff;
        auto sptr = grad;
        std::memset(sptr, 0, sizeof(ctype) * N * C * IH * IW);
        rep(n, N) {
            rep(oh, OH) rep(ow, OW) {
                switch (imode) {
                    case InterpolationMode::INTER_LINEAR: {
                        int ih0, ih1, iw0, iw1;
                        float ah0, ah1, aw0, aw1;

                        std::tie(ah0, ih0, ah1, ih1) =
                                get_nearest_linear_coord(imode, scale_h, IH, oh);
                        std::tie(aw0, iw0, aw1, iw1) =
                                get_nearest_linear_coord(imode, scale_w, IW, ow);

                        if (is_nhwc) {
                            rep(c, C) {
                                sptr[(ih0 * IW + iw0) * C + c] += output_converter(
                                        hptr[(oh * OW + ow) * C + c] * ah0 * aw0);
                                sptr[(ih0 * IW + iw1) * C + c] += output_converter(
                                        hptr[(oh * OW + ow) * C + c] * ah0 * aw1);
                                sptr[(ih1 * IW + iw0) * C + c] += output_converter(
                                        hptr[(oh * OW + ow) * C + c] * ah1 * aw0);
                                sptr[(ih1 * IW + iw1) * C + c] += output_converter(
                                        hptr[(oh * OW + ow) * C + c] * ah1 * aw1);
                            }
                        } else {
                            rep(c, C) {
                                float hidden = hptr[c * OH * OW + oh * OW + ow];
                                sptr[c * IH * IW + ih0 * IW + iw0] +=
                                        output_converter(ah0 * aw0 * hidden);
                                sptr[c * IH * IW + ih1 * IW + iw0] +=
                                        output_converter(ah1 * aw0 * hidden);
                                sptr[c * IH * IW + ih0 * IW + iw1] +=
                                        output_converter(ah0 * aw1 * hidden);
                                sptr[c * IH * IW + ih1 * IW + iw1] +=
                                        output_converter(ah1 * aw1 * hidden);
                            }
                        }
                        break;
                    }
                    case InterpolationMode::NEAREST: {
                        auto ih = get_nearest_src(scale_h, IH, oh);
                        auto iw = get_nearest_src(scale_w, IW, ow);
                        rep(c, static_cast<int>(C)) {
                            sptr[c * IH * IW + ih * IW + iw] +=
                                    output_converter(hptr[c * OH * OW + oh * OW + ow]);
                        }
                        break;
                    }
                    case InterpolationMode::INTER_CUBIC: {
                        auto coord_h = get_cubic_coord(scale_h, oh);
                        auto coord_w = get_cubic_coord(scale_w, ow);

                        float alphah = coord_h.first;
                        float alphaw = coord_w.first;

                        int ih0 = coord_h.second - 1;
                        int iw0 = coord_w.second - 1;
                        float h_coeff[4], w_coeff[4];
                        interpolate_cubic(alphah, h_coeff);
                        interpolate_cubic(alphaw, w_coeff);

                        rep(c, static_cast<int>(C)) {
                            constexpr int ksize = 4;
                            rep(kh, ksize) {
                                int h = saturate<int, int>(ih0 + kh, 0, IH - 1);
                                rep(kw, ksize) {
                                    int w = saturate<int, int>(iw0 + kw, 0, IW - 1);
                                    sptr[c * IH * IW + h * IW + w] += output_converter(
                                            hptr[c * OH * OW + oh * OW + ow] *
                                            h_coeff[kh] * w_coeff[kw]);
                                }
                            }
                        }
                        break;
                    }
                    default: {
                        megdnn_throw("unsupported mode in ResizeBackwardImpl");
                        break;
                    }
                }
            }
            sptr += C * IH * IW;
            hptr += C * OH * OW;
        }
    };
    MEGDNN_DISPATCH_CPU_KERN_OPR(kern());
}

#define INST(ctype)                                                                 \
    template void ResizeBackwardImpl::kern_naive(                                   \
            bool, InterpolationMode, const ctype*, ctype*, int, int, int, int, int, \
            int);
INST(dt_float32);
DNN_INC_FLOAT16(INST(dt_float16));
#undef INST

void ResizeBackwardImpl::exec(
        _megdnn_tensor_in diff, _megdnn_tensor_out grad, _megdnn_workspace workspace) {
    check_exec(diff.layout, grad.layout, workspace.size);
    megdnn_assert(
            param().format == param::Resize::Format::NCHW ||
                    param().format == param::Resize::Format::NHWC,
            "invalid resize format");
    size_t N, C, IH, IW, OH, OW;
    bool is_nhwc = param().format == param::Resize::Format::NHWC;
    if (is_nhwc) {
        if (param().imode != Param::InterpolationMode::LINEAR &&
            is_nhwc_contig_wc(grad.layout)) {
            megdnn_assert(
                    0,
                    "unsupport mode in resizeBackward, only support param().imode = "
                    "LINEAR");
        }
        N = grad.layout.shape[0];
        C = grad.layout.shape[3];
        IH = grad.layout.shape[1];
        IW = grad.layout.shape[2];
        OH = diff.layout.shape[1];
        OW = diff.layout.shape[2];
    } else {
        N = grad.layout.shape[0], C = grad.layout.shape[1], IH = grad.layout.shape[2],
        IW = grad.layout.shape[3];
        OH = diff.layout.shape[2], OW = diff.layout.shape[3];
    }
    switch (grad.layout.dtype.enumv()) {
#define cb(_t)                                                                     \
    case DTypeTrait<_t>::enumv: {                                                  \
        typedef DTypeTrait<_t>::ctype ct;                                          \
        ct* diff_ptr = diff.ptr<ct>();                                             \
        ct* grad_ptr = grad.ptr<ct>();                                             \
        ResizeBackwardImpl::kern_naive(                                            \
                is_nhwc, param().imode, diff_ptr, grad_ptr, N, C, IH, IW, OH, OW); \
        break;                                                                     \
    }
        cb(megdnn::dtype::Float32);
        DNN_INC_FLOAT16(cb(megdnn::dtype::Float16));
        default:
            megdnn_throw(ssprintf(
                    "unsupported dtype: %s in resize backward",
                    grad.layout.dtype.name()));
    }
}

// vim: syntax=cpp.doxygen
