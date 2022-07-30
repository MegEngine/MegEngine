#include "src/cuda/warp_perspective/opr_impl.h"
#include "src/cuda/warp_perspective/warp_perspective_cv.cuh"

#include "src/cuda/utils.h"
#include "src/cuda/warp_perspective/common.h"
#include "src/cuda/warp_perspective/helper.h"

#include "src/common/cv/common.h"
#include "src/common/warp_common.h"

namespace megdnn {
namespace cuda {

namespace {
inline void deduce_reformat_layout(
        std::unique_ptr<RelayoutFormat>& relayout, const TensorLayout& src_layout,
        TensorLayout& dst_layout, RelayoutFormat::Param::Mode mode, const int oc = 0,
        const int group = 1) {
    if (src_layout.ndim > 0) {
        RelayoutFormat::Param trans_param;
        trans_param.mode = mode;
        trans_param.oc = oc;
        trans_param.group = group;
        relayout->param() = trans_param;
        relayout->deduce_layout(src_layout, dst_layout);
    } else {
        dst_layout = src_layout;
    }
}

void get_inner_layout(
        const TensorLayout& src, const TensorLayout& dst, TensorLayout& inner_src,
        TensorLayout& inner_dst, Handle* handle,
        WarpPerspectiveForwardImpl::Param::Format format) {
    if ((src.dtype.enumv() == DTypeEnum::QuantizedS4 ||
         src.dtype.enumv() == DTypeEnum::Quantized4Asymm) &&
        dst.dtype.enumv() == src.dtype.enumv() &&
        format == param::WarpPerspective::Format::NCHW) {
        auto relayout_opr = handle->create_operator<RelayoutFormat>();
        deduce_reformat_layout(
                relayout_opr, src, inner_src, RelayoutFormat::Param::Mode::NCHW_NCHW64,
                0, 1);
        deduce_reformat_layout(
                relayout_opr, dst, inner_dst, RelayoutFormat::Param::Mode::NCHW_NCHW64,
                0, 1);
    } else {
        megdnn_assert(0, "not support");
    }
}

}  // namespace

namespace warp_perspective {

void warp_perspective_cv_exec(
        _megdnn_tensor_in src, _megdnn_tensor_in mat, _megdnn_tensor_in dst,
        float border_val, BorderMode bmode, InterpolationMode imode,
        _megdnn_workspace workspace, cudaStream_t stream) {
    megdnn_assert(src.layout[3] == 1 || src.layout[3] == 3, "unsupported src channel");
    megdnn_assert(
            src.layout.dtype != dtype::Float32() || src.layout.dtype != dtype::Uint8(),
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
                        src_mat.ptr(), dst_mat.ptr(), src_mat.rows(), src_mat.cols(),
                        dst_mat.rows(), dst_mat.cols(), src_mat.step(), dst_mat.step(),
                        bmode, imode, trans_ptr, border_val, workspace_ptr, stream);
            } else {
                warp_perspective_cv_proxy<float, 3>(
                        src_mat.ptr(), dst_mat.ptr(), src_mat.rows(), src_mat.cols(),
                        dst_mat.rows(), dst_mat.cols(), src_mat.step(), dst_mat.step(),
                        bmode, imode, trans_ptr, border_val, workspace_ptr, stream);
            }
        } else if (dst.layout.dtype == dtype::Uint8()) {
            Mat<uchar> src_mat = TensorND2Mat<uchar>(src, i);
            Mat<uchar> dst_mat = TensorND2Mat<uchar>(dst, i);
            if (src_mat.channels() == 1) {
                warp_perspective_cv_proxy<uchar, 1>(
                        src_mat.ptr(), dst_mat.ptr(), src_mat.rows(), src_mat.cols(),
                        dst_mat.rows(), dst_mat.cols(), src_mat.step(), dst_mat.step(),
                        bmode, imode, trans_ptr, static_cast<uchar>(border_val),
                        workspace_ptr, stream);
            } else {
                warp_perspective_cv_proxy<uchar, 3>(
                        src_mat.ptr(), dst_mat.ptr(), src_mat.rows(), src_mat.cols(),
                        dst_mat.rows(), dst_mat.cols(), src_mat.step(), dst_mat.step(),
                        bmode, imode, trans_ptr, static_cast<uchar>(border_val),
                        workspace_ptr, stream);
            }

        } else {
            megdnn_throw("Unsupported datatype of WarpPerspective optr.");
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
    if ((src.dtype.enumv() == DTypeEnum::QuantizedS4 ||
         src.dtype.enumv() == DTypeEnum::Quantized4Asymm) &&
        param().format == param::WarpPerspective::Format::NCHW) {
        get_inner_layout(src, dst, fsrc, fdst, handle(), param().format);
        sizes.push_back(fsrc.span().dist_byte());
        sizes.push_back(fdst.span().dist_byte());
    } else {
        auto get_workspace = [&sizes](TensorLayout& layout) {
            if (layout.dtype == dtype::BFloat16()) {
                layout.dtype = dtype::Float32();
                sizes.push_back(layout.span().dist_byte());
            }
        };
        get_workspace(fsrc);
        get_workspace(fmat);
        get_workspace(fdst);
    }
    if (param().format == param::WarpPerspective::Format::NHWC) {
        //! use double for the workspace dtype as float may cause
        //! accuracy problems
        sizes.push_back(mat.total_nr_elems() * sizeof(double));
    }

    return {ptr, std::move(sizes)};
}

WorkspaceBundle WarpPerspectiveForwardImpl::get_workspace_bundle(
        void* ptr, const TensorLayoutArray& srcs, const TensorLayout& mat,
        const TensorLayout& mat_idx, const TensorLayout& dst) const {
    MEGDNN_MARK_USED_VAR(mat_idx);
    SmallVector<size_t> sizes;
    TensorLayoutArray fsrcs = srcs;
    TensorLayout fmat = mat;
    TensorLayout fdst = dst;
    auto get_workspace = [&sizes](TensorLayout& layout) {
        if (layout.dtype == dtype::BFloat16()) {
            layout.dtype = dtype::Float32();
            sizes.push_back(layout.span().dist_byte());
        }
    };
    for (auto&& fsrc : fsrcs) {
        get_workspace(fsrc);
    }
    get_workspace(fmat);
    get_workspace(fdst);
    sizes.push_back(sizeof(dt_float32*) * srcs.size());
    if (param().format == param::WarpPerspective::Format::NHWC) {
        //! use double for the workspace dtype as float may cause
        //! accuracy problems
        sizes.push_back(mat.total_nr_elems() * sizeof(double));
    }
    return {ptr, std::move(sizes)};
}

void WarpPerspectiveForwardImpl::exec(
        _megdnn_tensor_in ssrc, _megdnn_tensor_in smat, _megdnn_tensor_in smat_idx,
        _megdnn_tensor_out sdst, _megdnn_workspace sworkspace) {
    check_exec_allow_nhwc_mat_idx(
            ssrc.layout, smat.layout, smat_idx.layout, sdst.layout, sworkspace.size);

    TensorND src = ssrc;
    TensorND mat = smat;
    TensorND mat_idx = smat_idx;
    TensorND dst = sdst;
    Param::Format inner_format = param().format;
    auto bundle = get_workspace_bundle(
            sworkspace.raw_ptr, ssrc.layout, smat.layout, smat_idx.layout, sdst.layout);
    auto ctypecvt = CompTypeCvter<dtype::BFloat16, dtype::Float32>(
            concrete_handle(this->handle()), &bundle);
    if (ssrc.layout.dtype.enumv() == DTypeTrait<dtype::BFloat16>::enumv) {
        ctypecvt.src_to_comp_type(ssrc, src)
                .src_to_comp_type(smat, mat)
                .src_to_comp_type(sdst, dst);
    } else if (
            (ssrc.layout.dtype.enumv() == DTypeEnum::QuantizedS4 ||
             ssrc.layout.dtype.enumv() == DTypeEnum::Quantized4Asymm) &&
            param().format == Param::Format::NCHW) {
        auto handle_ptr = handle();
        get_inner_layout(
                ssrc.layout, sdst.layout, src.layout, dst.layout, handle_ptr,
                param().format);
        src = TensorND{bundle.get(0), src.layout};
        dst = TensorND{bundle.get(1), dst.layout};
        auto relayout_opr = handle_ptr->create_operator<RelayoutFormat>();
        RelayoutFormat::Param trans_param;
        trans_param.mode = RelayoutFormat::Param::Mode::NCHW_NCHW64;
        relayout_opr->param() = trans_param;
        relayout_opr->exec(ssrc, src, {});
        inner_format = Param::Format::NCHW64;
    }

    {
        auto stream = cuda_stream(this->handle());
        bool is_nhwc = inner_format == param::WarpPerspective::Format::NHWC;

        if (is_nhwc && param().imode != Param::InterpolationMode::LINEAR) {
            // use opencv impl only for nhwc and non-linear interp
            megdnn_assert(
                    !mat_idx.raw_ptr(),
                    "mat_idx is not supported in NHWC case with "
                    "non-linear interpolation");
            warp_perspective::warp_perspective_cv_exec(
                    src, mat, dst, param().border_val,
                    warp_perspective::get_bmode(param().bmode),
                    warp_perspective::get_imode(param().imode), ctypecvt.workspace(),
                    stream);

        } else {
            megdnn_assert(warp::is_dnn_available(
                    src.layout, mat.layout, dst.layout, param().imode, inner_format));
            size_t C, IH, IW, OH, OW;
            if (is_nhwc) {
                C = src.layout.shape[3];
                IH = src.layout.shape[1];
                IW = src.layout.shape[2];
                OH = dst.layout.shape[1];
                OW = dst.layout.shape[2];
            } else if (inner_format == Param::Format::NCHW4) {
                C = src.layout.shape[1] * 4;
                IH = src.layout.shape[2];
                IW = src.layout.shape[3];
                OH = dst.layout.shape[2];
                OW = dst.layout.shape[3];
            } else if (inner_format == Param::Format::NHWC_NCHW) {
                C = src.layout.shape[3];
                IH = src.layout.shape[1];
                IW = src.layout.shape[2];
                OH = dst.layout.shape[2];
                OW = dst.layout.shape[3];
            } else if (inner_format == Param::Format::NHWC_NCHW4_IC_SMALL) {
                C = src.layout.shape[3];
                IH = src.layout.shape[1];
                IW = src.layout.shape[2];
                OH = dst.layout.shape[2];
                OW = dst.layout.shape[3];
                megdnn_assert(
                        (C == 1) || (C == 3),
                        "NHWC_NCHW4_IC_SMALL only support C == 1 or C == 3");
            } else if (inner_format == Param::Format::NCHW_NCHW4_IC_SMALL) {
                C = src.layout.shape[1];
                IH = src.layout.shape[2];
                IW = src.layout.shape[3];
                OH = dst.layout.shape[2];
                OW = dst.layout.shape[3];
                megdnn_assert(
                        (C == 1) || (C == 3),
                        "NCHW_NCHW4_IC_SMALL only support C == 1 or C == 3");
            } else if (inner_format == Param::Format::NCHW64) {
                C = src.layout.shape[1] * 64;
                IH = src.layout.shape[2];
                IW = src.layout.shape[3];
                OH = dst.layout.shape[2];
                OW = dst.layout.shape[3];
            } else {
                megdnn_assert(
                        inner_format == param::WarpPerspective::Format::NCHW,
                        "invalid warp_perspective format");
                C = src.layout.shape[1];
                IH = src.layout.shape[2];
                IW = src.layout.shape[3];
                OH = dst.layout.shape[2];
                OW = dst.layout.shape[3];
            }
            megdnn_assert(
                    param().imode == Param::InterpolationMode::LINEAR,
                    "unsupported interpolation mode for NCHW format");
            auto bval = param().border_val;
            auto bmode = warp_perspective::get_bmode(param().bmode);
            if (src.layout.dtype == dst.layout.dtype) {
                if (src.layout.dtype == dtype::Float32{}) {
                    warp_perspective::forward_proxy(
                            is_nhwc, src.ptr<dt_float32>(), mat.ptr<dt_float32>(),
                            mat_idx.raw_ptr() ? mat_idx.ptr<int>() : nullptr,
                            dst.ptr<dt_float32>(), src.layout[0], mat.layout[0], C, IH,
                            IW, OH, OW, bval, bmode, async_error_info(handle()),
                            m_error_tracker, stream);
                } else if (DNN_FLOAT16_SELECT(
                                   src.layout.dtype == dtype::Float16(), false)) {
#ifndef MEGDNN_DISABLE_FLOAT16
                    warp_perspective::forward_proxy(
                            is_nhwc, src.ptr<dt_float16>(), mat.ptr<dt_float32>(),
                            mat_idx.raw_ptr() ? mat_idx.ptr<int>() : nullptr,
                            dst.ptr<dt_float16>(), src.layout[0], mat.layout[0], C, IH,
                            IW, OH, OW, static_cast<dt_float16>(bval), bmode,
                            async_error_info(handle()), m_error_tracker, stream);
#endif
                } else if (src.layout.dtype == dtype::Uint8()) {
                    warp_perspective::forward_proxy<dt_uint8>(
                            is_nhwc, src.ptr<dt_uint8>(), mat.ptr<dt_float32>(),
                            mat_idx.raw_ptr() ? mat_idx.ptr<int>() : nullptr,
                            dst.ptr<dt_uint8>(), src.layout[0], mat.layout[0], C, IH,
                            IW, OH, OW, bval, bmode, async_error_info(handle()),
                            m_error_tracker, stream);
                } else if (src.layout.dtype == dtype::Int8()) {
                    megdnn_assert(
                            !is_nhwc,
                            "WarpPerspective on CUDA does not support "
                            "NHWC + Int8");
                    warp_perspective::forward_proxy<dt_int8>(
                            false, src.ptr<dt_int8>(), mat.ptr<dt_float32>(),
                            mat_idx.raw_ptr() ? mat_idx.ptr<int>() : nullptr,
                            dst.ptr<dt_int8>(), src.layout[0], mat.layout[0], C, IH, IW,
                            OH, OW, bval /* implicit float -> int8 conversion,
                                            should be safe */
                            ,
                            bmode, async_error_info(handle()), m_error_tracker, stream);
                } else if (src.layout.dtype.enumv() == DTypeEnum::QuantizedS8) {
                    megdnn_assert(
                            param().format == Param::Format::NCHW4,
                            "WarpPerspective on CUDA supports NCHW4 + "
                            "QuantizedS8 only");
                    warp_perspective::forward_proxy_nchw4<dt_int8>(
                            src.compatible_ptr<dt_int8>(), mat.ptr<dt_float32>(),
                            mat_idx.raw_ptr() ? mat_idx.ptr<int>() : nullptr,
                            dst.compatible_ptr<dt_int8>(), src.layout[0], mat.layout[0],
                            C, IH, IW, OH, OW, bval, bmode, async_error_info(handle()),
                            m_error_tracker, stream);
                } else if (
                        (src.layout.dtype.enumv() == DTypeEnum::QuantizedS4) &&
                        (param().format == Param::Format::NCHW64 ||
                         param().format == Param::Format::NCHW)) {
                    bval = roundf(bval);
                    bval = fmin(fmax(-8.f, bval), 7.f);
                    warp_perspective::forward_proxy_nchw64<dt_qint4>(
                            src.compatible_ptr<dt_qint4>(), mat.ptr<dt_float32>(),
                            mat_idx.raw_ptr() ? mat_idx.ptr<int>() : nullptr,
                            dst.compatible_ptr<dt_qint4>(), src.layout[0],
                            mat.layout[0], C, IH, IW, OH, OW,
                            static_cast<dt_qint4>(bval), bmode,
                            async_error_info(handle()), m_error_tracker, stream);
                    if (param().format == Param::Format::NCHW) {
                        auto relayout_opr = handle()->create_operator<RelayoutFormat>();
                        RelayoutFormat::Param trans_param;
                        trans_param.mode = RelayoutFormat::Param::Mode::NCHW64_NCHW;
                        trans_param.oc = sdst.layout[1];
                        relayout_opr->param() = trans_param;
                        relayout_opr->exec(dst, sdst, {});
                    }
                } else if (
                        (src.layout.dtype.enumv() == DTypeEnum::Quantized4Asymm) &&
                        (param().format == Param::Format::NCHW64 ||
                         param().format == Param::Format::NCHW)) {
                    bval = roundf(bval);
                    bval = fmin(fmax(0, bval), 15);
                    warp_perspective::forward_proxy_nchw64<dt_quint4>(
                            src.compatible_ptr<dt_quint4>(), mat.ptr<dt_float32>(),
                            mat_idx.raw_ptr() ? mat_idx.ptr<int>() : nullptr,
                            dst.compatible_ptr<dt_quint4>(), src.layout[0],
                            mat.layout[0], C, IH, IW, OH, OW,
                            static_cast<dt_quint4>(bval), bmode,
                            async_error_info(handle()), m_error_tracker, stream);
                    if (param().format == Param::Format::NCHW) {
                        auto relayout_opr = handle()->create_operator<RelayoutFormat>();
                        RelayoutFormat::Param trans_param;
                        trans_param.mode = RelayoutFormat::Param::Mode::NCHW64_NCHW;
                        trans_param.oc = sdst.layout[1];
                        relayout_opr->param() = trans_param;
                        relayout_opr->exec(dst, sdst, {});
                    }
                } else if (
                        (src.layout.dtype.enumv() == DTypeEnum::QuantizedS4 ||
                         src.layout.dtype.enumv() == DTypeEnum::Quantized4Asymm) &&
                        (param().format == Param::Format::NHWC)) {
                    constexpr int pack_c = 8;
                    megdnn_assert(C % pack_c == 0);
                    bval = roundf(bval);
                    if (src.layout.dtype.enumv() == DTypeEnum::QuantizedS4) {
                        bval = fmin(fmax(-8.f, bval), 7.f);
                        if (C % 16 == 0) {
                            warp_perspective::forward_proxy_nhwc_bit4<dt_qint4, 16>(
                                    src.ptr<dt_qint4>(), mat.ptr<dt_float32>(),
                                    mat_idx.raw_ptr() ? mat_idx.ptr<int>() : nullptr,
                                    dst.ptr<dt_qint4>(), src.layout[0], mat.layout[0],
                                    C, IH, IW, OH, OW, static_cast<dt_qint4>(bval),
                                    bmode, async_error_info(handle()), m_error_tracker,
                                    stream);
                        } else {
                            warp_perspective::forward_proxy_nhwc_bit4<dt_qint4, pack_c>(
                                    src.ptr<dt_qint4>(), mat.ptr<dt_float32>(),
                                    mat_idx.raw_ptr() ? mat_idx.ptr<int>() : nullptr,
                                    dst.ptr<dt_qint4>(), src.layout[0], mat.layout[0],
                                    C, IH, IW, OH, OW, static_cast<dt_qint4>(bval),
                                    bmode, async_error_info(handle()), m_error_tracker,
                                    stream);
                        }
                    } else {
                        bval = fmin(fmax(0.f, bval), 15.f);
                        if (C % 16 == 0) {
                            warp_perspective::forward_proxy_nhwc_bit4<dt_quint4, 16>(
                                    src.ptr<dt_quint4>(), mat.ptr<dt_float32>(),
                                    mat_idx.raw_ptr() ? mat_idx.ptr<int>() : nullptr,
                                    dst.ptr<dt_quint4>(), src.layout[0], mat.layout[0],
                                    C, IH, IW, OH, OW, static_cast<dt_quint4>(bval),
                                    bmode, async_error_info(handle()), m_error_tracker,
                                    stream);
                        } else {
                            warp_perspective::forward_proxy_nhwc_bit4<
                                    dt_quint4, pack_c>(
                                    src.ptr<dt_quint4>(), mat.ptr<dt_float32>(),
                                    mat_idx.raw_ptr() ? mat_idx.ptr<int>() : nullptr,
                                    dst.ptr<dt_quint4>(), src.layout[0], mat.layout[0],
                                    C, IH, IW, OH, OW, static_cast<dt_quint4>(bval),
                                    bmode, async_error_info(handle()), m_error_tracker,
                                    stream);
                        }
                    }
                }
            } else if ((src.layout.dtype.enumv() == DTypeEnum::Quantized8Asymm ||
                        src.layout.dtype.enumv() == DTypeEnum::Uint8)) {
                uint8_t zero_point = 0;
                float scale = 1.f;
                if (src.layout.dtype.enumv() == DTypeEnum::Quantized8Asymm) {
                    zero_point =
                            src.layout.dtype.param<dtype::Quantized8Asymm>().zero_point;
                    scale = src.layout.dtype.param<dtype::Quantized8Asymm>().scale;
                } else if (
                        src.layout.dtype.enumv() == DTypeEnum::Uint8 &&
                        dst.layout.dtype.enumv() == DTypeEnum::QuantizedS8) {
                    zero_point = 128;
                    scale = 1.f;
                }
                DTypeParamImpl<dt_quint8> src_dtype_param(scale, zero_point);

                if ((dst.layout.dtype.enumv() == DTypeEnum::QuantizedS8 &&
                     dst.layout.dtype.param<dtype::QuantizedS8>().scale == scale) &&
                    ((param().format == Param::Format::NCHW_NCHW4_IC_SMALL) ||
                     (param().format == Param::Format::NHWC_NCHW4_IC_SMALL))) {
                    bool is_nhwc_ic_small =
                            (param().format == Param::Format::NHWC_NCHW4_IC_SMALL);
                    warp_perspective::forward_proxy_quint8_dimshuffle_typecvt_nchw4<
                            dt_quint8, dt_uint8, dt_int8>(
                            is_nhwc_ic_small, src.compatible_ptr<dt_uint8>(),
                            mat.ptr<dt_float32>(),
                            mat_idx.raw_ptr() ? mat_idx.ptr<int>() : nullptr,
                            dst.compatible_ptr<dt_int8>(), src.layout[0], mat.layout[0],
                            C, IH, IW, OH, OW, bval, src_dtype_param, bmode,
                            async_error_info(handle()), m_error_tracker, stream);
                } else {
                    megdnn_assert(
                            ((dst.layout.dtype.enumv() == DTypeEnum::Float32) &&
                             ((param().format == Param::Format::NCHW) ||
                              (param().format == Param::Format::NHWC_NCHW))),
                            "invalid format for Quantized8Asymm input");
                    bool is_nhwc = (param().format == Param::Format::NHWC_NCHW);
                    warp_perspective::forward_proxy_quint8_dimshuffle_typecvt_nchw<
                            dt_quint8, dt_uint8, dt_float32>(
                            is_nhwc, src.compatible_ptr<dt_uint8>(),
                            mat.ptr<dt_float32>(),
                            mat_idx.raw_ptr() ? mat_idx.ptr<int>() : nullptr,
                            dst.compatible_ptr<dt_float32>(), src.layout[0],
                            mat.layout[0], C, IH, IW, OH, OW, bval, src_dtype_param,
                            bmode, async_error_info(handle()), m_error_tracker, stream);
                }
            } else {
                megdnn_throw(
                        ssprintf("unsupported dtype: %s", src.layout.dtype.name()));
            }
        }
    }
    if (ssrc.layout.dtype.enumv() == DTypeTrait<dtype::BFloat16>::enumv) {
        ctypecvt.comp_to_dst_type(dst, sdst);
    }
}

void WarpPerspectiveForwardImpl::exec(
        _megdnn_in const TensorNDArray& ssrcs, _megdnn_tensor_in smat,
        _megdnn_tensor_in smat_idx, _megdnn_tensor_out sdst,
        _megdnn_workspace sworkspace) {
    TensorLayoutArray ssrcs_layout;
    for (auto&& s : ssrcs) {
        ssrcs_layout.push_back(s.layout);
    }
    check_exec_allow_nhwc_mat_idx(
            ssrcs_layout, smat.layout, smat_idx.layout, sdst.layout, sworkspace.size);

    TensorNDArray srcs = ssrcs;
    TensorND mat = smat;
    TensorND mat_idx = smat_idx;
    TensorND dst = sdst;
    Param::Format inner_format = param().format;
    auto bundle = get_workspace_bundle(
            sworkspace.raw_ptr, ssrcs_layout, smat.layout, smat_idx.layout,
            sdst.layout);
    auto ctypecvt = CompTypeCvter<dtype::BFloat16, dtype::Float32>(
            concrete_handle(this->handle()), &bundle);
    if (ssrcs.front().layout.dtype.enumv() == DTypeTrait<dtype::BFloat16>::enumv) {
        for (size_t i = 0; i < ssrcs.size(); i++) {
            ctypecvt.src_to_comp_type(ssrcs[i], srcs[i]);
        }
        ctypecvt.src_to_comp_type(smat, mat).src_to_comp_type(sdst, dst);
    }

    {
        auto stream = cuda_stream(this->handle());
        bool is_nhwc = inner_format == param::WarpPerspective::Format::NHWC;
        TensorND src = srcs.front();
        megdnn_assert(warp::is_dnn_available(
                ssrcs_layout, mat.layout, dst.layout, param().imode, inner_format));
        size_t C, IH, IW, OH, OW;
        if (is_nhwc) {
            C = src.layout.shape[3];
            IH = src.layout.shape[1];
            IW = src.layout.shape[2];
            OH = dst.layout.shape[1];
            OW = dst.layout.shape[2];
        } else {
            megdnn_assert(
                    inner_format == param::WarpPerspective::Format::NCHW,
                    "invalid warp_perspective format");
            C = src.layout.shape[1];
            IH = src.layout.shape[2];
            IW = src.layout.shape[3];
            OH = dst.layout.shape[2];
            OW = dst.layout.shape[3];
        }
        megdnn_assert(
                param().imode == Param::InterpolationMode::LINEAR,
                "unsupported interpolation mode form NCHW format");
        auto bval = param().border_val;
        auto bmode = warp_perspective::get_bmode(param().bmode);
        if (src.layout.dtype == dst.layout.dtype) {
            if (src.layout.dtype == dtype::Float32{}) {
                SmallVector<size_t> workspace_sizes{sizeof(dt_float32*) * srcs.size()};
                WorkspaceBundle workspace_cpu(nullptr, workspace_sizes);
                auto total_workspace_size = workspace_cpu.total_size_in_bytes();
                void* workspace_cpu_raw = malloc(total_workspace_size);
                workspace_cpu = WorkspaceBundle(workspace_cpu_raw, workspace_sizes);
                auto srcs_cpu = static_cast<const dt_float32**>(workspace_cpu.get(0));
                size_t i =
                        is_nhwc ? bundle.nr_workspace() - 2 : bundle.nr_workspace() - 1;
                auto srcs_gpu = static_cast<const dt_float32**>(bundle.get(i));
                for (size_t i = 0; i < srcs.size(); ++i) {
                    srcs_cpu[i] = srcs[i].ptr<dt_float32>();
                }
                cuda_check(cudaMemcpyAsync(
                        bundle.get(i), workspace_cpu.get(0), workspace_cpu.get_size(0),
                        cudaMemcpyHostToDevice, stream));
                cuda_check(cudaStreamAddCallback(
                        stream, callback_free, static_cast<void*>(workspace_cpu_raw),
                        0));
                warp_perspective::forward_proxy_multi_src(
                        is_nhwc, srcs_gpu, mat.ptr<dt_float32>(),
                        mat_idx.raw_ptr() ? mat_idx.ptr<int>() : nullptr,
                        dst.ptr<dt_float32>(), srcs.size(), mat.layout[0], C, IH, IW,
                        OH, OW, bval, bmode, async_error_info(handle()),
                        m_error_tracker, stream);
            } else if (DNN_FLOAT16_SELECT(
                               src.layout.dtype == dtype::Float16(), false)) {
#ifndef MEGDNN_DISABLE_FLOAT16
                SmallVector<size_t> workspace_sizes{sizeof(dt_float16*) * srcs.size()};
                WorkspaceBundle workspace_cpu(nullptr, workspace_sizes);
                auto total_workspace_size = workspace_cpu.total_size_in_bytes();
                void* workspace_cpu_raw = malloc(total_workspace_size);
                workspace_cpu = WorkspaceBundle(workspace_cpu_raw, workspace_sizes);
                auto srcs_cpu = static_cast<const dt_float16**>(workspace_cpu.get(0));
                auto srcs_gpu = static_cast<const dt_float16**>(bundle.get(0));
                for (size_t i = 0; i < srcs.size(); ++i) {
                    srcs_cpu[i] = srcs[i].ptr<dt_float16>();
                }
                cuda_check(cudaMemcpyAsync(
                        bundle.get(0), workspace_cpu.get(0), workspace_cpu.get_size(0),
                        cudaMemcpyHostToDevice, stream));
                cuda_check(cudaStreamAddCallback(
                        stream, callback_free, static_cast<void*>(workspace_cpu_raw),
                        0));
                warp_perspective::forward_proxy_multi_src(
                        is_nhwc, srcs_gpu, mat.ptr<dt_float32>(),
                        mat_idx.raw_ptr() ? mat_idx.ptr<int>() : nullptr,
                        dst.ptr<dt_float16>(), srcs.size(), mat.layout[0], C, IH, IW,
                        OH, OW, static_cast<dt_float16>(bval), bmode,
                        async_error_info(handle()), m_error_tracker, stream);
#endif
            }
        } else {
            megdnn_throw(ssprintf("unsupported dtype: %s", src.layout.dtype.name()));
        }
    }
    if (ssrcs.front().layout.dtype.enumv() == DTypeTrait<dtype::BFloat16>::enumv) {
        ctypecvt.comp_to_dst_type(dst, sdst);
    }
}

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
