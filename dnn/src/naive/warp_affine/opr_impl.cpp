/**
 * \file dnn/src/naive/warp_affine/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/common/warp_common.h"
#include "src/naive/handle.h"
#include "src/naive/warp_affine/opr_impl.h"
#include "src/naive/warp_affine/warp_affine_cv.h"

#include "midout.h"

MIDOUT_DECL(megdnn_naive_warpaffine)
MIDOUT_DECL(megdnn_naive_warpaffine_dtype)

using namespace megdnn;
using namespace naive;

template <typename ctype, typename mtype>
void WarpAffineImpl::kern_naive(const KernParam<ctype, mtype>& kern_param,
                                size_t task_id) {
    if (kern_param.format == Format::NHWC) {
        kern_naive_nhwc(kern_param, task_id);
        return;
    } else if (kern_param.format == Format::NHWCD4) {
        kern_naive_nhwcd4(kern_param, task_id);
        return;
    }

    UNPACK_WARP_AFFINE_FWD_KERN_PARAM(kern_param);
    MEGDNN_MARK_USED_VAR(N_SRC);
    MEGDNN_MARK_USED_VAR(N_MAT);
    rounding::RoundingConverter<ctype> output_converter;
    auto bmode = param().border_mode;
    auto border_val = param().border_val;
    size_t n = task_id / OH;
    size_t oh = task_id % OH;
    mptr += n * 2 * 3;
    dptr += n * C * OH * OW;
    sptr += n * C * IH * IW;

    rep(ow, OW) {
        float alphaw = mptr[0] * ow + mptr[1] * oh + mptr[2];
        float alphah = mptr[3] * ow + mptr[4] * oh + mptr[5];

        int iw0 = get_real_coord(std::floor(alphaw) + 0, IW);
        int iw1 = get_real_coord(std::floor(alphaw) + 1, IW);
        int ih0 = get_real_coord(std::floor(alphah) + 0, IH);
        int ih1 = get_real_coord(std::floor(alphah) + 1, IH);

        alphaw -= floor(alphaw);
        alphah -= floor(alphah);
        if (bmode != BorderMode::CONSTANT) {
            rep(c, C) {
                dptr[c * OH * OW + oh * OW + ow] = output_converter(
                        sptr[c * IH * IW + ih0 * IW + iw0] * (1.0f - alphaw) *
                                (1.0f - alphah) +
                        sptr[c * IH * IW + ih0 * IW + iw1] * alphaw *
                                (1.0f - alphah) +
                        sptr[c * IH * IW + ih1 * IW + iw0] * (1.0f - alphaw) *
                                alphah +
                        sptr[c * IH * IW + ih1 * IW + iw1] * alphaw * alphah);
            }
        } else {
            rep(c, C) {
                const float b = border_val;
                auto val = (ih0 != -1 && iw0 != -1
                                    ? sptr[c * IH * IW + ih0 * IW + iw0]
                                    : b) *
                                   (1.0f - alphaw) * (1.0f - alphah) +
                           (ih0 != -1 && iw1 != -1
                                    ? sptr[c * IH * IW + ih0 * IW + iw1]
                                    : b) *
                                   alphaw * (1.0f - alphah) +
                           (ih1 != -1 && iw0 != -1
                                    ? sptr[c * IH * IW + ih1 * IW + iw0]
                                    : b) *
                                   (1.0f - alphaw) * alphah +
                           (ih1 != -1 && iw1 != -1
                                    ? sptr[c * IH * IW + ih1 * IW + iw1]
                                    : b) *
                                   alphaw * alphah;
                dptr[c * OH * OW + oh * OW + ow] =
                        output_converter(std::isfinite(val) ? val : b);
            }
        }
    }
}

template <typename ctype, typename mtype>
void WarpAffineImpl::kern_naive_nhwcd4(
        const KernParam<ctype, mtype>& kern_param, size_t task_id) {
    UNPACK_WARP_AFFINE_FWD_KERN_PARAM(kern_param);
    MEGDNN_MARK_USED_VAR(N_SRC);
    MEGDNN_MARK_USED_VAR(N_MAT);
    rounding::RoundingConverter<ctype> output_converter;
    auto bmode = param().border_mode;
    auto border_val = param().border_val;
    size_t n = task_id / OH;
    size_t oh = task_id % OH;
    mptr += n * 2 * 3;
    dptr += n * C * OH * OW * 4;
    sptr += n * C * IH * IW * 4;
    rep(ow, OW) {
        float alphaw = mptr[0] * ow + mptr[1] * oh + mptr[2];
        float alphah = mptr[3] * ow + mptr[4] * oh + mptr[5];
        int iw0 = get_real_coord(std::floor(alphaw) + 0, IW);
        int iw1 = get_real_coord(std::floor(alphaw) + 1, IW);
        int ih0 = get_real_coord(std::floor(alphah) + 0, IH);
        int ih1 = get_real_coord(std::floor(alphah) + 1, IH);
        alphaw -= floor(alphaw);
        alphah -= floor(alphah);
        if (bmode != BorderMode::CONSTANT) {
            rep(c, C) {
                for (int i = 0; i < 4; i++) {
                    dptr[((oh * C + c) * OW + ow) * 4 + i] = output_converter(
                            sptr[((ih0 * C + c) * IW + iw0) * 4 + i] *
                                    (1.0f - alphaw) * (1.0f - alphah) +
                            sptr[((ih0 * C + c) * IW + iw1) * 4 + i] * alphaw *
                                    (1.0f - alphah) +
                            sptr[((ih1 * C + c) * IW + iw0) * 4 + i] *
                                    (1.0f - alphaw) * alphah +
                            sptr[((ih1 * C + c) * IW + iw1) * 4 + i] * alphaw *
                                    alphah);
                }
            }
        } else {
            rep(c, C) {
                const float b = border_val;
                for (int i = 0; i < 4; i++) {
                    auto val =
                            (ih0 != -1 && iw0 != -1
                                     ? sptr[(((ih0 * C + c) * IW + iw0)) * 4 +
                                            i]
                                     : b) *
                                    (1.0f - alphaw) * (1.0f - alphah) +
                            (ih0 != -1 && iw1 != -1
                                     ? sptr[((ih0 * C + c) * IW + iw1) * 4 + i]
                                     : b) *
                                    alphaw * (1.0f - alphah) +
                            (ih1 != -1 && iw0 != -1
                                     ? sptr[((ih1 * C + c) * IW + iw0) * 4 + i]
                                     : b) *
                                    (1.0f - alphaw) * alphah +
                            (ih1 != -1 && iw1 != -1
                                     ? sptr[((ih1 * C + c) * IW + iw1) * 4 + i]
                                     : b) *
                                    alphaw * alphah;
                    dptr[((oh * C + c) * OW + ow) * 4 + i] =
                            output_converter(std::isfinite(val) ? val : b);
                }
            }
        }
    }
}

template <typename ctype, typename mtype>
void WarpAffineImpl::kern_naive_nhwc(const KernParam<ctype, mtype>& kern_param,
                                     size_t task_id) {
    UNPACK_WARP_AFFINE_FWD_KERN_PARAM(kern_param);
    MEGDNN_MARK_USED_VAR(N_SRC);
    MEGDNN_MARK_USED_VAR(N_MAT);
    rounding::RoundingConverter<ctype> output_converter;
    auto bmode = param().border_mode;
    auto border_val = param().border_val;
    size_t n = task_id / OH;
    size_t oh = task_id % OH;
    mptr += n * 2 * 3;
    dptr += n * C * OH * OW;
    sptr += n * C * IH * IW;
    rep(ow, OW) {
        float alphaw = mptr[0] * ow + mptr[1] * oh + mptr[2];
        float alphah = mptr[3] * ow + mptr[4] * oh + mptr[5];

        int iw0 = get_real_coord(std::floor(alphaw) + 0, IW);
        int iw1 = get_real_coord(std::floor(alphaw) + 1, IW);
        int ih0 = get_real_coord(std::floor(alphah) + 0, IH);
        int ih1 = get_real_coord(std::floor(alphah) + 1, IH);

        alphaw -= floor(alphaw);
        alphah -= floor(alphah);
        if (bmode != BorderMode::CONSTANT) {
            rep(c, C) {
                dptr[(oh * OW + ow) * C + c] = output_converter(
                        sptr[(ih0 * IW + iw0) * C + c] * (1.0f - alphaw) *
                                (1.0f - alphah) +
                        sptr[(ih0 * IW + iw1) * C + c] * alphaw *
                                (1.0f - alphah) +
                        sptr[(ih1 * IW + iw0) * C + c] * (1.0f - alphaw) *
                                alphah +
                        sptr[(ih1 * IW + iw1) * C + c] * alphaw * alphah);
            }
        } else {
            rep(c, C) {
                const float b = border_val;
                auto val =
                        (ih0 != -1 && iw0 != -1 ? sptr[(ih0 * IW + iw0) * C + c]
                                                : b) *
                                (1.0f - alphaw) * (1.0f - alphah) +
                        (ih0 != -1 && iw1 != -1 ? sptr[(ih0 * IW + iw1) * C + c]
                                                : b) *
                                alphaw * (1.0f - alphah) +
                        (ih1 != -1 && iw0 != -1 ? sptr[(ih1 * IW + iw0) * C + c]
                                                : b) *
                                (1.0f - alphaw) * alphah +
                        (ih1 != -1 && iw1 != -1 ? sptr[(ih1 * IW + iw1) * C + c]
                                                : b) *
                                alphaw * alphah;
                dptr[(oh * OW + ow) * C + c] =
                        output_converter(std::isfinite(val) ? val : b);
            }
        }
    }
}

void WarpAffineImpl::exec(_megdnn_tensor_in src, _megdnn_tensor_in mat,
                          _megdnn_tensor_out dst, _megdnn_workspace workspace) {
    check_exec(src.layout, mat.layout, dst.layout, workspace.size);

    if (warp::is_cv_available(src.layout, mat.layout, dst.layout, param().imode,
                              param().format)) {
        MIDOUT_BEGIN(megdnn_naive_warpaffine, void) {
            warp_affine_cv_exec(src, mat, dst, param().border_val,
                                param().border_mode, param().imode, handle());
        }
        MIDOUT_END();
    } else {
        size_t batch = dst.layout[0];
        size_t oh = dst.layout[1];
        if (param().format == Format::NCHW) {
            oh = dst.layout[2];
        }
        megdnn_assert(warp::is_dnn_available(src.layout, mat.layout, dst.layout,
                                             param().imode, param().format));
        // We currently use floating point for all WarpAffine computation,
        // so even if the input ctype is one of the integer type, mtype should
        // still be float32. However, if the input dtype is one of the floating
        // point type (float16, ...), we should use the same type as the input
        // type.
#define cb(dt, ct, mct, _midout_iv)                                          \
    case DTypeTrait<dt>::enumv: {                                            \
        auto kparam = KernParam<ct, mct>::from_tensors(param().format, src,  \
                                                       mat, dst, workspace); \
        MIDOUT_BEGIN(megdnn_naive_warpaffine_dtype, midout_iv(_midout_iv)) { \
            auto run = [kparam, this](size_t index, size_t) {                \
                kern_naive(kparam, index);                                   \
            };                                                               \
            MEGDNN_DISPATCH_MULTI_THREAD_CPU_KERN_OPR(run, batch* oh);       \
        }                                                                    \
        MIDOUT_END();                                                        \
        return;                                                              \
    }

        switch (src.layout.dtype.enumv()) {
            cb(dtype::Float32, float, float, 0);
            MEGDNN_INC_FLOAT16(cb(dtype::Float16, dt_float16, dt_float16, 1));
            cb(dtype::Int8, int8_t, float, 2);
            cb(dtype::QuantizedS8, int8_t, float, 3);
            cb(dtype::Uint8, uint8_t, float, 4);
            cb(dtype::Quantized8Asymm, uint8_t, float, 5);
            default:
                megdnn_throw(
                        ssprintf("Unsupported input DType in WarpAffine: %s",
                                 src.layout.dtype.name())
                                .c_str());
                return;
        }
#undef cb
    }
}

// vim: syntax=cpp.doxygen
