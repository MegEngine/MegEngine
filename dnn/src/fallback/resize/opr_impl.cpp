/**
 * \file dnn/src/fallback/resize/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/fallback/resize/opr_impl.h"
#include <vector>
#include "src/common/rounding_converter.cuh"
#include "src/fallback/handle.h"

#include "src/fallback/resize/gi/direct_nchwxx.h"
#include "src/fallback/resize/gi/resize_cv.h"
#include "src/fallback/resize/gi/upsample2_nchw.h"
#include "src/fallback/resize/gi/upsample2_nchwxx.h"

#include "midout.h"
MIDOUT_DECL(megdnn_fallback_resize)

using namespace megdnn;
using namespace fallback;

template <typename ctype>
void ResizeImpl::kern_fallback(const KernParam<ctype>& kern_param) {
    if (kern_param.format == Format::NHWC) {
        kern_fallback_nhwc(kern_param);
        return;
    }
    megdnn_assert(kern_param.format == Format::NCHW);

    UNPACK_RESIZE_FWD_KERN_PARAM_WITH_STRIDE(kern_param);
    rounding::RoundingConverter<ctype> output_converter;
    float scale_h = static_cast<float>(OH) / IH;
    float scale_w = static_cast<float>(OW) / IW;

    auto build_table = [this](InterpolationMode imode, float scale, int isize,
                              int osize) {
        std::vector<std::tuple<float, int, float, int>> table;
        rep(i, osize) {
            table.push_back(get_nearest_linear_coord(imode, scale, isize, i));
        }
        return table;
    };

    auto table_h = build_table(kern_param.imode, scale_h, IH, OH);
    auto table_w = build_table(kern_param.imode, scale_w, IW, OW);

    rep(n, N) {
        rep(c, static_cast<int>(C)) {
            rep(oh, OH) {
                float ah0, ah1, aw0, aw1;
                int ih0, ih1, iw0, iw1;

                std::tie(ah0, ih0, ah1, ih1) = table_h[oh];
                rep(ow, OW) {
                    std::tie(aw0, iw0, aw1, iw1) = table_w[ow];
                    dptr[c * OH * OW + oh * OW + ow] = output_converter(
                            sptr[c * S_IC + ih0 * S_IH + iw0 * S_IW] * ah0 * aw0 +
                            sptr[c * S_IC + ih0 * S_IH + iw1 * S_IW] * ah0 * aw1 +
                            sptr[c * S_IC + ih1 * S_IH + iw0 * S_IW] * ah1 * aw0 +
                            sptr[c * S_IC + ih1 * S_IH + iw1 * S_IW] * ah1 * aw1);
                }
            }
        }
        sptr += S_IN;
        dptr += C * OH * OW;
    }
}

template <typename ctype>
void ResizeImpl::kern_fallback_nhwc(const KernParam<ctype>& kern_param) {
    UNPACK_RESIZE_FWD_KERN_PARAM(kern_param);
    rounding::RoundingConverter<ctype> output_converter;
    float scale_h = static_cast<float>(OH) / IH;
    float scale_w = static_cast<float>(OW) / IW;

    auto build_table = [this](InterpolationMode imode, float scale, int isize,
                              int osize) {
        std::vector<std::tuple<float, int, float, int>> table;
        rep(i, osize) {
            table.push_back(get_nearest_linear_coord(imode, scale, isize, i));
        }
        return table;
    };
    auto table_h = build_table(kern_param.imode, scale_h, IH, OH);
    auto table_w = build_table(kern_param.imode, scale_w, IW, OW);

    rep(n, N) {
        rep(oh, OH) {
            float ah0, ah1, aw0, aw1;
            int ih0, ih1, iw0, iw1;

            std::tie(ah0, ih0, ah1, ih1) = table_h[oh];
            rep(ow, OW) {
                std::tie(aw0, iw0, aw1, iw1) = table_w[ow];
                rep(c, C) {
                    dptr[(oh * OW + ow) * C + c] = output_converter(
                            sptr[(ih0 * IW + iw0) * C + c] * ah0 * aw0 +
                            sptr[(ih0 * IW + iw1) * C + c] * ah0 * aw1 +
                            sptr[(ih1 * IW + iw0) * C + c] * ah1 * aw0 +
                            sptr[(ih1 * IW + iw1) * C + c] * ah1 * aw1);
                }
            }
        }
        sptr += C * IH * IW;
        dptr += C * OH * OW;
    }
}

void ResizeImpl::exec(
        _megdnn_tensor_in src, _megdnn_tensor_in dst, _megdnn_workspace workspace) {
    check_exec(src.layout, dst.layout, workspace.size);
    exec_gi(src, dst, workspace);
}

void ResizeImpl::exec_fallback(
        _megdnn_tensor_in src, _megdnn_tensor_in dst, _megdnn_workspace workspace) {
    if (param().format == param::Resize::Format::NCHW4 ||
        param().format == param::Resize::Format::NCHW44 ||
        param().format == param::Resize::Format::NCHW88 ||
        (param().format == param::Resize::Format::NCHW &&
         param().imode != param::Resize::InterpolationMode::INTER_LINEAR)) {
        naive::ResizeImpl::exec(src, dst, workspace);
        return;
    }
    if ((param().format == param::Resize::Format::NCHW ||
         (src.layout[3] != 1 && src.layout[3] != 3)) ||
        (param().imode == param::Resize::InterpolationMode::LINEAR)) {
#define cb(dt, ct)                                                   \
    case DTypeTrait<dt>::enumv: {                                    \
        auto kparam = KernParam<ct>::from_tensors(                   \
                param().format, param().imode, src, dst, workspace); \
        MEGDNN_DISPATCH_CPU_KERN_OPR(kern_fallback(kparam));         \
        return;                                                      \
    }

        switch (src.layout.dtype.enumv()) {
            cb(dtype::Float32, float);
            DNN_INC_FLOAT16(cb(dtype::Float16, dt_float16));
            cb(dtype::Int8, int8_t);
            cb(dtype::QuantizedS8, int8_t);
            cb(dtype::Uint8, uint8_t);
            cb(dtype::Quantized8Asymm, uint8_t);
            default:
                megdnn_throw(ssprintf(
                                     "Unsupported input DType in Resize: %s",
                                     src.layout.dtype.name())
                                     .c_str());
                return;
        }

#undef cb
    }

    naive::ResizeImpl::exec(src, dst, workspace);
}

void ResizeImpl::exec_gi(
        _megdnn_tensor_in src, _megdnn_tensor_in dst, _megdnn_workspace workspace) {
    bool is_contiguous = src.layout.is_contiguous() && dst.layout.is_contiguous();
    bool is_dtype_same = src.layout.dtype == dst.layout.dtype;
    bool is_dtype_fp32 = src.layout.dtype == dtype::Float32();
    bool is_dtype_supported = is_dtype_same && is_dtype_fp32;

    bool is_nchw_fp32 = param().format == param::Resize::Format::NCHW && is_dtype_fp32;
    bool is_nchw44_fp32 =
            param().format == param::Resize::Format::NCHW44 && is_dtype_fp32;
    bool is_imode_nearest =
            param().imode == param::Resize::InterpolationMode::INTER_NEAREST;
    bool is_imode_linear =
            param().imode == param::Resize::InterpolationMode::INTER_LINEAR;
    bool is_imode_supported = is_imode_nearest || is_imode_linear;

    bool is_upsample2 = src.layout.shape[2] * 2 == dst.layout.shape[2] &&
                        src.layout.shape[3] * 2 == dst.layout.shape[3];
    bool usable = is_contiguous && is_dtype_supported && is_imode_supported;

    if (param().format == param::Resize::Format::NHWC &&
        (src.layout[3] == 1 || src.layout[3] == 3) && is_nhwc_contig_wc(src.layout) &&
        is_dtype_fp32) {
        MEGDNN_DISPATCH_CPU_KERN_OPR(resize_cv_gi_exec(src, dst, param().imode));
    } else if (!usable) {
        exec_fallback(src, dst, workspace);
    } else if (is_dtype_fp32) {
        auto kern_param = KernParam<float>::from_tensors(
                param().format, param().imode, src, dst, workspace);
        if (is_nchw44_fp32) {
            if (is_upsample2) {
                if (is_imode_nearest) {
                    MIDOUT_BEGIN(megdnn_fallback_resize, midout_iv(0)) {
                        MEGDNN_DISPATCH_CPU_KERN_OPR(
                                resize_nearest_upsample2_nchw44_gi_fp32(kern_param));
                    }
                    MIDOUT_END();
                } else {
                    megdnn_assert(is_imode_linear, "invalid imode");
                    MIDOUT_BEGIN(megdnn_fallback_resize, midout_iv(1)) {
                        MEGDNN_DISPATCH_CPU_KERN_OPR(
                                resize_linear_upsample2_nchw44_gi_fp32(kern_param));
                    }
                    MIDOUT_END();
                }
            } else {
                if (is_imode_nearest) {
                    MIDOUT_BEGIN(megdnn_fallback_resize, midout_iv(2)) {
                        MEGDNN_DISPATCH_CPU_KERN_OPR(
                                resize_direct_nearest_nchw44_gi_fp32(kern_param));
                    }
                    MIDOUT_END();
                } else {
                    megdnn_assert(is_imode_linear, "invalid imode");
                    MIDOUT_BEGIN(megdnn_fallback_resize, midout_iv(3)) {
                        MEGDNN_DISPATCH_CPU_KERN_OPR(
                                resize_direct_linear_nchw44_gi_fp32(kern_param));
                    }
                    MIDOUT_END();
                }
            }
        } else if (is_nchw_fp32) {
            if (is_upsample2) {
                if (is_imode_nearest) {
                    MIDOUT_BEGIN(megdnn_fallback_resize, midout_iv(4)) {
                        MEGDNN_DISPATCH_CPU_KERN_OPR(
                                resize_nearest_upsample2_nchw_gi_fp32(kern_param));
                    }
                    MIDOUT_END();
                } else {
                    megdnn_assert(is_imode_linear, "invalid imode");
                    MIDOUT_BEGIN(megdnn_fallback_resize, midout_iv(5)) {
                        MEGDNN_DISPATCH_CPU_KERN_OPR(
                                resize_linear_upsample2_nchw_gi_fp32(kern_param));
                    }
                    MIDOUT_END();
                }
            } else {
                exec_fallback(src, dst, workspace);
            }
        } else {
            exec_fallback(src, dst, workspace);
        }
    } else {
        exec_fallback(src, dst, workspace);
    }
}
// vim: syntax=cpp.doxygen
// vim: syntax=cpp.doxygen
