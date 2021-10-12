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

// vim: syntax=cpp.doxygen
