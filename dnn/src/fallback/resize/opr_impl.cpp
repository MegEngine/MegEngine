/**
 * \file dnn/src/fallback/resize/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/fallback/resize/opr_impl.h"
#include <vector>
#include "src/fallback/handle.h"
#include "src/common/rounding_converter.cuh"

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

    auto build_table = [this](float scale, int isize,
                              int osize) -> std::vector<std::pair<float, int>> {
        std::vector<std::pair<float, int>> table;
        rep(i, osize) { table.push_back(get_origin_coord(scale, isize, i)); }
        return table;
    };

    auto table_h = build_table(scale_h, IH, OH);
    auto table_w = build_table(scale_w, IW, OW);

    rep(n, N) {
        rep(c, static_cast<int>(C)) {
            rep(oh, OH) {
                auto coord_h = table_h[oh];
                float alphah = coord_h.first;
                int ih0 = coord_h.second;
                int ih1 = ih0 + 1;
                rep(ow, OW) {
                    auto coord_w = table_w[ow];
                    float alphaw = coord_w.first;
                    int iw0 = coord_w.second;
                    int iw1 = iw0 + 1;
                    dptr[c * OH * OW + oh * OW + ow] = output_converter(
                            sptr[c * S_IC + ih0 * S_IH + iw0 * S_IW] *
                                    (1.0f - alphaw) * (1.0f - alphah) +
                            sptr[c * S_IC + ih0 * S_IH + iw1 * S_IW] *
                                    alphaw * (1.0f - alphah) +
                            sptr[c * S_IC + ih1 * S_IH + iw0 * S_IW] *
                                    (1.0f - alphaw) * alphah +
                            sptr[c * S_IC + ih1 * S_IH + iw1 * S_IW] *
                                    alphaw * alphah);
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

    auto build_table = [this](float scale, int isize,
                              int osize) -> std::vector<std::pair<float, int>> {
        std::vector<std::pair<float, int>> table;
        rep(i, osize) { table.push_back(get_origin_coord(scale, isize, i)); }
        return table;
    };
    auto table_h = build_table(scale_h, IH, OH);
    auto table_w = build_table(scale_w, IW, OW);

    rep(n, N) {
        rep(oh, OH) {
            auto coord_h = table_h[oh];
            float alphah = coord_h.first;
            int ih0 = coord_h.second;
            int ih1 = ih0 + 1;
            rep(ow, OW) {
                auto coord_w = table_w[ow];
                float alphaw = coord_w.first;
                int iw0 = coord_w.second;
                int iw1 = iw0 + 1;
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
            }
        }
        sptr += C * IH * IW;
        dptr += C * OH * OW;
    }
}

void ResizeImpl::exec(_megdnn_tensor_in src, _megdnn_tensor_in dst,
                      _megdnn_workspace workspace) {
    check_exec(src.layout, dst.layout, workspace.size);
    if (param().format == param::Resize::Format::NCHW4) {
        naive::ResizeImpl::exec(src, dst, workspace);
        return;
    }
    if ((param().format == param::Resize::Format::NCHW ||
         (src.layout[3] != 1 && src.layout[3] != 3)) ||
        (param().imode == param::Resize::InterpolationMode::LINEAR)) {
#define cb(dt, ct)                                                          \
    case DTypeTrait<dt>::enumv: {                                           \
        auto kparam = KernParam<ct>::from_tensors(param().format, src, dst, \
                                                  workspace);               \
        MEGDNN_DISPATCH_CPU_KERN_OPR(kern_fallback(kparam));                \
        return;                                                             \
    }

        switch (src.layout.dtype.enumv()) {
            cb(dtype::Float32, float);
            MEGDNN_INC_FLOAT16(cb(dtype::Float16, dt_float16));
            cb(dtype::Int8, int8_t);
            cb(dtype::QuantizedS8, int8_t);
            cb(dtype::Uint8, uint8_t);
            cb(dtype::Quantized8Asymm, uint8_t);
            default:
                megdnn_throw(
                        ssprintf("Unsupported input DType in Resize: %s",
                                 src.layout.dtype.name())
                                .c_str());
                return;
        }

#undef cb
    }

    naive::ResizeImpl::exec(src, dst, workspace);
}

// vim: syntax=cpp.doxygen
