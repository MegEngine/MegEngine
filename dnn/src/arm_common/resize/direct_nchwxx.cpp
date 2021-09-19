/**
 * \file dnn/src/arm_common/resize/direct_nchwxx.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/arm_common/resize/direct_nchwxx.h"

#include "src/arm_common/resize/helper.h"
#include "src/arm_common/simd_macro/marm_neon.h"

using namespace megdnn;
using namespace arm_common;
using namespace resize;

namespace {

template <typename ctype, InterpolationMode imode>
void resize_direct_nchwxx(const ctype* sptr, ctype* dptr, size_t N, size_t IH,
                          size_t IW, size_t OH, size_t OW) {
    using simd_helper = SIMDHelper<ctype>;
    constexpr size_t PC = simd_helper::simd_width;
    using simd_type = typename simd_helper::simd_type;

    float scale_h = static_cast<float>(OH) / IH;
    float scale_w = static_cast<float>(OW) / IW;

    for (size_t n = 0; n < N; ++n) {
        for (size_t oh = 0; oh < OH; ++oh) {
            for (size_t ow = 0; ow < OW; ++ow) {
                int ih0, ih1, iw0, iw1;
                float ah0, ah1, aw0, aw1;

                std::tie(ah0, ih0, ah1, ih1) =
                        get_nearest_linear_coord(imode, scale_h, IH, oh);
                std::tie(aw0, iw0, aw1, iw1) =
                        get_nearest_linear_coord(imode, scale_w, IW, ow);

                simd_type r0 = simd_helper::load(sptr + (ih0 * IW + iw0) * PC);
                simd_type r1 = simd_helper::load(sptr + (ih0 * IW + iw1) * PC);
                simd_type r2 = simd_helper::load(sptr + (ih1 * IW + iw0) * PC);
                simd_type r3 = simd_helper::load(sptr + (ih1 * IW + iw1) * PC);

                // FIXME: weight fp16 may cause precision problem
                ctype a0 = ah0 * aw0;
                ctype a1 = ah0 * aw1;
                ctype a2 = ah1 * aw0;
                ctype a3 = ah1 * aw1;

                simd_type c = simd_helper::dup(0);
                c = simd_helper::fma(c, r0, a0);
                c = simd_helper::fma(c, r1, a1);
                c = simd_helper::fma(c, r2, a2);
                c = simd_helper::fma(c, r3, a3);

                simd_helper::store(dptr + (oh * OW + ow) * PC, c);
            }
        }
        sptr += IH * IW * PC;
        dptr += OH * OW * PC;
    }
}
}

void megdnn::arm_common::resize_direct_nearest_nchw44_fp32(
        const ResizeImpl::KernParam<float>& kern_param) {
    resize_direct_nchwxx<float, InterpolationMode::INTER_NEAREST>(
            kern_param.sptr, kern_param.dptr, kern_param.n * kern_param.c / 4,
            kern_param.ih, kern_param.iw, kern_param.oh, kern_param.ow);
}

void megdnn::arm_common::resize_direct_linear_nchw44_fp32(
        const ResizeImpl::KernParam<float>& kern_param) {
    resize_direct_nchwxx<float, InterpolationMode::INTER_LINEAR>(
            kern_param.sptr, kern_param.dptr, kern_param.n * kern_param.c / 4,
            kern_param.ih, kern_param.iw, kern_param.oh, kern_param.ow);
}

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

void megdnn::arm_common::resize_direct_nearest_nchw88_fp16(
        const ResizeImpl::KernParam<dt_float16>& kern_param) {
    auto sptr = reinterpret_cast<const __fp16*>(kern_param.sptr);
    auto dptr = reinterpret_cast<__fp16*>(kern_param.dptr);
    resize_direct_nchwxx<__fp16, InterpolationMode::INTER_NEAREST>(
            sptr, dptr, kern_param.n * kern_param.c / 8, kern_param.ih,
            kern_param.iw, kern_param.oh, kern_param.ow);
}

void megdnn::arm_common::resize_direct_linear_nchw88_fp16(
        const ResizeImpl::KernParam<dt_float16>& kern_param) {
    auto sptr = reinterpret_cast<const __fp16*>(kern_param.sptr);
    auto dptr = reinterpret_cast<__fp16*>(kern_param.dptr);
    resize_direct_nchwxx<__fp16, InterpolationMode::INTER_LINEAR>(
            sptr, dptr, kern_param.n * kern_param.c / 8, kern_param.ih,
            kern_param.iw, kern_param.oh, kern_param.ow);
}

#endif
