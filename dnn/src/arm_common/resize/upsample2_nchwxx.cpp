/**
 * \file dnn/src/arm_common/resize/upsample2_nchwxx.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/arm_common/resize/upsample2_nchwxx.h"

#include "src/arm_common/resize/helper.h"
#include "src/arm_common/simd_macro/marm_neon.h"

using namespace megdnn;
using namespace arm_common;
using namespace resize;

namespace {

template <typename simd_helper, size_t fh, size_t fw>
static inline typename simd_helper::simd_type compute_linear_element(
        const typename simd_helper::simd_type src[4],
        const typename simd_helper::simd_type alpha[2][2]) {
    typename simd_helper::simd_type c = simd_helper::dup(0);
    c = simd_helper::fma(c, src[0], alpha[0 ^ fh][0 ^ fw]);
    c = simd_helper::fma(c, src[1], alpha[0 ^ fh][1 ^ fw]);
    c = simd_helper::fma(c, src[2], alpha[1 ^ fh][0 ^ fw]);
    c = simd_helper::fma(c, src[3], alpha[1 ^ fh][1 ^ fw]);
    return c;
}

template <typename simd_helper, bool has_right, bool has_bottom>
static inline void compute_linear_2x2_element(
        const typename simd_helper::ctype* src, typename simd_helper::ctype* dst,
        size_t IW, size_t OW, const typename simd_helper::simd_type alpha[2][2]) {
    constexpr size_t PC = simd_helper::simd_width;
    const typename simd_helper::ctype* src_ptr[4] = {src, src, src, src};

    if (has_right) {
        src_ptr[1] += PC;
        src_ptr[3] += PC;
    }
    if (has_bottom) {
        src_ptr[2] += IW * PC;
        src_ptr[3] += IW * PC;
    }

    typename simd_helper::simd_type rsrc[4];
    rsrc[0] = simd_helper::load(src_ptr[0]);
    rsrc[1] = simd_helper::load(src_ptr[1]);
    rsrc[2] = simd_helper::load(src_ptr[2]);
    rsrc[3] = simd_helper::load(src_ptr[3]);

    typename simd_helper::simd_type rdst[4];
    rdst[0] = compute_linear_element<simd_helper, 0, 0>(rsrc, alpha);
    rdst[1] = compute_linear_element<simd_helper, 0, 1>(rsrc, alpha);
    rdst[2] = compute_linear_element<simd_helper, 1, 0>(rsrc, alpha);
    rdst[3] = compute_linear_element<simd_helper, 1, 1>(rsrc, alpha);

    simd_helper::store(dst, rdst[0]);
    if (has_right) {
        simd_helper::store(dst + PC, rdst[1]);
    }
    if (has_bottom) {
        simd_helper::store(dst + OW * PC, rdst[2]);
    }
    if (has_right && has_bottom) {
        simd_helper::store(dst + (OW + 1) * PC, rdst[3]);
    }
}

template <typename ctype>
void linear_upsample2_nchwxx(
        const ctype* src_ptr, ctype* dst_ptr, size_t N, size_t IH, size_t IW) {
    using simd_helper = SIMDHelper<ctype>;
    size_t OW = IW * 2;
    constexpr size_t PC = simd_helper::simd_width;

    typename simd_helper::simd_type alpha[2][2];
    alpha[0][0] = simd_helper::dup(0.75 * 0.75);
    alpha[0][1] = simd_helper::dup(0.75 * 0.25);
    alpha[1][0] = simd_helper::dup(0.25 * 0.75);
    alpha[1][1] = simd_helper::dup(0.25 * 0.25);

    for (size_t i = 0; i < N; ++i) {
        compute_linear_2x2_element<simd_helper, false, false>(
                src_ptr, dst_ptr, IW, OW, alpha);

        {
            for (size_t iw = 0; iw + 1 < IW; ++iw) {
                compute_linear_2x2_element<simd_helper, true, false>(
                        src_ptr + iw * PC, dst_ptr + (iw * 2 + 1) * PC, IW, OW, alpha);
            }
        }
        compute_linear_2x2_element<simd_helper, false, false>(
                src_ptr + (IW - 1) * PC, dst_ptr + (OW - 1) * PC, IW, OW, alpha);
        dst_ptr += OW * PC;

        for (size_t ih = 0; ih + 1 < IH; ++ih) {
            compute_linear_2x2_element<simd_helper, false, true>(
                    src_ptr, dst_ptr, IW, OW, alpha);
            for (size_t iw = 0; iw + 1 < IW; ++iw) {
                compute_linear_2x2_element<simd_helper, true, true>(
                        src_ptr + iw * PC, dst_ptr + (iw * 2 + 1) * PC, IW, OW, alpha);
            }
            compute_linear_2x2_element<simd_helper, false, true>(
                    src_ptr + (IW - 1) * PC, dst_ptr + (OW - 1) * PC, IW, OW, alpha);

            src_ptr += IW * PC;
            dst_ptr += 2 * OW * PC;
        }

        compute_linear_2x2_element<simd_helper, false, false>(
                src_ptr, dst_ptr, IW, OW, alpha);
        {
            for (size_t iw = 0; iw + 1 < IW; ++iw) {
                compute_linear_2x2_element<simd_helper, true, false>(
                        src_ptr + iw * PC, dst_ptr + (iw * 2 + 1) * PC, IW, OW, alpha);
            }
        }

        compute_linear_2x2_element<simd_helper, false, false>(
                src_ptr + (IW - 1) * PC, dst_ptr + (OW - 1) * PC, IW, OW, alpha);
        src_ptr += IW * PC;
        dst_ptr += OW * PC;
    }
}

template <typename ctype>
void nearest_upsample2_nchwxx(
        const ctype* src_ptr, ctype* dst_ptr, size_t N, size_t IH, size_t IW) {
    using simd_helper = SIMDHelper<ctype>;
    size_t OW = IW * 2;
    constexpr size_t PC = simd_helper::simd_width;

    for (size_t i = 0; i < N; ++i) {
        for (size_t ih = 0; ih < IH; ++ih) {
            for (size_t iw = 0; iw < IW; ++iw) {
                typename simd_helper::simd_type r0 =
                        simd_helper::load(src_ptr + iw * PC);

                simd_helper::store(dst_ptr + (iw * 2) * PC, r0);
                simd_helper::store(dst_ptr + (iw * 2 + 1) * PC, r0);
                simd_helper::store(dst_ptr + (OW + iw * 2) * PC, r0);
                simd_helper::store(dst_ptr + (OW + iw * 2 + 1) * PC, r0);
            }
            src_ptr += IW * PC;
            dst_ptr += 2 * OW * PC;
        }
    }
}
}  // namespace

void megdnn::arm_common::resize_linear_upsample2_nchw44_fp32(
        const ResizeImpl::KernParam<float>& kern_param) {
    linear_upsample2_nchwxx(
            kern_param.sptr, kern_param.dptr, kern_param.n * kern_param.c / 4,
            kern_param.ih, kern_param.iw);
}

void megdnn::arm_common::resize_nearest_upsample2_nchw44_fp32(
        const ResizeImpl::KernParam<float>& kern_param) {
    nearest_upsample2_nchwxx(
            kern_param.sptr, kern_param.dptr, kern_param.n * kern_param.c / 4,
            kern_param.ih, kern_param.iw);
}

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

void megdnn::arm_common::resize_linear_upsample2_nchw88_fp16(
        const ResizeImpl::KernParam<dt_float16>& kern_param) {
    auto sptr = reinterpret_cast<const __fp16*>(kern_param.sptr);
    auto dptr = reinterpret_cast<__fp16*>(kern_param.dptr);
    linear_upsample2_nchwxx(
            sptr, dptr, kern_param.n * kern_param.c / 8, kern_param.ih, kern_param.iw);
}

void megdnn::arm_common::resize_nearest_upsample2_nchw88_fp16(
        const ResizeImpl::KernParam<dt_float16>& kern_param) {
    auto sptr = reinterpret_cast<const __fp16*>(kern_param.sptr);
    auto dptr = reinterpret_cast<__fp16*>(kern_param.dptr);
    nearest_upsample2_nchwxx(
            sptr, dptr, kern_param.n * kern_param.c / 8, kern_param.ih, kern_param.iw);
}

#endif
