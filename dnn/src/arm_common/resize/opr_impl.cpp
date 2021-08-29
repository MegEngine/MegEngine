/**
 * \file dnn/src/arm_common/resize/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/arm_common/resize/opr_impl.h"
#include "src/arm_common/handle.h"
#include "src/arm_common/resize/resize_cv.h"
#include "src/arm_common/simd_macro/marm_neon.h"

using namespace megdnn;
using namespace arm_common;

void ResizeImpl::exec(_megdnn_tensor_in src, _megdnn_tensor_in dst,
                      _megdnn_workspace workspace) {
    check_exec(src.layout, dst.layout, workspace.size);

    if (param().format == param::Resize::Format::NCHW44 ||
        param().format == param::Resize::Format::NCHW88) {
        bool is_contiguous =
                src.layout.is_contiguous() && dst.layout.is_contiguous();
        bool dtype_same = src.layout.dtype == dst.layout.dtype;
        bool nchw44_enable = param().format == param::Resize::Format::NCHW44 &&
                             src.layout.dtype == dtype::Float32();
        bool nchw88_enable =
                param().format == param::Resize::Format::NCHW88 &&
                DNN_FLOAT16_SELECT(src.layout.dtype == dtype::Float16(), false);
        bool interp_supported =
                param().imode ==
                        param::Resize::InterpolationMode::INTER_NEAREST ||
                param().imode == param::Resize::InterpolationMode::INTER_LINEAR;
        bool is_upsample2 =
                param().imode ==
                        param::Resize::InterpolationMode::INTER_NEAREST &&
                src.layout.shape[2] * 2 == dst.layout.shape[2] &&
                src.layout.shape[3] * 2 == dst.layout.shape[3];
        bool need_fallback = !is_contiguous || !dtype_same ||
                             !interp_supported ||
                             (!nchw44_enable && !nchw88_enable);

        if (need_fallback) {
            fallback::ResizeImpl::exec(src, dst, workspace);
        } else if (nchw44_enable) {
            auto kern_param = KernParam<float>::from_tensors(
                    param().format, param().imode, src, dst, workspace);
            if (is_upsample2) {
                MEGDNN_DISPATCH_CPU_KERN_OPR(
                        kern_nearest_upsample2_pack_simd_width(src, dst));
            } else {
                MEGDNN_DISPATCH_CPU_KERN_OPR(kern_nchw44_fp32(kern_param));
            }
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
        } else if (nchw88_enable) {
            auto kern_param = KernParam<dt_float16>::from_tensors(
                    param().format, param().imode, src, dst, workspace);
            if (is_upsample2) {
                MEGDNN_DISPATCH_CPU_KERN_OPR(
                        kern_nearest_upsample2_pack_simd_width(src, dst));
            } else {
                MEGDNN_DISPATCH_CPU_KERN_OPR(kern_nchw88_fp16(kern_param));
            }
#endif
        } else {
            fallback::ResizeImpl::exec(src, dst, workspace);
        }
    } else if (param().format == param::Resize::Format::NCHW ||
               (src.layout[3] != 1 && src.layout[3] != 3) ||
               !is_nhwc_contig_wc(src.layout)) {
        fallback::ResizeImpl::exec(src, dst, workspace);
    } else {
        megdnn_assert(param().format == param::Resize::Format::NHWC,
                      "invalid resize format");
        MEGDNN_DISPATCH_CPU_KERN_OPR(resize_cv_exec(src, dst, param().imode));
    }
}

template <typename ctype>
void ResizeImpl::kern_nchw44_fp32(const KernParam<ctype>& kern_param) {
    UNPACK_RESIZE_FWD_KERN_PARAM(kern_param);
    float scale_h = static_cast<float>(OH) / IH;
    float scale_w = static_cast<float>(OW) / IW;

    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C / 4; ++c) {
            for (size_t oh = 0; oh < OH; ++oh) {
                for (size_t ow = 0; ow < OW; ++ow) {
                    int ih0, ih1, iw0, iw1;
                    float ah0, ah1, aw0, aw1;

                    std::tie(ah0, ih0, ah1, ih1) = get_nearest_linear_coord(
                            kern_param.imode, scale_h, IH, oh);
                    std::tie(aw0, iw0, aw1, iw1) = get_nearest_linear_coord(
                            kern_param.imode, scale_w, IW, ow);

#define SRC_ADDRESS(ih, iw) \
    (sptr + n * C * IH * IW + (c * IH * IW + ih * IW + iw) * 4)
#define DST_ADDRESS(oh, ow) \
    (dptr + n * C * OH * OW + (c * OH * OW + oh * OW + ow) * 4)
                    float32x4_t r0 = vld1q_f32(SRC_ADDRESS(ih0, iw0));
                    float32_t a0 = ah0 * aw0;
                    float32x4_t r1 = vld1q_f32(SRC_ADDRESS(ih0, iw1));
                    float32_t a1 = ah0 * aw1;
                    float32x4_t r2 = vld1q_f32(SRC_ADDRESS(ih1, iw0));
                    float32_t a2 = ah1 * aw0;
                    float32x4_t r3 = vld1q_f32(SRC_ADDRESS(ih1, iw1));
                    float32_t a3 = ah1 * aw1;

                    r0 = vmulq_n_f32(r0, a0);
#if defined(__ARM_FEATURE_FMA) && defined(__aarch64__)
                    r0 = vfmaq_n_f32(r0, r1, a1);
                    r0 = vfmaq_n_f32(r0, r2, a2);
                    r0 = vfmaq_n_f32(r0, r3, a3);
#else
                    r0 = vaddq_f32(r0, vmulq_n_f32(r1, a1));
                    r0 = vaddq_f32(r0, vmulq_n_f32(r2, a2));
                    r0 = vaddq_f32(r0, vmulq_n_f32(r3, a3));
#endif

                    vst1q_f32(DST_ADDRESS(oh, ow), r0);
#undef SRC_ADDRESS
#undef DST_ADDRESS
                }
            }
        }
    }
}

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
template <typename ctype>
void ResizeImpl::kern_nchw88_fp16(const KernParam<ctype>& kern_param) {
    UNPACK_RESIZE_FWD_KERN_PARAM(kern_param);
    float scale_h = static_cast<float>(OH) / IH;
    float scale_w = static_cast<float>(OW) / IW;
    const float16_t* src_ptr = reinterpret_cast<float16_t*>(sptr);
    float16_t* dst_ptr = reinterpret_cast<float16_t*>(dptr);

    for (size_t n = 0; n < N; ++n) {
        for (size_t c = 0; c < C / 8; ++c) {
            for (size_t oh = 0; oh < OH; ++oh) {
                for (size_t ow = 0; ow < OW; ++ow) {
                    int ih0, ih1, iw0, iw1;
                    float ah0, ah1, aw0, aw1;

                    std::tie(ah0, ih0, ah1, ih1) = get_nearest_linear_coord(
                            kern_param.imode, scale_h, IH, oh);
                    std::tie(aw0, iw0, aw1, iw1) = get_nearest_linear_coord(
                            kern_param.imode, scale_w, IW, ow);

#define SRC_ADDRESS(ih, iw) \
    (src_ptr + n * C * IH * IW + (c * IH * IW + ih * IW + iw) * 8)
#define DST_ADDRESS(oh, ow) \
    (dst_ptr + n * C * OH * OW + (c * OH * OW + oh * OW + ow) * 8)
                    float16x8_t r0 = vld1q_f16(SRC_ADDRESS(ih0, iw0));
                    float32_t a0 = ah0 * aw0;
                    float16x8_t r1 = vld1q_f16(SRC_ADDRESS(ih0, iw1));
                    float32_t a1 = ah0 * aw1;
                    float16x8_t r2 = vld1q_f16(SRC_ADDRESS(ih1, iw0));
                    float32_t a2 = ah1 * aw0;
                    float16x8_t r3 = vld1q_f16(SRC_ADDRESS(ih1, iw1));
                    float32_t a3 = ah1 * aw1;

                    r0 = vmulq_n_f16(r0, a0);
#if defined(__ARM_FEATURE_FMA) && defined(__aarch64__)
                    r0 = vfmaq_n_f16(r0, r1, a1);
                    r0 = vfmaq_n_f16(r0, r2, a2);
                    r0 = vfmaq_n_f16(r0, r3, a3);
#else
                    r0 = vaddq_f16(r0, vmulq_n_f16(r1, a1));
                    r0 = vaddq_f16(r0, vmulq_n_f16(r2, a2));
                    r0 = vaddq_f16(r0, vmulq_n_f16(r3, a3));
#endif

                    vst1q_f16(DST_ADDRESS(oh, ow), r0);
#undef SRC_ADDRESS
#undef DST_ADDRESS
                }
            }
        }
    }
}
#endif

void ResizeImpl::kern_nearest_upsample2_pack_simd_width(
        _megdnn_tensor_in src, _megdnn_tensor_out dst) {
    const uint8_t* src_ptr = reinterpret_cast<uint8_t*>(src.raw_ptr);
    uint8_t* dst_ptr = reinterpret_cast<uint8_t*>(dst.raw_ptr);

    size_t S = 2;
    size_t N = src.layout.shape[0];
    size_t IC = src.layout.shape[1];
    size_t IH = src.layout.shape[2];
    size_t IW = src.layout.shape[3];
    size_t OH = dst.layout.shape[2];
    size_t OW = dst.layout.shape[3];

    for (size_t i = 0; i < N * IC; ++i) {
        for (size_t ih = 0; ih < IH; ++ih) {
            for (size_t iw = 0; iw < IW; ++iw) {
                size_t oh = ih * S;
                size_t ow = iw * S;
                uint8x16_t r0 = vld1q_u8(src_ptr + i * IH * IW * 16 +
                                         ih * IW * 16 + iw * 16);

                for (size_t fh = 0; fh < S; ++fh) {
                    for (size_t fw = 0; fw < S; ++fw) {
                        vst1q_u8(dst_ptr + i * OH * OW * 16 +
                                         (oh + fh) * OW * 16 + (ow + fw) * 16,
                                 r0);
                    }
                }
            }
        }
    }
}

// vim: syntax=cpp.doxygen
