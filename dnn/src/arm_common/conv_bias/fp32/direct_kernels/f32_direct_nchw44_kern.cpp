/**
 * \file
 * dnn/src/arm_common/conv_bias/fp32/direct_kernels/f32_direct_nchw44_kern.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/arm_common/conv_bias/fp32/f32_direct_nchw44_kern.h"
#include "src/arm_common/conv_bias/opr_impl.h"
#include "src/arm_common/simd_macro/marm_neon.h"
#include "src/fallback/conv_bias/common.h"
namespace megdnn {
namespace arm_common {
namespace conv_bias {
template <>
void pack_src_fp32_nchw44<1>(float* sptr_base, const float* sptr_origin,
                             const int, const int pw, const int pad_right,
                             const int ih, const int iw, const int iw2,
                             const int pad_top, const int pad_bottom,
                             const int ic, const int ic_stride) {
    constexpr int ic_step = 4;
    rep_step(ic_idx, ic, ic_step) {
        const float* sptr = sptr_origin + ic_idx * ic_stride;
        memset(sptr_base, 0, sizeof(float) * iw2 * pad_top * ic_step);
        sptr_base += iw2 * pad_top * ic_step;
        rep(ih_idx, ih) {
            memset(sptr_base, 0, sizeof(float) * pw * ic_step);
            sptr_base += pw * ic_step;
            memcpy(sptr_base, sptr, sizeof(float) * iw * ic_step);
            sptr_base += iw * ic_step;
            sptr += iw * ic_step;
            memset(sptr_base, 0, sizeof(float) * pad_right * ic_step);
            sptr_base += pad_right * ic_step;
        }
        memset(sptr_base, 0, sizeof(float) * iw2 * pad_bottom * ic_step);
        sptr_base += iw2 * pad_bottom * ic_step;
    }
}

namespace {

static inline void odd_even_split_iw8_even(float* sptr_base, const float* sptr,
                                           const int odd_start,
                                           const int src_idx,
                                           const int iw_idx) {
    constexpr int ic_step = 4;
    const int src_offset = src_idx * ic_step;
    const int even_offset = iw_idx / 2 * ic_step;
    const int odd_offset = (odd_start + iw_idx / 2) * ic_step;
    float32x4_t temp[8];
    temp[0] = vld1q_f32(sptr + src_offset + 0 * ic_step);
    temp[1] = vld1q_f32(sptr + src_offset + 1 * ic_step);
    temp[2] = vld1q_f32(sptr + src_offset + 2 * ic_step);
    temp[3] = vld1q_f32(sptr + src_offset + 3 * ic_step);
    temp[4] = vld1q_f32(sptr + src_offset + 4 * ic_step);
    temp[5] = vld1q_f32(sptr + src_offset + 5 * ic_step);
    temp[6] = vld1q_f32(sptr + src_offset + 6 * ic_step);
    temp[7] = vld1q_f32(sptr + src_offset + 7 * ic_step);
    vst1q_f32(sptr_base + even_offset + 0 * ic_step, temp[0]);
    vst1q_f32(sptr_base + even_offset + 1 * ic_step, temp[2]);
    vst1q_f32(sptr_base + even_offset + 2 * ic_step, temp[4]);
    vst1q_f32(sptr_base + even_offset + 3 * ic_step, temp[6]);
    vst1q_f32(sptr_base + odd_offset + 0 * ic_step, temp[1]);
    vst1q_f32(sptr_base + odd_offset + 1 * ic_step, temp[3]);
    vst1q_f32(sptr_base + odd_offset + 2 * ic_step, temp[5]);
    vst1q_f32(sptr_base + odd_offset + 3 * ic_step, temp[7]);
}

static inline void odd_even_split_iw8_odd(float* sptr_base, const float* sptr,
                                          const int odd_start,
                                          const int src_idx, const int iw_idx) {
    constexpr int ic_step = 4;
    const int src_offset = src_idx * ic_step;
    const int even_offset = (iw_idx + 1) / 2 * ic_step;
    const int odd_offset = (odd_start + iw_idx / 2) * ic_step;
    float32x4_t temp[8];
    temp[0] = vld1q_f32(sptr + src_offset + 0 * ic_step);
    temp[1] = vld1q_f32(sptr + src_offset + 1 * ic_step);
    temp[2] = vld1q_f32(sptr + src_offset + 2 * ic_step);
    temp[3] = vld1q_f32(sptr + src_offset + 3 * ic_step);
    temp[4] = vld1q_f32(sptr + src_offset + 4 * ic_step);
    temp[5] = vld1q_f32(sptr + src_offset + 5 * ic_step);
    temp[6] = vld1q_f32(sptr + src_offset + 6 * ic_step);
    temp[7] = vld1q_f32(sptr + src_offset + 7 * ic_step);
    vst1q_f32(sptr_base + odd_offset + 0 * ic_step, temp[0]);
    vst1q_f32(sptr_base + odd_offset + 1 * ic_step, temp[2]);
    vst1q_f32(sptr_base + odd_offset + 2 * ic_step, temp[4]);
    vst1q_f32(sptr_base + odd_offset + 3 * ic_step, temp[6]);
    vst1q_f32(sptr_base + even_offset + 0 * ic_step, temp[1]);
    vst1q_f32(sptr_base + even_offset + 1 * ic_step, temp[3]);
    vst1q_f32(sptr_base + even_offset + 2 * ic_step, temp[5]);
    vst1q_f32(sptr_base + even_offset + 3 * ic_step, temp[7]);
}
}  // namespace

template <>
void pack_src_fp32_nchw44<2>(float* sptr_base, const float* sptr_origin,
                             const int ph, const int pw, const int pad_right,
                             const int ih, const int iw, const int iw2,
                             const int pad_top, const int pad_bottom,
                             const int ic, const int ic_stride) {
    constexpr int ic_step = 4;
    int odd_start = megdnn::div_ceil(iw2, 2);
    float32x4_t zero_v = vdupq_n_f32(0.f);
    MEGDNN_MARK_USED_VAR(ph);
    bool even_start = pw % 2 == 0;
    rep_step(ic_idx, ic, ic_step) {
        const float* sptr = sptr_origin + ic_idx * ic_stride;
        memset(sptr_base, 0, sizeof(float) * iw2 * pad_top * ic_step);
        sptr_base += iw2 * pad_top * ic_step;
        rep(ih_idx, ih) {
            int iw_idx = 0;
            rep(idx, pw) {
                if (iw_idx % 2 == 0) {
                    vst1q_f32(sptr_base + iw_idx / 2 * ic_step, zero_v);
                } else {
                    vst1q_f32(sptr_base + (odd_start + iw_idx / 2) * ic_step,
                              zero_v);
                }
                ++iw_idx;
            }
            int src_idx = 0;
            if (even_start) {
                for (; src_idx + 7 < iw; src_idx += 8) {
                    odd_even_split_iw8_even(sptr_base, sptr, odd_start, src_idx,
                                            iw_idx);
                    iw_idx += 8;
                }
            } else {
                for (; src_idx + 7 < iw; src_idx += 8) {
                    odd_even_split_iw8_odd(sptr_base, sptr, odd_start, src_idx,
                                           iw_idx);
                    iw_idx += 8;
                }
            }
            for (; src_idx < iw; ++src_idx) {
                if (iw_idx % 2 == 0) {
                    vst1q_f32(sptr_base + iw_idx / 2 * ic_step,
                              vld1q_f32(sptr + src_idx * ic_step));
                } else {
                    vst1q_f32(sptr_base + (odd_start + iw_idx / 2) * ic_step,
                              vld1q_f32(sptr + src_idx * ic_step));
                }
                ++iw_idx;
            }
            rep(idx, pad_right) {
                if (iw_idx % 2 == 0) {
                    vst1q_f32(sptr_base + iw_idx / 2 * ic_step, zero_v);
                } else {
                    vst1q_f32(sptr_base + (odd_start + iw_idx / 2) * ic_step,
                              zero_v);
                }
                ++iw_idx;
            }
            sptr_base += iw2 * ic_step;
            sptr += iw * ic_step;
        }
        memset(sptr_base, 0, sizeof(float) * iw2 * pad_bottom * ic_step);
        sptr_base += iw2 * pad_bottom * ic_step;
    }
}

}  // namespace conv_bias
}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
