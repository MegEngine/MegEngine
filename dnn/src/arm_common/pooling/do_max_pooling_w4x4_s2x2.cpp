/**
 * \file dnn/src/arm_common/pooling/do_max_pooling_w4x4_s2x2.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#include "src/arm_common/pooling/do_max_pooling_w4x4_s2x2.h"
#include "src/arm_common/pooling/pooling_helper.h"

namespace megdnn {
namespace arm_common {

void do_max_pooling_w4x4_s2x2_float_NEON(const dt_float32* src, dt_float32* dst,
                                         DType src_dtype, const int IH,
                                         const int IW, const int OH,
                                         const int OW, const int PH,
                                         const int PW) {
    const int window = 4;
    const int stride = 2;
    using Pooler = MaxPooler<16, dt_float32, float, float>;
    int oh = 0;
    for (; oh < OH && -PH + stride * oh < 0; ++oh) {
        int ow = 0;
        for (; ow < OW; ++ow) {
            do_pxl_naive<Pooler, window>(oh, ow, src, dst, src_dtype, IH, IW,
                                         OH, OW, PH, PW, stride, stride);
        }
    }
    for (; oh < OH && -PH + stride * oh + window <= IH; ++oh) {
        int ow = 0;
        for (; ow < OW && -PW + stride * ow < 0; ++ow) {
            do_pxl_naive<Pooler, window>(oh, ow, src, dst, src_dtype, IH, IW,
                                         OH, OW, PH, PW, stride, stride);
        }
        dt_float32 last_hf_res = -std::numeric_limits<dt_float32>::infinity();
        int ih = -PH + stride * oh, iw = -PW + stride * ow;
        if (-PW + stride * ow + window <= IW) {
            float32x4_t i0 = vld1q_f32(src + (ih + 0) * IW + iw),
                        i1 = vld1q_f32(src + (ih + 1) * IW + iw),
                        i2 = vld1q_f32(src + (ih + 2) * IW + iw),
                        i3 = vld1q_f32(src + (ih + 3) * IW + iw);
            float32x4_t sum0 = vmaxq_f32(vmaxq_f32(i0, i1), vmaxq_f32(i2, i3));
            float32x2_t t = vpmax_f32(vget_low_f32(sum0), vget_high_f32(sum0));
            dst[oh * OW + ow] =
                    std::max(vget_lane_f32(t, 0), vget_lane_f32(t, 1));
            last_hf_res = vget_lane_f32(t, 1);
            ow += 1;
        }
        for (; ow + 1 < OW && -PW + stride * (ow + 1) + window <= IW; ow += 2) {
            iw = -PW + stride * (ow + 1);
            float32x4_t i0 = vld1q_f32(src + (ih + 0) * IW + iw),
                        i1 = vld1q_f32(src + (ih + 1) * IW + iw),
                        i2 = vld1q_f32(src + (ih + 2) * IW + iw),
                        i3 = vld1q_f32(src + (ih + 3) * IW + iw);
            float32x4_t sum0 = vmaxq_f32(vmaxq_f32(i0, i1), vmaxq_f32(i2, i3));
            float32x2_t t = vpmax_f32(vget_low_f32(sum0), vget_high_f32(sum0));
            dst[oh * OW + ow + 0] = std::max(vget_lane_f32(t, 0), last_hf_res);
            dst[oh * OW + ow + 1] =
                    std::max(vget_lane_f32(t, 0), vget_lane_f32(t, 1));
            last_hf_res = vget_lane_f32(t, 1);
        }
        for (; ow < OW; ++ow) {
            do_pxl_naive<Pooler, window>(oh, ow, src, dst, src_dtype, IH, IW,
                                         OH, OW, PH, PW, stride, stride);
        }
    }
    for (; oh < OH; ++oh) {
        int ow = 0;
        for (; ow < OW; ++ow) {
            do_pxl_naive<Pooler, window>(oh, ow, src, dst, src_dtype, IH, IW,
                                         OH, OW, PH, PW, stride, stride);
        }
    }
}

void do_max_pooling_w4x4_s2x2_int8_NEON(const int8_t* src, int8_t* dst,
                                        DType src_dtype, const int IH,
                                        const int IW, const int OH,
                                        const int OW, const int PH,
                                        const int PW) {
    const int window = 4;
    const int stride = 2;
    using Pooler = MaxPooler<16, dt_qint8, int8_t, float>;
    int oh = 0;
    for (; oh < OH && -PH + stride * oh < 0; ++oh) {
        int ow = 0;
        for (; ow < OW; ++ow) {
            do_pxl_naive<Pooler, window>(oh, ow, src, dst, src_dtype, IH, IW,
                                         OH, OW, PH, PW, stride, stride);
        }
    }
    for (; oh < OH && -PH + stride * oh + window <= IH; ++oh) {
        int ow = 0;
        for (; ow < OW && -PW + stride * ow < 0; ++ow) {
            do_pxl_naive<Pooler, window>(oh, ow, src, dst, src_dtype, IH, IW,
                                         OH, OW, PH, PW, stride, stride);
        }
        int8_t last_res = std::numeric_limits<int8_t>::lowest();
        int ih = -PH + stride * oh, iw = -PW + stride * ow;
        if (-PW + stride * (ow + 6) + window <= IW) {
            int8x16_t i0 = vld1q_s8(src + (ih + 0) * IW + iw),
                      i1 = vld1q_s8(src + (ih + 1) * IW + iw),
                      i2 = vld1q_s8(src + (ih + 2) * IW + iw),
                      i3 = vld1q_s8(src + (ih + 3) * IW + iw);
            int8x16_t sum0 = vmaxq_s8(vmaxq_s8(i0, i1), vmaxq_s8(i2, i3));
            int8x8_t t = vpmax_s8(vget_low_s8(sum0), vget_high_s8(sum0));
#define cb(i)               \
    dst[oh * OW + ow + i] = \
            std::max(vget_lane_s8(t, i), vget_lane_s8(t, i + 1));
            UNROLL_CALL_NOWRAPPER(7, cb)
#undef cb
            last_res = vget_lane_s8(t, 7);
            ow += 7;
        }
        for (; ow + 7 < OW && -PW + stride * (ow + 7) + window <= IW; ow += 8) {
            iw = -PW + stride * (ow + 1);
            int8x16_t i0 = vld1q_s8(src + (ih + 0) * IW + iw),
                      i1 = vld1q_s8(src + (ih + 1) * IW + iw),
                      i2 = vld1q_s8(src + (ih + 2) * IW + iw),
                      i3 = vld1q_s8(src + (ih + 3) * IW + iw);
            int8x16_t sum0 = vmaxq_s8(vmaxq_s8(i0, i1), vmaxq_s8(i2, i3));
            int8x8_t t = vpmax_s8(vget_low_s8(sum0), vget_high_s8(sum0));
            dst[oh * OW + ow + 0] = std::max(vget_lane_s8(t, 0), last_res);
#define cb(i)                   \
    dst[oh * OW + ow + i + 1] = \
            std::max(vget_lane_s8(t, i), vget_lane_s8(t, i + 1));
            UNROLL_CALL_NOWRAPPER(7, cb)
#undef cb
            last_res = vget_lane_s8(t, 7);
        }
        for (; ow < OW; ++ow) {
            do_pxl_naive<Pooler, window>(oh, ow, src, dst, src_dtype, IH, IW,
                                         OH, OW, PH, PW, stride, stride);
        }
    }
    for (; oh < OH; ++oh) {
        int ow = 0;
        for (; ow < OW; ++ow) {
            do_pxl_naive<Pooler, window>(oh, ow, src, dst, src_dtype, IH, IW,
                                         OH, OW, PH, PW, stride, stride);
        }
    }
}

void do_max_pooling_w4x4_s2x2_uint8_NEON(const uint8_t* src, uint8_t* dst,
                                         DType src_dtype, const int IH,
                                         const int IW, const int OH,
                                         const int OW, const int PH,
                                         const int PW) {
    const int window = 4;
    const int stride = 2;
    using Pooler = MaxPooler<16, dt_quint8, uint8_t, float>;
    int oh = 0;
    for (; oh < OH && -PH + stride * oh < 0; ++oh) {
        int ow = 0;
        for (; ow < OW; ++ow) {
            do_pxl_naive<Pooler, window>(oh, ow, src, dst, src_dtype, IH, IW,
                                         OH, OW, PH, PW, stride, stride);
        }
    }
    for (; oh < OH && -PH + stride * oh + window <= IH; ++oh) {
        int ow = 0;
        for (; ow < OW && -PW + stride * ow < 0; ++ow) {
            do_pxl_naive<Pooler, window>(oh, ow, src, dst, src_dtype, IH, IW,
                                         OH, OW, PH, PW, stride, stride);
        }
        uint8_t last_res = std::numeric_limits<uint8_t>::lowest();
        int ih = -PH + stride * oh, iw = -PW + stride * ow;
        if (-PW + stride * (ow + 6) + window <= IW) {
            uint8x16_t i0 = vld1q_u8(src + (ih + 0) * IW + iw),
                       i1 = vld1q_u8(src + (ih + 1) * IW + iw),
                       i2 = vld1q_u8(src + (ih + 2) * IW + iw),
                       i3 = vld1q_u8(src + (ih + 3) * IW + iw);
            uint8x16_t sum0 = vmaxq_u8(vmaxq_u8(i0, i1), vmaxq_u8(i2, i3));
            uint8x8_t t = vpmax_u8(vget_low_u8(sum0), vget_high_u8(sum0));
#define cb(i)               \
    dst[oh * OW + ow + i] = \
            std::max(vget_lane_u8(t, i), vget_lane_u8(t, i + 1));
            UNROLL_CALL_NOWRAPPER(7, cb)
#undef cb
            last_res = vget_lane_u8(t, 7);
            ow += 7;
        }
        for (; ow + 7 < OW && -PW + stride * (ow + 7) + window <= IW; ow += 8) {
            iw = -PW + stride * (ow + 1);
            uint8x16_t i0 = vld1q_u8(src + (ih + 0) * IW + iw),
                       i1 = vld1q_u8(src + (ih + 1) * IW + iw),
                       i2 = vld1q_u8(src + (ih + 2) * IW + iw),
                       i3 = vld1q_u8(src + (ih + 3) * IW + iw);
            uint8x16_t sum0 = vmaxq_u8(vmaxq_u8(i0, i1), vmaxq_u8(i2, i3));
            uint8x8_t t = vpmax_u8(vget_low_u8(sum0), vget_high_u8(sum0));
            dst[oh * OW + ow + 0] = std::max(vget_lane_u8(t, 0), last_res);
#define cb(i)                   \
    dst[oh * OW + ow + i + 1] = \
            std::max(vget_lane_u8(t, i), vget_lane_u8(t, i + 1));
            UNROLL_CALL_NOWRAPPER(7, cb)
#undef cb
            last_res = vget_lane_u8(t, 7);
        }
        for (; ow < OW; ++ow) {
            do_pxl_naive<Pooler, window>(oh, ow, src, dst, src_dtype, IH, IW,
                                         OH, OW, PH, PW, stride, stride);
        }
    }
    for (; oh < OH; ++oh) {
        int ow = 0;
        for (; ow < OW; ++ow) {
            do_pxl_naive<Pooler, window>(oh, ow, src, dst, src_dtype, IH, IW,
                                         OH, OW, PH, PW, stride, stride);
        }
    }
}
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
void do_max_pooling_w4x4_s2x2_float16_NEON(const __fp16* src, __fp16* dst,
                                           DType src_dtype, const int IH,
                                           const int IW, const int OH,
                                           const int OW, const int PH,
                                           const int PW) {
    const int window = 4;
    const int stride = 2;
    using Pooler = MaxPooler<16, dt_float16, __fp16, __fp16>;
    int oh = 0;
    for (; oh < OH && -PH + stride * oh < 0; ++oh) {
        int ow = 0;
        for (; ow < OW; ++ow) {
            do_pxl_naive<Pooler, window>(oh, ow, src, dst, src_dtype, IH, IW,
                                         OH, OW, PH, PW, stride, stride);
        }
    }
    for (; oh < OH && -PH + stride * oh + window <= IH; ++oh) {
        int ow = 0;
        for (; ow < OW && -PW + stride * ow < 0; ++ow) {
            do_pxl_naive<Pooler, window>(oh, ow, src, dst, src_dtype, IH, IW,
                                         OH, OW, PH, PW, stride, stride);
        }
        __fp16 last_hf_res = -std::numeric_limits<dt_float16>::infinity();
        int ih = -PH + stride * oh, iw = -PW + stride * ow;
        if (-PW + stride * (ow + 2) + window <= IW) {
            float16x8_t i0 = vld1q_f16(src + (ih + 0) * IW + iw),
                        i1 = vld1q_f16(src + (ih + 1) * IW + iw),
                        i2 = vld1q_f16(src + (ih + 2) * IW + iw),
                        i3 = vld1q_f16(src + (ih + 3) * IW + iw);
            float16x8_t sum0 = vmaxq_f16(vmaxq_f16(i0, i1), vmaxq_f16(i2, i3));
            float16x4_t t = vpmax_f16(vget_low_f16(sum0), vget_high_f16(sum0));
            dst[oh * OW + ow] =
                    std::max(vget_lane_f16(t, 0), vget_lane_f16(t, 1));
            dst[oh * OW + ow + 1] =
                    std::max(vget_lane_f16(t, 1), vget_lane_f16(t, 2));
            dst[oh * OW + ow + 2] =
                    std::max(vget_lane_f16(t, 2), vget_lane_f16(t, 3));
            last_hf_res = vget_lane_f16(t, 3);
            ow += 3;
        }
        for (; ow + 3 < OW && -PW + stride * (ow + 3) + window <= IW; ow += 4) {
            iw = -PW + stride * (ow + 1);
            float16x8_t i0 = vld1q_f16(src + (ih + 0) * IW + iw),
                        i1 = vld1q_f16(src + (ih + 1) * IW + iw),
                        i2 = vld1q_f16(src + (ih + 2) * IW + iw),
                        i3 = vld1q_f16(src + (ih + 3) * IW + iw);
            float16x8_t sum0 = vmaxq_f16(vmaxq_f16(i0, i1), vmaxq_f16(i2, i3));
            float16x4_t t = vpmax_f16(vget_low_f16(sum0), vget_high_f16(sum0));
            dst[oh * OW + ow + 0] = std::max(vget_lane_f16(t, 0), last_hf_res);
            dst[oh * OW + ow + 1] =
                    std::max(vget_lane_f16(t, 0), vget_lane_f16(t, 1));
            dst[oh * OW + ow + 2] =
                    std::max(vget_lane_f16(t, 1), vget_lane_f16(t, 2));
            dst[oh * OW + ow + 3] =
                    std::max(vget_lane_f16(t, 2), vget_lane_f16(t, 3));
            last_hf_res = vget_lane_f16(t, 3);
        }
        for (; ow < OW; ++ow) {
            do_pxl_naive<Pooler, window>(oh, ow, src, dst, src_dtype, IH, IW,
                                         OH, OW, PH, PW, stride, stride);
        }
    }
    for (; oh < OH; ++oh) {
        int ow = 0;
        for (; ow < OW; ++ow) {
            do_pxl_naive<Pooler, window>(oh, ow, src, dst, src_dtype, IH, IW,
                                         OH, OW, PH, PW, stride, stride);
        }
    }
}
#endif
}  // namespace arm_common
}  // namespace megdnn
// vim: syntax=cpp.doxygen

