/**
 * \file dnn/src/arm_common/elemwise_multi_type/kernels.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "kernels.h"
#include "src/arm_common/simd_macro/marm_neon.h"

namespace megdnn {
namespace arm_common {

#if defined(__ARM_FEATURE_FMA)
#define Vfmaq_f32(d, n, m) vfmaq_f32(d, n, m)
#else
#define Vfmaq_f32(d, n, m) vmlaq_f32(d, n, m)
#endif

void neon_fuse_mul_add3_int16xf32xf32xf32_vec_bcast111c_bcast111c(
        size_t batch_size, size_t channel_stride, size_t channel_size,
        const int16_t* src0, const float* src1, const float* src2, float* dst) {
    const int16_t* __restrict sptr0 = src0;
    const float* __restrict sptr1 = src1;
    const float* __restrict sptr2 = src2;
    float* __restrict dst_ptr = dst;
    for (size_t batch = 0; batch < batch_size; ++batch) {
        for (size_t s = 0; s < channel_stride; ++s) {
            size_t i = 0;
            for (; i + 15 < channel_size; i += 16, sptr0 += 16, dst_ptr += 16) {
                auto vec0_01 = vld1q_s16(sptr0);
                auto vec0_23 = vld1q_s16(sptr0 + 8);
                auto vec1_0 = vld1q_f32(sptr1 + i);
                auto vec1_1 = vld1q_f32(sptr1 + i + 4);
                auto vec1_2 = vld1q_f32(sptr1 + i + 8);
                auto vec1_3 = vld1q_f32(sptr1 + i + 12);
                auto vec2_0 = vld1q_f32(sptr2 + i);
                auto vec2_1 = vld1q_f32(sptr2 + i + 4);
                auto vec2_2 = vld1q_f32(sptr2 + i + 8);
                auto vec2_3 = vld1q_f32(sptr2 + i + 12);

                auto vec0_0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vec0_01)));
                auto vec0_1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vec0_01)));
                auto vec0_2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vec0_23)));
                auto vec0_3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vec0_23)));

                auto dst_vec_0 = Vfmaq_f32(vec2_0, vec0_0, vec1_0);
                auto dst_vec_1 = Vfmaq_f32(vec2_1, vec0_1, vec1_1);
                auto dst_vec_2 = Vfmaq_f32(vec2_2, vec0_2, vec1_2);
                auto dst_vec_3 = Vfmaq_f32(vec2_3, vec0_3, vec1_3);

                vst1q_f32(dst_ptr, dst_vec_0);
                vst1q_f32(dst_ptr + 4, dst_vec_1);
                vst1q_f32(dst_ptr + 8, dst_vec_2);
                vst1q_f32(dst_ptr + 12, dst_vec_3);
            }
            for (; i + 7 < channel_size; i += 8, sptr0 += 8, dst_ptr += 8) {
                auto vec0_01 = vld1q_s16(sptr0);
                auto vec1_0 = vld1q_f32(sptr1 + i);
                auto vec1_1 = vld1q_f32(sptr1 + i + 4);
                auto vec2_0 = vld1q_f32(sptr2 + i);
                auto vec2_1 = vld1q_f32(sptr2 + i + 4);

                auto vec0_0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vec0_01)));
                auto vec0_1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vec0_01)));

                auto dst_vec_0 = Vfmaq_f32(vec2_0, vec0_0, vec1_0);
                auto dst_vec_1 = Vfmaq_f32(vec2_1, vec0_1, vec1_1);

                vst1q_f32(dst_ptr, dst_vec_0);
                vst1q_f32(dst_ptr + 4, dst_vec_1);
            }
            for (; i + 3 < channel_size; i += 4, sptr0 += 4, dst_ptr += 4) {
                auto vec0_0 = vld1_s16(sptr0);
                auto vec1_0 = vld1q_f32(sptr1 + i);
                auto vec2_0 = vld1q_f32(sptr2 + i);

                auto vec0_0_f32 = vcvtq_f32_s32(vmovl_s16(vec0_0));

                auto dst_vec_0 = Vfmaq_f32(vec2_0, vec0_0_f32, vec1_0);

                vst1q_f32(dst_ptr, dst_vec_0);
            }
            for (; i < channel_size; ++i, ++sptr0, ++dst_ptr) {
                *dst_ptr = (float)(*sptr0) * sptr1[i] + sptr2[i];
            }
        }
    }
}

void neon_fuse_mul_add3_uint8xf32xf32xf32_vec_bcast111c_bcast111c(
        size_t batch_size, size_t channel_stride, size_t channel_size,
        const uint8_t* src0, const float* src1, const float* src2, float* dst) {
    const uint8_t* __restrict sptr0 = src0;
    const float* __restrict sptr1 = src1;
    const float* __restrict sptr2 = src2;
    float* __restrict dst_ptr = dst;
    for (size_t batch = 0; batch < batch_size; ++batch) {
        for (size_t s = 0; s < channel_stride; ++s) {
            size_t i = 0;
            for (; i + 15 < channel_size; i += 16, sptr0 += 16, dst_ptr += 16) {
                auto vec0_0123_u8 = vld1q_u8(sptr0);
                auto vec1_0 = vld1q_f32(sptr1 + i);
                auto vec1_1 = vld1q_f32(sptr1 + i + 4);
                auto vec1_2 = vld1q_f32(sptr1 + i + 8);
                auto vec1_3 = vld1q_f32(sptr1 + i + 12);
                auto vec2_0 = vld1q_f32(sptr2 + i);
                auto vec2_1 = vld1q_f32(sptr2 + i + 4);
                auto vec2_2 = vld1q_f32(sptr2 + i + 8);
                auto vec2_3 = vld1q_f32(sptr2 + i + 12);

                auto vec0_01 =
                        vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(vec0_0123_u8)));
                auto vec0_23 =
                        vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(vec0_0123_u8)));

                auto vec0_0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vec0_01)));
                auto vec0_1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vec0_01)));
                auto vec0_2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vec0_23)));
                auto vec0_3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vec0_23)));

                auto dst_vec_0 = Vfmaq_f32(vec2_0, vec0_0, vec1_0);
                auto dst_vec_1 = Vfmaq_f32(vec2_1, vec0_1, vec1_1);
                auto dst_vec_2 = Vfmaq_f32(vec2_2, vec0_2, vec1_2);
                auto dst_vec_3 = Vfmaq_f32(vec2_3, vec0_3, vec1_3);

                vst1q_f32(dst_ptr, dst_vec_0);
                vst1q_f32(dst_ptr + 4, dst_vec_1);
                vst1q_f32(dst_ptr + 8, dst_vec_2);
                vst1q_f32(dst_ptr + 12, dst_vec_3);
            }
            for (; i + 7 < channel_size; i += 8, sptr0 += 8, dst_ptr += 8) {
                auto vec0_01_u8 = vld1_u8(sptr0);
                auto vec1_0 = vld1q_f32(sptr1 + i);
                auto vec1_1 = vld1q_f32(sptr1 + i + 4);
                auto vec2_0 = vld1q_f32(sptr2 + i);
                auto vec2_1 = vld1q_f32(sptr2 + i + 4);

                auto vec0_01 = vreinterpretq_s16_u16(vmovl_u8(vec0_01_u8));

                auto vec0_0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vec0_01)));
                auto vec0_1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vec0_01)));

                auto dst_vec_0 = Vfmaq_f32(vec2_0, vec0_0, vec1_0);
                auto dst_vec_1 = Vfmaq_f32(vec2_1, vec0_1, vec1_1);

                vst1q_f32(dst_ptr, dst_vec_0);
                vst1q_f32(dst_ptr + 4, dst_vec_1);
            }
            for (; i < channel_size; ++i, ++sptr0, ++dst_ptr) {
                *dst_ptr = (float)(*sptr0) * sptr1[i] + sptr2[i];
            }
        }
    }
}

void neon_fuse_mul_add3_int16xf32xf32xf32_vec_bcast101_bcast101(
        size_t batch_size, size_t channel_size, size_t channel_stride,
        const int16_t* src0, const float* src1, const float* src2, float* dst) {
    const int16_t* __restrict sptr0 = src0;
    const float* __restrict sptr1 = src1;
    const float* __restrict sptr2 = src2;
    float* __restrict dst_ptr = dst;
    for (size_t batch = 0; batch < batch_size; ++batch) {
        for (size_t chan = 0; chan < channel_size; ++chan) {
            auto vec1 = vdupq_n_f32(sptr1[chan]);
            auto vec2 = vdupq_n_f32(sptr2[chan]);
            size_t i = 0;
            for (; i + 15 < channel_stride; i += 16, sptr0 += 16, dst_ptr += 16) {
                auto vec0_01 = vld1q_s16(sptr0);
                auto vec0_23 = vld1q_s16(sptr0 + 8);

                auto vec0_0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vec0_01)));
                auto vec0_1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vec0_01)));
                auto vec0_2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vec0_23)));
                auto vec0_3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vec0_23)));

                auto dst_vec_0 = Vfmaq_f32(vec2, vec0_0, vec1);
                auto dst_vec_1 = Vfmaq_f32(vec2, vec0_1, vec1);
                auto dst_vec_2 = Vfmaq_f32(vec2, vec0_2, vec1);
                auto dst_vec_3 = Vfmaq_f32(vec2, vec0_3, vec1);

                vst1q_f32(dst_ptr, dst_vec_0);
                vst1q_f32(dst_ptr + 4, dst_vec_1);
                vst1q_f32(dst_ptr + 8, dst_vec_2);
                vst1q_f32(dst_ptr + 12, dst_vec_3);
            }
            for (; i + 7 < channel_stride; i += 8, sptr0 += 8, dst_ptr += 8) {
                auto vec0_01 = vld1q_s16(sptr0);

                auto vec0_0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vec0_01)));
                auto vec0_1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vec0_01)));

                auto dst_vec_0 = Vfmaq_f32(vec2, vec0_0, vec1);
                auto dst_vec_1 = Vfmaq_f32(vec2, vec0_1, vec1);

                vst1q_f32(dst_ptr, dst_vec_0);
                vst1q_f32(dst_ptr + 4, dst_vec_1);
            }
            for (; i + 3 < channel_stride; i += 4, sptr0 += 4, dst_ptr += 4) {
                auto vec0_0 = vld1_s16(sptr0);
                auto vec0_0_f32 = vcvtq_f32_s32(vmovl_s16(vec0_0));
                auto dst_vec_0 = Vfmaq_f32(vec2, vec0_0_f32, vec1);

                vst1q_f32(dst_ptr, dst_vec_0);
            }
            for (; i < channel_stride; ++i, ++sptr0, ++dst_ptr) {
                *dst_ptr = (float)(*sptr0) * sptr1[chan] + sptr2[chan];
            }
        }
    }
}

void neon_fuse_mul_add3_uint8xf32xf32xf32_vec_bcast101_bcast101(
        size_t batch_size, size_t channel_size, size_t channel_stride,
        const uint8_t* src0, const float* src1, const float* src2, float* dst) {
    const uint8_t* __restrict sptr0 = src0;
    const float* __restrict sptr1 = src1;
    const float* __restrict sptr2 = src2;
    float* __restrict dst_ptr = dst;
    for (size_t batch = 0; batch < batch_size; ++batch) {
        for (size_t chan = 0; chan < channel_size; ++chan) {
            auto vec1 = vdupq_n_f32(sptr1[chan]);
            auto vec2 = vdupq_n_f32(sptr2[chan]);
            size_t i = 0;
            for (; i + 15 < channel_stride; i += 16, sptr0 += 16, dst_ptr += 16) {
                auto vec0_0123_u8 = vld1q_u8(sptr0);

                auto vec0_01 =
                        vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(vec0_0123_u8)));
                auto vec0_23 =
                        vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(vec0_0123_u8)));

                auto vec0_0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vec0_01)));
                auto vec0_1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vec0_01)));
                auto vec0_2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vec0_23)));
                auto vec0_3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vec0_23)));

                auto dst_vec_0 = Vfmaq_f32(vec2, vec0_0, vec1);
                auto dst_vec_1 = Vfmaq_f32(vec2, vec0_1, vec1);
                auto dst_vec_2 = Vfmaq_f32(vec2, vec0_2, vec1);
                auto dst_vec_3 = Vfmaq_f32(vec2, vec0_3, vec1);

                vst1q_f32(dst_ptr, dst_vec_0);
                vst1q_f32(dst_ptr + 4, dst_vec_1);
                vst1q_f32(dst_ptr + 8, dst_vec_2);
                vst1q_f32(dst_ptr + 12, dst_vec_3);
            }
            for (; i + 7 < channel_stride; i += 8, sptr0 += 8, dst_ptr += 8) {
                auto vec0_01_u8 = vld1_u8(sptr0);
                auto vec0_01 = vreinterpretq_s16_u16(vmovl_u8(vec0_01_u8));

                auto vec0_0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vec0_01)));
                auto vec0_1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vec0_01)));

                auto dst_vec_0 = Vfmaq_f32(vec2, vec0_0, vec1);
                auto dst_vec_1 = Vfmaq_f32(vec2, vec0_1, vec1);

                vst1q_f32(dst_ptr, dst_vec_0);
                vst1q_f32(dst_ptr + 4, dst_vec_1);
            }
            for (; i < channel_stride; ++i, ++sptr0, ++dst_ptr) {
                *dst_ptr = (float)(*sptr0) * sptr1[chan] + sptr2[chan];
            }
        }
    }
}

void neon_fuse_mul_add3_int16xf32xf32xf32_vec_vec_vec(
        size_t size, const int16_t* src0, const float* src1, const float* src2,
        float* dst) {
    const int16_t* __restrict sptr0 = src0;
    const float* __restrict sptr1 = src1;
    const float* __restrict sptr2 = src2;
    float* __restrict dst_ptr = dst;
    size_t i = 0;
    for (; i + 15 < size;
         i += 16, sptr0 += 16, sptr1 += 16, sptr2 += 16, dst_ptr += 16) {
        auto vec0_01 = vld1q_s16(sptr0);
        auto vec0_23 = vld1q_s16(sptr0 + 8);
        auto vec1_0 = vld1q_f32(sptr1);
        auto vec1_1 = vld1q_f32(sptr1 + 4);
        auto vec1_2 = vld1q_f32(sptr1 + 8);
        auto vec1_3 = vld1q_f32(sptr1 + 12);
        auto vec2_0 = vld1q_f32(sptr2);
        auto vec2_1 = vld1q_f32(sptr2 + 4);
        auto vec2_2 = vld1q_f32(sptr2 + 8);
        auto vec2_3 = vld1q_f32(sptr2 + 12);

        auto vec0_0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vec0_01)));
        auto vec0_1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vec0_01)));
        auto vec0_2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vec0_23)));
        auto vec0_3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vec0_23)));

        auto dst_vec_0 = Vfmaq_f32(vec2_0, vec0_0, vec1_0);
        auto dst_vec_1 = Vfmaq_f32(vec2_1, vec0_1, vec1_1);
        auto dst_vec_2 = Vfmaq_f32(vec2_2, vec0_2, vec1_2);
        auto dst_vec_3 = Vfmaq_f32(vec2_3, vec0_3, vec1_3);

        vst1q_f32(dst_ptr, dst_vec_0);
        vst1q_f32(dst_ptr + 4, dst_vec_1);
        vst1q_f32(dst_ptr + 8, dst_vec_2);
        vst1q_f32(dst_ptr + 12, dst_vec_3);
    }
    for (; i + 7 < size; i += 8, sptr0 += 8, sptr1 += 8, sptr2 += 8, dst_ptr += 8) {
        auto vec0_01 = vld1q_s16(sptr0);
        auto vec1_0 = vld1q_f32(sptr1);
        auto vec1_1 = vld1q_f32(sptr1 + 4);
        auto vec2_0 = vld1q_f32(sptr2);
        auto vec2_1 = vld1q_f32(sptr2 + 4);

        auto vec0_0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vec0_01)));
        auto vec0_1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vec0_01)));

        auto dst_vec_0 = Vfmaq_f32(vec2_0, vec0_0, vec1_0);
        auto dst_vec_1 = Vfmaq_f32(vec2_1, vec0_1, vec1_1);

        vst1q_f32(dst_ptr, dst_vec_0);
        vst1q_f32(dst_ptr + 4, dst_vec_1);
    }
    for (; i + 3 < size; i += 4, sptr0 += 4, sptr1 += 4, sptr2 += 4, dst_ptr += 4) {
        auto vec0_0 = vld1_s16(sptr0);
        auto vec1_0 = vld1q_f32(sptr1);
        auto vec2_0 = vld1q_f32(sptr2);

        auto vec0_0_f32 = vcvtq_f32_s32(vmovl_s16(vec0_0));

        auto dst_vec_0 = Vfmaq_f32(vec2_0, vec0_0_f32, vec1_0);

        vst1q_f32(dst_ptr, dst_vec_0);
    }
    for (; i < size; ++i, ++sptr0, ++sptr1, ++sptr2, ++dst_ptr) {
        *dst_ptr = (float)(*sptr0) * (*sptr1) + (*sptr2);
    }
}

void neon_fuse_mul_add3_uint8xf32xf32xf32_vec_vec_vec(
        size_t size, const uint8_t* src0, const float* src1, const float* src2,
        float* dst) {
    const uint8_t* __restrict sptr0 = src0;
    const float* __restrict sptr1 = src1;
    const float* __restrict sptr2 = src2;
    float* __restrict dst_ptr = dst;
    size_t i = 0;
    for (; i + 15 < size;
         i += 16, sptr0 += 16, sptr1 += 16, sptr2 += 16, dst_ptr += 16) {
        auto vec0_0123 = vld1q_u8(sptr0);
        auto vec1_0 = vld1q_f32(sptr1);
        auto vec1_1 = vld1q_f32(sptr1 + 4);
        auto vec1_2 = vld1q_f32(sptr1 + 8);
        auto vec1_3 = vld1q_f32(sptr1 + 12);
        auto vec2_0 = vld1q_f32(sptr2);
        auto vec2_1 = vld1q_f32(sptr2 + 4);
        auto vec2_2 = vld1q_f32(sptr2 + 8);
        auto vec2_3 = vld1q_f32(sptr2 + 12);

        auto vec0_01 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(vec0_0123)));
        auto vec0_23 = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(vec0_0123)));

        auto vec0_0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vec0_01)));
        auto vec0_1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vec0_01)));
        auto vec0_2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vec0_23)));
        auto vec0_3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vec0_23)));

        auto dst_vec_0 = Vfmaq_f32(vec2_0, vec0_0, vec1_0);
        auto dst_vec_1 = Vfmaq_f32(vec2_1, vec0_1, vec1_1);
        auto dst_vec_2 = Vfmaq_f32(vec2_2, vec0_2, vec1_2);
        auto dst_vec_3 = Vfmaq_f32(vec2_3, vec0_3, vec1_3);

        vst1q_f32(dst_ptr, dst_vec_0);
        vst1q_f32(dst_ptr + 4, dst_vec_1);
        vst1q_f32(dst_ptr + 8, dst_vec_2);
        vst1q_f32(dst_ptr + 12, dst_vec_3);
    }

    for (; i + 7 < size; i += 8, sptr0 += 8, sptr1 += 8, sptr2 += 8, dst_ptr += 8) {
        auto vec0_01_u8 = vld1_u8(sptr0);
        auto vec1_0 = vld1q_f32(sptr1);
        auto vec1_1 = vld1q_f32(sptr1 + 4);
        auto vec2_0 = vld1q_f32(sptr2);
        auto vec2_1 = vld1q_f32(sptr2 + 4);

        auto vec0_01 = vreinterpretq_s16_u16(vmovl_u8(vec0_01_u8));

        auto vec0_0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vec0_01)));
        auto vec0_1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vec0_01)));

        auto dst_vec_0 = Vfmaq_f32(vec2_0, vec0_0, vec1_0);
        auto dst_vec_1 = Vfmaq_f32(vec2_1, vec0_1, vec1_1);

        vst1q_f32(dst_ptr, dst_vec_0);
        vst1q_f32(dst_ptr + 4, dst_vec_1);
    }
    for (; i < size; ++i, ++sptr0, ++sptr1, ++sptr2, ++dst_ptr) {
        *dst_ptr = (float)(*sptr0) * (*sptr1) + (*sptr2);
    }
}

void neon_fuse_mul_add3_int16xf32xf32xf32_vec_scaler_scaler(
        size_t size, const int16_t* src0, const float* src1, const float* src2,
        float* dst) {
    const int16_t* __restrict sptr0 = src0;
    const float* __restrict sptr1 = src1;
    const float* __restrict sptr2 = src2;
    auto vec1 = vdupq_n_f32(sptr1[0]);
    auto vec2 = vdupq_n_f32(sptr2[0]);
    float* __restrict dst_ptr = dst;
    size_t i = 0;
    for (; i + 15 < size; i += 16, sptr0 += 16, dst_ptr += 16) {
        auto vec0_01 = vld1q_s16(sptr0);
        auto vec0_23 = vld1q_s16(sptr0 + 8);

        auto vec0_0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vec0_01)));
        auto vec0_1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vec0_01)));
        auto vec0_2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vec0_23)));
        auto vec0_3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vec0_23)));

        auto dst_vec_0 = Vfmaq_f32(vec2, vec0_0, vec1);
        auto dst_vec_1 = Vfmaq_f32(vec2, vec0_1, vec1);
        auto dst_vec_2 = Vfmaq_f32(vec2, vec0_2, vec1);
        auto dst_vec_3 = Vfmaq_f32(vec2, vec0_3, vec1);

        vst1q_f32(dst_ptr, dst_vec_0);
        vst1q_f32(dst_ptr + 4, dst_vec_1);
        vst1q_f32(dst_ptr + 8, dst_vec_2);
        vst1q_f32(dst_ptr + 12, dst_vec_3);
    }
    for (; i + 7 < size; i += 8, sptr0 += 8, sptr1 += 8, sptr2 += 8, dst_ptr += 8) {
        auto vec0_01 = vld1q_s16(sptr0);

        auto vec0_0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vec0_01)));
        auto vec0_1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vec0_01)));

        auto dst_vec_0 = Vfmaq_f32(vec2, vec0_0, vec1);
        auto dst_vec_1 = Vfmaq_f32(vec2, vec0_1, vec1);

        vst1q_f32(dst_ptr, dst_vec_0);
        vst1q_f32(dst_ptr + 4, dst_vec_1);
    }
    for (; i + 3 < size; i += 4, sptr0 += 4, sptr1 += 4, sptr2 += 4, dst_ptr += 4) {
        auto vec0_0 = vld1_s16(sptr0);

        auto vec0_0_f32 = vcvtq_f32_s32(vmovl_s16(vec0_0));

        auto dst_vec_0 = Vfmaq_f32(vec2, vec0_0_f32, vec1);

        vst1q_f32(dst_ptr, dst_vec_0);
    }
    for (; i < size; ++i, ++sptr0, ++dst_ptr) {
        *dst_ptr = (float)(*sptr0) * (*sptr1) + (*sptr2);
    }
}

void neon_fuse_mul_add3_uint8xf32xf32xf32_vec_scaler_scaler(
        size_t size, const uint8_t* src0, const float* src1, const float* src2,
        float* dst) {
    const uint8_t* __restrict sptr0 = src0;
    const float* __restrict sptr1 = src1;
    const float* __restrict sptr2 = src2;
    auto vec1 = vdupq_n_f32(sptr1[0]);
    auto vec2 = vdupq_n_f32(sptr2[0]);
    float* __restrict dst_ptr = dst;
    size_t i = 0;
    for (; i + 15 < size; i += 16, sptr0 += 16, dst_ptr += 16) {
        auto vec0_0123 = vld1q_u8(sptr0);

        auto vec0_01 = vreinterpretq_s16_u16(vmovl_u8(vget_low_u8(vec0_0123)));
        auto vec0_23 = vreinterpretq_s16_u16(vmovl_u8(vget_high_u8(vec0_0123)));

        auto vec0_0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vec0_01)));
        auto vec0_1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vec0_01)));
        auto vec0_2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vec0_23)));
        auto vec0_3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vec0_23)));

        auto dst_vec_0 = Vfmaq_f32(vec2, vec0_0, vec1);
        auto dst_vec_1 = Vfmaq_f32(vec2, vec0_1, vec1);
        auto dst_vec_2 = Vfmaq_f32(vec2, vec0_2, vec1);
        auto dst_vec_3 = Vfmaq_f32(vec2, vec0_3, vec1);

        vst1q_f32(dst_ptr, dst_vec_0);
        vst1q_f32(dst_ptr + 4, dst_vec_1);
        vst1q_f32(dst_ptr + 8, dst_vec_2);
        vst1q_f32(dst_ptr + 12, dst_vec_3);
    }

    for (; i + 7 < size; i += 8, sptr0 += 8, dst_ptr += 8) {
        auto vec0_01_u8 = vld1_u8(sptr0);

        auto vec0_01 = vreinterpretq_s16_u16(vmovl_u8(vec0_01_u8));

        auto vec0_0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vec0_01)));
        auto vec0_1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vec0_01)));

        auto dst_vec_0 = Vfmaq_f32(vec2, vec0_0, vec1);
        auto dst_vec_1 = Vfmaq_f32(vec2, vec0_1, vec1);

        vst1q_f32(dst_ptr, dst_vec_0);
        vst1q_f32(dst_ptr + 4, dst_vec_1);
    }
    for (; i < size; ++i, ++sptr0, ++dst_ptr) {
        *dst_ptr = (float)(*sptr0) * (*sptr1) + (*sptr2);
    }
}

void neon_mul_int16xf32xf32_vec_bcast111c(
        size_t batch_size, size_t channel_stride, size_t channel_size,
        const int16_t* src0, const float* src1, float* dst) {
    const int16_t* __restrict sptr0 = src0;
    const float* __restrict sptr1 = src1;
    float* __restrict dst_ptr = dst;
    for (size_t batch = 0; batch < batch_size; ++batch) {
        for (size_t s = 0; s < channel_stride; ++s) {
            size_t i = 0;
            for (; i + 15 < channel_size; i += 16, sptr0 += 16, dst_ptr += 16) {
                auto vec0_01 = vld1q_s16(sptr0);
                auto vec0_23 = vld1q_s16(sptr0 + 8);
                auto vec1_0 = vld1q_f32(sptr1 + i);
                auto vec1_1 = vld1q_f32(sptr1 + i + 4);
                auto vec1_2 = vld1q_f32(sptr1 + i + 8);
                auto vec1_3 = vld1q_f32(sptr1 + i + 12);

                auto vec0_0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vec0_01)));
                auto vec0_1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vec0_01)));
                auto vec0_2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vec0_23)));
                auto vec0_3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vec0_23)));

                auto dst_vec_0 = vmulq_f32(vec0_0, vec1_0);
                auto dst_vec_1 = vmulq_f32(vec0_1, vec1_1);
                auto dst_vec_2 = vmulq_f32(vec0_2, vec1_2);
                auto dst_vec_3 = vmulq_f32(vec0_3, vec1_3);

                vst1q_f32(dst_ptr, dst_vec_0);
                vst1q_f32(dst_ptr + 4, dst_vec_1);
                vst1q_f32(dst_ptr + 8, dst_vec_2);
                vst1q_f32(dst_ptr + 12, dst_vec_3);
            }
            for (; i + 7 < channel_size; i += 8, sptr0 += 8, dst_ptr += 8) {
                auto vec0_01 = vld1q_s16(sptr0);
                auto vec1_0 = vld1q_f32(sptr1 + i);
                auto vec1_1 = vld1q_f32(sptr1 + i + 4);

                auto vec0_0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vec0_01)));
                auto vec0_1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vec0_01)));

                auto dst_vec_0 = vmulq_f32(vec0_0, vec1_0);
                auto dst_vec_1 = vmulq_f32(vec0_1, vec1_1);

                vst1q_f32(dst_ptr, dst_vec_0);
                vst1q_f32(dst_ptr + 4, dst_vec_1);
            }
            for (; i < channel_size; ++i, ++sptr0, ++dst_ptr) {
                *dst_ptr = (float)(*sptr0) * sptr1[i];
            }
        }
    }
}

void neon_mul_int16xf32xf32_vec_bcast101(
        size_t batch_size, size_t channel_size, size_t channel_stride,
        const int16_t* src0, const float* src1, float* dst) {
    const int16_t* __restrict sptr0 = src0;
    const float* __restrict sptr1 = src1;
    float* __restrict dst_ptr = dst;
    for (size_t batch = 0; batch < batch_size; ++batch) {
        for (size_t chan = 0; chan < channel_size; ++chan) {
            auto vec1 = vdupq_n_f32(sptr1[chan]);
            size_t i = 0;
            for (; i + 15 < channel_stride; i += 16, sptr0 += 16, dst_ptr += 16) {
                auto vec0_01 = vld1q_s16(sptr0);
                auto vec0_23 = vld1q_s16(sptr0 + 8);

                auto vec0_0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vec0_01)));
                auto vec0_1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vec0_01)));
                auto vec0_2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vec0_23)));
                auto vec0_3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vec0_23)));

                auto dst_vec_0 = vmulq_f32(vec0_0, vec1);
                auto dst_vec_1 = vmulq_f32(vec0_1, vec1);
                auto dst_vec_2 = vmulq_f32(vec0_2, vec1);
                auto dst_vec_3 = vmulq_f32(vec0_3, vec1);

                vst1q_f32(dst_ptr, dst_vec_0);
                vst1q_f32(dst_ptr + 4, dst_vec_1);
                vst1q_f32(dst_ptr + 8, dst_vec_2);
                vst1q_f32(dst_ptr + 12, dst_vec_3);
            }
            for (; i + 7 < channel_stride; i += 8, sptr0 += 8, dst_ptr += 8) {
                auto vec0_01 = vld1q_s16(sptr0);

                auto vec0_0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vec0_01)));
                auto vec0_1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vec0_01)));

                auto dst_vec_0 = vmulq_f32(vec0_0, vec1);
                auto dst_vec_1 = vmulq_f32(vec0_1, vec1);

                vst1q_f32(dst_ptr, dst_vec_0);
                vst1q_f32(dst_ptr + 4, dst_vec_1);
            }
            for (; i < channel_stride; ++i, ++sptr0, ++dst_ptr) {
                *dst_ptr = (float)(*sptr0) * sptr1[chan];
            }
        }
    }
}

void neon_mul_int16xf32xf32_vec_vec(
        size_t size, const int16_t* src0, const float* src1, float* dst) {
    const int16_t* __restrict sptr0 = src0;
    const float* __restrict sptr1 = src1;
    float* __restrict dst_ptr = dst;
    size_t i = 0;
    for (; i + 15 < size; i += 16, sptr0 += 16, sptr1 += 16, dst_ptr += 16) {
        auto vec0_01 = vld1q_s16(sptr0);
        auto vec0_23 = vld1q_s16(sptr0 + 8);
        auto vec1_0 = vld1q_f32(sptr1);
        auto vec1_1 = vld1q_f32(sptr1 + 4);
        auto vec1_2 = vld1q_f32(sptr1 + 8);
        auto vec1_3 = vld1q_f32(sptr1 + 12);

        auto vec0_0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vec0_01)));
        auto vec0_1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vec0_01)));
        auto vec0_2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vec0_23)));
        auto vec0_3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vec0_23)));

        auto dst_vec_0 = vmulq_f32(vec0_0, vec1_0);
        auto dst_vec_1 = vmulq_f32(vec0_1, vec1_1);
        auto dst_vec_2 = vmulq_f32(vec0_2, vec1_2);
        auto dst_vec_3 = vmulq_f32(vec0_3, vec1_3);

        vst1q_f32(dst_ptr, dst_vec_0);
        vst1q_f32(dst_ptr + 4, dst_vec_1);
        vst1q_f32(dst_ptr + 8, dst_vec_2);
        vst1q_f32(dst_ptr + 12, dst_vec_3);
    }

    for (; i + 7 < size; i += 8, sptr0 += 8, sptr1 += 8, dst_ptr += 8) {
        auto vec0_01 = vld1q_s16(sptr0);
        auto vec1_0 = vld1q_f32(sptr1);
        auto vec1_1 = vld1q_f32(sptr1 + 4);

        auto vec0_0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vec0_01)));
        auto vec0_1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vec0_01)));

        auto dst_vec_0 = vmulq_f32(vec0_0, vec1_0);
        auto dst_vec_1 = vmulq_f32(vec0_1, vec1_1);

        vst1q_f32(dst_ptr, dst_vec_0);
        vst1q_f32(dst_ptr + 4, dst_vec_1);
    }
    for (; i < size; ++i, ++sptr0, ++sptr1, ++dst_ptr) {
        *dst_ptr = (float)(*sptr0) * (*sptr1);
    }
}

void neon_mul_int16xf32xf32_vec_scaler(
        size_t size, const int16_t* src0, const float* src1, float* dst) {
    const int16_t* __restrict sptr0 = src0;
    const float* __restrict sptr1 = src1;
    auto vec1 = vdupq_n_f32(sptr1[0]);
    float* __restrict dst_ptr = dst;
    size_t i = 0;
    for (; i + 15 < size; i += 16, sptr0 += 16, dst_ptr += 16) {
        auto vec0_01 = vld1q_s16(sptr0);
        auto vec0_23 = vld1q_s16(sptr0 + 8);

        auto vec0_0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vec0_01)));
        auto vec0_1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vec0_01)));
        auto vec0_2 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vec0_23)));
        auto vec0_3 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vec0_23)));

        auto dst_vec_0 = vmulq_f32(vec0_0, vec1);
        auto dst_vec_1 = vmulq_f32(vec0_1, vec1);
        auto dst_vec_2 = vmulq_f32(vec0_2, vec1);
        auto dst_vec_3 = vmulq_f32(vec0_3, vec1);

        vst1q_f32(dst_ptr, dst_vec_0);
        vst1q_f32(dst_ptr + 4, dst_vec_1);
        vst1q_f32(dst_ptr + 8, dst_vec_2);
        vst1q_f32(dst_ptr + 12, dst_vec_3);
    }

    for (; i + 7 < size; i += 8, sptr0 += 8, dst_ptr += 8) {
        auto vec0_01 = vld1q_s16(sptr0);

        auto vec0_0 = vcvtq_f32_s32(vmovl_s16(vget_low_s16(vec0_01)));
        auto vec0_1 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(vec0_01)));

        auto dst_vec_0 = vmulq_f32(vec0_0, vec1);
        auto dst_vec_1 = vmulq_f32(vec0_1, vec1);

        vst1q_f32(dst_ptr, dst_vec_0);
        vst1q_f32(dst_ptr + 4, dst_vec_1);
    }
    for (; i < size; ++i, ++sptr0, ++dst_ptr) {
        *dst_ptr = (float)(*sptr0) * (*sptr1);
    }
}

}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
