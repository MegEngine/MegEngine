/**
 * \file dnn/src/arm_common/conv_bias/fp32/helper.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once
#include "src/common/unroll_macro.h"
#include "src/arm_common/simd_macro/marm_neon.h"

namespace megdnn {
namespace arm_common {
inline void transpose_4x4(const float* src, float* dst, int lda, int ldb) {
    float32x4x2_t a0, a1;
    a0.val[0] = vld1q_f32(src + 0 * lda);
    a0.val[1] = vld1q_f32(src + 1 * lda);
    a1.val[0] = vld1q_f32(src + 2 * lda);
    a1.val[1] = vld1q_f32(src + 3 * lda);
    float32x4x2_t b0 = vzipq_f32(a0.val[0], a1.val[0]);
    float32x4x2_t b1 = vzipq_f32(a0.val[1], a1.val[1]);
    float32x4x2_t c0 = vzipq_f32(b0.val[0], b1.val[0]);
    float32x4x2_t c1 = vzipq_f32(b0.val[1], b1.val[1]);
    vst1q_f32(dst + 0 * ldb, c0.val[0]);
    vst1q_f32(dst + 1 * ldb, c0.val[1]);
    vst1q_f32(dst + 2 * ldb, c1.val[0]);
    vst1q_f32(dst + 3 * ldb, c1.val[1]);
}
}  // namespace arm_common
}  // namespace megdnn

#define MATRIX_MUL4x4(sum, a, b)                         \
    sum##0 = vmlaq_low_lane_f32(sum##0, b##0, a##0, 0);  \
    sum##0 = vmlaq_low_lane_f32(sum##0, b##1, a##0, 1);  \
    sum##0 = vmlaq_high_lane_f32(sum##0, b##2, a##0, 2); \
    sum##0 = vmlaq_high_lane_f32(sum##0, b##3, a##0, 3); \
    sum##1 = vmlaq_low_lane_f32(sum##1, b##0, a##1, 0);  \
    sum##1 = vmlaq_low_lane_f32(sum##1, b##1, a##1, 1);  \
    sum##1 = vmlaq_high_lane_f32(sum##1, b##2, a##1, 2); \
    sum##1 = vmlaq_high_lane_f32(sum##1, b##3, a##1, 3); \
    sum##2 = vmlaq_low_lane_f32(sum##2, b##0, a##2, 0);  \
    sum##2 = vmlaq_low_lane_f32(sum##2, b##1, a##2, 1);  \
    sum##2 = vmlaq_high_lane_f32(sum##2, b##2, a##2, 2); \
    sum##2 = vmlaq_high_lane_f32(sum##2, b##3, a##2, 3); \
    sum##3 = vmlaq_low_lane_f32(sum##3, b##0, a##3, 0);  \
    sum##3 = vmlaq_low_lane_f32(sum##3, b##1, a##3, 1);  \
    sum##3 = vmlaq_high_lane_f32(sum##3, b##2, a##3, 2); \
    sum##3 = vmlaq_high_lane_f32(sum##3, b##3, a##3, 3);

#define CONCAT(a, idx) a##idx

#if MEGDNN_AARCH64
//! ret and a are type Vector<float, 8>
#define TRANSPOSE_8x8(a, ret)                                  \
    do {                                                       \
        auto b0 = vzipq_f32(CONCAT(a, 0).value.val[0],         \
                            CONCAT(a, 1).value.val[0]);        \
        auto b1 = vzipq_f32(CONCAT(a, 0).value.val[1],         \
                            CONCAT(a, 1).value.val[1]);        \
        auto b2 = vzipq_f32(CONCAT(a, 2).value.val[0],         \
                            CONCAT(a, 3).value.val[0]);        \
        auto b3 = vzipq_f32(CONCAT(a, 2).value.val[1],         \
                            CONCAT(a, 3).value.val[1]);        \
        auto b4 = vzipq_f32(CONCAT(a, 4).value.val[0],         \
                            CONCAT(a, 5).value.val[0]);        \
        auto b5 = vzipq_f32(CONCAT(a, 4).value.val[1],         \
                            CONCAT(a, 5).value.val[1]);        \
        auto b6 = vzipq_f32(CONCAT(a, 6).value.val[0],         \
                            CONCAT(a, 7).value.val[0]);        \
        auto b7 = vzipq_f32(CONCAT(a, 6).value.val[1],         \
                            CONCAT(a, 7).value.val[1]);        \
        CONCAT(ret, 0).value.val[0] = vreinterpretq_f32_s64(   \
                vzip1q_s64(vreinterpretq_s64_f32(b0.val[0]),   \
                           vreinterpretq_s64_f32(b2.val[0]))); \
        CONCAT(ret, 0).value.val[1] = vreinterpretq_f32_s64(   \
                vzip1q_s64(vreinterpretq_s64_f32(b4.val[0]),   \
                           vreinterpretq_s64_f32(b6.val[0]))); \
        CONCAT(ret, 1).value.val[0] = vreinterpretq_f32_s64(   \
                vzip2q_s64(vreinterpretq_s64_f32(b0.val[0]),   \
                           vreinterpretq_s64_f32(b2.val[0]))); \
        CONCAT(ret, 1).value.val[1] = vreinterpretq_f32_s64(   \
                vzip2q_s64(vreinterpretq_s64_f32(b4.val[0]),   \
                           vreinterpretq_s64_f32(b6.val[0]))); \
        CONCAT(ret, 2).value.val[0] = vreinterpretq_f32_s64(   \
                vzip1q_s64(vreinterpretq_s64_f32(b0.val[1]),   \
                           vreinterpretq_s64_f32(b2.val[1]))); \
        CONCAT(ret, 2).value.val[1] = vreinterpretq_f32_s64(   \
                vzip1q_s64(vreinterpretq_s64_f32(b4.val[1]),   \
                           vreinterpretq_s64_f32(b6.val[1]))); \
        CONCAT(ret, 3).value.val[0] = vreinterpretq_f32_s64(   \
                vzip2q_s64(vreinterpretq_s64_f32(b0.val[1]),   \
                           vreinterpretq_s64_f32(b2.val[1]))); \
        CONCAT(ret, 3).value.val[1] = vreinterpretq_f32_s64(   \
                vzip2q_s64(vreinterpretq_s64_f32(b4.val[1]),   \
                           vreinterpretq_s64_f32(b6.val[1]))); \
        CONCAT(ret, 4).value.val[0] = vreinterpretq_f32_s64(   \
                vzip1q_s64(vreinterpretq_s64_f32(b1.val[0]),   \
                           vreinterpretq_s64_f32(b3.val[0]))); \
        CONCAT(ret, 4).value.val[1] = vreinterpretq_f32_s64(   \
                vzip1q_s64(vreinterpretq_s64_f32(b5.val[0]),   \
                           vreinterpretq_s64_f32(b7.val[0]))); \
        CONCAT(ret, 5).value.val[0] = vreinterpretq_f32_s64(   \
                vzip2q_s64(vreinterpretq_s64_f32(b1.val[0]),   \
                           vreinterpretq_s64_f32(b3.val[0]))); \
        CONCAT(ret, 5).value.val[1] = vreinterpretq_f32_s64(   \
                vzip2q_s64(vreinterpretq_s64_f32(b5.val[0]),   \
                           vreinterpretq_s64_f32(b7.val[0]))); \
        CONCAT(ret, 6).value.val[0] = vreinterpretq_f32_s64(   \
                vzip1q_s64(vreinterpretq_s64_f32(b1.val[1]),   \
                           vreinterpretq_s64_f32(b3.val[1]))); \
        CONCAT(ret, 6).value.val[1] = vreinterpretq_f32_s64(   \
                vzip1q_s64(vreinterpretq_s64_f32(b5.val[1]),   \
                           vreinterpretq_s64_f32(b7.val[1]))); \
        CONCAT(ret, 7).value.val[0] = vreinterpretq_f32_s64(   \
                vzip2q_s64(vreinterpretq_s64_f32(b1.val[1]),   \
                           vreinterpretq_s64_f32(b3.val[1]))); \
        CONCAT(ret, 7).value.val[1] = vreinterpretq_f32_s64(   \
                vzip2q_s64(vreinterpretq_s64_f32(b5.val[1]),   \
                           vreinterpretq_s64_f32(b7.val[1]))); \
    } while (0);

#define TRANSPOSE_8x3(a, ret)                                    \
    auto b0 = vzipq_f32(CONCAT(a, 0).value, CONCAT(a, 1).value); \
    auto b1 = vzipq_f32(CONCAT(a, 2).value, CONCAT(a, 3).value); \
    auto b2 = vzipq_f32(CONCAT(a, 4).value, CONCAT(a, 5).value); \
    auto b3 = vzipq_f32(CONCAT(a, 6).value, CONCAT(a, 7).value); \
    CONCAT(ret, 0).value.val[0] = vreinterpretq_f32_s64(         \
            vzip1q_s64(vreinterpretq_s64_f32(b0.val[0]),         \
                       vreinterpretq_s64_f32(b1.val[0])));       \
    CONCAT(ret, 0).value.val[1] = vreinterpretq_f32_s64(         \
            vzip1q_s64(vreinterpretq_s64_f32(b2.val[0]),         \
                       vreinterpretq_s64_f32(b3.val[0])));       \
    CONCAT(ret, 1).value.val[0] = vreinterpretq_f32_s64(         \
            vzip2q_s64(vreinterpretq_s64_f32(b0.val[0]),         \
                       vreinterpretq_s64_f32(b1.val[0])));       \
    CONCAT(ret, 1).value.val[1] = vreinterpretq_f32_s64(         \
            vzip2q_s64(vreinterpretq_s64_f32(b2.val[0]),         \
                       vreinterpretq_s64_f32(b3.val[0])));       \
    CONCAT(ret, 2).value.val[0] = vreinterpretq_f32_s64(         \
            vzip1q_s64(vreinterpretq_s64_f32(b0.val[1]),         \
                       vreinterpretq_s64_f32(b1.val[1])));       \
    CONCAT(ret, 2).value.val[1] = vreinterpretq_f32_s64(         \
            vzip1q_s64(vreinterpretq_s64_f32(b2.val[1]),         \
                       vreinterpretq_s64_f32(b3.val[1])));

#define TRANSPOSE_8x4(a, ret)                                    \
    auto b0 = vzipq_f32(CONCAT(a, 0).value, CONCAT(a, 1).value); \
    auto b1 = vzipq_f32(CONCAT(a, 2).value, CONCAT(a, 3).value); \
    auto b2 = vzipq_f32(CONCAT(a, 4).value, CONCAT(a, 5).value); \
    auto b3 = vzipq_f32(CONCAT(a, 6).value, CONCAT(a, 7).value); \
    CONCAT(ret, 0).value.val[0] = vreinterpretq_f32_s64(         \
            vzip1q_s64(vreinterpretq_s64_f32(b0.val[0]),         \
                       vreinterpretq_s64_f32(b1.val[0])));       \
    CONCAT(ret, 0).value.val[1] = vreinterpretq_f32_s64(         \
            vzip1q_s64(vreinterpretq_s64_f32(b2.val[0]),         \
                       vreinterpretq_s64_f32(b3.val[0])));       \
    CONCAT(ret, 1).value.val[0] = vreinterpretq_f32_s64(         \
            vzip2q_s64(vreinterpretq_s64_f32(b0.val[0]),         \
                       vreinterpretq_s64_f32(b1.val[0])));       \
    CONCAT(ret, 1).value.val[1] = vreinterpretq_f32_s64(         \
            vzip2q_s64(vreinterpretq_s64_f32(b2.val[0]),         \
                       vreinterpretq_s64_f32(b3.val[0])));       \
    CONCAT(ret, 2).value.val[0] = vreinterpretq_f32_s64(         \
            vzip1q_s64(vreinterpretq_s64_f32(b0.val[1]),         \
                       vreinterpretq_s64_f32(b1.val[1])));       \
    CONCAT(ret, 2).value.val[1] = vreinterpretq_f32_s64(         \
            vzip1q_s64(vreinterpretq_s64_f32(b2.val[1]),         \
                       vreinterpretq_s64_f32(b3.val[1])));       \
    CONCAT(ret, 3).value.val[0] = vreinterpretq_f32_s64(         \
            vzip2q_s64(vreinterpretq_s64_f32(b0.val[1]),         \
                       vreinterpretq_s64_f32(b1.val[1])));       \
    CONCAT(ret, 3).value.val[1] = vreinterpretq_f32_s64(         \
            vzip2q_s64(vreinterpretq_s64_f32(b2.val[1]),         \
                       vreinterpretq_s64_f32(b3.val[1])));

#elif MEGDNN_ARMV7
#define TRANSPOSE_8x4(a, ret)                                                 \
    auto b0 = vzipq_f32(CONCAT(a, 0).value, CONCAT(a, 1).value);              \
    auto b1 = vzipq_f32(CONCAT(a, 2).value, CONCAT(a, 3).value);              \
    auto b2 = vzipq_f32(CONCAT(a, 4).value, CONCAT(a, 5).value);              \
    auto b3 = vzipq_f32(CONCAT(a, 6).value, CONCAT(a, 7).value);              \
    CONCAT(ret, 0).value.val[0] =                                             \
            vcombine_f32(vget_low_f32(b0.val[0]), vget_low_f32(b1.val[0]));   \
    CONCAT(ret, 1).value.val[0] =                                             \
            vcombine_f32(vget_high_f32(b0.val[0]), vget_high_f32(b1.val[0])); \
    CONCAT(ret, 2).value.val[0] =                                             \
            vcombine_f32(vget_low_f32(b0.val[1]), vget_low_f32(b1.val[1]));   \
    CONCAT(ret, 3).value.val[0] =                                             \
            vcombine_f32(vget_high_f32(b0.val[1]), vget_high_f32(b1.val[1])); \
    CONCAT(ret, 0).value.val[1] =                                             \
            vcombine_f32(vget_low_f32(b2.val[0]), vget_low_f32(b3.val[0]));   \
    CONCAT(ret, 1).value.val[1] =                                             \
            vcombine_f32(vget_high_f32(b2.val[0]), vget_high_f32(b3.val[0])); \
    CONCAT(ret, 2).value.val[1] =                                             \
            vcombine_f32(vget_low_f32(b2.val[1]), vget_low_f32(b3.val[1]));   \
    CONCAT(ret, 3).value.val[1] =                                             \
            vcombine_f32(vget_high_f32(b2.val[1]), vget_high_f32(b3.val[1]));

#endif
// vim: syntax=cpp.doxygen
