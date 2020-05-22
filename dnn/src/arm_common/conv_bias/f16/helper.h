/**
 * \file dnn/src/arm_common/conv_bias/f16/helper.h
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
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#define MATRIX_MUL4x4_fp16(sum, a, b)                        \
    sum##0 = vmul_lane_f16(b##0, a##0, 0);                   \
    sum##1 = vmul_lane_f16(b##0, a##1, 0);                   \
    sum##2 = vmul_lane_f16(b##0, a##2, 0);                   \
    sum##3 = vmul_lane_f16(b##0, a##3, 0);                   \
    sum##0 = vadd_f16(sum##0, vmul_lane_f16(b##1, a##0, 1)); \
    sum##1 = vadd_f16(sum##1, vmul_lane_f16(b##1, a##1, 1)); \
    sum##2 = vadd_f16(sum##2, vmul_lane_f16(b##1, a##2, 1)); \
    sum##3 = vadd_f16(sum##3, vmul_lane_f16(b##1, a##3, 1)); \
    sum##0 = vadd_f16(sum##0, vmul_lane_f16(b##2, a##0, 2)); \
    sum##1 = vadd_f16(sum##1, vmul_lane_f16(b##2, a##1, 2)); \
    sum##2 = vadd_f16(sum##2, vmul_lane_f16(b##2, a##2, 2)); \
    sum##3 = vadd_f16(sum##3, vmul_lane_f16(b##2, a##3, 2)); \
    sum##0 = vadd_f16(sum##0, vmul_lane_f16(b##3, a##0, 3)); \
    sum##1 = vadd_f16(sum##1, vmul_lane_f16(b##3, a##1, 3)); \
    sum##2 = vadd_f16(sum##2, vmul_lane_f16(b##3, a##2, 3)); \
    sum##3 = vadd_f16(sum##3, vmul_lane_f16(b##3, a##3, 3));

#define CONCAT(a, id) a##id

#if MEGDNN_AARCH64

#define TRANSPOSE_4x4(a, ret)                                    \
    do {                                                         \
        auto b00 = vzip1_f16(CONCAT(a, 0).value,                 \
                             CONCAT(a, 1).value); /*a1b1a2b2*/   \
        auto b01 = vzip2_f16(CONCAT(a, 0).value,                 \
                             CONCAT(a, 1).value); /*a3b3a4b4*/   \
        auto b10 = vzip1_f16(CONCAT(a, 2).value,                 \
                             CONCAT(a, 3).value); /*c1d1c2d2*/   \
        auto b11 = vzip2_f16(CONCAT(a, 2).value,                 \
                             CONCAT(a, 3).value); /*c3d3c4d4*/   \
        auto s32b00 = vreinterpret_s32_f16(b00);                 \
        auto s32b01 = vreinterpret_s32_f16(b01);                 \
        auto s32b10 = vreinterpret_s32_f16(b10);                 \
        auto s32b11 = vreinterpret_s32_f16(b11);                 \
        CONCAT(ret, 0).value =                                   \
                vreinterpret_f16_s32(vzip1_s32(s32b00, s32b10)); \
        CONCAT(ret, 1).value =                                   \
                vreinterpret_f16_s32(vzip2_s32(s32b00, s32b10)); \
        CONCAT(ret, 2).value =                                   \
                vreinterpret_f16_s32(vzip1_s32(s32b01, s32b11)); \
        CONCAT(ret, 3).value =                                   \
                vreinterpret_f16_s32(vzip2_s32(s32b01, s32b11)); \
    } while (0);

#define TRANSPOSE_4x8(a, ret)                                           \
    do {                                                                \
        auto b00 = vzip1q_f16(CONCAT(a, 0).value,                       \
                              CONCAT(a, 1).value); /*a1b1a2b2a3b3a4b4*/ \
        auto b01 = vzip2q_f16(CONCAT(a, 0).value,                       \
                              CONCAT(a, 1).value); /*a5b5a6b6a7b7a8b8*/ \
        auto b10 = vzip1q_f16(CONCAT(a, 2).value,                       \
                              CONCAT(a, 3).value); /*c1d1c2d2c3d3c4d4*/ \
        auto b11 = vzip2q_f16(CONCAT(a, 2).value,                       \
                              CONCAT(a, 3).value); /*c5d5c6d6c7d7c8d8*/ \
        auto s32b00 = vreinterpretq_s32_f16(b00);                       \
        auto s32b01 = vreinterpretq_s32_f16(b01);                       \
        auto s32b10 = vreinterpretq_s32_f16(b10);                       \
        auto s32b11 = vreinterpretq_s32_f16(b11);                       \
        auto f16b00 = vreinterpretq_f16_s32(                            \
                vzip1q_s32(s32b00, s32b10)); /*a1b1c1d1a2b2c2d2*/       \
        auto f16b01 = vreinterpretq_f16_s32(                            \
                vzip2q_s32(s32b00, s32b10)); /*a3b3c3d3a4b4a4d4*/       \
        auto f16b10 = vreinterpretq_f16_s32(                            \
                vzip1q_s32(s32b01, s32b11)); /*a5b5c5d5a6b6c6d6*/       \
        auto f16b11 = vreinterpretq_f16_s32(                            \
                vzip2q_s32(s32b01, s32b11)); /*a7b7c7d7a8b8c8d8*/       \
        CONCAT(ret, 0).value = vget_low_f16(f16b00);                    \
        CONCAT(ret, 1).value = vget_high_f16(f16b00);                   \
        CONCAT(ret, 2).value = vget_low_f16(f16b01);                    \
        CONCAT(ret, 3).value = vget_high_f16(f16b01);                   \
        CONCAT(ret, 4).value = vget_low_f16(f16b10);                    \
        CONCAT(ret, 5).value = vget_high_f16(f16b10);                   \
        CONCAT(ret, 6).value = vget_low_f16(f16b11);                    \
        CONCAT(ret, 7).value = vget_high_f16(f16b11);                   \
    } while (0);

#define TRANSPOSE_8x4(a, ret)                                                  \
    do {                                                                       \
        auto b00 = vzip1_f16(CONCAT(a, 0).value,                               \
                             CONCAT(a, 1).value); /*a1b1a2b2*/                 \
        auto b01 = vzip2_f16(CONCAT(a, 0).value,                               \
                             CONCAT(a, 1).value); /*a3b3a4b4*/                 \
        auto b10 = vzip1_f16(CONCAT(a, 2).value,                               \
                             CONCAT(a, 3).value); /*c1d1c2d2*/                 \
        auto b11 = vzip2_f16(CONCAT(a, 2).value,                               \
                             CONCAT(a, 3).value); /*c3d3c4d4*/                 \
        auto b20 = vzip1_f16(CONCAT(a, 4).value,                               \
                             CONCAT(a, 5).value); /*e1f1e2f2*/                 \
        auto b21 = vzip2_f16(CONCAT(a, 4).value,                               \
                             CONCAT(a, 5).value); /*e3f3e4f4*/                 \
        auto b30 = vzip1_f16(CONCAT(a, 6).value,                               \
                             CONCAT(a, 7).value); /*g1h1g2h2*/                 \
        auto b31 = vzip2_f16(CONCAT(a, 6).value,                               \
                             CONCAT(a, 7).value); /*g3h3g4h4*/                 \
        auto s32b00 = vreinterpret_s32_f16(b00);                               \
        auto s32b01 = vreinterpret_s32_f16(b01);                               \
        auto s32b10 = vreinterpret_s32_f16(b10);                               \
        auto s32b11 = vreinterpret_s32_f16(b11);                               \
        auto s32b20 = vreinterpret_s32_f16(b20);                               \
        auto s32b21 = vreinterpret_s32_f16(b21);                               \
        auto s32b30 = vreinterpret_s32_f16(b30);                               \
        auto s32b31 = vreinterpret_s32_f16(b31);                               \
        CONCAT(ret, 0).value =                                                 \
                vcombine_f16(vreinterpret_f16_s32(vzip1_s32(s32b00, s32b10)),  \
                             vreinterpret_f16_s32(vzip1_s32(s32b20, s32b30))); \
        CONCAT(ret, 1).value =                                                 \
                vcombine_f16(vreinterpret_f16_s32(vzip2_s32(s32b00, s32b10)),  \
                             vreinterpret_f16_s32(vzip2_s32(s32b20, s32b30))); \
        CONCAT(ret, 2).value =                                                 \
                vcombine_f16(vreinterpret_f16_s32(vzip1_s32(s32b01, s32b11)),  \
                             vreinterpret_f16_s32(vzip1_s32(s32b21, s32b31))); \
        CONCAT(ret, 3).value =                                                 \
                vcombine_f16(vreinterpret_f16_s32(vzip2_s32(s32b01, s32b11)),  \
                             vreinterpret_f16_s32(vzip2_s32(s32b21, s32b31))); \
    } while (0);

#define TRANSPOSE_8x8(a, ret)                                            \
    do {                                                                 \
        auto b00 = vzip1q_f16(CONCAT(a, 0).value,                        \
                              CONCAT(a, 1).value); /*a1b1a2b2 a3b3a4b4*/ \
        auto b01 = vzip2q_f16(CONCAT(a, 0).value,                        \
                              CONCAT(a, 1).value); /*a5b5a6b6 a7b7a8b8*/ \
        auto b10 = vzip1q_f16(CONCAT(a, 2).value,                        \
                              CONCAT(a, 3).value); /*c1d1c2d2 c3d3c4d4*/ \
        auto b11 = vzip2q_f16(CONCAT(a, 2).value,                        \
                              CONCAT(a, 3).value); /*c5d5c6d6 c7d7c8d8*/ \
        auto b20 = vzip1q_f16(CONCAT(a, 4).value,                        \
                              CONCAT(a, 5).value); /*e1f1e2f2 e3f3e4f4*/ \
        auto b21 = vzip2q_f16(CONCAT(a, 4).value,                        \
                              CONCAT(a, 5).value); /*e5f5e6f6 e7f7e8f8*/ \
        auto b30 = vzip1q_f16(CONCAT(a, 6).value,                        \
                              CONCAT(a, 7).value); /*g1h1g2h2 g3h3g4h4*/ \
        auto b31 = vzip2q_f16(CONCAT(a, 6).value,                        \
                              CONCAT(a, 7).value); /*g5h5g6h6 g7h7g8h8*/ \
        auto s32b00 = vreinterpretq_s32_f16(b00);                        \
        auto s32b01 = vreinterpretq_s32_f16(b01);                        \
        auto s32b10 = vreinterpretq_s32_f16(b10);                        \
        auto s32b11 = vreinterpretq_s32_f16(b11);                        \
        auto s32b20 = vreinterpretq_s32_f16(b20);                        \
        auto s32b21 = vreinterpretq_s32_f16(b21);                        \
        auto s32b30 = vreinterpretq_s32_f16(b30);                        \
        auto s32b31 = vreinterpretq_s32_f16(b31);                        \
        auto s64b00 = vreinterpretq_s64_s32(                             \
                vzip1q_s32(s32b00, s32b10)); /*a1b1c1d1 a2b2c2d2*/       \
        auto s64b01 = vreinterpretq_s64_s32(                             \
                vzip2q_s32(s32b00, s32b10)); /*a3b3c3d3 a4b4c4d4*/       \
        auto s64b10 = vreinterpretq_s64_s32(                             \
                vzip1q_s32(s32b01, s32b11)); /*a5b5c5d5 a6b6c6d6*/       \
        auto s64b11 = vreinterpretq_s64_s32(                             \
                vzip2q_s32(s32b01, s32b11)); /*a7b7c7d7 a8b8c8d8*/       \
        auto s64b20 = vreinterpretq_s64_s32(                             \
                vzip1q_s32(s32b20, s32b30)); /*e1f1g1h1 e2f2g2h2*/       \
        auto s64b21 = vreinterpretq_s64_s32(                             \
                vzip2q_s32(s32b20, s32b30)); /*e3f3g3h3 e4f4g4h4*/       \
        auto s64b30 = vreinterpretq_s64_s32(                             \
                vzip1q_s32(s32b21, s32b31)); /*e5f5g5h5 e6f6g6h6*/       \
        auto s64b31 = vreinterpretq_s64_s32(                             \
                vzip2q_s32(s32b21, s32b31)); /*e7f7g7h7 e8f8g8h8*/       \
        CONCAT(ret, 0).value =                                           \
                vreinterpretq_f16_s64(vzip1q_s64(s64b00, s64b20));       \
        CONCAT(ret, 1).value =                                           \
                vreinterpretq_f16_s64(vzip2q_s64(s64b00, s64b20));       \
        CONCAT(ret, 2).value =                                           \
                vreinterpretq_f16_s64(vzip1q_s64(s64b01, s64b21));       \
        CONCAT(ret, 3).value =                                           \
                vreinterpretq_f16_s64(vzip2q_s64(s64b01, s64b21));       \
        CONCAT(ret, 4).value =                                           \
                vreinterpretq_f16_s64(vzip1q_s64(s64b10, s64b30));       \
        CONCAT(ret, 5).value =                                           \
                vreinterpretq_f16_s64(vzip2q_s64(s64b10, s64b30));       \
        CONCAT(ret, 6).value =                                           \
                vreinterpretq_f16_s64(vzip1q_s64(s64b11, s64b31));       \
        CONCAT(ret, 7).value =                                           \
                vreinterpretq_f16_s64(vzip2q_s64(s64b11, s64b31));       \
    } while (0);

#else

#define TRANSPOSE_4x4(a, ret)                                            \
    do {                                                                 \
        auto b0_01 = vzip_f16(CONCAT(a, 0).value,                        \
                              CONCAT(a, 1).value); /*a1b1a2b2 a3b3a4b4*/ \
        auto b1_01 = vzip_f16(CONCAT(a, 2).value,                        \
                              CONCAT(a, 3).value); /*c1d1c2d2 c3d3c4d4*/ \
        auto s32b00 = vreinterpret_s32_f16(b0_01.val[0]);                \
        auto s32b01 = vreinterpret_s32_f16(b0_01.val[1]);                \
        auto s32b10 = vreinterpret_s32_f16(b1_01.val[0]);                \
        auto s32b11 = vreinterpret_s32_f16(b1_01.val[1]);                \
        auto s32b00b10 = vzip_s32(s32b00, s32b10); /*a1b1c1d1 a2b2c2d2*/ \
        auto s32b01b11 = vzip_s32(s32b01, s32b11); /*a3b3c3d3 a4b4c4d4*/ \
        CONCAT(ret, 0).value = vreinterpret_f16_s32(s32b00b10.val[0]);   \
        CONCAT(ret, 1).value = vreinterpret_f16_s32(s32b00b10.val[1]);   \
        CONCAT(ret, 2).value = vreinterpret_f16_s32(s32b01b11.val[0]);   \
        CONCAT(ret, 3).value = vreinterpret_f16_s32(s32b01b11.val[1]);   \
    } while (0);

#define TRANSPOSE_4x8(a, ret)                                              \
    do {                                                                   \
        auto b0_01 = vzipq_f16(                                            \
                CONCAT(a, 0).value,                                        \
                CONCAT(a, 1).value); /*a1b1a2b2a3b3a4b4 a5b5a6b6a7b7a8b8*/ \
        auto b1_01 = vzipq_f16(                                            \
                CONCAT(a, 2).value,                                        \
                CONCAT(a, 3).value); /*c1d1c2d2c3d3c4d4 c5d6c6d6c7d7c8d8*/ \
        auto s32b00 = vreinterpretq_s32_f16(b0_01.val[0]);                 \
        auto s32b01 = vreinterpretq_s32_f16(b0_01.val[1]);                 \
        auto s32b10 = vreinterpretq_s32_f16(b1_01.val[0]);                 \
        auto s32b11 = vreinterpretq_s32_f16(b1_01.val[1]);                 \
        auto s32b00b10 = vzipq_s32(                                        \
                s32b00, s32b10); /*a1b1c1d1a2b2c2d2 a3b3c3d3a4b4c4d4*/     \
        auto s32b01b11 = vzipq_s32(                                        \
                s32b01, s32b11); /*a5b5c5d5a6b6c6d6 a7b7c7d7a8b8c8d8*/     \
        CONCAT(ret, 0).value =                                             \
                vreinterpret_f16_s32(vget_low_f16(s32b00b10.val[0]));      \
        CONCAT(ret, 1).value =                                             \
                vreinterpret_f16_s32(vget_high_f16(s32b00b10.val[0]));     \
        CONCAT(ret, 2).value =                                             \
                vreinterpret_f16_s32(vget_low_f16(s32b00b10.val[1]));      \
        CONCAT(ret, 3).value =                                             \
                vreinterpret_f16_s32(vget_high_f16(s32b00b10.val[1]));     \
        CONCAT(ret, 4).value =                                             \
                vreinterpret_f16_s32(vget_low_f16(s32b01b11.val[0]));      \
        CONCAT(ret, 5).value =                                             \
                vreinterpret_f16_s32(vget_high_f16(s32b01b11.val[0]));     \
        CONCAT(ret, 6).value =                                             \
                vreinterpret_f16_s32(vget_low_f16(s32b01b11.val[1]));      \
        CONCAT(ret, 7).value =                                             \
                vreinterpret_f16_s32(vget_high_f16(s32b01b11.val[1]));     \
    } while (0);

#define TRANSPOSE_8x4(a, ret)                                            \
    do {                                                                 \
        auto b0_01 = vzip_f16(CONCAT(a, 0).value,                        \
                              CONCAT(a, 1).value); /*a1b1a2b2 a3b3a4b4*/ \
        auto b1_01 = vzip_f16(CONCAT(a, 2).value,                        \
                              CONCAT(a, 3).value); /*c1d1c2d2 c3d3c4d4*/ \
        auto b2_01 = vzip_f16(CONCAT(a, 4).value,                        \
                              CONCAT(a, 5).value); /*e1f1e2f2 e3f3e4f4*/ \
        auto b3_01 = vzip_f16(CONCAT(a, 6).value,                        \
                              CONCAT(a, 7).value); /*g1h1g2h2 g3h3g4h4*/ \
        auto s32b00 = vreinterpret_s32_f16(b0_01.val[0]);                \
        auto s32b01 = vreinterpret_s32_f16(b0_01.val[1]);                \
        auto s32b10 = vreinterpret_s32_f16(b1_01.val[0]);                \
        auto s32b11 = vreinterpret_s32_f16(b1_01.val[1]);                \
        auto s32b20 = vreinterpret_s32_f16(b2_01.val[0]);                \
        auto s32b21 = vreinterpret_s32_f16(b2_01.val[1]);                \
        auto s32b30 = vreinterpret_s32_f16(b3_01.val[0]);                \
        auto s32b31 = vreinterpret_s32_f16(b3_01.val[1]);                \
        auto s32b00b10 = vzip_s32(s32b00, s32b10);                       \
        auto s32b01b11 = vzip_s32(s32b01, s32b11);                       \
        auto s32b20b30 = vzip_s32(s32b20, s32b30);                       \
        auto s32b21b31 = vzip_s32(s32b21, s32b31);                       \
        CONCAT(ret, 0).value =                                           \
                vcombine_f16(vreinterpret_f16_s32(s32b00b10.val[0]),     \
                             vreinterpret_f16_s32(s32b20b30.val[0]));    \
        CONCAT(ret, 1).value =                                           \
                vcombine_f16(vreinterpret_f16_s32(s32b00b10.val[1]),     \
                             vreinterpret_f16_s32(s32b20b30.val[1]));    \
        CONCAT(ret, 2).value =                                           \
                vcombine_f16(vreinterpret_f16_s32(s32b01b11.val[0]),     \
                             vreinterpret_f16_s32(s32b21b31.val[0]));    \
        CONCAT(ret, 3).value =                                           \
                vcombine_f16(vreinterpret_f16_s32(s32b01b11.val[1]),     \
                             vreinterpret_f16_s32(s32b21b31.val[1]));    \
    } while (0);

#define TRANSPOSE_8x8(a, ret)                                           \
    do {                                                                \
        auto b00 = vzipq_f16(CONCAT(a, 0).value,                        \
                             CONCAT(a, 1).value); /*a1b1a2b2 a3b3a4b4*/ \
        auto b01 = vzipq_f16(CONCAT(a, 0).value,                        \
                             CONCAT(a, 1).value); /*a5b5a6b6 a7b7a8b8*/ \
        auto b10 = vzipq_f16(CONCAT(a, 2).value,                        \
                             CONCAT(a, 3).value); /*c1d1c2d2 c3d3c4d4*/ \
        auto b11 = vzipq_f16(CONCAT(a, 2).value,                        \
                             CONCAT(a, 3).value); /*c5d5c6d6 c7d7c8d8*/ \
        auto b20 = vzipq_f16(CONCAT(a, 4).value,                        \
                             CONCAT(a, 5).value); /*e1f1e2f2 e3f3e4f4*/ \
        auto b21 = vzipq_f16(CONCAT(a, 4).value,                        \
                             CONCAT(a, 5).value); /*e5f5e6f6 e7f7e8f8*/ \
        auto b30 = vzipq_f16(CONCAT(a, 6).value,                        \
                             CONCAT(a, 7).value); /*g1h1g2h2 g3h3g4h4*/ \
        auto b31 = vzipq_f16(CONCAT(a, 6).value,                        \
                             CONCAT(a, 7).value); /*g5h5g6h6 g7h7g8h8*/ \
        auto s32b00 = vreinterpretq_s32_f16(b00.val[0]);                \
        auto s32b01 = vreinterpretq_s32_f16(b01.val[1]);                \
        auto s32b10 = vreinterpretq_s32_f16(b10.val[0]);                \
        auto s32b11 = vreinterpretq_s32_f16(b11.val[1]);                \
        auto s32b20 = vreinterpretq_s32_f16(b20.val[0]);                \
        auto s32b21 = vreinterpretq_s32_f16(b21.val[1]);                \
        auto s32b30 = vreinterpretq_s32_f16(b30.val[0]);                \
        auto s32b31 = vreinterpretq_s32_f16(b31.val[1]);                \
        auto s32b00b10 = vzipq_s32(s32b00, s32b10);                     \
        auto s32b01b11 = vzipq_s32(s32b01, s32b11);                     \
        auto s32b20b30 = vzipq_s32(s32b20, s32b30);                     \
        auto s32b21b31 = vzipq_s32(s32b21, s32b31);                     \
        CONCAT(ret, 0).value = vreinterpretq_f16_s32(                   \
                vcombine_s32(vget_low_s32(s32b00b10.val[0]),            \
                             vget_low_s32(s32b20b30.val[0])));          \
        CONCAT(ret, 1).value = vreinterpretq_f16_s32(                   \
                vcombine_s32(vget_high_s32(s32b00b10.val[0]),           \
                             vget_high_s32(s32b20b30.val[0])));         \
        CONCAT(ret, 2).value = vreinterpretq_f16_s32(                   \
                vcombine_s32(vget_low_s32(s32b00b10.val[1]),            \
                             vget_low_s32(s32b20b30.val[1])));          \
        CONCAT(ret, 3).value = vreinterpretq_f16_s32(                   \
                vcombine_s32(vget_high_s32(s32b00b10.val[1]),           \
                             vget_high_s32(s32b20b30.val[1])));         \
        CONCAT(ret, 4).value = vreinterpretq_f16_s32(                   \
                vcombine_s32(vget_low_s32(s32b01b11.val[0]),            \
                             vget_low_s32(s32b21b31.val[0])));          \
        CONCAT(ret, 5).value = vreinterpretq_f16_s32(                   \
                vcombine_s32(vget_high_s32(s32b01b11.val[0]),           \
                             vget_high_s32(s32b21b31.val[0])));         \
        CONCAT(ret, 6).value = vreinterpretq_f16_s32(                   \
                vcombine_s32(vget_low_s32(s32b01b11.val[1]),            \
                             vget_low_s32(s32b21b31.val[1])));          \
        CONCAT(ret, 7).value = vreinterpretq_f16_s32(                   \
                vcombine_s32(vget_high_s32(s32b01b11.val[1]),           \
                             vget_high_s32(s32b21b31.val[1])));         \
    } while (0);

#endif
#endif
// vim: syntax=cpp.doxygen
