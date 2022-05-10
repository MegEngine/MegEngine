/**
 * \file
 * dnn/src/arm_common/conv_bias/int8/direct_kernels/dot_direct_nchw44_common.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#pragma once
#include "megdnn/arch.h"
#if MGB_ENABLE_DOT
#include "src/arm_common/simd_macro/marm_neon.h"

static inline void quant_store_s8(
        float32x4_t v0, float32x4_t v1, float32x4_t v2, float32x4_t v3, int8_t* ptr,
        int8x16_t relu_reg) {
    int32x4_t i0 = vcvtaq_s32_f32(v0);
    int32x4_t i1 = vcvtaq_s32_f32(v1);
    int32x4_t i2 = vcvtaq_s32_f32(v2);
    int32x4_t i3 = vcvtaq_s32_f32(v3);

    int16x4_t i16_0 = vqmovn_s32(i0);
    int16x4_t i16_1 = vqmovn_s32(i1);
    int16x4_t i16_2 = vqmovn_s32(i2);
    int16x4_t i16_3 = vqmovn_s32(i3);

    int8x8_t i8_0 = vqmovn_s16(vcombine_s16(i16_0, i16_1));
    int8x8_t i8_1 = vqmovn_s16(vcombine_s16(i16_2, i16_3));
    int8x16_t rst = vcombine_s8(i8_0, i8_1);

    rst = vmaxq_s8(rst, relu_reg);

    vst1q_s8(ptr, rst);
}

#endif