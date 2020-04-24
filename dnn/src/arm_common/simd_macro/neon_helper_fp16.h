/**
 * \file dnn/src/arm_common/simd_macro/neon_helper_fp16.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/arm_common/simd_macro/marm_neon.h"

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#define MEGDNN_SIMD_NAME NEON
#define MEGDNN_SIMD_TARGET neon
#define MEGDNN_SIMD_ATTRIBUTE_TARGET
#define MEGDNN_SIMD_WIDTH 4
#define MEGDNN_SIMD_TYPE float16x8_t
#define MEGDNN_SIMD_TYPE2 float16x8x2_t
#define MEGDNN_SIMD_LOADU(addr) vld1q_f16(addr)
#define MEGDNN_SIMD_STOREU(addr, reg) vst1q_f16(addr, reg)
#define MEGDNN_SIMD_SETZERO() vdupq_n_f16(0.0f)
#define MEGDNN_SIMD_SET1(num) vdupq_n_f16(num)

#endif
