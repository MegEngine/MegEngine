/**
 * \file dnn/src/x86/matrix_mul/int8/strategy.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#pragma once
#include "src/fallback/matrix_mul/gemm_common.h"

namespace megdnn {
namespace x86 {
namespace matmul {

#if MEGDNN_X86_WITH_VNNI

MEGDNN_REG_GEMM_STRATEGY_WITH_PACK_A_TYPE(dt_int8, dt_uint8, dt_int32, dt_int32,
                                          12, 32, 4, false, false,
                                          gemm_int8_vnni_12x32x4);
#endif

MEGDNN_REG_GEMM_STRATEGY(dt_int8, dt_int32, dt_int32, 2, 4, 16, false, false,
                         gemm_avx2_s8s8s32_2x4x16);

MEGDNN_REG_GEMM_STRATEGY_WITH_PACK_A_TYPE(dt_int8, dt_int16, dt_int32, dt_int32,
                                          4, 16, 2, false, false,
                                          gemm_avx2_s8s8s32_4x16x2);

MEGDNN_REG_GEMM_STRATEGY_WITH_PACK_A_TYPE(dt_int8, dt_int16, dt_int16, dt_int32,
                                          4, 16, 2, false, false,
                                          gemm_avx2_s8s8s16_4x16x2);

MEGDNN_REG_GEMM_STRATEGY_WITH_PACK_A_TYPE(dt_int8, dt_int16, dt_int32, dt_int32,
                                          4, 8, 2, false, false,
                                          gemm_sse_s8s8s32_4x8x2);

MEGDNN_REG_GEMM_STRATEGY_WITH_PACK_A_TYPE(dt_int8, dt_int16, dt_int16, dt_int32,
                                          4, 8, 2, false, false,
                                          gemm_sse_s8s8s16_4x8x2);

}  // namespace matmul
}  // namespace x86
}  // namespace megdnn

// vim: syntax=cpp.doxygen
