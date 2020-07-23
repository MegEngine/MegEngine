/**
 * \file dnn/src/armv7/matrix_mul/int8x8x16/strategy.h
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
namespace armv7 {
namespace matmul {

MEGDNN_REG_GEMM_STRATEGY(int8_t, int16_t, int16_t, 4, 2, 16, false, true,
                         gemm_s8x8x16_4x2);

MEGDNN_REG_GEMM_STRATEGY(int8_t, int16_t, int16_t, 4, 8, 8, false, true,
                         gemm_s8x8x16_4x8);

MEGDNN_REG_GEMM_STRATEGY_WITH_PACK_A_TYPE(int8_t, int16_t, int16_t, int16_t, 8,
                                          8, 4, false, false,
                                          gemm_s8x8x16_mk4_8x8);

}  // namespace matmul
}  // namespace armv7
}  // namespace megdnn

// vim: syntax=cpp.doxygen
