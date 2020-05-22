/**
 * \file dnn/src/armv7/matrix_mul/int16x16x32/strategy.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "src/fallback/matrix_mul/gemm_common.h"

namespace megdnn {
namespace armv7 {
namespace matmul {

MEGDNN_REG_GEMM_STRATEGY(int16_t, int32_t, int32_t, 12, 4, 1, false, true,
                         gemm_s16x16x32_12x4);

MEGDNN_REG_GEMM_STRATEGY_NOPACK(dt_int16, dt_int32, dt_int32, 4, 8, 1, false,
                                true, gemm_nopack_s16_4x8);

}  // namespace matmul
}  // namespace armv7
}  // namespace megdnn

// vim: syntax=cpp.doxygen
