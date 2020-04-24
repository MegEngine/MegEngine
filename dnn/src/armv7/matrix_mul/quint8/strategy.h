/**
 * \file dnn/src/armv7/matrix_mul/quint8/strategy.h
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

MEGDNN_REG_GEMM_STRATEGY(dt_uint8, dt_int32, dt_int32, 4, 8, 8, false, true,
                         gemm_u8_4x8);
#if __ARM_FEATURE_DOTPROD
MEGDNN_REG_GEMM_STRATEGY(dt_uint8, dt_int32, dt_int32, 4, 8, 4, false, false,
                         gemm_dot_quint8_4x8);
#endif

}  // namespace matmul
}  // namespace armv7
}  // namespace megdnn

// vim: syntax=cpp.doxygen
