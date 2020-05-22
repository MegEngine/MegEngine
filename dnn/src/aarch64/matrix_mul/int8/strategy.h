/**
 * \file dnn/src/aarch64/matrix_mul/int8/strategy.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#if !(__ARM_FEATURE_DOTPROD)
#include "src/fallback/matrix_mul/gemm_common.h"

namespace megdnn {
namespace aarch64 {
namespace matmul {

MEGDNN_REG_GEMM_STRATEGY(dt_int8, dt_int32, dt_int32, 4, 4, 16, false, true,
                         gemm_s8_4x4);

MEGDNN_REG_GEMM_STRATEGY(dt_int8, dt_int32, dt_int32, 4, 4, 16, false, false,
                         gemm_mk4_s8_4x4);

MEGDNN_REG_GEMM_STRATEGY(dt_int8, dt_int32, dt_int32, 8, 8, 8, false, true,
                         gemm_s8_8x8);

}  // namespace matmul
}  // namespace aarch64
}  // namespace megdnn

#endif
// vim: syntax=cpp.doxygen
