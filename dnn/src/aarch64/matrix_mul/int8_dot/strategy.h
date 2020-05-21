/**
 * \file dnn/src/aarch64/matrix_mul/int8_dot/strategy.h
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

#if __ARM_FEATURE_DOTPROD
namespace megdnn {
namespace aarch64 {
namespace matmul {

MEGDNN_REG_GEMM_STRATEGY(dt_int8, dt_int32, dt_int32, 8, 12, 4, false, true,
                         gemm_s8_8x12);

MEGDNN_REG_GEMM_STRATEGY(dt_int8, dt_int32, dt_int32, 8, 12, 4, false, true,
                         gemm_mk4_s8_8x12);

}  // namespace aarch64
}  // namespace matmul
}  // namespace megdnn
#endif
// vim: syntax=cpp.doxygen
