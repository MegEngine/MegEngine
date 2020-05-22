/**
 * \file dnn/src/aarch64/matrix_mul/fp16/strategy.h
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

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
namespace megdnn {
namespace aarch64 {
namespace matmul {

MEGDNN_REG_GEMM_STRATEGY(dt_float16, dt_float16, dt_float16, 8, 24, 1, false,
                         true, hgemm_8x24);

MEGDNN_REG_GEMM_STRATEGY_NOPACK(dt_float16, dt_float16, dt_float16, 8, 8, 1,
                                false, true, gemm_nopack_f16_8x8);

}  // namespace matmul
}  // namespace aarch64
}  // namespace megdnn
#endif
// vim: syntax=cpp.doxygen
