/**
 * \file dnn/src/aarch64/matrix_mul/fp32/strategy.h
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
namespace aarch64 {
namespace matmul {
MEGDNN_REG_GEMM_STRATEGY(float, float, float, 8, 12, 1, false, true,
                         sgemm_8x12);

MEGDNN_REG_GEMM_STRATEGY(float, float, float, 4, 16, 1, false, true,
                         sgemm_4x16);

MEGDNN_REG_GEMM_STRATEGY(float, float, float, 8, 12, 1, false, false,
                         sgemm_mk4_8x12);

MEGDNN_REG_GEMM_STRATEGY_NOPACK(float, float, float, 4, 16, 1, false, true,
                                sgemm_nopack_4x16);

}  // namespace matmul
}  // namespace aarch64
}  // namespace megdnn

// vim: syntax=cpp.doxygen
