/**
 * \file dnn/src/armv7/matrix_mul/fp32/strategy.h
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

MEGDNN_REG_GEMM_STRATEGY(float, float, float, 4, 12, 1, false, true,
                         sgemm_4x12);

MEGDNN_REG_GEMM_STRATEGY(float, float, float, 4, 12, 1, false, false,
                         sgemm_mk4_pack_4x12);

MEGDNN_REG_GEMM_STRATEGY_NOPACK(float, float, float, 4, 8, 1, false, true,
                                sgemm_nopack_4x8);

}  // namespace matmul
}  // namespace armv7
}  // namespace megdnn

// vim: syntax=cpp.doxygen
