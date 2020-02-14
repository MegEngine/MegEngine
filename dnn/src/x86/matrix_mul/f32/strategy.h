/**
 * \file dnn/src/x86/matrix_mul/f32/strategy.h
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
namespace x86 {
namespace matmul {

MEGDNN_REG_GEMM_STRATEGY_NOPACK(float, float, float, 8, 8, 8, false, true,
                                sgemm_nopack_8x8_avx2);

}  // namespace matmul
}  // namespace x86
}  // namespace megdnn