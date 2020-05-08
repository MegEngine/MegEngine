/**
 * \file dnn/src/aarch64/matrix_mul/quint8_dot/strategy.h
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

MEGDNN_REG_GEMM_STRATEGY(uint8_t, int32_t, int32_t, 8, 8, 4, false, true,
                         gemm_u8_8x8);

}  // namespace aarch64
}  // namespace matmul
}  // namespace megdnn
#endif
// vim: syntax=cpp.doxygen
