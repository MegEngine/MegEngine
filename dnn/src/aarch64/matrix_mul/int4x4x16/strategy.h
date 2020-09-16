/**
 * \file dnn/src/aarch64/matrix_mul/int4x4x16/strategy.h
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

MEGDNN_REG_GEMM_STRATEGY(dt_int8, dt_int16, dt_int16, 8, 8, 8, false, true,
                         gemm_s4x4x16_s4_8x8x8);

}  // namespace matmul
}  // namespace aarch64
}  // namespace megdnn
// vim: syntax=cpp.doxygen
