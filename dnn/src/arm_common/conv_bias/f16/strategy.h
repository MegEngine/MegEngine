/**
 * \file dnn/src/arm_common/conv_bias/f16/strategy.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "src/arm_common/conv_bias/postprocess_helper.h"
#include "src/fallback/conv_bias/winograd/winograd.h"
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
namespace megdnn {
namespace arm_common {
namespace winograd {

MEGDNN_REG_WINOGRAD_STRATEGY(dt_float16, dt_float16, dt_float16, dt_float16, 2,
                             3, 4, 4, winograd_2x3_4x4_f16)
MEGDNN_REG_WINOGRAD_STRATEGY(dt_float16, dt_float16, dt_float16, dt_float16, 4,
                             5, 1, 1, winograd_4x5_1x1_f16)
MEGDNN_REG_WINOGRAD_STRATEGY(dt_float16, dt_float16, dt_float16, dt_float16, 6,
                             3, 1, 1, winograd_6x3_1x1_f16)
MEGDNN_REG_WINOGRAD_STRATEGY(dt_float16, dt_float16, dt_float16, dt_float16, 2,
                             3, 8, 8, winograd_2x3_8x8_f16)
}  // namespace winograd
}  // namespace arm_common
}  // namespace megdnn
#endif
// vim: syntax=cpp.doxygen
