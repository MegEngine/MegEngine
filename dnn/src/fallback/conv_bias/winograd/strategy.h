/**
 * \file dnn/src/fallback/conv_bias/winograd/strategy.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "src/fallback/conv_bias/winograd/winograd.h"

namespace megdnn {
namespace fallback {
namespace winograd {

MEGDNN_REG_WINOGRAD_STRATEGY(float, float, float, float, 2, 3, 1, 1,
                             winograd_2x3_1x1_f)

MEGDNN_REG_WINOGRAD_STRATEGY(float, float, float, float, 2, 3, 4, 4,
                             winograd_2x3_4x4_f)

MEGDNN_REG_WINOGRAD_STRATEGY(int8_t, int8_t, int16_t, int, 2, 3, 1, 1,
                             winograd_2x3_1x1_qs8)

MEGDNN_REG_WINOGRAD_STRATEGY(int8_t, int8_t, int16_t, int, 2, 3, 8, 8,
                             winograd_2x3_8x8_qs8)

}
}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
