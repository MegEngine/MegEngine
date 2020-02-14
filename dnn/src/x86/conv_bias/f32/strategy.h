/**
 * \file dnn/src/x86/conv_bias/f32/strategy.h
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
#include "src/x86/conv_bias/postprocess_helper.h"

namespace megdnn {
namespace x86 {
namespace winograd {

MEGDNN_REG_WINOGRAD_STRATEGY(float, float, float, float, 6, 3, 8, 8,
                             winograd_nchw88_6x3_8x8_f)

MEGDNN_REG_WINOGRAD_STRATEGY(float, float, float, float, 2, 3, 8, 8,
                             winograd_nchw88_2x3_8x8_f)
}  // namespace winograd
}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
