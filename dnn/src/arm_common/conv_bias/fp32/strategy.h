/**
 * \file dnn/src/arm_common/conv_bias/fp32/strategy.h
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

namespace megdnn {
namespace arm_common {
namespace winograd {

MEGDNN_REG_WINOGRAD_STRATEGY(float, float, float, float, 2, 3, 4, 4,
                             winograd_2x3_4x4_f)

MEGDNN_REG_WINOGRAD_STRATEGY(float, float, float, float, 6, 3, 1, 1,
                             winograd_6x3_1x1_f)

MEGDNN_REG_WINOGRAD_STRATEGY(float, float, float, float, 6, 3, 4, 4,
                             winograd_6x3_4x4_f)

MEGDNN_REG_WINOGRAD_STRATEGY(float, float, float, float, 5, 4, 1, 1,
                             winograd_5x4_1x1_f)

MEGDNN_REG_WINOGRAD_STRATEGY(float, float, float, float, 4, 5, 1, 1,
                             winograd_4x5_1x1_f)

MEGDNN_REG_WINOGRAD_STRATEGY(float, float, float, float, 2, 3, 4, 4,
                             winograd_F23_mk4_f_nchw44)

MEGDNN_REG_WINOGRAD_STRATEGY(float, float, float, float, 6, 3, 4, 4,
                             winograd_F63_mk4_f_nchw44)

MEGDNN_REG_WINOGRAD_STRATEGY(float, float, float, float, 7, 3, 4, 4,
                             winograd_F73_mk4_f_nchw44)
}  // namespace winograd
}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
