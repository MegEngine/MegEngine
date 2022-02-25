#pragma once

#include "src/arm_common/conv_bias/postprocess_helper.h"
#include "src/fallback/conv_bias/winograd/winograd.h"

namespace megdnn {
namespace arm_common {
namespace winograd {

MEGDNN_REG_WINOGRAD_STRATEGY(
        int8_t, int8_t, int16_t, int, 2, 3, 8, 8, winograd_2x3_8x8_s8)
MEGDNN_REG_WINOGRAD_STRATEGY(
        int8_t, int8_t, int16_t, int, 2, 3, 8, 8, winograd_2x3_8x8_s8_nchw44)
MEGDNN_REG_WINOGRAD_STRATEGY(
        int8_t, int8_t, float, float, 2, 3, 4, 4, winograd_2x3_4x4_s8_f32_nchw44)
}  // namespace winograd
}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
