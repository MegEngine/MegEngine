#pragma once

#include "src/fallback/conv_bias/winograd/winograd.h"
#include "src/x86/conv_bias/postprocess_helper.h"

namespace megdnn {
namespace x86 {
namespace winograd {

MEGDNN_REG_WINOGRAD_STRATEGY(
        float, float, float, float, 6, 3, 8, 8, winograd_nchw88_6x3_8x8_f)

MEGDNN_REG_WINOGRAD_STRATEGY(
        float, float, float, float, 2, 3, 8, 8, winograd_nchw88_2x3_8x8_f)
}  // namespace winograd
}  // namespace x86
}  // namespace megdnn

// vim: syntax=cpp.doxygen
