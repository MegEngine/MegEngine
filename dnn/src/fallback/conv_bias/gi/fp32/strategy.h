#pragma once

#include "src/fallback/conv_bias/gi/postprocess_helper.h"
#include "src/fallback/conv_bias/winograd/winograd.h"

namespace megdnn {
namespace fallback {
namespace winograd {

MEGDNN_REG_WINOGRAD_STRATEGY(
        float, float, float, float, 2, 3, 4, 4, winograd_gi_2x3_4x4_f)

MEGDNN_REG_WINOGRAD_STRATEGY(float, float, float, float, 6, 3, 1, 1, winograd_6x3_1x1_f)

MEGDNN_REG_WINOGRAD_STRATEGY(float, float, float, float, 4, 3, 1, 1, winograd_4x3_1x1_f)

MEGDNN_REG_WINOGRAD_STRATEGY(float, float, float, float, 6, 3, 4, 4, winograd_6x3_4x4_f)

MEGDNN_REG_WINOGRAD_STRATEGY(float, float, float, float, 4, 3, 4, 4, winograd_4x3_4x4_f)

MEGDNN_REG_WINOGRAD_STRATEGY(float, float, float, float, 5, 4, 1, 1, winograd_5x4_1x1_f)

MEGDNN_REG_WINOGRAD_STRATEGY(float, float, float, float, 4, 5, 1, 1, winograd_4x5_1x1_f)

MEGDNN_REG_WINOGRAD_STRATEGY(
        float, float, float, float, 2, 3, 4, 4, winograd_F23_mk4_f_nchw44)

MEGDNN_REG_WINOGRAD_STRATEGY(
        float, float, float, float, 6, 3, 4, 4, winograd_F63_mk4_f_nchw44)

MEGDNN_REG_WINOGRAD_STRATEGY(
        float, float, float, float, 4, 3, 4, 4, winograd_F43_mk4_f_nchw44)

MEGDNN_REG_WINOGRAD_STRATEGY(
        float, float, float, float, 7, 3, 4, 4, winograd_F73_mk4_f_nchw44)
}  // namespace winograd
}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
