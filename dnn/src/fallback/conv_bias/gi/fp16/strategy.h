#pragma once
#include "src/fallback/general_intrinsic/gi_common.h"

#if defined(GI_SUPPORT_F16)

#include "src/fallback/conv_bias/gi/postprocess_helper.h"
#include "src/fallback/conv_bias/winograd/winograd.h"
#include "src/fallback/general_intrinsic/gi_float16.h"

namespace megdnn {
namespace fallback {
namespace winograd {

MEGDNN_REG_WINOGRAD_STRATEGY(
        dt_float16, dt_float16, dt_float16, dt_float16, 4, 3, 8, 8,
        winograd_F43_mk8_f16_nchw88)
}  // namespace winograd
}  // namespace fallback
}  // namespace megdnn

#endif
// vim: syntax=cpp.doxygen