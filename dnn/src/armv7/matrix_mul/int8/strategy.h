#pragma once
#include "src/fallback/matrix_mul/gemm_common.h"

namespace megdnn {
namespace armv7 {
namespace matmul {

MEGDNN_REG_GEMM_STRATEGY(
        dt_int8, dt_int32, dt_int32, 4, 2, 16, false, true, gemm_s8_4x2);

MEGDNN_REG_GEMM_STRATEGY(
        dt_int8, dt_int32, dt_int32, 4, 8, 8, false, true, gemm_s8_4x8);

MEGDNN_REG_GEMM_STRATEGY(
        dt_int8, dt_int32, dt_int32, 4, 2, 16, false, false, gemm_mk4_s8_4x2);
#if MGB_ENABLE_DOT
MEGDNN_REG_GEMM_STRATEGY(
        dt_int8, dt_int32, dt_int32, 6, 8, 4, false, false, gemm_dots8_6x8);

MEGDNN_REG_GEMM_STRATEGY(
        dt_int8, dt_int32, dt_int32, 8, 4, 4, false, false, gemm_mk4_dots8_8x4);
#endif
}  // namespace matmul
}  // namespace armv7
}  // namespace megdnn

// vim: syntax=cpp.doxygen
