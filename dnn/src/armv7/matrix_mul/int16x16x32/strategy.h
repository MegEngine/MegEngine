#pragma once
#include "src/fallback/matrix_mul/gemm_common.h"

namespace megdnn {
namespace armv7 {
namespace matmul {

MEGDNN_REG_GEMM_STRATEGY(
        int16_t, int32_t, int32_t, 12, 4, 1, false, true, gemm_s16x16x32_12x4);

MEGDNN_REG_GEMM_STRATEGY_NOPACK(
        dt_int16, dt_int32, dt_int32, 4, 8, 1, false, true, gemm_nopack_s16_4x8);

}  // namespace matmul
}  // namespace armv7
}  // namespace megdnn

// vim: syntax=cpp.doxygen
