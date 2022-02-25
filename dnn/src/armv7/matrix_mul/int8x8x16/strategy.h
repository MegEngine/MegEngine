#pragma once
#include "src/fallback/matrix_mul/gemm_common.h"

namespace megdnn {
namespace armv7 {
namespace matmul {

MEGDNN_REG_GEMM_STRATEGY(
        int8_t, int16_t, int16_t, 4, 2, 16, false, true, gemm_s8x8x16_4x2);

MEGDNN_REG_GEMM_STRATEGY(
        int8_t, int16_t, int16_t, 4, 8, 8, false, true, gemm_s8x8x16_4x8);

MEGDNN_REG_GEMM_STRATEGY(
        int8_t, int16_t, int16_t, 8, 8, 4, false, true, gemm_s8x8x16_8x8);

MEGDNN_REG_GEMM_STRATEGY_WITH_PACK_A_TYPE(
        int8_t, int16_t, int16_t, int16_t, 8, 8, 4, false, false, gemm_s8x8x16_mk4_8x8);

}  // namespace matmul
}  // namespace armv7
}  // namespace megdnn

// vim: syntax=cpp.doxygen
