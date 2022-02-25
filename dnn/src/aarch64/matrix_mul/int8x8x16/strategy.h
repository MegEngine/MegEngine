#pragma once

#include "src/fallback/matrix_mul/gemm_common.h"

namespace megdnn {
namespace aarch64 {
namespace matmul {

MEGDNN_REG_GEMM_STRATEGY(
        dt_int8, dt_int16, dt_int16, 8, 8, 8, false, true, gemm_s8x8x16_8x8);
MEGDNN_REG_GEMM_STRATEGY(
        dt_int8, dt_int16, dt_int16, 4, 4, 16, false, true, gemm_s8x8x16_4x4);
MEGDNN_REG_GEMM_STRATEGY(
        dt_int8, dt_int16, dt_int16, 4, 4, 8, false, false, gemm_s8x8x16_mk4_4x4_a72);
MEGDNN_REG_GEMM_STRATEGY_WITH_PACK_A_TYPE(
        dt_int8, dt_int16, dt_int16, dt_int16, 16, 12, 4, false, false,
        gemm_s8x8x16_mk4_16x12_a53);
MEGDNN_REG_GEMM_STRATEGY(
        dt_int8, dt_int16, dt_int16, 8, 8, 8, false, false, gemm_s8x8x16_mk4_8x8x8);

}  // namespace matmul
}  // namespace aarch64
}  // namespace megdnn
// vim: syntax=cpp.doxygen
