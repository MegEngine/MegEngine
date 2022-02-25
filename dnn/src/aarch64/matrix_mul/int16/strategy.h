#pragma once

#include "src/fallback/matrix_mul/gemm_common.h"

namespace megdnn {
namespace aarch64 {
namespace matmul {

MEGDNN_REG_GEMM_STRATEGY(
        dt_int16, dt_int32, dt_int32, 12, 8, 1, false, true, gemm_s16_12x8x1);

MEGDNN_REG_GEMM_STRATEGY_NOPACK(
        dt_int16, dt_int32, dt_int32, 8, 8, 1, false, true, gemm_nopack_s16_8x8);

}  // namespace matmul
}  // namespace aarch64
}  // namespace megdnn

// vim: syntax=cpp.doxygen
