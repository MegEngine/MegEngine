#pragma once

#include "src/fallback/matrix_mul/gemm_common.h"

namespace megdnn {
namespace aarch64 {
namespace matmul {

MEGDNN_REG_GEMM_STRATEGY(
        dt_int8, dt_int32, dt_int32, 4, 4, 16, false, true, gemm_s8_4x4);

MEGDNN_REG_GEMM_STRATEGY(
        dt_int8, dt_int32, dt_int32, 4, 4, 16, false, false, gemm_mk4_s8_4x4);

MEGDNN_REG_GEMM_STRATEGY(
        dt_int8, dt_int32, dt_int32, 8, 8, 8, false, true, gemm_s8_8x8);

}  // namespace matmul
}  // namespace aarch64
}  // namespace megdnn

// vim: syntax=cpp.doxygen
