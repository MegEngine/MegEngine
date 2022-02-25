#pragma once
#include "src/fallback/matrix_mul/gemm_common.h"

namespace megdnn {
namespace armv7 {
namespace matmul {

MEGDNN_REG_GEMM_STRATEGY(
        dt_uint8, dt_int32, dt_int32, 4, 8, 8, false, true, gemm_u8_4x8);
#if MGB_ENABLE_DOT
MEGDNN_REG_GEMM_STRATEGY(
        dt_uint8, dt_int32, dt_int32, 4, 8, 4, false, false, gemm_dot_quint8_4x8);
#endif

}  // namespace matmul
}  // namespace armv7
}  // namespace megdnn

// vim: syntax=cpp.doxygen
