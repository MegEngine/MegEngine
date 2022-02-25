#pragma once
#include "src/fallback/matrix_mul/gemm_common.h"

#if MGB_ENABLE_DOT
namespace megdnn {
namespace aarch64 {
namespace matmul {

MEGDNN_REG_GEMM_STRATEGY(
        dt_int8, dt_int32, dt_int32, 8, 12, 4, false, true, gemm_s8_8x12);

MEGDNN_REG_GEMM_STRATEGY(
        dt_int8, dt_int32, dt_int32, 8, 12, 4, false, true, gemm_mk4_s8_8x12);

}  // namespace matmul
}  // namespace aarch64
}  // namespace megdnn
#endif
// vim: syntax=cpp.doxygen
