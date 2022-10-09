#pragma once
#include "src/fallback/matrix_mul/gemm_common.h"

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
namespace megdnn {
namespace aarch64 {
namespace matmul {

MEGDNN_REG_GEMM_STRATEGY(
        dt_float16, dt_float16, dt_float16, 8, 24, 1, false, true, hgemm_8x24);

MEGDNN_REG_GEMM_STRATEGY(
        dt_float16, dt_float16, dt_float16, 16, 12, 1, false, false, hgemm_mk8_16x12);

MEGDNN_REG_GEMM_STRATEGY_NOPACK(
        dt_float16, dt_float16, dt_float16, 8, 8, 1, false, true, gemm_nopack_f16_8x8);

}  // namespace matmul
}  // namespace aarch64
}  // namespace megdnn
#endif
// vim: syntax=cpp.doxygen
