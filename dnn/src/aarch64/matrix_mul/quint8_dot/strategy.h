#pragma once
#include "src/fallback/matrix_mul/gemm_common.h"

#if MGB_ENABLE_DOT
namespace megdnn {
namespace aarch64 {
namespace matmul {

MEGDNN_REG_GEMM_STRATEGY(
        uint8_t, int32_t, int32_t, 8, 8, 4, false, true, gemm_u8_8x8_dot);

}  // namespace matmul
}  // namespace aarch64
}  // namespace megdnn
#endif
// vim: syntax=cpp.doxygen
