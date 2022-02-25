#pragma once

#include "src/fallback/matrix_mul/gemm_common.h"

namespace megdnn {
namespace aarch64 {
namespace matmul {

MEGDNN_REG_GEMM_STRATEGY(
        dt_uint8, dt_int32, dt_int32, 8, 8, 8, false, true, gemm_u8_8x8);

}  // namespace matmul
}  // namespace aarch64
}  // namespace megdnn

// vim: syntax=cpp.doxygen
