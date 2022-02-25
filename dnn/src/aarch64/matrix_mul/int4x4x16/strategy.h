#pragma once

#include "src/fallback/matrix_mul/gemm_common.h"

namespace megdnn {
namespace aarch64 {
namespace matmul {

MEGDNN_REG_GEMM_STRATEGY(
        dt_int8, dt_int16, dt_int16, 8, 8, 8, false, true, gemm_s4x4x16_s4_8x8x8);

}  // namespace matmul
}  // namespace aarch64
}  // namespace megdnn
// vim: syntax=cpp.doxygen
