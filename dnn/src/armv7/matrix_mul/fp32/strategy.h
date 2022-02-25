#pragma once
#include "src/fallback/matrix_mul/gemm_common.h"

namespace megdnn {
namespace armv7 {
namespace matmul {

MEGDNN_REG_GEMM_STRATEGY(float, float, float, 4, 12, 1, false, true, sgemm_4x12);

MEGDNN_REG_GEMM_STRATEGY(
        float, float, float, 4, 12, 1, false, false, sgemm_mk4_pack_4x12);

MEGDNN_REG_GEMM_STRATEGY_NOPACK(
        float, float, float, 4, 8, 1, false, true, sgemm_nopack_4x8);

}  // namespace matmul
}  // namespace armv7
}  // namespace megdnn

// vim: syntax=cpp.doxygen
