#pragma once
#include "src/fallback/matrix_mul/gemm_common.h"

namespace megdnn {
namespace x86 {
namespace matmul {

MEGDNN_REG_GEMM_STRATEGY_NOPACK(
        float, float, float, 8, 8, 8, false, true, sgemm_nopack_8x8_avx2);

MEGDNN_REG_GEMM_STRATEGY_WITH_PACK_A_TYPE(
        float, float, float, float, 6, 16, 1, false, false, sgemm_pack_6x16_avx2);

}  // namespace matmul
}  // namespace x86
}  // namespace megdnn