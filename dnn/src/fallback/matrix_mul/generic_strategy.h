#pragma once
#include "src/fallback/general_intrinsic/gi_common.h"
#include "src/fallback/matrix_mul/gemm_common.h"

namespace megdnn {
namespace matmul {
namespace fallback {

MEGDNN_REG_GEMM_STRATEGY(float, float, float, 8, 12, 1, false, true, sgemm_8x12);
MEGDNN_REG_GEMM_STRATEGY_NOPACK(
        float, float, float, 4, 8, 1, false, true, gi_sgemm_nopack_4x8);
#if defined(GI_SUPPORT_F16)
MEGDNN_REG_GEMM_STRATEGY_NOPACK(
        dt_float16, dt_float16, dt_float16, 8, 8, 1, false, true,
        gi_sgemm_nopack_mk8_8x8_fp16);
#endif
MEGDNN_REG_GEMM_STRATEGY(float, float, float, 4, 12, 1, false, true, gi_sgemm_4x12);
MEGDNN_REG_GEMM_STRATEGY(
        float, float, float, 4, 12, 1, false, false, gi_sgemm_mk4_pack_4x12);

}  // namespace fallback
}  // namespace matmul
}  // namespace megdnn

// vim: syntax=cpp.doxygen
