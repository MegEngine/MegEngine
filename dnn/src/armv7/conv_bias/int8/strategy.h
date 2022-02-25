#pragma once
#include "src/fallback/matrix_mul/gemm_common.h"

namespace megdnn {
namespace armv7 {
namespace matmul {

/**
 * \brief base strategy of gemm.
 *
 * \name gemm_<type>_<block>_biasmode_nolinemode
 */
MEGDNN_REG_GEMM_STRATEGY_WITH_WRITEBACK(
        dt_int8, dt_int8, dt_int32, 4, 2, 16, false, true, gemm_s8_4x2_nobias_identity);

MEGDNN_REG_GEMM_STRATEGY_WITH_SUPER(
        gemm_s8_4x2_nobias_relu, gemm_s8_4x2_nobias_identity);

MEGDNN_REG_GEMM_STRATEGY_WITH_SUPER(
        gemm_s8_4x2_nobias_hswish, gemm_s8_4x2_nobias_identity);

MEGDNN_REG_GEMM_STRATEGY_WITH_SUPER(
        gemm_s8_4x2_bias_channel_identity, gemm_s8_4x2_nobias_identity);

MEGDNN_REG_GEMM_STRATEGY_WITH_SUPER(
        gemm_s8_4x2_bias_channel_relu, gemm_s8_4x2_nobias_identity);

MEGDNN_REG_GEMM_STRATEGY_WITH_SUPER(
        gemm_s8_4x2_bias_channel_hswish, gemm_s8_4x2_nobias_identity);

}  // namespace matmul
}  // namespace armv7
}  // namespace megdnn

// vim: syntax=cpp.doxygen
