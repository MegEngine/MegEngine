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
        dt_uint8, dt_uint8, dt_int32, 4, 8, 8, false, true,
        gemm_u8_4x8_nobias_identity);

MEGDNN_REG_GEMM_STRATEGY_WITH_SUPER(
        gemm_u8_4x8_nobias_relu, gemm_u8_4x8_nobias_identity);

MEGDNN_REG_GEMM_STRATEGY_WITH_SUPER(
        gemm_u8_4x8_nobias_hswish, gemm_u8_4x8_nobias_identity);

MEGDNN_REG_GEMM_STRATEGY_WITH_SUPER(
        gemm_u8_4x8_bias_channel_identity, gemm_u8_4x8_nobias_identity);

MEGDNN_REG_GEMM_STRATEGY_WITH_SUPER(
        gemm_u8_4x8_bias_channel_relu, gemm_u8_4x8_nobias_identity);

MEGDNN_REG_GEMM_STRATEGY_WITH_SUPER(
        gemm_u8_4x8_bias_channel_hswish, gemm_u8_4x8_nobias_identity);

}  // namespace matmul
}  // namespace armv7
}  // namespace megdnn

// vim: syntax=cpp.doxygen
