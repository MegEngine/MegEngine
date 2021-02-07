/**
 * \file dnn/src/aarch64/conv_bias/quint8/strategy.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "src/fallback/matrix_mul/gemm_common.h"

namespace megdnn {
namespace aarch64 {
namespace matmul {

#if MGB_ENABLE_DOT
MEGDNN_REG_GEMM_STRATEGY_WITH_WRITEBACK(dt_uint8, dt_uint8, dt_int32, 8, 8, 4,
                                        false, true,
                                        gemm_u8_8x8_dot_nobias_identity);

MEGDNN_REG_GEMM_STRATEGY_WITH_SUPER(gemm_u8_8x8_dot_nobias_relu,
                                    gemm_u8_8x8_dot_nobias_identity);

MEGDNN_REG_GEMM_STRATEGY_WITH_SUPER(gemm_u8_8x8_dot_nobias_hswish,
                                    gemm_u8_8x8_dot_nobias_identity);

MEGDNN_REG_GEMM_STRATEGY_WITH_SUPER(gemm_u8_8x8_dot_bias_channel_identity,
                                    gemm_u8_8x8_dot_nobias_identity);

MEGDNN_REG_GEMM_STRATEGY_WITH_SUPER(gemm_u8_8x8_dot_bias_channel_relu,
                                    gemm_u8_8x8_dot_nobias_identity);

MEGDNN_REG_GEMM_STRATEGY_WITH_SUPER(gemm_u8_8x8_dot_bias_channel_hswish,
                                    gemm_u8_8x8_dot_nobias_identity);


#endif
MEGDNN_REG_GEMM_STRATEGY_WITH_WRITEBACK(dt_uint8, dt_uint8, dt_int32, 8, 8, 8,
                                        false, true,
                                        gemm_u8_8x8_nodot_nobias_identity);

MEGDNN_REG_GEMM_STRATEGY_WITH_SUPER(gemm_u8_8x8_nodot_nobias_relu,
                                    gemm_u8_8x8_nodot_nobias_identity);

MEGDNN_REG_GEMM_STRATEGY_WITH_SUPER(gemm_u8_8x8_nodot_nobias_hswish,
                                    gemm_u8_8x8_nodot_nobias_identity);

MEGDNN_REG_GEMM_STRATEGY_WITH_SUPER(gemm_u8_8x8_nodot_bias_channel_identity,
                                    gemm_u8_8x8_nodot_nobias_identity);

MEGDNN_REG_GEMM_STRATEGY_WITH_SUPER(gemm_u8_8x8_nodot_bias_channel_relu,
                                    gemm_u8_8x8_nodot_nobias_identity);

MEGDNN_REG_GEMM_STRATEGY_WITH_SUPER(gemm_u8_8x8_nodot_bias_channel_hswish,
                                    gemm_u8_8x8_nodot_nobias_identity);


}  // namespace matmul
}  // namespace aarch64
}  // namespace megdnn

// vim: syntax=cpp.doxygen
