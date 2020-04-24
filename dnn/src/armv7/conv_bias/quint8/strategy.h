/**
 * \file dnn/src/armv7/conv_bias/quint8/strategy.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
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
MEGDNN_REG_GEMM_STRATEGY_WITH_WRITEBACK(dt_uint8, dt_uint8, dt_int32, 4, 8, 8,
                                        false, true,
                                        gemm_u8_4x8_nobias_identity);

MEGDNN_REG_GEMM_STRATEGY_WITH_SUPER(gemm_u8_4x8_nobias_relu,
                                    gemm_u8_4x8_nobias_identity);

MEGDNN_REG_GEMM_STRATEGY_WITH_SUPER(gemm_u8_4x8_nobias_hswish,
                                    gemm_u8_4x8_nobias_identity);

MEGDNN_REG_GEMM_STRATEGY_WITH_SUPER(gemm_u8_4x8_bias_channel_identity,
                                    gemm_u8_4x8_nobias_identity);

MEGDNN_REG_GEMM_STRATEGY_WITH_SUPER(gemm_u8_4x8_bias_channel_relu,
                                    gemm_u8_4x8_nobias_identity);

MEGDNN_REG_GEMM_STRATEGY_WITH_SUPER(gemm_u8_4x8_bias_channel_hswish,
                                    gemm_u8_4x8_nobias_identity);

}  // namespace matmul
}  // namespace armv7
}  // namespace megdnn

// vim: syntax=cpp.doxygen
