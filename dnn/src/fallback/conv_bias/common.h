/**
 * \file dnn/src/fallback/conv_bias/common.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include <stdint.h>
#include "megdnn/oprs.h"
#include "src/common/utils.h"

namespace megdnn {
using NonlineMode = ConvBias::Param::NonlineMode;
using BiasMode = ConvBiasForward::BiasMode;

#define DISPATCH_GEMM_NONLINE(_gemm, _gemm_midout_enum, _bias,      \
                              _bias_midout_enum)                    \
    switch (param.nonlineMode) {                                    \
        case param::ConvBias::NonlineMode::IDENTITY: {              \
            DISPATCH_GEMM_STRATEGY(_gemm, _gemm_midout_enum, _bias, \
                                   _bias_midout_enum, identity, 0); \
            break;                                                  \
        }                                                           \
        case param::ConvBias::NonlineMode::RELU: {                  \
            DISPATCH_GEMM_STRATEGY(_gemm, _gemm_midout_enum, _bias, \
                                   _bias_midout_enum, relu, 1);     \
            break;                                                  \
        }                                                           \
        case param::ConvBias::NonlineMode::H_SWISH: {               \
            DISPATCH_GEMM_STRATEGY(_gemm, _gemm_midout_enum, _bias, \
                                   _bias_midout_enum, hswish, 2);   \
            break;                                                  \
        }                                                           \
        default:                                                    \
            megdnn_assert(0);                                       \
            break;                                                  \
    }

#define DISPATCH_GEMM_BIAS(_gemm, _gemm_midout_enum)                         \
    switch (param.bias_mode) {                                               \
        case BiasMode::NO_BIAS:                                              \
            DISPATCH_GEMM_NONLINE(_gemm, _gemm_midout_enum, nobias, 0)       \
            break;                                                           \
        case BiasMode::BROADCAST_CHANNEL_BIAS:                               \
            DISPATCH_GEMM_NONLINE(_gemm, _gemm_midout_enum, bias_channel, 1) \
            break;                                                           \
        default:                                                             \
            megdnn_assert(0);                                                \
            break;                                                           \
    }

#define DISPATCH_CONV_NONLINE(i, midout_tag, stride, _conv, BIAS_MODE,         \
                              dst_type)                                        \
    switch (param.nonlineMode) {                                               \
        case param::ConvBias::NonlineMode::IDENTITY: {                         \
            DISPATCH_CONV_STRATEGY(i, midout_tag, stride, _conv, BIAS_MODE,    \
                                   TypeCvtOp<dt_qint32 MEGDNN_COMMA dst_type>, \
                                   0);                                         \
            break;                                                             \
        }                                                                      \
        case param::ConvBias::NonlineMode::RELU: {                             \
            DISPATCH_CONV_STRATEGY(i, midout_tag, stride, _conv, BIAS_MODE,    \
                                   ReluOp<dt_qint32 MEGDNN_COMMA dst_type>,    \
                                   1);                                         \
            break;                                                             \
        }                                                                      \
        case param::ConvBias::NonlineMode::H_SWISH: {                          \
            DISPATCH_CONV_STRATEGY(i, midout_tag, stride, _conv, BIAS_MODE,    \
                                   HSwishOp<dt_qint32 MEGDNN_COMMA dst_type>,  \
                                   2);                                         \
            break;                                                             \
        }                                                                      \
        default:                                                               \
            megdnn_assert(0);                                                  \
            break;                                                             \
    }

#define DISPATCH_CONV_BIAS(i, midout_tag, stride, _conv, dst_type)            \
    switch (param.bias_mode) {                                                \
        case BiasMode::NO_BIAS:                                               \
            DISPATCH_CONV_NONLINE(i, midout_tag, stride, _conv,               \
                                  BiasMode::NO_BIAS, dst_type)                \
            break;                                                            \
        case BiasMode::BROADCAST_CHANNEL_BIAS:                                \
            DISPATCH_CONV_NONLINE(i, midout_tag, stride, _conv,               \
                                  BiasMode::BROADCAST_CHANNEL_BIAS, dst_type) \
            break;                                                            \
        default:                                                              \
            megdnn_assert(0);                                                 \
            break;                                                            \
    }

#define DISPATCH_CONV_STRATEGY(i, midout_tag, stride, conv, BIAS_MODE, Op, \
                               _nonline_midout_enum)                       \
    MIDOUT_BEGIN(midout_tag, i, stride, midout_iv(BIAS_MODE),              \
                 _nonline_midout_enum) {                                   \
        return {{conv<i, BIAS_MODE, Op>, {1_z, 1_z, 1_z}}};                \
    }                                                                      \
    MIDOUT_END()

#define DISPATCH_FILTER(filter, kern, arg...) \
    switch (filter) {                         \
        case 2:                               \
            kern(2, ##arg);                   \
            break;                            \
        case 3:                               \
            kern(3, ##arg);                   \
            break;                            \
        case 5:                               \
            kern(5, ##arg);                   \
            break;                            \
        case 7:                               \
            kern(7, ##arg);                   \
            break;                            \
        default:                              \
            megdnn_assert(0);                 \
            break;                            \
    }

enum class PostprocessMode : uint8_t {
    FLOAT = 0,  ///< support all biasmode and no_nonlinemode
    NO_PROCESS, ///<support  non bias and identity
    QUANTIZED,///<support  NOBIAS ,BROADCAST_CHANNEL_BIAS and relu hswish identify nonline mode   
};
}  // namespace megdnn

// vim: syntax=cpp.doxygen
