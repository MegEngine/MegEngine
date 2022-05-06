/**
 * \file dnn/src/fallback/conv_bias/gi/postprocess_helper.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2022 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#pragma once

#include "megdnn/basic_types.h"
#include "src/fallback/conv_bias/opr_impl.h"

#include "midout.h"

MIDOUT_DECL(fallback_gi_conv_bias_postprocess_helper)

namespace {

#define GI_DISPATCH_CONV_WINOGRAD_NONLINE(                                            \
        _midout_tag, cb, _bias_id, _src_type, _dst_type, _bmode, _nonline_mode, ...)  \
    switch (_nonline_mode) {                                                          \
        case param::ConvBias::NonlineMode::IDENTITY: {                                \
            MIDOUT_BEGIN(_midout_tag, _bias_id, 0) {                                  \
                cb(_bmode, NoneOp<_src_type MEGDNN_COMMA _dst_type>, __VA_ARGS__);    \
            }                                                                         \
            MIDOUT_END();                                                             \
            break;                                                                    \
        }                                                                             \
        case param::ConvBias::NonlineMode::RELU: {                                    \
            MIDOUT_BEGIN(_midout_tag, _bias_id, 1) {                                  \
                cb(_bmode, ReluOp<_src_type MEGDNN_COMMA _dst_type>, __VA_ARGS__);    \
            }                                                                         \
            MIDOUT_END();                                                             \
            break;                                                                    \
        }                                                                             \
        case param::ConvBias::NonlineMode::SIGMOID: {                                 \
            MIDOUT_BEGIN(_midout_tag, _bias_id, 2) {                                  \
                cb(_bmode, SigmoidOp<_src_type MEGDNN_COMMA _dst_type>, __VA_ARGS__); \
            }                                                                         \
            MIDOUT_END();                                                             \
            break;                                                                    \
        }                                                                             \
        case param::ConvBias::NonlineMode::H_SWISH: {                                 \
            MIDOUT_BEGIN(_midout_tag, _bias_id, 3) {                                  \
                cb(_bmode, HSwishOp<_src_type MEGDNN_COMMA _dst_type>, __VA_ARGS__);  \
            }                                                                         \
            MIDOUT_END();                                                             \
            break;                                                                    \
        }                                                                             \
        default:                                                                      \
            megdnn_assert(0);                                                         \
            break;                                                                    \
    }

#define GI_DISPATCH_CONV_WINOGRAD_BIAS(                                           \
        _midout_tag, cb, _src_type, _dst_type, _bmode, _nonline_mode, ...)        \
    switch (_bmode) {                                                             \
        case BiasMode::BIAS: {                                                    \
            GI_DISPATCH_CONV_WINOGRAD_NONLINE(                                    \
                    _midout_tag, cb, 0, _src_type, _dst_type, BiasMode::BIAS,     \
                    _nonline_mode, __VA_ARGS__)                                   \
            break;                                                                \
        }                                                                         \
        case BiasMode::NO_BIAS: {                                                 \
            GI_DISPATCH_CONV_WINOGRAD_NONLINE(                                    \
                    _midout_tag, cb, 1, _src_type, _dst_type, BiasMode::NO_BIAS,  \
                    _nonline_mode, __VA_ARGS__)                                   \
            break;                                                                \
        }                                                                         \
        case BiasMode::BROADCAST_CHANNEL_BIAS: {                                  \
            GI_DISPATCH_CONV_WINOGRAD_NONLINE(                                    \
                    _midout_tag, cb, 2, _src_type, _dst_type,                     \
                    BiasMode::BROADCAST_CHANNEL_BIAS, _nonline_mode, __VA_ARGS__) \
            break;                                                                \
        }                                                                         \
        default:                                                                  \
            megdnn_assert(0);                                                     \
            break;                                                                \
    }

}  // namespace
