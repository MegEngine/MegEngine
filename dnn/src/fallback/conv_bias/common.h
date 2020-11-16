/**
 * \file dnn/src/fallback/conv_bias/common.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#pragma once

#include <stdint.h>
#include "megdnn/oprs.h"
#include "src/common/postprocess.h"
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

#define DISPATCH_FILTER_CHANNEL_WISE(filter, kern, arg...) \
    switch (filter) {                                      \
        case 2:                                            \
            kern(2, ##arg);                                \
            break;                                         \
        case 3:                                            \
            kern(3, ##arg);                                \
            break;                                         \
        case 5:                                            \
            kern(5, ##arg);                                \
            break;                                         \
        default:                                           \
            megdnn_assert(0);                              \
            break;                                         \
    }

#define MEGDNN_WINOGRAD_ALGO_FUN_DECLARE(_algo_data_type)                      \
    bool is_reproducible() const override { return true; }                     \
    bool usable(const NCBKernSizeParam& param,                                 \
                AlgoSelectionStrategy algo_selection_strategy) const override; \
    size_t get_workspace(const NCBKernSizeParam& param) const override;        \
    virtual SmallVector<NCBKern> dispatch_kerns(const NCBKernSizeParam& param) \
            const override;                                                    \
    SmallVector<TensorLayout> deduce_preprocessed_filter_layout(               \
            const NCBKernSizeParam& param) const override;                     \
    size_t get_preprocess_workspace(const NCBKernSizeParam& param)             \
            const override;                                                    \
    virtual SmallVector<NCBKern> dispatch_preprocess_kerns(                    \
            const NCBKernSizeParam& param) const override;                     \
    ConvAlgoTypePack get_algo_type() const override {                          \
        return {_algo_data_type, AlgoCategory::WINOGRAD};                      \
    }                                                                          \
    std::string param() const override {                                       \
        std::string ret;                                                       \
        serialize_write_pod(m_matmul_algo, ret);                               \
        serialize_write_pod(m_tile_size, ret);                                 \
        return ret;                                                            \
    }                                                                          \
                                                                               \
private:                                                                       \
    fallback::MatrixMulImpl::AlgoBase* m_matmul_algo;                          \
    mutable std::string m_name;                                                \
    uint32_t m_tile_size;

}  // namespace megdnn

// vim: syntax=cpp.doxygen
