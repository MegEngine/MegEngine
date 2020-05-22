/**
 * \file dnn/src/arm_common/conv_bias/quint8/algos.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/arm_common/conv_bias/quint8/algos.h"
#include "src/arm_common/conv_bias/quint8/stride1.h"
#include "src/arm_common/conv_bias/quint8/stride2.h"
#include "src/arm_common/conv_bias/quint8/stride1_dotprod.h"
#include "src/arm_common/conv_bias/quint8/stride2_dotprod.h"
#include "src/arm_common/elemwise_op.h"
#include "src/fallback/conv_bias/common.h"
#include "midout.h"

MIDOUT_DECL(megdnn_arm_common_conv_bias_quint8)

using namespace megdnn;
using namespace arm_common;

/* ===================== stride1 algo ===================== */
bool ConvBiasImpl::AlgoQU8DirectStride1::usable(
        fallback::ConvBiasImpl*, const NCBKernSizeParam& param,
        AlgoSelectionStrategy algo_selection_strategy) const {
    bool avaible = direct_quint8_stride1::can_conv_direct_stride1_quint8(param);
    if (algo_selection_strategy ==
        ConvBiasImpl::AlgoSelectionStrategy::HEURISTIC) {
        bool large_group = param.filter_meta.group >= param.nr_threads;
        avaible &= (large_group == m_large_group);
    }
    return avaible;
}

size_t ConvBiasImpl::AlgoQU8DirectStride1::get_workspace(
        fallback::ConvBiasImpl*, const NCBKernSizeParam& param) const {
    auto bundle = direct_quint8_stride1::get_bundle(param, m_large_group);
    return bundle.total_size_in_bytes();
}

SmallVector<ConvBiasImpl::NCBKern>
ConvBiasImpl::AlgoQU8DirectStride1::dispatch_kerns(
        fallback::ConvBiasImpl*, const NCBKernSizeParam& param) const {
    MIDOUT_BEGIN(megdnn_arm_common_conv_bias_quint8, 0, 0) {
        return direct_quint8_stride1::get_kimpls(param, m_large_group);
    }
    MIDOUT_END();
    return {};
}

/* ===================== stride2 algo ===================== */
bool ConvBiasImpl::AlgoQU8DirectStride2::usable(
        fallback::ConvBiasImpl*, const NCBKernSizeParam& param,
        AlgoSelectionStrategy algo_selection_strategy) const {
    bool avaible = direct_quint8_stride2::can_conv_direct_stride2_quint8(param);
    if (algo_selection_strategy ==
        ConvBiasImpl::AlgoSelectionStrategy::HEURISTIC) {
        bool large_group = param.filter_meta.group >= param.nr_threads;
        avaible &= (large_group == m_large_group);
    }
    return avaible;
}

size_t ConvBiasImpl::AlgoQU8DirectStride2::get_workspace(
        fallback::ConvBiasImpl*, const NCBKernSizeParam& param) const {
    auto bundle = direct_quint8_stride2::get_bundle(param, m_large_group);
    return bundle.total_size_in_bytes();
}

SmallVector<ConvBiasImpl::NCBKern>
ConvBiasImpl::AlgoQU8DirectStride2::dispatch_kerns(
        fallback::ConvBiasImpl*, const NCBKernSizeParam& param) const {
    MIDOUT_BEGIN(megdnn_arm_common_conv_bias_quint8, 0, 1) {
        return direct_quint8_stride2::get_kimpls(param, m_large_group);
    }
    MIDOUT_END();
    return {};
}
#if __ARM_FEATURE_DOTPROD
/* ===================== stride1 algo ===================== */
bool ConvBiasImpl::AlgoDotU8DirectStride1::usable(
        fallback::ConvBiasImpl*, const NCBKernSizeParam& param,
        AlgoSelectionStrategy algo_selection_strategy) const {
    bool avaible =
            direct_dotprod_quint8_stride1::can_conv_direct_stride1_quint8(
                    param);
    if (algo_selection_strategy ==
        ConvBiasImpl::AlgoSelectionStrategy::HEURISTIC) {
        bool large_group = param.filter_meta.group >= param.nr_threads;
        avaible &= (large_group == m_large_group);
    }
    return avaible;
}

size_t ConvBiasImpl::AlgoDotU8DirectStride1::get_workspace(
        fallback::ConvBiasImpl*, const NCBKernSizeParam& param) const {
    auto bundle =
            direct_dotprod_quint8_stride1::get_bundle(param, m_large_group);
    return bundle.total_size_in_bytes();
}

SmallVector<ConvBiasImpl::NCBKern>
ConvBiasImpl::AlgoDotU8DirectStride1::dispatch_kerns(
        fallback::ConvBiasImpl*, const NCBKernSizeParam& param) const {
    MIDOUT_BEGIN(megdnn_arm_common_conv_bias_quint8, 1, 0) {
        return direct_dotprod_quint8_stride1::get_kimpls(param, m_large_group);
    }
    MIDOUT_END();
    return {};
}

/* ===================== stride2 algo ===================== */
bool ConvBiasImpl::AlgoDotU8DirectStride2::usable(
        fallback::ConvBiasImpl*, const NCBKernSizeParam& param,
        AlgoSelectionStrategy algo_selection_strategy) const {
    bool avaible =
            direct_dotprod_quint8_stride2::can_conv_direct_stride2_quint8(
                    param);
    if (algo_selection_strategy ==
        ConvBiasImpl::AlgoSelectionStrategy::HEURISTIC) {
        bool large_group = param.filter_meta.group >= param.nr_threads;
        avaible &= (large_group == m_large_group);
    }
    return avaible;
}

size_t ConvBiasImpl::AlgoDotU8DirectStride2::get_workspace(
        fallback::ConvBiasImpl*, const NCBKernSizeParam& param) const {
    auto bundle =
            direct_dotprod_quint8_stride2::get_bundle(param, m_large_group);
    return bundle.total_size_in_bytes();
}

SmallVector<ConvBiasImpl::NCBKern>
ConvBiasImpl::AlgoDotU8DirectStride2::dispatch_kerns(
        fallback::ConvBiasImpl*, const NCBKernSizeParam& param) const {
    MIDOUT_BEGIN(megdnn_arm_common_conv_bias_quint8, 1, 1) {
        return direct_dotprod_quint8_stride2::get_kimpls(param, m_large_group);
    }
    MIDOUT_END();
    return {};
}

#endif
// vim: syntax=cpp.doxygen
