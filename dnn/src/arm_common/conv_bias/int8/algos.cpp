/**
 * \file dnn/src/arm_common/conv_bias/int8/algos.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/arm_common/conv_bias/int8/algos.h"
#include "src/arm_common/conv_bias/int8/channel_wise_nchw44.h"
#include "src/arm_common/conv_bias/int8/strategy.h"
#include "src/arm_common/conv_bias/int8/stride1.h"
#include "src/arm_common/conv_bias/int8/stride1_dotprod.h"
#include "src/arm_common/conv_bias/int8/stride2.h"
#include "src/arm_common/conv_bias/int8/stride2_dotprod.h"
#include "src/arm_common/elemwise_op.h"
#include "src/fallback/conv_bias/common.h"

#include "midout.h"

using namespace megdnn;
using namespace arm_common;

MIDOUT_DECL(megdnn_arm_common_conv_bias_int8)
/* ===================== stride1 algo ===================== */
bool ConvBiasImpl::AlgoS8DirectStride1::usable(
        fallback::ConvBiasImpl*, const NCBKernSizeParam& param,
        AlgoSelectionStrategy algo_selection_strategy) const {
    bool avaible = direct_int8_stride1::can_conv_direct_stride1_int8(param);
    auto fm = param.filter_meta;
    if (algo_selection_strategy ==
        ConvBiasImpl::AlgoSelectionStrategy::HEURISTIC) {
        bool large_group = fm.group >= param.nr_threads;
        avaible &= (large_group == m_large_group);
    }
    return avaible;
}
bool ConvBiasImpl::AlgoS8DirectStride1::is_preferred(
        megdnn::fallback::ConvBiasImpl*, const NCBKernSizeParam& param) const {
    auto&& fm = param.filter_meta;
    auto FH = fm.spatial[0];
    auto OC = fm.ocpg;
    auto IC = fm.icpg;
    bool preferred = ((FH == 2 && (OC <= 10 || IC <= 8)) ||
                      ((FH == 3 || FH == 5 || FH == 7) &&
                       (OC <= 16 || (IC <= 4 && OC <= 32)))) &&
                     param.bias_mode != BiasMode::BIAS;
    return preferred;
}

size_t ConvBiasImpl::AlgoS8DirectStride1::get_workspace(
        fallback::ConvBiasImpl*, const NCBKernSizeParam& param) const {
    auto bundle = direct_int8_stride1::get_bundle(param, m_large_group);
    return bundle.total_size_in_bytes();
}

SmallVector<ConvBiasImpl::NCBKern>
ConvBiasImpl::AlgoS8DirectStride1::dispatch_kerns(
        fallback::ConvBiasImpl*, const NCBKernSizeParam& param) const {
    MIDOUT_BEGIN(megdnn_arm_common_conv_bias_int8, 1, 0) {
        return direct_int8_stride1::get_kimpls(param, m_large_group);
    }
    MIDOUT_END();
    return {};
}

/* ===================== stride1 algo ===================== */
bool ConvBiasImpl::AlgoS8ChanWiseStride1NCHW44::usable(
        fallback::ConvBiasImpl*, const NCBKernSizeParam& param,
        AlgoSelectionStrategy) const {
    return channel_wise_nchw44::stride1::is_available(param);
}

size_t ConvBiasImpl::AlgoS8ChanWiseStride1NCHW44::get_workspace(
        fallback::ConvBiasImpl*, const NCBKernSizeParam& param) const {
    auto bundle = channel_wise_nchw44::stride1::get_bundle(param);
    return bundle.total_size_in_bytes();
}

SmallVector<ConvBiasImpl::NCBKern>
ConvBiasImpl::AlgoS8ChanWiseStride1NCHW44::dispatch_kerns(
        fallback::ConvBiasImpl*, const NCBKernSizeParam& param) const {
    MIDOUT_BEGIN(megdnn_arm_common_conv_bias_int8,
                 midout_iv("AlgoS8ChanWiseStride1NCHW44"_hash)) {
        return channel_wise_nchw44::stride1::get_kimpls(param);
    }
    MIDOUT_END();
    return {};
}

/* ===================== stride2 algo ===================== */
bool ConvBiasImpl::AlgoS8ChanWiseStride2NCHW44::usable(
        fallback::ConvBiasImpl*, const NCBKernSizeParam& param,
        AlgoSelectionStrategy) const {
    return channel_wise_nchw44::stride2::is_available(param);
}

size_t ConvBiasImpl::AlgoS8ChanWiseStride2NCHW44::get_workspace(
        fallback::ConvBiasImpl*, const NCBKernSizeParam& param) const {
    auto bundle = channel_wise_nchw44::stride2::get_bundle(param);
    return bundle.total_size_in_bytes();
}

SmallVector<ConvBiasImpl::NCBKern>
ConvBiasImpl::AlgoS8ChanWiseStride2NCHW44::dispatch_kerns(
        fallback::ConvBiasImpl*, const NCBKernSizeParam& param) const {
    MIDOUT_BEGIN(megdnn_arm_common_conv_bias_int8,
                 midout_iv("AlgoS8ChanWiseStride2NCHW44"_hash)) {
        return channel_wise_nchw44::stride2::get_kimpls(param);
    }
    MIDOUT_END();
    return {};
}

/* ===================== stride2 algo ===================== */
bool ConvBiasImpl::AlgoS8DirectStride2::usable(
        fallback::ConvBiasImpl*, const NCBKernSizeParam& param,
        AlgoSelectionStrategy algo_selection_strategy) const {
    bool avaible = direct_int8_stride2::can_conv_direct_stride2_int8(param);
    if (algo_selection_strategy ==
        ConvBiasImpl::AlgoSelectionStrategy::HEURISTIC) {
        bool large_group = param.filter_meta.group >= param.nr_threads;
        avaible &= (large_group == m_large_group);
    }
    return avaible;
}

size_t ConvBiasImpl::AlgoS8DirectStride2::get_workspace(
        fallback::ConvBiasImpl*, const NCBKernSizeParam& param) const {
    auto bundle = direct_int8_stride2::get_bundle(param, m_large_group);
    return bundle.total_size_in_bytes();
}

SmallVector<ConvBiasImpl::NCBKern>
ConvBiasImpl::AlgoS8DirectStride2::dispatch_kerns(
        fallback::ConvBiasImpl*, const NCBKernSizeParam& param) const {
    MIDOUT_BEGIN(megdnn_arm_common_conv_bias_int8, 1, 1) {
        return direct_int8_stride2::get_kimpls(param, m_large_group);
    }
    MIDOUT_END();
    return {};
}

#if __ARM_FEATURE_DOTPROD
/* ===================== dot stride1 algo ======================== */
bool ConvBiasImpl::AlgoDotS8DirectStride1::usable(
        FallbackConvBiasImpl*, const NCBKernSizeParam& param,
        AlgoSelectionStrategy algo_selection_strategy) const {
    bool avaible =
            direct_dotprod_int8_stride1::can_conv_direct_stride1_int8(param);

    if (algo_selection_strategy ==
        ConvBiasImpl::AlgoSelectionStrategy::HEURISTIC) {
        bool large_group = param.filter_meta.group >= param.nr_threads;
        avaible &= (large_group == m_large_group);
    }

    return avaible;
}

size_t ConvBiasImpl::AlgoDotS8DirectStride1::get_workspace(
        FallbackConvBiasImpl*, const NCBKernSizeParam& param) const {
    auto bundle = direct_dotprod_int8_stride1::get_bundle(param, m_large_group);
    return bundle.total_size_in_bytes();
}

SmallVector<ConvBiasImpl::NCBKern>
ConvBiasImpl::AlgoDotS8DirectStride1::dispatch_kerns(
        FallbackConvBiasImpl*, const NCBKernSizeParam& param) const {
    MIDOUT_BEGIN(megdnn_arm_common_conv_bias_int8, 2, 1) {
        return direct_dotprod_int8_stride1::get_kimpls(param, m_large_group);
    }
    MIDOUT_END();
    return {};
}

/* ===================== dot stride2 algo ======================== */
bool ConvBiasImpl::AlgoDotS8DirectStride2::usable(
        FallbackConvBiasImpl*, const NCBKernSizeParam& param,
        AlgoSelectionStrategy algo_selection_strategy) const {
    bool avaible =
            direct_dotprod_int8_stride2::can_conv_direct_stride2_int8(param);
    if (algo_selection_strategy ==
        ConvBiasImpl::AlgoSelectionStrategy::HEURISTIC) {
        bool large_group = param.filter_meta.group >= param.nr_threads;
        avaible &= (large_group == m_large_group);
    }
    return avaible;
}

size_t ConvBiasImpl::AlgoDotS8DirectStride2::get_workspace(
        FallbackConvBiasImpl*, const NCBKernSizeParam& param) const {
    auto bundle = direct_dotprod_int8_stride2::get_bundle(param, m_large_group);
    return bundle.total_size_in_bytes();
}

SmallVector<ConvBiasImpl::NCBKern>
ConvBiasImpl::AlgoDotS8DirectStride2::dispatch_kerns(
        FallbackConvBiasImpl*, const NCBKernSizeParam& param) const {
    MIDOUT_BEGIN(megdnn_arm_common_conv_bias_int8, 2, 2) {
        return direct_dotprod_int8_stride2::get_kimpls(param, m_large_group);
    }
    MIDOUT_END();
    return {};
}
#endif

/* ======================= AlgoS8WinogradF23_8x8 ======================== */

bool ConvBiasImpl::AlgoS8WinogradF23_8x8::usable(
        fallback::ConvBiasImpl* opr, const NCBKernSizeParam& param,
        AlgoSelectionStrategy /*algo_selection_strategy*/) const {
    if (param.filter_meta.icpg % 8 != 0 || param.filter_meta.ocpg % 8 != 0)
        return false;
    using Strategy = winograd::winograd_2x3_8x8_s8;
    Strategy strategy(param.src_type, param.filter_type, param.dst_type);
    auto&& matmul_param =
            megdnn::winograd::ConvBias<Strategy, param::MatrixMul::Format::MK8>(
                    strategy, m_tile_size, param.nr_threads, param.osz[0],
                    param.osz[1], param.filter_meta.ocpg)
                    .get_matmul_kern_param(param);
    return m_matmul_algo->usable(matmul_param) &&
           ((opr->param().format == param::ConvBias::Format::NCHW &&
             param.filter_type.enumv() == DTypeEnum::QuantizedS8) ||
            (opr->param().format == param::ConvBias::Format::NCHW_WINOGRAD &&
             opr->param().output_block_size == 2 &&
             param.winograd_matmul_format == param::MatrixMul::Format::MK8 &&
             param.filter_type.enumv() == DTypeEnum::QuantizedS16)) &&
           opr->param().mode == param::ConvBias::Mode::CROSS_CORRELATION &&
           (param.filter_meta.spatial[0] == param.filter_meta.spatial[1] &&
            param.filter_meta.spatial[0] == 3) &&
           (param.filter_meta.stride[0] == param.filter_meta.stride[1] &&
            param.filter_meta.stride[0] == 1) &&
           (param.filter_meta.dilation[0] == param.filter_meta.dilation[1] &&
            param.filter_meta.dilation[0] == 1) &&
           param.compute_mode == param::ConvBias::ComputeMode::DEFAULT &&
           param.src_type.enumv() == DTypeEnum::QuantizedS8 &&
           param.bias_type.enumv() == DTypeEnum::QuantizedS32 &&
           param.dst_type.enumv() == DTypeEnum::QuantizedS8;
}

size_t ConvBiasImpl::AlgoS8WinogradF23_8x8::get_workspace(
        fallback::ConvBiasImpl*, const NCBKernSizeParam& param) const {
    winograd::winograd_2x3_8x8_s8 strategy(param.src_type, param.filter_type,
                                           param.dst_type);
    return megdnn::winograd::ConvBias<winograd::winograd_2x3_8x8_s8,
                                      param::MatrixMul::Format::MK8>(
                   strategy, m_tile_size, param.nr_threads, param.osz[0],
                   param.osz[1], param.filter_meta.ocpg)
            .get_workspace_size(param, m_matmul_algo);
}

SmallVector<ConvBiasImpl::NCBKern>
ConvBiasImpl::AlgoS8WinogradF23_8x8::dispatch_kerns(
        fallback::ConvBiasImpl*, const NCBKernSizeParam& param) const {
    MEGDNN_MARK_USED_VAR(param);
    MIDOUT_BEGIN(megdnn_arm_common_conv_bias_int8, 0, 2) {
        winograd::winograd_2x3_8x8_s8 strategy(
                param.src_type, param.filter_type, param.dst_type);
        auto winograd_impl =
                megdnn::winograd::ConvBias<winograd::winograd_2x3_8x8_s8,
                                           param::MatrixMul::Format::MK8>(
                        strategy, m_tile_size, param.nr_threads, param.osz[0],
                        param.osz[1], param.filter_meta.ocpg);
        return winograd_impl.get_kerns(param, m_matmul_algo);
    }
    MIDOUT_END();
    return {};
}

// vim: syntax=cpp.doxygen
