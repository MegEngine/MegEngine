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

bool ConvBiasImpl::AlgoS8DirectStride1::usable(const NCBKernSizeParam& param,
                                               AlgoSelectionStrategy) const {
    return direct_int8_stride1::can_conv_direct_stride1_int8(param);
}
bool ConvBiasImpl::AlgoS8DirectStride1::is_preferred(
        const NCBKernSizeParam& param) const {
    MIDOUT_BEGIN(megdnn_arm_common_conv_bias_int8,
                 midout_iv("AlgoS8DirectStride1::is_preferred"_hash)) {
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
    MIDOUT_END();
}

size_t ConvBiasImpl::AlgoS8DirectStride1::get_workspace(
        const NCBKernSizeParam& param) const {
    MIDOUT_BEGIN(megdnn_arm_common_conv_bias_int8,
                 midout_iv("AlgoS8DirectStride1::get_workspace"_hash)) {
        bool large_group = param.filter_meta.group >= param.nr_threads;
        auto bundle = direct_int8_stride1::get_bundle(param, large_group);
        return bundle.total_size_in_bytes();
    }
    MIDOUT_END();
    return 0;
}

SmallVector<ConvBiasImpl::NCBKern>
ConvBiasImpl::AlgoS8DirectStride1::dispatch_kerns(
         const NCBKernSizeParam& param) const {
    MIDOUT_BEGIN(megdnn_arm_common_conv_bias_int8,
                 midout_iv("AlgoS8DirectStride1::dispatch_kerns"_hash)) {
        bool large_group = param.filter_meta.group >= param.nr_threads;
        return direct_int8_stride1::get_kimpls(param, large_group);
    }
    MIDOUT_END();
    return {};
}

/* ===================== stride1 algo ===================== */
bool ConvBiasImpl::AlgoS8ChanWiseStride1NCHW44::usable(
         const NCBKernSizeParam& param,
        AlgoSelectionStrategy) const {
    return channel_wise_nchw44::stride1::is_available(param);
}

size_t ConvBiasImpl::AlgoS8ChanWiseStride1NCHW44::get_workspace(
         const NCBKernSizeParam& param) const {
    MIDOUT_BEGIN(megdnn_arm_common_conv_bias_int8,
            midout_iv("AlgoS8ChanWiseStride1NCHW44::get_workspace"_hash)) {
        auto bundle = channel_wise_nchw44::stride1::get_bundle(param);
        return bundle.total_size_in_bytes();
    }
    MIDOUT_END();
    return 0;
}

SmallVector<ConvBiasImpl::NCBKern>
ConvBiasImpl::AlgoS8ChanWiseStride1NCHW44::dispatch_kerns(
         const NCBKernSizeParam& param) const {
    MIDOUT_BEGIN(megdnn_arm_common_conv_bias_int8,
                 midout_iv("AlgoS8ChanWiseStride1NCHW44::dispatch_kerns"_hash)) {
        return channel_wise_nchw44::stride1::get_kimpls(param);
    }
    MIDOUT_END();
    return {};
}

/* ===================== stride2 algo ===================== */
bool ConvBiasImpl::AlgoS8ChanWiseStride2NCHW44::usable(
         const NCBKernSizeParam& param,
        AlgoSelectionStrategy) const {
    return channel_wise_nchw44::stride2::is_available(param);
}

size_t ConvBiasImpl::AlgoS8ChanWiseStride2NCHW44::get_workspace(
         const NCBKernSizeParam& param) const {
    MIDOUT_BEGIN(megdnn_arm_common_conv_bias_int8,
                 midout_iv("AlgoS8ChanWiseStride2NCHW44::get_workspace"_hash)) {
        auto bundle = channel_wise_nchw44::stride2::get_bundle(param);
        return bundle.total_size_in_bytes();
    }
    MIDOUT_END();
    return 0;
}

SmallVector<ConvBiasImpl::NCBKern>
ConvBiasImpl::AlgoS8ChanWiseStride2NCHW44::dispatch_kerns(
         const NCBKernSizeParam& param) const {
    MIDOUT_BEGIN(megdnn_arm_common_conv_bias_int8,
                 midout_iv("AlgoS8ChanWiseStride2NCHW44::dispatch_kerns"_hash)) {
        return channel_wise_nchw44::stride2::get_kimpls(param);
    }
    MIDOUT_END();
    return {};
}

/* ===================== stride2 algo ===================== */
bool ConvBiasImpl::AlgoS8DirectStride2::usable(const NCBKernSizeParam& param,
                                               AlgoSelectionStrategy) const {
    return direct_int8_stride2::can_conv_direct_stride2_int8(param);
}

size_t ConvBiasImpl::AlgoS8DirectStride2::get_workspace(
        const NCBKernSizeParam& param) const {
    MIDOUT_BEGIN(megdnn_arm_common_conv_bias_int8,
                 midout_iv("AlgoS8DirectStride2::get_workspace"_hash)) {
        bool large_group = param.filter_meta.group >= param.nr_threads;
        auto bundle = direct_int8_stride2::get_bundle(param, large_group);
        return bundle.total_size_in_bytes();
    }
    MIDOUT_END();
    return 0;
}

SmallVector<ConvBiasImpl::NCBKern>
ConvBiasImpl::AlgoS8DirectStride2::dispatch_kerns(
         const NCBKernSizeParam& param) const {
    MIDOUT_BEGIN(megdnn_arm_common_conv_bias_int8,
                 midout_iv("AlgoS8DirectStride2::dispatch_kerns"_hash)) {
        bool large_group = param.filter_meta.group >= param.nr_threads;
        return direct_int8_stride2::get_kimpls(param, large_group);
    }
    MIDOUT_END();
    return {};
}

#if __ARM_FEATURE_DOTPROD
/* ===================== dot stride1 algo ======================== */
bool ConvBiasImpl::AlgoDotS8DirectStride1::usable(const NCBKernSizeParam& param,
                                                  AlgoSelectionStrategy) const {
    return direct_dotprod_int8_stride1::can_conv_direct_stride1_int8(param);
}

size_t ConvBiasImpl::AlgoDotS8DirectStride1::get_workspace(
        const NCBKernSizeParam& param) const {
    MIDOUT_BEGIN(megdnn_arm_common_conv_bias_int8,
                 midout_iv("AlgoDotS8DirectStride1::get_workspace"_hash)) {
        bool large_group = param.filter_meta.group >= param.nr_threads;
        auto bundle = direct_dotprod_int8_stride1::get_bundle(param, large_group);
        return bundle.total_size_in_bytes();
    }
    MIDOUT_END();
    return 0;
}

SmallVector<ConvBiasImpl::NCBKern>
ConvBiasImpl::AlgoDotS8DirectStride1::dispatch_kerns(
         const NCBKernSizeParam& param) const {
    MIDOUT_BEGIN(megdnn_arm_common_conv_bias_int8,
                 midout_iv("AlgoDotS8DirectStride1::dispatch_kerns"_hash)) {
        bool large_group = param.filter_meta.group >= param.nr_threads;
        return direct_dotprod_int8_stride1::get_kimpls(param, large_group);
    }
    MIDOUT_END();
    return {};
}

/* ===================== dot stride2 algo ======================== */
bool ConvBiasImpl::AlgoDotS8DirectStride2::usable(const NCBKernSizeParam& param,
                                                  AlgoSelectionStrategy) const {
    return direct_dotprod_int8_stride2::can_conv_direct_stride2_int8(param);
}

size_t ConvBiasImpl::AlgoDotS8DirectStride2::get_workspace(
        const NCBKernSizeParam& param) const {
    MIDOUT_BEGIN(megdnn_arm_common_conv_bias_int8,
                 midout_iv("AlgoDotS8DirectStride2::get_workspace"_hash)) {
        bool large_group = param.filter_meta.group >= param.nr_threads;
        auto bundle = direct_dotprod_int8_stride2::get_bundle(param, large_group);
        return bundle.total_size_in_bytes();
    }
    MIDOUT_END();
    return 0;
}

SmallVector<ConvBiasImpl::NCBKern>
ConvBiasImpl::AlgoDotS8DirectStride2::dispatch_kerns(
         const NCBKernSizeParam& param) const {
    MIDOUT_BEGIN(megdnn_arm_common_conv_bias_int8,
                 midout_iv("AlgoDotS8DirectStride2::dispatch_kerns"_hash)) {
        bool large_group = param.filter_meta.group >= param.nr_threads;
        return direct_dotprod_int8_stride2::get_kimpls(param, large_group);
    }
    MIDOUT_END();
    return {};
}
#endif

/* ======================= AlgoS8WinogradF23_8x8 ======================== */

bool ConvBiasImpl::AlgoS8WinogradF23_8x8::usable(
        const NCBKernSizeParam& param,
        AlgoSelectionStrategy /*algo_selection_strategy*/) const {
    MIDOUT_BEGIN(megdnn_arm_common_conv_bias_int8,
                 midout_iv("AlgoS8WinogradF23_8x8::usable"_hash)) {
        if (param.filter_meta.icpg % 8 != 0 || param.filter_meta.ocpg % 8 != 0)
            return false;
        using Strategy = winograd::winograd_2x3_8x8_s8;
        using PackMode = fallback::MatrixMulImpl::AlgoBase::PackMode;
        Strategy strategy(param.src_type, param.filter_type, param.dst_type);
        auto&& matmul_param =
                megdnn::winograd::ConvBias<Strategy,
                                           param::MatrixMul::Format::MK8>(
                        strategy, m_tile_size, param)
                        .get_matmul_kern_param(param);
        return m_matmul_algo->usable(matmul_param) &&
               m_matmul_algo->packmode() == PackMode::NO_PACK &&
               ((param.filter_meta.format == param::ConvBias::Format::NCHW &&
                 param.filter_type.enumv() == DTypeEnum::QuantizedS8) ||
                (param.filter_meta.format ==
                         param::ConvBias::Format::NCHW_WINOGRAD &&
                 param.output_block_size == 2 &&
                 param.winograd_matmul_format ==
                         param::MatrixMul::Format::MK8 &&
                 param.filter_type.enumv() == DTypeEnum::QuantizedS16)) &&
               !param.filter_meta.should_flip &&
               (param.filter_meta.spatial[0] == param.filter_meta.spatial[1] &&
                param.filter_meta.spatial[0] == 3) &&
               (param.filter_meta.stride[0] == param.filter_meta.stride[1] &&
                param.filter_meta.stride[0] == 1) &&
               (param.filter_meta.dilation[0] ==
                        param.filter_meta.dilation[1] &&
                param.filter_meta.dilation[0] == 1) &&
               param.compute_mode == param::ConvBias::ComputeMode::DEFAULT &&
               param.src_type.enumv() == DTypeEnum::QuantizedS8 &&
               param.bias_type.enumv() == DTypeEnum::QuantizedS32 &&
               param.dst_type.enumv() == DTypeEnum::QuantizedS8;
    }
    MIDOUT_END();
    return false;
}

MEGDNN_WINOGRAD_ALGO_FUN_DEFINE_ALL(AlgoS8WinogradF23_8x8,
                                    winograd::winograd_2x3_8x8_s8,
                                    megdnn_arm_common_conv_bias_int8,
                                    param::MatrixMul::Format::MK8);

//=========================== input int8 compute float32 =========
bool ConvBiasImpl::AlgoS8CF32WinogradF23_4x4_NCHW44::usable(
        const NCBKernSizeParam& param,
        AlgoSelectionStrategy /*algo_selection_strategy*/) const {
    MEGDNN_MARK_USED_VAR(param);
    MIDOUT_BEGIN(megdnn_arm_common_conv_bias_int8,
                 midout_iv("arm_common_AlgoS8CF32WinogradF23_4x4::usable"_hash)) {
        if (param.filter_meta.icpg % 4 != 0 || param.filter_meta.ocpg % 4 != 0)
            return false;
        bool is_matmul_usable = false;

        using Strategy = winograd::winograd_2x3_4x4_s8_f32_nchw44;
        using PackMode = fallback::MatrixMulImpl::AlgoBase::PackMode;
        Strategy strategy(param.src_type, param.filter_type, param.dst_type);
        is_matmul_usable = m_matmul_algo->usable(
                megdnn::winograd::ConvBias<Strategy,
                                           param::MatrixMul::Format::MK4>(
                        strategy, m_tile_size, param)
                        .get_matmul_kern_param(param));
        return is_matmul_usable &&
               m_matmul_algo->packmode() == PackMode::NO_PACK &&
               ((param.filter_meta.format == param::ConvBias::Format::NCHW44 &&
                 param.filter_type.enumv() == DTypeEnum::QuantizedS8) ||
                ((param.filter_meta.format ==
                  param::ConvBias::Format::NCHW44_WINOGRAD) &&
                 param.output_block_size == 2 &&
                 param.winograd_matmul_format ==
                         param::MatrixMul::Format::MK4)) &&
               !param.filter_meta.should_flip &&
               (param.filter_meta.spatial[0] == param.filter_meta.spatial[1] &&
                param.filter_meta.spatial[0] == 3) &&
               (param.filter_meta.stride[0] == param.filter_meta.stride[1] &&
                param.filter_meta.stride[0] == 1) &&
               (param.filter_meta.dilation[0] ==
                        param.filter_meta.dilation[1] &&
                param.filter_meta.dilation[0] == 1) &&
               (param.compute_mode == param::ConvBias::ComputeMode::FLOAT32 ||
                param.compute_mode == param::ConvBias::ComputeMode::DEFAULT) &&
               param.src_type.enumv() == DTypeEnum::QuantizedS8 &&
               param.bias_type.enumv() == DTypeEnum::QuantizedS32 &&
               param.dst_type.enumv() == DTypeEnum::QuantizedS8;
    }
    MIDOUT_END();
    return false;
}

MEGDNN_WINOGRAD_ALGO_FUN_DEFINE_ALL(AlgoS8CF32WinogradF23_4x4_NCHW44,
                                    winograd::winograd_2x3_4x4_s8_f32_nchw44,
                                    megdnn_arm_common_conv_bias_int8,
                                    param::MatrixMul::Format::MK4);

/* ======================= AlgoS8WinogradF23_8x8_NCHW44 ======================== */
bool ConvBiasImpl::AlgoS8WinogradF23_8x8_NCHW44::usable(
        const NCBKernSizeParam& param,
        AlgoSelectionStrategy /*algo_selection_strategy*/) const {
    MIDOUT_BEGIN(
            megdnn_arm_common_conv_bias_int8,
            midout_iv("arm_common_AlgoS8WinogradF23_8x8_NCHW44::usable"_hash)) {
        if (param.filter_meta.icpg % 8 != 0 || param.filter_meta.ocpg % 8 != 0)
            return false;
        using Strategy = winograd::winograd_2x3_8x8_s8_nchw44;
        Strategy strategy(param.src_type, param.filter_type, param.dst_type);
        auto&& matmul_param =
                megdnn::winograd::ConvBias<Strategy,
                                           param::MatrixMul::Format::MK8>(
                        strategy, m_tile_size, param)
                        .get_matmul_kern_param(param);
        bool is_matmul_usable = m_matmul_algo->usable(matmul_param);
        return is_matmul_usable &&
               ((param.filter_meta.format == param::ConvBias::Format::NCHW44 &&
                 param.filter_type.enumv() == DTypeEnum::QuantizedS8) ||
                (param.filter_meta.format ==
                         param::ConvBias::Format::NCHW44_WINOGRAD &&
                 param.output_block_size == 2 &&
                 param.winograd_matmul_format ==
                         param::MatrixMul::Format::MK8 &&
                 param.filter_type.enumv() == DTypeEnum::QuantizedS16)) &&
               !param.filter_meta.should_flip &&
               (param.filter_meta.spatial[0] == param.filter_meta.spatial[1] &&
                param.filter_meta.spatial[0] == 3) &&
               (param.filter_meta.stride[0] == param.filter_meta.stride[1] &&
                param.filter_meta.stride[0] == 1) &&
               (param.filter_meta.dilation[0] ==
                        param.filter_meta.dilation[1] &&
                param.filter_meta.dilation[0] == 1) &&
               param.compute_mode == param::ConvBias::ComputeMode::DEFAULT &&
               param.src_type.enumv() == DTypeEnum::QuantizedS8 &&
               param.bias_type.enumv() == DTypeEnum::QuantizedS32 &&
               param.dst_type.enumv() == DTypeEnum::QuantizedS8;
    }
    MIDOUT_END();
    return false;
}

MEGDNN_WINOGRAD_ALGO_FUN_DEFINE_ALL(AlgoS8WinogradF23_8x8_NCHW44,
                                    winograd::winograd_2x3_8x8_s8_nchw44,
                                    megdnn_arm_common_conv_bias_int8,
                                    param::MatrixMul::Format::MK8);

// vim: syntax=cpp.doxygen
