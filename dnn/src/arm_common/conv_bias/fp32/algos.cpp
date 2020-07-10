/**
 * \file dnn/src/arm_common/conv_bias/fp32/algos.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/arm_common/conv_bias/fp32/algos.h"
#include "src/arm_common/conv_bias/direct/multi_thread_common.h"
#include "src/arm_common/conv_bias/fp32/direct.h"
#include "src/arm_common/conv_bias/fp32/do_conv_stride1.h"
#include "src/arm_common/conv_bias/fp32/do_conv_stride2.h"
#include "src/arm_common/conv_bias/fp32/strategy.h"
#include "src/arm_common/conv_bias/img2col_helper.h"
#include "src/arm_common/conv_bias/postprocess_helper.h"
#include "src/common/opr_delegate.h"
#include "src/fallback/conv_bias/common.h"

#include "midout.h"

MIDOUT_DECL(megdnn_arm_common_winograd_fp32)

using namespace megdnn;
using namespace arm_common;

/* ======================= AlgoFP32WinogradF23_4x4 ======================== */

bool ConvBiasImpl::AlgoFP32WinogradF23_4x4::usable(
        const NCBKernSizeParam& param,
        AlgoSelectionStrategy /*algo_selection_strategy*/) const {
    MEGDNN_MARK_USED_VAR(param);
    MIDOUT_BEGIN(megdnn_arm_common_winograd_fp32, 0, 0) {
        if (param.filter_meta.icpg % 4 != 0 || param.filter_meta.ocpg % 4 != 0)
            return false;
        using Strategy = winograd::winograd_2x3_4x4_f;
        using PackMode = fallback::MatrixMulImpl::AlgoBase::PackMode;
        Strategy strategy(param.src_type, param.filter_type, param.dst_type);
        auto&& matmul_param =
                megdnn::winograd::ConvBias<Strategy,
                                           param::MatrixMul::Format::MK4>(
                        strategy, m_tile_size, param)
                        .get_matmul_kern_param(param);
        return m_matmul_algo->usable(matmul_param) &&
               m_matmul_algo->packmode() == PackMode::NO_PACK &&
               (param.filter_meta.format == param::ConvBias::Format::NCHW ||
                (param.filter_meta.format ==
                         param::ConvBias::Format::NCHW_WINOGRAD &&
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
               param.compute_mode == param::ConvBias::ComputeMode::DEFAULT &&
               param.src_type.enumv() == DTypeEnum::Float32;
    }
    MIDOUT_END();
    return false;
}

MEGDNN_WINOGRAD_ALGO_FUN_DEFINE_ALL(AlgoFP32WinogradF23_4x4,
                                    winograd::winograd_2x3_4x4_f,
                                    megdnn_arm_common_winograd_fp32,
                                    param::MatrixMul::Format::MK4);

/* ======================= AlgoFP32WinogradF63 ======================== */

bool ConvBiasImpl::AlgoFP32WinogradF63::usable(
        const NCBKernSizeParam& param,
        AlgoSelectionStrategy /*algo_selection_strategy*/) const {
    MEGDNN_MARK_USED_VAR(param);
    MIDOUT_BEGIN(megdnn_arm_common_winograd_fp32, 1, 0) {
        using Strategy = winograd::winograd_6x3_1x1_f;
        Strategy strategy(param.src_type, param.filter_type, param.dst_type);
        auto&& matmul_param = megdnn::winograd::ConvBias<Strategy>(
                                      strategy, m_tile_size, param)
                                      .get_matmul_kern_param(param);
        return m_matmul_algo->usable(matmul_param) &&
               (param.filter_meta.format == param::ConvBias::Format::NCHW ||
                (param.filter_meta.format ==
                         param::ConvBias::Format::NCHW_WINOGRAD &&
                 param.output_block_size == 6 &&
                 param.winograd_matmul_format ==
                         param::MatrixMul::Format::DEFAULT)) &&
               !param.filter_meta.should_flip &&
               (param.filter_meta.spatial[0] == param.filter_meta.spatial[1] &&
                param.filter_meta.spatial[0] == 3) &&
               (param.filter_meta.stride[0] == param.filter_meta.stride[1] &&
                param.filter_meta.stride[0] == 1) &&
               (param.filter_meta.dilation[0] ==
                        param.filter_meta.dilation[1] &&
                param.filter_meta.dilation[0] == 1) &&
               param.compute_mode == param::ConvBias::ComputeMode::DEFAULT &&
               param.src_type.enumv() == DTypeEnum::Float32;
    }
    MIDOUT_END();
    return false;
}

MEGDNN_WINOGRAD_ALGO_FUN_DEFINE_ALL(AlgoFP32WinogradF63,
                                    winograd::winograd_6x3_1x1_f,
                                    megdnn_arm_common_winograd_fp32,
                                    param::MatrixMul::Format::DEFAULT);

/* ======================= AlgoFP32WinogradF54 ======================== */

bool ConvBiasImpl::AlgoFP32WinogradF54::usable(
        const NCBKernSizeParam& param,
        AlgoSelectionStrategy /*algo_selection_strategy*/) const {
    MEGDNN_MARK_USED_VAR(param);
    MIDOUT_BEGIN(megdnn_arm_common_winograd_fp32, 2, 0) {
        using Strategy = winograd::winograd_5x4_1x1_f;
        Strategy strategy(param.src_type, param.filter_type, param.dst_type);
        auto&& matmul_param = megdnn::winograd::ConvBias<Strategy>(
                                      strategy, m_tile_size, param)
                                      .get_matmul_kern_param(param);
        return m_matmul_algo->usable(matmul_param) &&
               (param.filter_meta.format == param::ConvBias::Format::NCHW ||
                (param.filter_meta.format ==
                         param::ConvBias::Format::NCHW_WINOGRAD &&
                 param.output_block_size == 5 &&
                 param.winograd_matmul_format ==
                         param::MatrixMul::Format::DEFAULT)) &&
               !param.filter_meta.should_flip &&
               (param.filter_meta.spatial[0] == param.filter_meta.spatial[1] &&
                param.filter_meta.spatial[0] == 4) &&
               (param.filter_meta.stride[0] == param.filter_meta.stride[1] &&
                param.filter_meta.stride[0] == 1) &&
               (param.filter_meta.dilation[0] ==
                        param.filter_meta.dilation[1] &&
                param.filter_meta.dilation[0] == 1) &&
               param.compute_mode == param::ConvBias::ComputeMode::DEFAULT &&
               param.src_type.enumv() == DTypeEnum::Float32;
    }
    MIDOUT_END();
    return false;
}

MEGDNN_WINOGRAD_ALGO_FUN_DEFINE_ALL(AlgoFP32WinogradF54,
                                    winograd::winograd_5x4_1x1_f,
                                    megdnn_arm_common_winograd_fp32,
                                    param::MatrixMul::Format::DEFAULT);

/* ======================= AlgoFP32WinogradF45 ======================== */

bool ConvBiasImpl::AlgoFP32WinogradF45::usable(
        const NCBKernSizeParam& param,
        AlgoSelectionStrategy /*algo_selection_strategy*/) const {
    MEGDNN_MARK_USED_VAR(param);
    MIDOUT_BEGIN(megdnn_arm_common_winograd_fp32, 3, 0) {
        using Strategy = winograd::winograd_4x5_1x1_f;
        Strategy strategy(param.src_type, param.filter_type, param.dst_type);
        auto&& matmul_param = megdnn::winograd::ConvBias<Strategy>(
                                      strategy, m_tile_size, param)
                                      .get_matmul_kern_param(param);
        return m_matmul_algo->usable(matmul_param) &&
               (param.filter_meta.format == param::ConvBias::Format::NCHW ||
                (param.filter_meta.format ==
                         param::ConvBias::Format::NCHW_WINOGRAD &&
                 param.output_block_size == 4 &&
                 param.winograd_matmul_format ==
                         param::MatrixMul::Format::DEFAULT)) &&
               !param.filter_meta.should_flip &&
               (param.filter_meta.spatial[0] == param.filter_meta.spatial[1] &&
                param.filter_meta.spatial[0] == 5) &&
               (param.filter_meta.stride[0] == param.filter_meta.stride[1] &&
                param.filter_meta.stride[0] == 1) &&
               (param.filter_meta.dilation[0] ==
                        param.filter_meta.dilation[1] &&
                param.filter_meta.dilation[0] == 1) &&
               param.compute_mode == param::ConvBias::ComputeMode::DEFAULT &&
               param.src_type.enumv() == DTypeEnum::Float32;
    }
    MIDOUT_END();
    return false;
}

MEGDNN_WINOGRAD_ALGO_FUN_DEFINE_ALL(AlgoFP32WinogradF45,
                                    winograd::winograd_4x5_1x1_f,
                                    megdnn_arm_common_winograd_fp32,
                                    param::MatrixMul::Format::DEFAULT);

/* ======================= AlgoFP32WinogradF63_4x4 ======================== */

bool ConvBiasImpl::AlgoFP32WinogradF63_4x4::usable(
        const NCBKernSizeParam& param,
        AlgoSelectionStrategy /*algo_selection_strategy*/) const {
    MEGDNN_MARK_USED_VAR(param);
    MIDOUT_BEGIN(megdnn_arm_common_winograd_fp32, 4, 0) {
        if (param.filter_meta.icpg % 4 != 0 || param.filter_meta.ocpg % 4 != 0)
            return false;
        using Strategy = winograd::winograd_6x3_4x4_f;
        using PackMode = fallback::MatrixMulImpl::AlgoBase::PackMode;
        Strategy strategy(param.src_type, param.filter_type, param.dst_type);
        auto&& matmul_param =
                megdnn::winograd::ConvBias<Strategy,
                                           param::MatrixMul::Format::MK4>(
                        strategy, m_tile_size, param)
                        .get_matmul_kern_param(param);
        return m_matmul_algo->usable(matmul_param) &&
               m_matmul_algo->packmode() == PackMode::NO_PACK &&
               (param.filter_meta.format == param::ConvBias::Format::NCHW ||
                (param.filter_meta.format ==
                         param::ConvBias::Format::NCHW_WINOGRAD &&
                 param.output_block_size == 6 &&
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
               param.compute_mode == param::ConvBias::ComputeMode::DEFAULT &&
               param.src_type.enumv() == DTypeEnum::Float32 &&
               param.filter_meta.icpg % 4 == 0 &&
               param.filter_meta.ocpg % 4 == 0;
    }
    MIDOUT_END();
    return false;
}

MEGDNN_WINOGRAD_ALGO_FUN_DEFINE_ALL(AlgoFP32WinogradF63_4x4,
                                    winograd::winograd_6x3_4x4_f,
                                    megdnn_arm_common_winograd_fp32,
                                    param::MatrixMul::Format::MK4);

/* =================== AlgoFP32WinogradF23_4x4_NCHW44 =================== */

bool ConvBiasImpl::AlgoFP32WinogradF23_4x4_NCHW44::usable(
        const NCBKernSizeParam& param,
        AlgoSelectionStrategy /*algo_selection_strategy*/) const {
    MEGDNN_MARK_USED_VAR(param);
    MIDOUT_BEGIN(megdnn_arm_common_winograd_fp32,
                 midout_iv("AlgoFP32WinogradF23_4x4_NCHW44"_hash)) {
        if (param.filter_meta.icpg % 4 != 0 || param.filter_meta.ocpg % 4 != 0)
            return false;
        using Strategy = winograd::winograd_F23_mk4_f_nchw44;
        Strategy strategy(param.src_type, param.filter_type, param.dst_type);
        auto&& matmul_param =
                megdnn::winograd::ConvBias<Strategy,
                                           param::MatrixMul::Format::MK4>(
                        strategy, m_tile_size, param)
                        .get_matmul_kern_param(param);
        return m_matmul_algo->usable(matmul_param) &&
               m_matmul_algo->packmode() ==
                       fallback::MatrixMulImpl::AlgoBase::PackMode::NO_PACK &&
               (param.filter_meta.format == param::ConvBias::Format::NCHW44 ||
                (param.filter_meta.format ==
                         param::ConvBias::Format::NCHW44_WINOGRAD &&
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
               param.compute_mode == param::ConvBias::ComputeMode::DEFAULT &&
               param.src_type.enumv() == DTypeEnum::Float32;
    }
    MIDOUT_END();
    return false;
}

MEGDNN_WINOGRAD_ALGO_FUN_DEFINE_ALL(AlgoFP32WinogradF23_4x4_NCHW44,
                                    winograd::winograd_F23_mk4_f_nchw44,
                                    megdnn_arm_common_winograd_fp32,
                                    param::MatrixMul::Format::MK4);

/* =================== AlgoFP32WinogradF63_4x4_NCHW44 ===================== */

bool ConvBiasImpl::AlgoFP32WinogradF63_4x4_NCHW44::usable(
        const NCBKernSizeParam& param,
        AlgoSelectionStrategy /*algo_selection_strategy*/) const {
    MEGDNN_MARK_USED_VAR(param);
    MIDOUT_BEGIN(megdnn_arm_common_winograd_fp32,
                 midout_iv("AlgoFP32WinogradF63_4x4_NCHW44"_hash)) {
        if (param.filter_meta.icpg % 4 != 0 || param.filter_meta.ocpg % 4 != 0)
            return false;
        using Strategy = winograd::winograd_F63_mk4_f_nchw44;
        Strategy strategy(param.src_type, param.filter_type, param.dst_type);
        auto&& matmul_param =
                megdnn::winograd::ConvBias<Strategy,
                                           param::MatrixMul::Format::MK4>(
                        strategy, m_tile_size, param)
                        .get_matmul_kern_param(param);
        return m_matmul_algo->usable(matmul_param) &&
               m_matmul_algo->packmode() ==
                       fallback::MatrixMulImpl::AlgoBase::PackMode::NO_PACK &&
               (param.filter_meta.format == param::ConvBias::Format::NCHW44 ||
                (param.filter_meta.format ==
                         param::ConvBias::Format::NCHW44_WINOGRAD &&
                 param.output_block_size == 6 &&
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
               param.compute_mode == param::ConvBias::ComputeMode::DEFAULT &&
               param.src_type.enumv() == DTypeEnum::Float32 &&
               param.filter_meta.icpg % 4 == 0 &&
               param.filter_meta.ocpg % 4 == 0;
    }
    MIDOUT_END();
    return false;
}

MEGDNN_WINOGRAD_ALGO_FUN_DEFINE_ALL(AlgoFP32WinogradF63_4x4_NCHW44,
                                    winograd::winograd_F63_mk4_f_nchw44,
                                    megdnn_arm_common_winograd_fp32,
                                    param::MatrixMul::Format::MK4);

/* =================== AlgoFP32WinogradF73_4x4_NCHW44 ===================== */

bool ConvBiasImpl::AlgoFP32WinogradF73_4x4_NCHW44::usable(
        const NCBKernSizeParam& param,
        AlgoSelectionStrategy /*algo_selection_strategy*/) const {
    MIDOUT_BEGIN(megdnn_arm_common_winograd_fp32,
                 midout_iv("AlgoFP32WinogradF73_4x4_NCHW44"_hash)) {
        if (param.filter_meta.icpg % 4 != 0 || param.filter_meta.ocpg % 4 != 0)
            return false;
        using Strategy = winograd::winograd_F73_mk4_f_nchw44;
        Strategy strategy(param.src_type, param.filter_type, param.dst_type);
        auto&& matmul_param =
                megdnn::winograd::ConvBias<Strategy,
                                           param::MatrixMul::Format::MK4>(
                        strategy, m_tile_size, param)
                        .get_matmul_kern_param(param);
        return m_matmul_algo->usable(matmul_param) &&
               m_matmul_algo->packmode() ==
                       fallback::MatrixMulImpl::AlgoBase::PackMode::NO_PACK &&
               (param.filter_meta.format == param::ConvBias::Format::NCHW44 ||
                (param.filter_meta.format ==
                         param::ConvBias::Format::NCHW44_WINOGRAD &&
                 param.output_block_size == 7 &&
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
               param.compute_mode == param::ConvBias::ComputeMode::DEFAULT &&
               param.src_type.enumv() == DTypeEnum::Float32;
    }
    MIDOUT_END();
    return false;
}

MEGDNN_WINOGRAD_ALGO_FUN_DEFINE_ALL(AlgoFP32WinogradF73_4x4_NCHW44,
                                    winograd::winograd_F73_mk4_f_nchw44,
                                    megdnn_arm_common_winograd_fp32,
                                    param::MatrixMul::Format::MK4);

/* ===================== direct algo ===================== */
MIDOUT_DECL(megdnn_arm_common_conv_bias_f32_kimpl);

bool ConvBiasImpl::AlgoF32Direct::usable(const NCBKernSizeParam& param,
                                         AlgoSelectionStrategy) const {
    MIDOUT_BEGIN(megdnn_arm_common_conv_bias_f32_kimpl, 0, 0) {
        auto&& fm = param.filter_meta;
        auto FH = fm.spatial[0];
        auto SH = fm.stride[0], SW = fm.stride[1];
        // the condition ``param.isz[0]*param.isz[1] >= 4'' and
        // ``param.osz[0]*param.osz[1] >= 4'' comes from the fact that the
        // kernel may have access to up to 4 floats after the end of the memory
        // chunk.
        return fm.format == param::ConvBias::Format::NCHW &&
               param.src_type.enumv() == DTypeEnum::Float32 &&
               param.filter_type.enumv() == DTypeEnum::Float32 &&
               param.dst_type.enumv() == DTypeEnum::Float32 &&
               fm.spatial_ndim == 2 && fm.dilation[0] == 1 &&
               fm.dilation[1] == 1 && param.isz[0] * param.isz[1] >= 4 &&
               param.osz[0] * param.osz[1] >= 4 && FH <= 7 && SH == 1 &&
               SW == 1;
    }
    MIDOUT_END();
    return false;
}
size_t ConvBiasImpl::AlgoF32Direct::get_workspace(
        const NCBKernSizeParam& param) const {
    MIDOUT_BEGIN(megdnn_arm_common_conv_bias_f32_kimpl, 0, 1) {
        bool large_group = param.filter_meta.group >= param.nr_threads;
        auto wbundle = MultithreadDirectConvCommon<float, float>::get_bundle(
                param, large_group);
        return wbundle.total_size_in_bytes();
    }
    MIDOUT_END();
    return 0;
}
SmallVector<ConvBiasImpl::NCBKern> ConvBiasImpl::AlgoF32Direct::get_kimpls(
        const NCBKernSizeParam& param) const {
    auto fm = param.filter_meta;
    size_t N = param.n;
    size_t IC = param.filter_meta.icpg;
    size_t OC = param.filter_meta.ocpg;
    size_t group = fm.group;
    bool large_group = group >= param.nr_threads;
    WorkspaceBundle bundle =
            MultithreadDirectConvCommon<float, float>::get_bundle(
                    param, large_group);
    SmallVector<NCBKern> ret_kerns;
    //! When group >= nr_threads, treat it as large_group, each thread process
    //! one group for better performance
    if (large_group) {
        //! Channel wise conv and big groups
        auto exec_one_group = [bundle](const NCBKernParam& kern_param,
                                       const NCBKernIndex& ncb_index) mutable {
            auto fm = kern_param.filter_meta;
            size_t IC = fm.icpg;
            size_t OC = fm.ocpg;
            bundle.set(kern_param.workspace_ptr);
            if (fm.should_flip) {
                for (size_t oc = 0; oc < OC; oc++) {
                    MultithreadDirectConvCommon<float, float>::weight_flip_kern(
                            bundle, kern_param, ncb_index,
                            {ncb_index.thread_id, 0, oc});
                }
            }
            for (size_t ic = 0; ic < IC; ic++) {
                MultithreadDirectConvCommon<float, float>::copy_padding_kern(
                        bundle, kern_param, ncb_index,
                        {ncb_index.thread_id, 0, ic});
            }
            for (size_t oc = 0; oc < OC; oc++) {
                MultithreadDirectConvCommon<float, float>::do_conv_kern(
                        bundle, kern_param, ncb_index,
                        fp32::conv_bias::kern_direct,
                        {ncb_index.thread_id, 0, oc});
            }
        };
        ret_kerns.push_back({exec_one_group, {group, N, 1_z}});
    } else {
        if (fm.should_flip) {
            auto weight_flip = [bundle](const NCBKernParam& kern_param,
                                        const NCBKernIndex& ncb_index) mutable {
                bundle.set(kern_param.workspace_ptr);
                MultithreadDirectConvCommon<float, float>::weight_flip_kern(
                        bundle, kern_param, ncb_index, ncb_index.ndrange_id);
            };
            ret_kerns.push_back({weight_flip, {group, 1_z, OC}});
        }
        auto copy_padding = [bundle](const NCBKernParam& kern_param,
                                     const NCBKernIndex& ncb_index) mutable {
            bundle.set(kern_param.workspace_ptr);
            MultithreadDirectConvCommon<float, float>::copy_padding_kern(
                    bundle, kern_param, ncb_index, ncb_index.ndrange_id);
        };
        ret_kerns.push_back({copy_padding, {group, N, IC}});
        auto do_conv = [bundle](const NCBKernParam& kern_param,
                                const NCBKernIndex& ncb_index) mutable {
            bundle.set(kern_param.workspace_ptr);
            MultithreadDirectConvCommon<float, float>::do_conv_kern(
                    bundle, kern_param, ncb_index, fp32::conv_bias::kern_direct,
                    ncb_index.ndrange_id);
        };
        ret_kerns.push_back({do_conv, {group, N, OC}});
    }
    return ret_kerns;
}

SmallVector<ConvBiasImpl::NCBKern> ConvBiasImpl::AlgoF32Direct::dispatch_kerns(
        const NCBKernSizeParam& param) const {
    MIDOUT_BEGIN(megdnn_arm_common_conv_bias_f32_kimpl, 0, 1) {
        return get_kimpls(param);
    }
    MIDOUT_END();
    return {};
}
/* ===================== stride-1 algo ===================== */
bool ConvBiasImpl::AlgoF32DirectStride1::usable(const NCBKernSizeParam& param,
                                                AlgoSelectionStrategy) const {
    MIDOUT_BEGIN(megdnn_arm_common_conv_bias_f32_kimpl, 1, 1) {
        auto&& fm = param.filter_meta;
        auto FH = fm.spatial[0];
        return param.filter_meta.format == param::ConvBias::Format::NCHW &&
               param.src_type.enumv() == DTypeEnum::Float32 &&
               param.filter_type.enumv() == DTypeEnum::Float32 &&
               param.dst_type.enumv() == DTypeEnum::Float32 &&
               !fm.should_flip && fm.spatial_ndim == 2 && fm.dilation[0] == 1 &&
               fm.dilation[1] == 1 && fm.stride[0] == 1 && fm.stride[1] == 1 &&
               FH == fm.spatial[1] &&
               (FH == 2 || FH == 3 || FH == 5 || FH == 7);
    }
    MIDOUT_END();
    return false;
}

size_t ConvBiasImpl::AlgoF32DirectStride1::get_workspace(
        const NCBKernSizeParam& param) const {
    MIDOUT_BEGIN(megdnn_arm_common_conv_bias_f32_kimpl, 1, 1) {
        bool large_group = param.filter_meta.group >= param.nr_threads;
        auto bundle =
                MultithreadDirectConvCommon<float, float>::get_bundle_stride(
                        param, large_group);
        return bundle.total_size_in_bytes();
    }
    MIDOUT_END();
    return 0;
}

SmallVector<ConvBiasImpl::NCBKern>
ConvBiasImpl::AlgoF32DirectStride1::get_kimpls(
        const NCBKernSizeParam& param) const {
    auto fm = param.filter_meta;
    auto FH = fm.spatial[0];
    size_t N = param.n;
    size_t IC = param.filter_meta.icpg;
    size_t OC = param.filter_meta.ocpg;
    size_t group = fm.group;
    bool large_group = group >= param.nr_threads;
    using Func = std::function<void(const float*, const float*, float*, size_t,
                                    size_t, size_t, size_t, size_t)>;
    Func conv_kern_function = nullptr;

#define SWITCH_KERN_STR1()                                                \
    switch (FH) {                                                         \
        case 2:                                                           \
            conv_kern_function = fp32::conv_stride1::do_conv_2x2_stride1; \
            break;                                                        \
        case 3:                                                           \
            conv_kern_function = fp32::conv_stride1::do_conv_3x3_stride1; \
            break;                                                        \
        case 5:                                                           \
            conv_kern_function = fp32::conv_stride1::do_conv_5x5_stride1; \
            break;                                                        \
        case 7:                                                           \
            conv_kern_function = fp32::conv_stride1::do_conv_7x7_stride1; \
            break;                                                        \
    }
    SWITCH_KERN_STR1();

    WorkspaceBundle bundle =
            MultithreadDirectConvCommon<float, float>::get_bundle_stride(
                    param, large_group);
    SmallVector<NCBKern> ret_kerns;
    //! When group >= nr_threads, treat it as large_group, each thread process
    //! one group for better performance
    if (large_group) {
        //! Channel wise conv and big groups
        auto exec_one_group = [bundle, conv_kern_function](
                                      const NCBKernParam& kern_param,
                                      const NCBKernIndex& ncb_index) mutable {
            auto fm = kern_param.filter_meta;
            size_t IC = fm.icpg;
            size_t OC = fm.ocpg;
            bundle.set(kern_param.workspace_ptr);
            for (size_t ic = 0; ic < IC; ic++) {
                MultithreadDirectConvCommon<float, float>::
                        copy_padding_kern_stride(bundle, kern_param, ncb_index,
                                                 {ncb_index.thread_id, 0, ic});
            }
            for (size_t oc = 0; oc < OC; oc++) {
                MultithreadDirectConvCommon<float, float>::do_conv_kern_stride(
                        bundle, kern_param, ncb_index, conv_kern_function,
                        {ncb_index.thread_id, 0, oc});
            }
        };
        ret_kerns.push_back({exec_one_group, {group, N, 1_z}});
    } else {
        auto copy_padding = [bundle](const NCBKernParam& kern_param,
                                     const NCBKernIndex& ncb_index) mutable {
            bundle.set(kern_param.workspace_ptr);
            MultithreadDirectConvCommon<float, float>::copy_padding_kern_stride(
                    bundle, kern_param, ncb_index, ncb_index.ndrange_id);
        };
        ret_kerns.push_back({copy_padding, {group, N, IC}});
        auto do_conv = [bundle, conv_kern_function](
                               const NCBKernParam& kern_param,
                               const NCBKernIndex& ncb_index) mutable {
            bundle.set(kern_param.workspace_ptr);
            MultithreadDirectConvCommon<float, float>::do_conv_kern_stride(
                    bundle, kern_param, ncb_index, conv_kern_function,
                    ncb_index.ndrange_id);
        };
        ret_kerns.push_back({do_conv, {group, N, OC}});
    }
    return ret_kerns;
}

SmallVector<ConvBiasImpl::NCBKern>
ConvBiasImpl::AlgoF32DirectStride1::dispatch_kerns(
        const NCBKernSizeParam& param) const {
    MIDOUT_BEGIN(megdnn_arm_common_conv_bias_f32_kimpl, 1, 2) {
        return get_kimpls(param);
    }
    MIDOUT_END();
    return {};
}

/* ===================== stride-2 algo ===================== */

bool ConvBiasImpl::AlgoF32DirectStride2::usable(const NCBKernSizeParam& param,
                                                AlgoSelectionStrategy) const {
    MIDOUT_BEGIN(megdnn_arm_common_conv_bias_f32_kimpl, 2, 0) {
        auto&& fm = param.filter_meta;
        auto FH = fm.spatial[0];
        return param.filter_meta.format == param::ConvBias::Format::NCHW &&
               param.src_type.enumv() == DTypeEnum::Float32 &&
               param.filter_type.enumv() == DTypeEnum::Float32 &&
               param.dst_type.enumv() == DTypeEnum::Float32 &&
               !fm.should_flip && fm.spatial_ndim == 2 && fm.dilation[0] == 1 &&
               fm.dilation[1] == 1 && fm.stride[0] == 2 && fm.stride[1] == 2 &&
               FH == fm.spatial[1] &&
               (FH == 2 || FH == 3 || FH == 5 || FH == 7);
    }
    MIDOUT_END();
    return false;
}
size_t ConvBiasImpl::AlgoF32DirectStride2::get_workspace(
        const NCBKernSizeParam& param) const {
    MIDOUT_BEGIN(megdnn_arm_common_conv_bias_f32_kimpl, 2, 1) {
        bool large_group = param.filter_meta.group >= param.nr_threads;
        auto bundle =
                MultithreadDirectConvCommon<float, float>::get_bundle_stride(
                        param, large_group);
        return bundle.total_size_in_bytes();
    }
    MIDOUT_END();
    return 0;
}
SmallVector<ConvBiasImpl::NCBKern>
ConvBiasImpl::AlgoF32DirectStride2::get_kimpls(
        const NCBKernSizeParam& param) const {
    auto fm = param.filter_meta;
    auto FH = fm.spatial[0];
    size_t N = param.n;
    size_t IC = param.filter_meta.icpg;
    size_t OC = param.filter_meta.ocpg;
    size_t group = fm.group;
    bool large_group = group >= param.nr_threads;
    using Func = std::function<void(const float*, const float*, float*, size_t,
                                    size_t, size_t, size_t, size_t)>;
    Func conv_kern_function = nullptr;

#define SWITCH_KERN_STR2()                                                \
    switch (FH) {                                                         \
        case 2:                                                           \
            conv_kern_function = fp32::conv_stride2::do_conv_2x2_stride2; \
            break;                                                        \
        case 3:                                                           \
            conv_kern_function = fp32::conv_stride2::do_conv_3x3_stride2; \
            break;                                                        \
        case 5:                                                           \
            conv_kern_function = fp32::conv_stride2::do_conv_5x5_stride2; \
            break;                                                        \
        case 7:                                                           \
            conv_kern_function = fp32::conv_stride2::do_conv_7x7_stride2; \
            break;                                                        \
    }
    SWITCH_KERN_STR2();

    WorkspaceBundle bundle =
            MultithreadDirectConvCommon<float, float>::get_bundle_stride(
                    param, large_group);
    SmallVector<NCBKern> ret_kerns;
    //! When group >= nr_threads, treat it as large_group, each thread process
    //! one group for better performance
    if (large_group) {
        //! Channel wise conv and big groups
        auto exec_one_group = [bundle, conv_kern_function](
                                      const NCBKernParam& kern_param,
                                      const NCBKernIndex& ncb_index) mutable {
            auto fm = kern_param.filter_meta;
            size_t IC = fm.icpg;
            size_t OC = fm.ocpg;
            bundle.set(kern_param.workspace_ptr);
            for (size_t ic = 0; ic < IC; ic++) {
                MultithreadDirectConvCommon<float, float>::
                        copy_padding_kern_stride(bundle, kern_param, ncb_index,
                                                 {ncb_index.thread_id, 0, ic});
            }
            for (size_t oc = 0; oc < OC; oc++) {
                MultithreadDirectConvCommon<float, float>::do_conv_kern_stride(
                        bundle, kern_param, ncb_index, conv_kern_function,
                        {ncb_index.thread_id, 0, oc});
            }
        };
        ret_kerns.push_back({exec_one_group, {group, N, 1_z}});
    } else {
        auto copy_padding = [bundle](const NCBKernParam& kern_param,
                                     const NCBKernIndex& ncb_index) mutable {
            bundle.set(kern_param.workspace_ptr);
            MultithreadDirectConvCommon<float, float>::copy_padding_kern_stride(
                    bundle, kern_param, ncb_index, ncb_index.ndrange_id);
        };
        ret_kerns.push_back({copy_padding, {group, N, IC}});
        auto do_conv = [bundle, conv_kern_function](
                               const NCBKernParam& kern_param,
                               const NCBKernIndex& ncb_index) mutable {
            bundle.set(kern_param.workspace_ptr);
            MultithreadDirectConvCommon<float, float>::do_conv_kern_stride(
                    bundle, kern_param, ncb_index, conv_kern_function,
                    ncb_index.ndrange_id);
        };
        ret_kerns.push_back({do_conv, {group, N, OC}});
    }
    return ret_kerns;
}

SmallVector<ConvBiasImpl::NCBKern>
ConvBiasImpl::AlgoF32DirectStride2::dispatch_kerns(
        const NCBKernSizeParam& param) const {
    MIDOUT_BEGIN(megdnn_arm_common_conv_bias_f32_kimpl, 2, 2) {
        return get_kimpls(param);
    }
    MIDOUT_END();
    return {};
}
// vim: syntax=cpp.doxygen
