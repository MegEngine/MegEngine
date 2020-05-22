/**
 * \file dnn/src/arm_common/conv_bias/f16/algos.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/arm_common/conv_bias/f16/algos.h"
#include "src/arm_common/conv_bias/direct/multi_thread_common.h"
#include "src/arm_common/conv_bias/f16/direct.h"
#include "src/arm_common/conv_bias/f16/do_conv_stride1.h"
#include "src/arm_common/conv_bias/f16/strategy.h"
#include "src/arm_common/conv_bias/img2col_helper.h"
#include "src/arm_common/conv_bias/postprocess_helper.h"
#include "src/common/opr_delegate.h"
#include "src/fallback/conv_bias/common.h"
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
#include "midout.h"
MIDOUT_DECL(megdnn_arm_common_winograd_fp16)
using namespace megdnn;
using namespace arm_common;

/* ======================= AlgoFP16WinogradF23 ======================== */

bool ConvBiasImpl::AlgoFP16WinogradF23::usable(
        fallback::ConvBiasImpl* opr, const NCBKernSizeParam& param,
        AlgoSelectionStrategy /*algo_selection_strategy*/) const {
    MEGDNN_MARK_USED_VAR(param);
    MEGDNN_MARK_USED_VAR(opr);
    MIDOUT_BEGIN(megdnn_arm_common_winograd_fp16, 0, 0) {
        using Strategy = winograd::winograd_2x3_4x4_f16;
        Strategy strategy(param.src_type, param.filter_type, param.dst_type);
        auto&& matmul_param =
                megdnn::winograd::ConvBias<Strategy>(
                        strategy, m_tile_size, param.nr_threads, param.osz[0],
                        param.osz[1], param.filter_meta.ocpg)
                        .get_matmul_kern_param(param);
        return m_matmul_algo->usable(matmul_param) &&
               (opr->param().format == param::ConvBias::Format::NCHW ||
                (opr->param().format ==
                         param::ConvBias::Format::NCHW_WINOGRAD &&
                 opr->param().output_block_size == 2 &&
                 param.winograd_matmul_format ==
                         param::MatrixMul::Format::DEFAULT)) &&
               opr->param().mode == param::ConvBias::Mode::CROSS_CORRELATION &&
               (param.filter_meta.spatial[0] == param.filter_meta.spatial[1] &&
                param.filter_meta.spatial[0] == 3) &&
               (param.filter_meta.stride[0] == param.filter_meta.stride[1] &&
                param.filter_meta.stride[0] == 1) &&
               (param.filter_meta.dilation[0] ==
                        param.filter_meta.dilation[1] &&
                param.filter_meta.dilation[0] == 1) &&
               param.compute_mode == param::ConvBias::ComputeMode::DEFAULT &&
               param.src_type.enumv() == DTypeEnum::Float16 &&
               param.filter_meta.icpg % 4 == 0 &&
               param.filter_meta.ocpg % 4 == 0;
    }
    MIDOUT_END();
    return false;
}

size_t ConvBiasImpl::AlgoFP16WinogradF23::get_workspace(
        fallback::ConvBiasImpl*, const NCBKernSizeParam& param) const {
    MEGDNN_MARK_USED_VAR(param);
    MIDOUT_BEGIN(megdnn_arm_common_winograd_fp16, 0, 1) {
        winograd::winograd_2x3_4x4_f16 strategy(
                param.src_type, param.filter_type, param.dst_type);
        return megdnn::winograd::ConvBias<winograd::winograd_2x3_4x4_f16>(
                       strategy, m_tile_size, param.nr_threads, param.osz[0],
                       param.osz[1], param.filter_meta.ocpg)
                .get_workspace_size(param, m_matmul_algo);
    }
    MIDOUT_END();
    return 0;
}

SmallVector<ConvBiasImpl::NCBKern>
ConvBiasImpl::AlgoFP16WinogradF23::dispatch_kerns(
        fallback::ConvBiasImpl*, const NCBKernSizeParam& param) const {
    MEGDNN_MARK_USED_VAR(param);
    MIDOUT_BEGIN(megdnn_arm_common_winograd_fp16, 0, 2) {
        winograd::winograd_2x3_4x4_f16 strategy(
                param.src_type, param.filter_type, param.dst_type);

        auto winograd_impl =
                megdnn::winograd::ConvBias<winograd::winograd_2x3_4x4_f16>(
                        strategy, m_tile_size, param.nr_threads, param.osz[0],
                        param.osz[1], param.filter_meta.ocpg);
        return winograd_impl.get_kerns(param, m_matmul_algo);
    }
    MIDOUT_END();
    return {};
}

/* ======================= AlgoFP16WinogradF45 ======================== */

bool ConvBiasImpl::AlgoFP16WinogradF45::usable(
        fallback::ConvBiasImpl* opr, const NCBKernSizeParam& param,
        AlgoSelectionStrategy /*algo_selection_strategy*/) const {
    MEGDNN_MARK_USED_VAR(param);
    MEGDNN_MARK_USED_VAR(opr);
    MIDOUT_BEGIN(megdnn_arm_common_winograd_fp16, 1, 0) {
        using Strategy = winograd::winograd_4x5_1x1_f16;
        Strategy strategy(param.src_type, param.filter_type, param.dst_type);
        auto&& matmul_param =
                megdnn::winograd::ConvBias<Strategy>(
                        strategy, m_tile_size, param.nr_threads, param.osz[0],
                        param.osz[1], param.filter_meta.ocpg)
                        .get_matmul_kern_param(param);
        return m_matmul_algo->usable(matmul_param) &&
               (opr->param().format == param::ConvBias::Format::NCHW ||
                (opr->param().format ==
                         param::ConvBias::Format::NCHW_WINOGRAD &&
                 opr->param().output_block_size == 4 &&
                 param.winograd_matmul_format ==
                         param::MatrixMul::Format::DEFAULT)) &&
               opr->param().mode == param::ConvBias::Mode::CROSS_CORRELATION &&
               (param.filter_meta.spatial[0] == param.filter_meta.spatial[1] &&
                param.filter_meta.spatial[0] == 5) &&
               (param.filter_meta.stride[0] == param.filter_meta.stride[1] &&
                param.filter_meta.stride[0] == 1) &&
               (param.filter_meta.dilation[0] ==
                        param.filter_meta.dilation[1] &&
                param.filter_meta.dilation[0] == 1) &&
               param.compute_mode == param::ConvBias::ComputeMode::DEFAULT &&
               param.src_type.enumv() == DTypeEnum::Float16;
    }
    MIDOUT_END();
    return false;
}

size_t ConvBiasImpl::AlgoFP16WinogradF45::get_workspace(
        fallback::ConvBiasImpl*, const NCBKernSizeParam& param) const {
    MEGDNN_MARK_USED_VAR(param);
    winograd::winograd_4x5_1x1_f16 strategy(param.src_type, param.filter_type,
                                            param.dst_type);
    MIDOUT_BEGIN(megdnn_arm_common_winograd_fp16, 1, 1) {
        return megdnn::winograd::ConvBias<winograd::winograd_4x5_1x1_f16>(
                       strategy, m_tile_size, param.nr_threads, param.osz[0],
                       param.osz[1], param.filter_meta.ocpg)
                .get_workspace_size(param, m_matmul_algo);
    }
    MIDOUT_END();
    return 0;
}

SmallVector<ConvBiasImpl::NCBKern>
ConvBiasImpl::AlgoFP16WinogradF45::dispatch_kerns(
        fallback::ConvBiasImpl*, const NCBKernSizeParam& param) const {
    MEGDNN_MARK_USED_VAR(param);
    MIDOUT_BEGIN(megdnn_arm_common_winograd_fp16, 1, 2) {
        winograd::winograd_4x5_1x1_f16 strategy(
                param.src_type, param.filter_type, param.dst_type);
        auto winograd_impl =
                megdnn::winograd::ConvBias<winograd::winograd_4x5_1x1_f16>(
                        strategy, m_tile_size, param.nr_threads, param.osz[0],
                        param.osz[1], param.filter_meta.ocpg);
        return winograd_impl.get_kerns(param, m_matmul_algo);
    }
    MIDOUT_END();
    return {};
}
/* ======================= AlgoFP16WinogradF63 ======================== */

bool ConvBiasImpl::AlgoFP16WinogradF63::usable(
        fallback::ConvBiasImpl* opr, const NCBKernSizeParam& param,
        AlgoSelectionStrategy /*algo_selection_strategy*/) const {
    MEGDNN_MARK_USED_VAR(param);
    MEGDNN_MARK_USED_VAR(opr);
    MIDOUT_BEGIN(megdnn_arm_common_winograd_fp16, 2, 0) {
        using Strategy = winograd::winograd_6x3_1x1_f16;
        Strategy strategy(param.src_type, param.filter_type, param.dst_type);
        auto&& matmul_param =
                megdnn::winograd::ConvBias<Strategy>(
                        strategy, m_tile_size, param.nr_threads, param.osz[0],
                        param.osz[1], param.filter_meta.ocpg)
                        .get_matmul_kern_param(param);
        return m_matmul_algo->usable(matmul_param) &&
               (opr->param().format == param::ConvBias::Format::NCHW ||
                (opr->param().format ==
                         param::ConvBias::Format::NCHW_WINOGRAD &&
                 opr->param().output_block_size == 6 &&
                 param.winograd_matmul_format ==
                         param::MatrixMul::Format::DEFAULT)) &&
               opr->param().mode == param::ConvBias::Mode::CROSS_CORRELATION &&
               (param.filter_meta.spatial[0] == param.filter_meta.spatial[1] &&
                param.filter_meta.spatial[0] == 3) &&
               (param.filter_meta.stride[0] == param.filter_meta.stride[1] &&
                param.filter_meta.stride[0] == 1) &&
               (param.filter_meta.dilation[0] ==
                        param.filter_meta.dilation[1] &&
                param.filter_meta.dilation[0] == 1) &&
               param.compute_mode == param::ConvBias::ComputeMode::DEFAULT &&
               param.src_type.enumv() == DTypeEnum::Float16;
    }
    MIDOUT_END();
    return false;
}

size_t ConvBiasImpl::AlgoFP16WinogradF63::get_workspace(
        fallback::ConvBiasImpl*, const NCBKernSizeParam& param) const {
    MEGDNN_MARK_USED_VAR(param);
    winograd::winograd_6x3_1x1_f16 strategy(param.src_type, param.filter_type,
                                            param.dst_type);
    MIDOUT_BEGIN(megdnn_arm_common_winograd_fp16, 2, 1) {
        return megdnn::winograd::ConvBias<winograd::winograd_6x3_1x1_f16>(
                       strategy, m_tile_size, param.nr_threads, param.osz[0],
                       param.osz[1], param.filter_meta.ocpg)
                .get_workspace_size(param, m_matmul_algo);
    }
    MIDOUT_END();
    return 0;
}

SmallVector<ConvBiasImpl::NCBKern>
ConvBiasImpl::AlgoFP16WinogradF63::dispatch_kerns(
        fallback::ConvBiasImpl*, const NCBKernSizeParam& param) const {
    MEGDNN_MARK_USED_VAR(param);
    MIDOUT_BEGIN(megdnn_arm_common_winograd_fp16, 2, 2) {
        winograd::winograd_6x3_1x1_f16 strategy(
                param.src_type, param.filter_type, param.dst_type);
        auto winograd_impl =
                megdnn::winograd::ConvBias<winograd::winograd_6x3_1x1_f16>(
                        strategy, m_tile_size, param.nr_threads, param.osz[0],
                        param.osz[1], param.filter_meta.ocpg);
        return winograd_impl.get_kerns(param, m_matmul_algo);
    }
    MIDOUT_END();
    return {};
}

/* ======================= AlgoFP16WinogradF23_8x8 ======================== */

bool ConvBiasImpl::AlgoFP16WinogradF23_8x8::usable(
        fallback::ConvBiasImpl* opr, const NCBKernSizeParam& param,
        AlgoSelectionStrategy /*algo_selection_strategy*/) const {
    MEGDNN_MARK_USED_VAR(param);
    MEGDNN_MARK_USED_VAR(opr);
    MIDOUT_BEGIN(megdnn_arm_common_winograd_fp16, 3, 0) {
        if (param.filter_meta.icpg % 8 != 0 || param.filter_meta.ocpg % 8 != 0)
            return false;
        using Strategy = winograd::winograd_2x3_8x8_f16;
        Strategy strategy(param.src_type, param.filter_type, param.dst_type);
        auto&& matmul_param =
                megdnn::winograd::ConvBias<Strategy,
                                           param::MatrixMul::Format::MK8>(
                        strategy, m_tile_size, param.nr_threads, param.osz[0],
                        param.osz[1], param.filter_meta.ocpg)
                        .get_matmul_kern_param(param);
        return m_matmul_algo->usable(matmul_param) &&
               (opr->param().format == param::ConvBias::Format::NCHW ||
                (opr->param().format ==
                         param::ConvBias::Format::NCHW_WINOGRAD &&
                 opr->param().output_block_size == 2 &&
                 param.winograd_matmul_format ==
                         param::MatrixMul::Format::MK8)) &&
               opr->param().mode == param::ConvBias::Mode::CROSS_CORRELATION &&
               (param.filter_meta.spatial[0] == param.filter_meta.spatial[1] &&
                param.filter_meta.spatial[0] == 3) &&
               (param.filter_meta.stride[0] == param.filter_meta.stride[1] &&
                param.filter_meta.stride[0] == 1) &&
               (param.filter_meta.dilation[0] ==
                        param.filter_meta.dilation[1] &&
                param.filter_meta.dilation[0] == 1) &&
               param.compute_mode == param::ConvBias::ComputeMode::DEFAULT &&
               param.src_type.enumv() == DTypeEnum::Float16;
    }
    MIDOUT_END();
    return false;
}

size_t ConvBiasImpl::AlgoFP16WinogradF23_8x8::get_workspace(
        fallback::ConvBiasImpl*, const NCBKernSizeParam& param) const {
    MEGDNN_MARK_USED_VAR(param);
    MIDOUT_BEGIN(megdnn_arm_common_winograd_fp16, 3, 1) {
        winograd::winograd_2x3_8x8_f16 strategy(
                param.src_type, param.filter_type, param.dst_type);
        return megdnn::winograd::ConvBias<winograd::winograd_2x3_8x8_f16,
                                          param::MatrixMul::Format::MK8>(
                       strategy, m_tile_size, param.nr_threads, param.osz[0],
                       param.osz[1], param.filter_meta.ocpg)
                .get_workspace_size(param, m_matmul_algo);
    }
    MIDOUT_END();
    return 0;
}

SmallVector<ConvBiasImpl::NCBKern>
ConvBiasImpl::AlgoFP16WinogradF23_8x8::dispatch_kerns(
        fallback::ConvBiasImpl*, const NCBKernSizeParam& param) const {
    MEGDNN_MARK_USED_VAR(param);
    MIDOUT_BEGIN(megdnn_arm_common_winograd_fp32, 3, 2) {
        winograd::winograd_2x3_8x8_f16 strategy(
                param.src_type, param.filter_type, param.dst_type);
        auto winograd_impl =
                megdnn::winograd::ConvBias<winograd::winograd_2x3_8x8_f16,
                                           param::MatrixMul::Format::MK8>(
                        strategy, m_tile_size, param.nr_threads, param.osz[0],
                        param.osz[1], param.filter_meta.ocpg);
        return winograd_impl.get_kerns(param, m_matmul_algo);
    }
    MIDOUT_END();
    return {};
}

/*========================from Convolution=============================*/

MIDOUT_DECL(megdnn_arm_common_conv_bias_fp16_kimpl)

bool ConvBiasImpl::AlgoF16Direct::usable(
        fallback::ConvBiasImpl*, const NCBKernSizeParam& param,
        AlgoSelectionStrategy algo_selection_strategy) const {
    MIDOUT_BEGIN(megdnn_arm_common_conv_bias_fp16_kimpl, 0, 0) {
        auto&& fm = param.filter_meta;
        auto FH = fm.spatial[0];
        auto SH = fm.stride[0], SW = fm.stride[1];
        // the condition ``param.isz[0]*param.isz[1] >= 8'' and
        // ``param.osz[0]*param.osz[1] >= 8'' comes from the fact that the
        // kernel may have access to up to 8 fp16 after the end of the memory
        // chunk.
        bool aviliable = fm.format == param::ConvBias::Format::NCHW &&
                         param.src_type.enumv() == DTypeEnum::Float16 &&
                         param.filter_type.enumv() == DTypeEnum::Float16 &&
                         param.dst_type.enumv() == DTypeEnum::Float16 &&
                         fm.spatial_ndim == 2 && fm.dilation[0] == 1 &&
                         fm.dilation[1] == 1 &&
                         param.isz[0] * param.isz[1] >= 8 &&
                         param.osz[0] * param.osz[1] >= 8 && FH <= 7 &&
                         SH == 1 && SW == 1;
        if (algo_selection_strategy == AlgoSelectionStrategy::HEURISTIC) {
            bool large_group = param.filter_meta.group >= param.nr_threads;
            aviliable &= (large_group == m_large_group);
        }
        return aviliable;
    }
    MIDOUT_END();
    return false;
}

size_t ConvBiasImpl::AlgoF16Direct::get_workspace(
        fallback::ConvBiasImpl*, const NCBKernSizeParam& param) const {
    MIDOUT_BEGIN(megdnn_arm_common_conv_bias_fp16_kimpl, 0, 1) {
        auto wbundle =
                MultithreadDirectConvCommon<dt_float16, __fp16>::get_bundle(
                        param, m_large_group);
        return wbundle.total_size_in_bytes();
    }
    MIDOUT_END();
    return 0;
}

SmallVector<ConvBiasImpl::NCBKern> ConvBiasImpl::AlgoF16Direct::get_kimpls(
        const NCBKernSizeParam& param) const {
    auto fm = param.filter_meta;
    size_t N = param.n;
    size_t IC = param.filter_meta.icpg;
    size_t OC = param.filter_meta.ocpg;
    size_t group = fm.group;
    WorkspaceBundle wbundle =
            MultithreadDirectConvCommon<dt_float16, __fp16>::get_bundle(
                    param, m_large_group);
    SmallVector<NCBKern> ret_kerns;
    //! When group >= nr_threads, treat it as large_group, each thread process
    //! one group for better performance
    if (m_large_group) {
        //! Channel wise conv and big groups
        auto exec_one_group = [wbundle](const NCBKernParam& kern_param,
                                        const NCBKernIndex& ncb_index) {
            auto fm = kern_param.filter_meta;
            size_t IC = fm.icpg;
            size_t OC = fm.ocpg;
            WorkspaceBundle bundle = wbundle;
            if (fm.should_flip) {
                for (size_t oc = 0; oc < OC; oc++) {
                    MultithreadDirectConvCommon<dt_float16, __fp16>::
                            weight_flip_kern(bundle, kern_param, ncb_index,
                                             {ncb_index.thread_id, 0, oc});
                }
            }
            for (size_t ic = 0; ic < IC; ic++) {
                MultithreadDirectConvCommon<dt_float16, __fp16>::
                        copy_padding_kern(bundle, kern_param, ncb_index,
                                          {ncb_index.thread_id, 0, ic});
            }
            for (size_t oc = 0; oc < OC; oc++) {
                MultithreadDirectConvCommon<dt_float16, __fp16>::do_conv_kern(
                        bundle, kern_param, ncb_index,
                        fp16::conv_bias::kern_direct_f16,
                        {ncb_index.thread_id, 0, oc});
            }
        };
        ret_kerns.push_back({exec_one_group, {group, N, 1_z}});
    } else {
        WorkspaceBundle bundle = wbundle;
        if (fm.should_flip) {
            auto weight_flip = [bundle](const NCBKernParam& kern_param,
                                        const NCBKernIndex& ncb_index) {
                MultithreadDirectConvCommon<dt_float16, __fp16>::
                        weight_flip_kern(bundle, kern_param, ncb_index,
                                         ncb_index.ndrange_id);
            };
            ret_kerns.push_back({weight_flip, {group, 1_z, OC}});
        }
        auto copy_padding = [bundle](const NCBKernParam& kern_param,
                                     const NCBKernIndex& ncb_index) {
            MultithreadDirectConvCommon<dt_float16, __fp16>::copy_padding_kern(
                    bundle, kern_param, ncb_index, ncb_index.ndrange_id);
        };
        ret_kerns.push_back({copy_padding, {group, N, IC}});
        auto do_conv = [bundle](const NCBKernParam& kern_param,
                                const NCBKernIndex& ncb_index) {
            MultithreadDirectConvCommon<dt_float16, __fp16>::do_conv_kern(
                    bundle, kern_param, ncb_index,
                    fp16::conv_bias::kern_direct_f16, ncb_index.ndrange_id);
        };
        ret_kerns.push_back({do_conv, {group, N, OC}});
    }
    return ret_kerns;
}

SmallVector<ConvBiasImpl::NCBKern> ConvBiasImpl::AlgoF16Direct::dispatch_kerns(
        fallback::ConvBiasImpl*, const NCBKernSizeParam& param) const {
    MIDOUT_BEGIN(megdnn_arm_common_conv_bias_fp16_kimpl, 0, 1) {
        return get_kimpls(param);
    }
    MIDOUT_END();
    return {};
}

/* ===================== stride-1 algo ===================== */

bool ConvBiasImpl::AlgoF16DirectStride1::usable(
        fallback::ConvBiasImpl*, const NCBKernSizeParam& param,
        AlgoSelectionStrategy algo_selection_strategy) const {
    MIDOUT_BEGIN(megdnn_arm_common_conv_bias_fp16_kimpl, 1, 0) {
        auto&& fm = param.filter_meta;
        auto FH = fm.spatial[0];
        bool aviliable =
                param.filter_meta.format == param::ConvBias::Format::NCHW &&
                param.src_type.enumv() == DTypeEnum::Float16 &&
                param.filter_type.enumv() == DTypeEnum::Float16 &&
                param.dst_type.enumv() == DTypeEnum::Float16 &&
                !fm.should_flip && fm.spatial_ndim == 2 &&
                fm.dilation[0] == 1 && fm.dilation[1] == 1 &&
                fm.stride[0] == 1 && fm.stride[1] == 1 && FH == fm.spatial[1] &&
                (FH == 2 || FH == 3 || FH == 5);
        if (algo_selection_strategy ==
            ConvBiasImpl::AlgoSelectionStrategy::HEURISTIC) {
            bool large_group = param.filter_meta.group >= param.nr_threads;
            aviliable &= (large_group == m_large_group);
        }
        return aviliable;
    }
    MIDOUT_END();
    return false;
}

SmallVector<ConvBiasImpl::NCBKern>
ConvBiasImpl::AlgoF16DirectStride1::get_kimpls(
        const NCBKernSizeParam& param) const {
    auto fm = param.filter_meta;
    auto FH = fm.spatial[0];
    size_t N = param.n;
    size_t IC = param.filter_meta.icpg;
    size_t OC = param.filter_meta.ocpg;
    size_t group = fm.group;
    using Func = std::function<void(const __fp16*, const __fp16*, __fp16*,
                                    size_t, size_t, size_t, size_t, size_t)>;
    Func conv_kern_function = nullptr;

#define SWITCH_KERN()                                                     \
    switch (FH) {                                                         \
        case 2:                                                           \
            conv_kern_function = fp16::conv_stride1::do_conv_2x2_stride1; \
            break;                                                        \
        case 3:                                                           \
            conv_kern_function = fp16::conv_stride1::do_conv_3x3_stride1; \
            break;                                                        \
        case 5:                                                           \
            conv_kern_function = fp16::conv_stride1::do_conv_5x5_stride1; \
            break;                                                        \
    }
    SWITCH_KERN();

    WorkspaceBundle wbundle =
            MultithreadDirectConvCommon<dt_float16, __fp16>::get_bundle_stride(
                    param, m_large_group);
    SmallVector<NCBKern> ret_kerns;
    //! When group >= nr_threads, treat it as large_group, each thread process
    //! one group for better performance
    if (m_large_group) {
        //! Channel wise conv and big groups
        auto exec_one_group = [wbundle, conv_kern_function](
                                      const NCBKernParam& kern_param,
                                      const NCBKernIndex& ncb_index) {
            auto fm = kern_param.filter_meta;
            size_t IC = fm.icpg;
            size_t OC = fm.ocpg;
            WorkspaceBundle bundle = wbundle;
            for (size_t ic = 0; ic < IC; ic++) {
                MultithreadDirectConvCommon<dt_float16, __fp16>::
                        copy_padding_kern_stride(bundle, kern_param, ncb_index,
                                                 {ncb_index.thread_id, 0, ic});
            }
            for (size_t oc = 0; oc < OC; oc++) {
                MultithreadDirectConvCommon<dt_float16, __fp16>::
                        do_conv_kern_stride(bundle, kern_param, ncb_index,
                                            conv_kern_function,
                                            {ncb_index.thread_id, 0, oc});
            }
        };
        ret_kerns.push_back({exec_one_group, {group, N, 1_z}});
    } else {
        WorkspaceBundle bundle = wbundle;
        auto copy_padding = [bundle](const NCBKernParam& kern_param,
                                     const NCBKernIndex& ncb_index) {
            MultithreadDirectConvCommon<dt_float16, __fp16>::
                    copy_padding_kern_stride(bundle, kern_param, ncb_index,
                                             ncb_index.ndrange_id);
        };
        ret_kerns.push_back({copy_padding, {group, N, IC}});
        auto do_conv = [bundle, conv_kern_function](
                               const NCBKernParam& kern_param,
                               const NCBKernIndex& ncb_index) {
            MultithreadDirectConvCommon<dt_float16, __fp16>::
                    do_conv_kern_stride(bundle, kern_param, ncb_index,
                                        conv_kern_function,
                                        ncb_index.ndrange_id);
        };
        ret_kerns.push_back({do_conv, {group, N, OC}});
    }
    return ret_kerns;
}

size_t ConvBiasImpl::AlgoF16DirectStride1::get_workspace(
        fallback::ConvBiasImpl*, const NCBKernSizeParam& param) const {
    MIDOUT_BEGIN(megdnn_arm_common_conv_bias_fp16_kimpl, 1, 1) {
        auto bundle = MultithreadDirectConvCommon<
                dt_float16, __fp16>::get_bundle_stride(param, m_large_group);
        return bundle.total_size_in_bytes();
    }
    MIDOUT_END();
    return 0;
}

SmallVector<ConvBiasImpl::NCBKern>
ConvBiasImpl::AlgoF16DirectStride1::dispatch_kerns(
        fallback::ConvBiasImpl*, const NCBKernSizeParam& param) const {
    MIDOUT_BEGIN(megdnn_arm_common_conv_bias_fp16_kimpl, 1, 2) {
        return get_kimpls(param);
    }
    MIDOUT_END();
    return {};
}

#endif
// vim: syntax=cpp.doxygen
