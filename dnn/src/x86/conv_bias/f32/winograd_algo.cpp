/**
 * \file dnn/src/x86/conv_bias/f32/winograd_algo.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/x86/conv_bias/f32/algos.h"
#include "src/common/utils.h"
#include "src/x86/conv_bias/opr_impl.h"
#include "src/x86/conv_bias/postprocess_helper.h"
#include "src/x86/handle.h"
#include "src/x86/profile.h"
#include "src/x86/conv_bias/f32/strategy.h"

#include "midout.h"

MIDOUT_DECL(megdnn_x86_winograd_fp32)

using namespace megdnn;
using namespace x86;

/* ======================= AlgoFP32WinogradF63_8*8 ======================== */

bool ConvBiasImpl::AlgoFP32WinogradF63_8x8::usable(
        fallback::ConvBiasImpl* opr, const NCBKernSizeParam& param,
        AlgoSelectionStrategy /*algo_selection_strategy*/) const {
    MEGDNN_MARK_USED_VAR(param);
    MEGDNN_MARK_USED_VAR(opr);
    MIDOUT_BEGIN(megdnn_x86_winograd_fp32, 1, 0) {
        //! TODO: now nchw88 winograd only support Dense mode
        if (param.filter_meta.icpg % 8 != 0 ||
            param.filter_meta.ocpg % 8 != 0 || param.filter_meta.group != 1)
            return false;
        using Strategy = winograd::winograd_nchw88_6x3_8x8_f;
        Strategy strategy(param.src_type, param.filter_type, param.dst_type);
        auto&& matmul_param =
                megdnn::winograd::ConvBias<Strategy,
                                           param::MatrixMul::Format::MK8>(
                        strategy, m_tile_size, param.nr_threads, param.osz[0],
                        param.osz[1], param.filter_meta.ocpg)
                        .get_matmul_kern_param(param);
        return m_matmul_algo->usable(matmul_param) &&
               (opr->param().format == param::ConvBias::Format::NCHW88 ||
                (opr->param().format ==
                         param::ConvBias::Format::NCHW88_WINOGRAD &&
                 opr->param().output_block_size == 6 &&
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
               param.src_type.enumv() == DTypeEnum::Float32 &&
               is_supported(SIMDType::AVX2);
    }
    MIDOUT_END();
    return false;
}

size_t ConvBiasImpl::AlgoFP32WinogradF63_8x8::get_workspace(
        fallback::ConvBiasImpl*, const NCBKernSizeParam& param) const {
    MEGDNN_MARK_USED_VAR(param);
    MIDOUT_BEGIN(megdnn_x86_winograd_fp32, 1, 1) {
        winograd::winograd_nchw88_6x3_8x8_f strategy(
                param.src_type, param.filter_type, param.dst_type);
        return megdnn::winograd::ConvBias<winograd::winograd_nchw88_6x3_8x8_f,
                                          param::MatrixMul::Format::MK8>(
                       strategy, m_tile_size, param.nr_threads, param.osz[0],
                       param.osz[1], param.filter_meta.ocpg)
                .get_workspace_size(param, m_matmul_algo);
    }
    MIDOUT_END();
    return 0;
}

SmallVector<ConvBiasImpl::NCBKern>
ConvBiasImpl::AlgoFP32WinogradF63_8x8::dispatch_kerns(
        fallback::ConvBiasImpl*, const NCBKernSizeParam& param) const {
    MEGDNN_MARK_USED_VAR(param);
    MIDOUT_BEGIN(megdnn_arm_common_winograd_fp32, 1, 2) {
        winograd::winograd_nchw88_6x3_8x8_f strategy(
                param.src_type, param.filter_type, param.dst_type);
        auto winograd_impl =
                megdnn::winograd::ConvBias<winograd::winograd_nchw88_6x3_8x8_f,
                                           param::MatrixMul::Format::MK8>(
                        strategy, m_tile_size, param.nr_threads, param.osz[0],
                        param.osz[1], param.filter_meta.ocpg);
        return winograd_impl.get_kerns(param, m_matmul_algo);
    }
    MIDOUT_END();
    return {};
}

/* ======================= AlgoFP32WinogradF23_8*8 ======================== */

bool ConvBiasImpl::AlgoFP32WinogradF23_8x8::usable(
        fallback::ConvBiasImpl* opr, const NCBKernSizeParam& param,
        AlgoSelectionStrategy /*algo_selection_strategy*/) const {
    MEGDNN_MARK_USED_VAR(param);
    MEGDNN_MARK_USED_VAR(opr);
    MIDOUT_BEGIN(megdnn_x86_winograd_fp32, 2, 0) {
        //! TODO: now nchw88 winograd only support Dense mode
        if (param.filter_meta.icpg % 8 != 0 ||
            param.filter_meta.ocpg % 8 != 0 || param.filter_meta.group != 1)
            return false;
        using Strategy = winograd::winograd_nchw88_2x3_8x8_f;
        Strategy strategy(param.src_type, param.filter_type, param.dst_type);
        auto&& matmul_param =
                megdnn::winograd::ConvBias<Strategy,
                                           param::MatrixMul::Format::MK8>(
                        strategy, m_tile_size, param.nr_threads, param.osz[0],
                        param.osz[1], param.filter_meta.ocpg)
                        .get_matmul_kern_param(param);
        return m_matmul_algo->usable(matmul_param) &&
               (opr->param().format == param::ConvBias::Format::NCHW88 ||
                (opr->param().format ==
                         param::ConvBias::Format::NCHW88_WINOGRAD &&
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
               param.src_type.enumv() == DTypeEnum::Float32 &&
               is_supported(SIMDType::AVX2);
    }
    MIDOUT_END();
    return false;
}

size_t ConvBiasImpl::AlgoFP32WinogradF23_8x8::get_workspace(
        fallback::ConvBiasImpl*, const NCBKernSizeParam& param) const {
    MEGDNN_MARK_USED_VAR(param);
    MIDOUT_BEGIN(megdnn_x86_winograd_fp32, 2, 1) {
        winograd::winograd_nchw88_2x3_8x8_f strategy(
                param.src_type, param.filter_type, param.dst_type);
        return megdnn::winograd::ConvBias<winograd::winograd_nchw88_2x3_8x8_f,
                                          param::MatrixMul::Format::MK8>(
                       strategy, m_tile_size, param.nr_threads, param.osz[0],
                       param.osz[1], param.filter_meta.ocpg)
                .get_workspace_size(param, m_matmul_algo);
    }
    MIDOUT_END();
    return 0;
}

SmallVector<ConvBiasImpl::NCBKern>
ConvBiasImpl::AlgoFP32WinogradF23_8x8::dispatch_kerns(
        fallback::ConvBiasImpl*, const NCBKernSizeParam& param) const {
    MEGDNN_MARK_USED_VAR(param);
    MIDOUT_BEGIN(megdnn_arm_common_winograd_fp32, 2, 2) {
        winograd::winograd_nchw88_2x3_8x8_f strategy(
                param.src_type, param.filter_type, param.dst_type);
        auto winograd_impl =
                megdnn::winograd::ConvBias<winograd::winograd_nchw88_2x3_8x8_f,
                                           param::MatrixMul::Format::MK8>(
                        strategy, m_tile_size, param.nr_threads, param.osz[0],
                        param.osz[1], param.filter_meta.ocpg);
        return winograd_impl.get_kerns(param, m_matmul_algo);
    }
    MIDOUT_END();
    return {};
}
// vim: syntax=cpp.doxygen
