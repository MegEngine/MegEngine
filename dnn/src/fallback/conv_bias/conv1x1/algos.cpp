/**
 * \file dnn/src/fallback/conv_bias/conv1x1/algos.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/common/opr_delegate.h"
#include "src/fallback/conv_bias/common.h"
#include "src/fallback/conv_bias/conv1x1/algos.h"
#include "src/fallback/conv_bias/conv1x1/conv1x1_dispatcher.h"
#include "src/fallback/conv_bias/conv1x1/conv1x1_strategy.h"
#include "src/fallback/conv_bias/opr_impl.h"

#include "megdnn/opr_param_defs.h"
#include "src/naive/convolution/helper.h"

#if MEGDNN_X86
#include "src/x86/conv_bias/postprocess_helper.h"
#elif (MEGDNN_ARMV7 || MEGDNN_AARCH64)
#include "src/arm_common/conv_bias/postprocess_helper.h"
#else
#include "src/common/postprocess_helper.h"
#endif

#include "midout.h"
MIDOUT_DECL(megdnn_fallback_conv1x1)

using namespace megdnn;
using namespace fallback;
#if MEGDNN_X86
using namespace x86;
#endif
using namespace conv1x1;

size_t ConvBiasImpl::AlgoConv1x1::get_oc_tile_size_heuristic(
        const NCBKernSizeParam& param) const {
    size_t OH = param.osz[0];
    size_t OW = param.osz[1];
    size_t OC = param.filter_meta.ocpg;
    if (OH * OW >= 56 * 56 || OC >= 64)
        return m_oc_block_size;
    size_t oc_block_size_one_thread = div_ceil(OC, param.nr_threads);
    return round_up<size_t>(oc_block_size_one_thread, 24);
}

WorkspaceBundle ConvBiasImpl::AlgoConv1x1::get_bundle_according_packmode(
        const NCBKernSizeParam& param) const {
    size_t OH = param.osz[0];
    size_t OW = param.osz[1];
    size_t compt_oc_block_size = get_oc_tile_size_heuristic(param);

    auto matmul_param =
            utils::get_matmul_kern_param(param, OH * OW, compt_oc_block_size);

    auto pack_mode = m_matmul_algo->packmode();
    if (pack_mode == MatrixMulImpl::AlgoBase::PackMode::DEFAULT) {
        MIDOUT_BEGIN(megdnn_fallback_conv1x1,
                     midout_iv("get_bundle_default"_hash)) {
            return Conv1x1Kerns<MatrixMulImpl::AlgoBase::PackMode::DEFAULT>()
                    .get_bundle(param, matmul_param, m_matmul_algo,
                                compt_oc_block_size);
        }
        MIDOUT_END();
    } else if (pack_mode == MatrixMulImpl::AlgoBase::PackMode::ONLY_PACKA) {
        MIDOUT_BEGIN(megdnn_fallback_conv1x1,
                     midout_iv("get_bundle_only_packa"_hash)) {
            return Conv1x1Kerns<MatrixMulImpl::AlgoBase::PackMode::ONLY_PACKA>()
                    .get_bundle(param, matmul_param, m_matmul_algo,
                                compt_oc_block_size);
        }
        MIDOUT_END();
    } else {
        MIDOUT_BEGIN(megdnn_fallback_conv1x1,
                     midout_iv("get_bundle_no_pack"_hash)) {
            return Conv1x1Kerns<MatrixMulImpl::AlgoBase::PackMode::NO_PACK>()
                    .get_bundle(param, matmul_param, m_matmul_algo,
                                compt_oc_block_size);
        }
        MIDOUT_END();
    }
    return {nullptr, {}};
}

size_t ConvBiasImpl::AlgoConv1x1::get_workspace(
        const NCBKernSizeParam& param) const {
    return get_bundle_according_packmode(param).total_size_in_bytes();
}

SmallVector<ConvBiasImpl::NCBKern>
ConvBiasImpl::AlgoConv1x1::get_kerns_according_packmode(
        const NCBKernSizeParam& param, bool weight_preprocess) const {
    size_t OH = param.osz[0];
    size_t OW = param.osz[1];
    size_t compt_oc_block_size = get_oc_tile_size_heuristic(param);
    auto pack_mode = m_matmul_algo->packmode();

    Conv1x1StrategyBase* conv1x1_strategy =
            Conv1x1Factory::make_conv1x1_strategy(param, pack_mode,
                                                  param.filter_meta.format);
    auto matmul_param =
            utils::get_matmul_kern_param(param, OH * OW, compt_oc_block_size);

    WorkspaceBundle whole_bundle = get_bundle_according_packmode(param);
    //! NO_PACK not implement get_bundle
    WorkspaceBundle matmul_bundle = {nullptr, {}};
    if (pack_mode == MatrixMulImpl::AlgoBase::PackMode::NO_PACK) {
        matmul_bundle = {nullptr,
                         {0, 0, m_matmul_algo->get_workspace(matmul_param)}};
    } else {
        matmul_bundle = m_matmul_algo->get_bundle(matmul_param);
    }
    WorkspaceBundle thread_bundle = utils::get_thread_bundle(
            param, matmul_bundle.get_size(2), compt_oc_block_size);

    if (pack_mode == MatrixMulImpl::AlgoBase::PackMode::DEFAULT) {
        MIDOUT_BEGIN(megdnn_fallback_conv1x1,
                     midout_iv("get_kern_default"_hash)) {
            if (!weight_preprocess) {
                return Conv1x1Kerns<
                               MatrixMulImpl::AlgoBase::PackMode::DEFAULT>()
                        .get_kern(param, whole_bundle, matmul_bundle,
                                  thread_bundle, conv1x1_strategy,
                                  m_matmul_algo, compt_oc_block_size);
            } else {
                return Conv1x1Kerns<
                               MatrixMulImpl::AlgoBase::PackMode::DEFAULT>()
                        .get_kern_preprocess(param, whole_bundle, matmul_bundle,
                                             conv1x1_strategy, m_matmul_algo,
                                             compt_oc_block_size);
            }
        }
        MIDOUT_END();
    } else if (pack_mode == MatrixMulImpl::AlgoBase::PackMode::ONLY_PACKA) {
        MIDOUT_BEGIN(megdnn_fallback_conv1x1,
                     midout_iv("get_kern_only_packa"_hash)) {
            if (!weight_preprocess) {
                return Conv1x1Kerns<
                               MatrixMulImpl::AlgoBase::PackMode::ONLY_PACKA>()
                        .get_kern(param, whole_bundle, matmul_bundle,
                                  thread_bundle, conv1x1_strategy,
                                  m_matmul_algo, compt_oc_block_size);
            } else {
                return Conv1x1Kerns<
                               MatrixMulImpl::AlgoBase::PackMode::ONLY_PACKA>()
                        .get_kern_preprocess(param, whole_bundle, matmul_bundle,
                                             conv1x1_strategy, m_matmul_algo,
                                             compt_oc_block_size);
            }
        }
        MIDOUT_END();
    } else {
        MIDOUT_BEGIN(megdnn_fallback_conv1x1,
                     midout_iv("get_kern_no_pack"_hash)) {
            if (!weight_preprocess) {
                return Conv1x1Kerns<
                               MatrixMulImpl::AlgoBase::PackMode::NO_PACK>()
                        .get_kern(param, whole_bundle, matmul_bundle,
                                  thread_bundle, conv1x1_strategy,
                                  m_matmul_algo, compt_oc_block_size);
            } else {
                return Conv1x1Kerns<
                               MatrixMulImpl::AlgoBase::PackMode::NO_PACK>()
                        .get_kern_preprocess(param, whole_bundle, matmul_bundle,
                                             conv1x1_strategy, m_matmul_algo,
                                             compt_oc_block_size);
            }
        }
        MIDOUT_END();
    }
}

SmallVector<ConvBiasImpl::NCBKern> ConvBiasImpl::AlgoConv1x1::dispatch_kerns(
        const NCBKernSizeParam& param) const {
    return get_kerns_according_packmode(param, false);
}

SmallVector<TensorLayout>
ConvBiasImpl::AlgoConv1x1::deduce_preprocessed_filter_layout(
        const NCBKernSizeParam& param) const {
    MIDOUT_BEGIN(megdnn_fallback_conv1x1,
                 midout_iv("deduce_preprocessed_filter_layout"_hash)) {
        WorkspaceBundle wb = get_bundle_according_packmode(param);

        size_t GROUP = param.filter_meta.group;
        SmallVector<TensorLayout> preprocessed_layouts;
        preprocessed_layouts.push_back(
                {{GROUP, wb.get_size(0)}, dtype::Int8()});
        return preprocessed_layouts;
    }
    MIDOUT_END();
    return {};
}

SmallVector<ConvBiasImpl::NCBKern>
ConvBiasImpl::AlgoConv1x1::dispatch_preprocess_kerns(
        const NCBKernSizeParam& param) const {
    return get_kerns_according_packmode(param, true);
}

bool ConvBiasImpl::AlgoConv1x1::usable(const NCBKernSizeParam& param,
                                       AlgoSelectionStrategy) const {
    MIDOUT_BEGIN(megdnn_fallback_conv1x1, 0, 2) {
        size_t FH = param.filter_meta.spatial[0],
               FW = param.filter_meta.spatial[1];
        size_t PH = param.filter_meta.padding[0],
               PW = param.filter_meta.padding[1];
        size_t SH = param.filter_meta.stride[0],
               SW = param.filter_meta.stride[1];
        auto format = param.filter_meta.format;
        size_t OH = param.osz[0];
        size_t OW = param.osz[1];
#if MEGDNN_AARCH64 || MEGDNN_ARMV7
        if (format != param::ConvBias::Format::NCHW &&
            format != param::ConvBias::Format::NCHW44 &&
            format != param::ConvBias::Format::NCHW44_DOT) {
            return false;
        }
        //! hybird mode is not support
        if (param.filter_meta.format == param::ConvBias::Format::NCHW44 ||
            param.filter_meta.format == param::ConvBias::Format::NCHW44_DOT) {
            if (param.filter_meta.icpg < 4_z || param.filter_meta.icpg == 1 ||
                param.filter_meta.ocpg == 1) {
                return false;
            }
        }
#else   //! x86 only support nchw mode
        if (format != param::ConvBias::Format::NCHW) {
            return false;
        }
#endif
        //! param
        if (FH != 1 || FW != 1 || PH || PW || SH != 1 || SW != 1) {
            return false;
        }
        //! data type
        if (param.src_type.enumv() != param.filter_type.enumv() ||
            (param.src_type.enumv() != DTypeEnum::Int8 &&
             param.src_type.enumv() != DTypeEnum::QuantizedS8 &&
             param.src_type.enumv() != DTypeEnum::Quantized8Asymm &&
#if !MEGDNN_DISABLE_FLOAT16
             param.src_type.enumv() != DTypeEnum::Float16 &&
#endif
             param.src_type.enumv() != DTypeEnum::Float32)) {
            return false;
        }
        //! x86 disable  Quntized8Asymm
#if MEGDNN_X86
        if (param.src_type.enumv() == DTypeEnum::Quantized8Asymm) {
            return false;
        }
#endif
        //! make sure 8x8x16 and 8x8x32 biasmode is nobias and nonlineMode
        //! is identity otherwise return false mean that 8x8x32 and 8x8x16
        //! not support PostProcess
        if (param.dst_type.enumv() == DTypeEnum::Int16 ||
            param.dst_type.enumv() == DTypeEnum::QuantizedS16 ||
            param.dst_type.enumv() == DTypeEnum::Int32 ||
            param.dst_type.enumv() == DTypeEnum::QuantizedS32) {
            if (param.nonlineMode != megdnn::NonlineMode::IDENTITY) {
                return false;
            }
        }
        MatrixMulImpl::KernSizeParam matmul_param =
                utils::get_matmul_kern_param(param, OH * OW,
                                             get_oc_tile_size_heuristic(param));
        bool matmul_usable = m_matmul_algo->usable(matmul_param);
        auto pack_mode = m_matmul_algo->packmode();
        bool strategy_usable = Conv1x1Factory::can_make_conv1x1_strategy(
                param, pack_mode, param.filter_meta.format);
        return matmul_usable && strategy_usable &&
               (param.filter_meta.dilation[0] ==
                        param.filter_meta.dilation[1] &&
                param.filter_meta.dilation[0] == 1) &&
               param.compute_mode == param::ConvBias::ComputeMode::DEFAULT;
    }
    MIDOUT_END();
    return false;
}

bool ConvBiasImpl::AlgoConv1x1::is_preferred(
        const NCBKernSizeParam& param) const {
    size_t OH = param.osz[0];
    size_t OW = param.osz[1];
    if (OH * OW != 1) {
        return m_matmul_algo->algoset() !=
               MatrixMulImpl::AlgoBase::AlgoSet::ALGO_TYPE_GEMV;
    } else {
#if (MEGDNN_ARMV7 || MEGDNN_AARCH64)
        if (param.src_type.enumv() == DTypeEnum::Int8 &&
            param.filter_type.enumv() == DTypeEnum::Int8 &&
            param.dst_type.enumv() == DTypeEnum::Int16) {
            return true;
        }
#elif MEGDNN_X86
        size_t OC = param.filter_meta.ocpg;
        if (OC > 2 || param.src_type.enumv() == DTypeEnum::Float32)
            return true;
#endif
        return false;
    }
}

// vim: syntax=cpp.doxygen
