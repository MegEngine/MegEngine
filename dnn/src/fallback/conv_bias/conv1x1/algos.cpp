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

#include "src/fallback/conv_bias/conv1x1/algos.h"
#include "src/common/opr_delegate.h"
#include "src/fallback/conv_bias/common.h"
#include "src/fallback/conv_bias/conv1x1/conv1x1_dispatcher.h"
#include "src/fallback/conv_bias/conv1x1/conv1x1_strategy.h"
#include "src/fallback/conv_bias/opr_impl.h"

#include "megdnn/opr_param_defs.h"
#include "src/naive/convolution/helper.h"

#if MEGDNN_X86
#include "src/x86/conv_bias/postprocess_helper.h"
#elif (MEGDNN_ARMV7 || MEGDNN_AARCH64)
#include "src/arm_common/conv_bias/postprocess_helper.h"
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

size_t ConvBiasImpl::AlgoConv1x1::get_workspace(
        ConvBiasImpl*, const NCBKernSizeParam& param) const {
    size_t OH = param.osz[0];
    size_t OW = param.osz[1];
    size_t compt_oc_block_size = get_oc_tile_size_heuristic(param);

    auto matmul_param =
            get_matmul_kern_param(param, OH * OW, compt_oc_block_size);

    auto pack_mode = m_matmul_algo->packmode();
    if (pack_mode == MatrixMulImpl::AlgoBase::PackMode::DEFAULT) {
        MIDOUT_BEGIN(megdnn_fallback_conv1x1, 0, 0, 0) {
            Conv1x1Kerns<MatrixMulImpl::AlgoBase::PackMode::DEFAULT> dispatcher;
            return dispatcher
                    .get_bundle(param, matmul_param, m_matmul_algo,
                                compt_oc_block_size)
                    .total_size_in_bytes();
        }
        MIDOUT_END();
    } else if (pack_mode == MatrixMulImpl::AlgoBase::PackMode::ONLY_PACKA) {
        MIDOUT_BEGIN(megdnn_fallback_conv1x1, 0, 0, 1) {
            Conv1x1Kerns<MatrixMulImpl::AlgoBase::PackMode::ONLY_PACKA>
                    dispatcher;
            return dispatcher
                    .get_bundle(param, matmul_param, m_matmul_algo,
                                compt_oc_block_size)
                    .total_size_in_bytes();
        }
        MIDOUT_END();
    } else {
        MIDOUT_BEGIN(megdnn_fallback_conv1x1, 0, 0, 2) {
            Conv1x1Kerns<MatrixMulImpl::AlgoBase::PackMode::NO_PACK> dispatcher;
            return dispatcher
                    .get_bundle(param, matmul_param, m_matmul_algo,
                                compt_oc_block_size)
                    .total_size_in_bytes();
        }
        MIDOUT_END();
    }
    return 0;
}

SmallVector<ConvBiasImpl::NCBKern> ConvBiasImpl::AlgoConv1x1::dispatch_kerns(
        ConvBiasImpl* opr, const NCBKernSizeParam& param) const {
    SmallVector<ConvBiasImpl::NCBKern> ret_kern;

    size_t OH = param.osz[0];
    size_t OW = param.osz[1];
    size_t OC = param.filter_meta.ocpg;
    size_t compt_oc_block_size = get_oc_tile_size_heuristic(param);
    size_t GROUP = param.filter_meta.group;
    size_t BATCH = param.n;
    size_t oc_blocks_per_group = div_ceil(OC, compt_oc_block_size);

    auto matmul_param =
            get_matmul_kern_param(param, OH * OW, compt_oc_block_size);
    WorkspaceBundle whole_bundle = {nullptr, {}};
    WorkspaceBundle thread_bundle = {nullptr, {}};
    WorkspaceBundle matmul_bundle = {nullptr, {}};

    auto pack_mode = m_matmul_algo->packmode();
    if (pack_mode == MatrixMulImpl::AlgoBase::PackMode::DEFAULT) {
        MIDOUT_BEGIN(megdnn_fallback_conv1x1, 0, 1, 0) {
            Conv1x1Kerns<MatrixMulImpl::AlgoBase::PackMode::DEFAULT> dispatcher;
            whole_bundle = dispatcher.get_bundle(
                    param, matmul_param, m_matmul_algo, compt_oc_block_size);
            matmul_bundle = m_matmul_algo->get_bundle(matmul_param);
        }
        MIDOUT_END();
    } else if (pack_mode == MatrixMulImpl::AlgoBase::PackMode::ONLY_PACKA) {
        MIDOUT_BEGIN(megdnn_fallback_conv1x1, 0, 1, 1) {
            Conv1x1Kerns<MatrixMulImpl::AlgoBase::PackMode::ONLY_PACKA>
                    dispatcher;
            whole_bundle = dispatcher.get_bundle(
                    param, matmul_param, m_matmul_algo, compt_oc_block_size);
            matmul_bundle = m_matmul_algo->get_bundle(matmul_param);
        }
        MIDOUT_END();
    } else {
        MIDOUT_BEGIN(megdnn_fallback_conv1x1, 0, 1, 2) {
            Conv1x1Kerns<MatrixMulImpl::AlgoBase::PackMode::NO_PACK> dispatcher;
            whole_bundle = dispatcher.get_bundle(
                    param, matmul_param, m_matmul_algo, compt_oc_block_size);
            matmul_bundle = {
                    nullptr,
                    {0, 0, m_matmul_algo->get_workspace(matmul_param)}};
        }
        MIDOUT_END();
    }

    //! get thread bundle
    thread_bundle = get_thread_bundle(param, matmul_bundle.get_size(2),
                                      compt_oc_block_size);

    Conv1x1StrategyBase* conv1x1_strategy =
            Conv1x1Factory::make_conv1x1_strategy(param, pack_mode,
                                                  opr->param().format);

    auto kern_packA = [this, whole_bundle, matmul_bundle, param,
                       compt_oc_block_size, conv1x1_strategy](
                              const NCBKernParam& ncb_param,
                              const NCBKernIndex& ncb_index) mutable {
        conv1x1_strategy->packA(whole_bundle, matmul_bundle,
                                compt_oc_block_size, this->m_matmul_algo, param,
                                ncb_param, std::move(ncb_index));
    };
    auto kern_packB = [this, whole_bundle, matmul_bundle, param,
                       conv1x1_strategy](
                              const NCBKernParam& ncb_param,
                              const NCBKernIndex& ncb_index) mutable {
        conv1x1_strategy->packB(whole_bundle, matmul_bundle,
                                this->m_matmul_algo, param, ncb_param,
                                std::move(ncb_index));
    };
    auto kern_compt = [this, whole_bundle, matmul_bundle, thread_bundle, param,
                       compt_oc_block_size, conv1x1_strategy](
                              const NCBKernParam& ncb_param,
                              const NCBKernIndex& ncb_index) mutable {
        conv1x1_strategy->exec(whole_bundle, matmul_bundle, thread_bundle,
                               compt_oc_block_size, this->m_matmul_algo, param,
                               ncb_param, std::move(ncb_index));
    };

    if (pack_mode == MatrixMulImpl::AlgoBase::PackMode::DEFAULT ||
        pack_mode == MatrixMulImpl::AlgoBase::PackMode::ONLY_PACKA) {
        ret_kern.push_back({kern_packA, {GROUP, oc_blocks_per_group}});
        if (pack_mode == MatrixMulImpl::AlgoBase::PackMode::DEFAULT) {
            ret_kern.push_back({kern_packB, {1}});
        }
    }
    ret_kern.push_back({kern_compt, {BATCH, GROUP, oc_blocks_per_group}});

    return ret_kern;
}

bool ConvBiasImpl::AlgoConv1x1::usable(ConvBiasImpl* opr,
                                       const NCBKernSizeParam& param,
                                       AlgoSelectionStrategy) const {
    MIDOUT_BEGIN(megdnn_fallback_conv1x1, 0, 2) {
        if (opr->param().format != param::ConvBias::Format::NCHW &&
            opr->param().format != param::ConvBias::Format::NCHW44 &&
            opr->param().format != param::ConvBias::Format::NCHW44_DOT)
            return false;

        size_t FH = param.filter_meta.spatial[0],
               FW = param.filter_meta.spatial[1];
        size_t PH = param.filter_meta.padding[0],
               PW = param.filter_meta.padding[1];
        size_t SH = param.filter_meta.stride[0],
               SW = param.filter_meta.stride[1];

        if (FH != 1 || FW != 1 || PH || PW || SH != 1 || SW != 1)
            return false;

        if (param.src_type.enumv() != param.filter_type.enumv() &&
            param.src_type.enumv() != DTypeEnum::Int8 &&
            param.src_type.enumv() != DTypeEnum::QuantizedS8 &&
            param.src_type.enumv() != DTypeEnum::Quantized8Asymm &&
#if !MEGDNN_DISABLE_FLOAT16
            param.src_type.enumv() != DTypeEnum::Float16 &&
#endif
            param.src_type.enumv() != DTypeEnum::Float32) {
            return false;
        }
        //! make sure 8x8x16 and 8x8x32 biasmode is nobias and nonlineMode
        //! is identity otherwise return false mean that 8x8x32 and 8x8x16
        //! not support PostProcess
        if (param.dst_type.enumv() == DTypeEnum::Int16 ||
            param.dst_type.enumv() == DTypeEnum::Int32 ||
            param.dst_type.enumv() == DTypeEnum::QuantizedS32) {
            if (param.bias_mode != megdnn::BiasMode::NO_BIAS ||
                param.nonlineMode != megdnn::NonlineMode::IDENTITY) {
                return false;
            }
        }

        if (opr->param().format == param::ConvBias::Format::NCHW44 ||
            opr->param().format == param::ConvBias::Format::NCHW44_DOT) {
            if (param.filter_meta.icpg < 4_z || param.filter_meta.icpg == 1 ||
                param.filter_meta.ocpg == 1) {
                return false;
            }
        }

        size_t OH = param.osz[0];
        size_t OW = param.osz[1];
        MatrixMulImpl::KernSizeParam matmul_param = get_matmul_kern_param(
                param, OH * OW, get_oc_tile_size_heuristic(param));
        bool matmul_usable = m_matmul_algo->usable(matmul_param);

        auto pack_mode = m_matmul_algo->packmode();
        bool strategy_usable = Conv1x1Factory::can_make_conv1x1_strategy(
                param, pack_mode, opr->param().format);

        return matmul_usable && strategy_usable &&
               (param.filter_meta.dilation[0] ==
                        param.filter_meta.dilation[1] &&
                param.filter_meta.dilation[0] == 1) &&
               param.compute_mode == param::ConvBias::ComputeMode::DEFAULT;
    }
    MIDOUT_END();
    return false;
}
