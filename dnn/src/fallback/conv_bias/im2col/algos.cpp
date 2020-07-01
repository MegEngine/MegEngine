/**
 * \file dnn/src/fallback/conv_bias/im2col/algos.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/fallback/conv_bias/im2col/algos.h"
#include "src/fallback/conv_bias/im2col/factory.h"
#include "megdnn/opr_param_defs.h"
#include "src/common/opr_delegate.h"
#include "src/fallback/conv_bias/common.h"
#include "src/fallback/conv_bias/opr_impl.h"
#include "src/naive/convolution/helper.h"

#include "midout.h"

MIDOUT_DECL(megdnn_fallback_im2col)

using namespace megdnn;
using namespace fallback;
using namespace im2col;

/*======================== AlgoIm2col=======================*/
/*!
 *  *\brief The index of all parts workspace in im2col workspace bundel
 *  *Through witch can convenient get the needed ptr
 */
struct Im2colBundelIndex {
    static constexpr size_t BUNDLE_PADDING_INDEX = 0_z;
    static constexpr size_t BUNDLE_PACKA_INDEX = 1_z;
    static constexpr size_t BUNDLE_THREAD_INDEX = 2_z;
};

using Pack_Mode=fallback::MatrixMulImpl::AlgoBase::PackMode;

//! Process one input channel copy padding
static void copy_padding_kern(WorkspaceBundle& bundle,
                              const ConvBiasImpl::NCBKernParam& param,
                              const ConvBiasImpl::NCBKernIndex& ncb_index,
                              StrategyBase* im2colstrategy, size_t pack_oc_size) {
    im2colstrategy->copy_padding_kern(bundle, param, ncb_index, pack_oc_size);
}

//! packA_kern
static void packA_kern(
        WorkspaceBundle& bundle,
        const fallback::ConvBiasImpl::NCBKernParam& param,
        fallback::MatrixMulImpl::KernSizeParam matmulparam,
        fallback::MatrixMulImpl::AlgoBase* matmul_algo,
        const fallback::ConvBiasImpl::NCBKernIndex& ncb_index,
        StrategyBase* im2colstrategy,
        const fallback::MatrixMulImpl::AlgoBase::MatmulDescription& matmul_desc,
        size_t pack_oc_size) {
    im2colstrategy->packA_kern(bundle, param, matmulparam, matmul_algo,
                               ncb_index, matmul_desc, pack_oc_size);
}

/*!
 * *\brief Im2colKerns collects all the im2col kerns in it
 */

template <Pack_Mode packmode>
class Im2colKerns;

template <>
class Im2colKerns<Pack_Mode::DEFAULT> {
public:
    //! conv kernel
    static void kerns(
            const WorkspaceBundle& bundle, WorkspaceBundle bundle_thread,
            const ConvBiasImpl::NCBKernParam& param,
            fallback::MatrixMulImpl::KernSizeParam matmul_kernsize_param,
            const fallback::MatrixMulImpl::AlgoBase* matmul_algo,
            const fallback::MatrixMulImpl::AlgoBase::MatmulDescription&
                    matmul_desc,
            StrategyParam strategyparam,
            fallback::ConvBiasImpl::NCBKernIndex ncb_index,
            size_t ohw_tile_size, StrategyBase* im2colstrategy) {
        size_t OC = param.filter_meta.ocpg;
        size_t output_block_size = std::min(
                ohw_tile_size,
                strategyparam.ohw - ncb_index.ndrange_id[2] * ohw_tile_size);
        size_t output_block_oc_size = std::min(
                strategyparam.oc_tile_size,
                OC - ncb_index.ndrange_id[3] * strategyparam.oc_tile_size);

        strategyparam.batch_id = ncb_index.ndrange_id[0];
        strategyparam.group_id = ncb_index.ndrange_id[1];
        strategyparam.oc_cur_index =
                ncb_index.ndrange_id[3] *
                strategyparam.oc_tile_size;
        strategyparam.oc_end_index = strategyparam.oc_cur_index +
                                     output_block_oc_size;
        strategyparam.ohw_cur_index =
                ncb_index.ndrange_id[2] * ohw_tile_size;
        strategyparam.output_block_oc_size = output_block_oc_size;
        strategyparam.output_block_size = output_block_size;

        bundle_thread.set(
                static_cast<int8_t*>(
                        bundle.get(Im2colBundelIndex::BUNDLE_THREAD_INDEX)) +
                bundle_thread.total_size_in_bytes() * ncb_index.thread_id);
        fallback::MatrixMulImpl::KernParam matmul_param;
        static_cast<fallback::MatrixMulImpl::KernSizeParam&>(matmul_param) =
                matmul_kernsize_param;

        //! 1.Im2col
        im2colstrategy->exec_im2col(bundle, bundle_thread, strategyparam, param,
                                    matmul_param, matmul_algo);

        //! 2.packb and matmul compute
        im2colstrategy->exec_matmul(param, strategyparam, bundle, bundle_thread,
                                    matmul_param, matmul_algo, ncb_index,
                                    matmul_desc);

        //! 3.postprocess and copy dst if need
        im2colstrategy->exec_postprocess(param, strategyparam, bundle_thread);
    }

    WorkspaceBundle get_thread_bundle(
            const fallback::ConvBiasImpl::NCBKernSizeParam& param,
            fallback::MatrixMulImpl::KernSizeParam im2col_kern_param,
            MatrixMulImpl::AlgoBase* matmul_algo, size_t ohw_tile_size,
            size_t oc_tile_size) {
        size_t IC = param.filter_meta.icpg, FH = param.filter_meta.spatial[0],
               FW = param.filter_meta.spatial[1];
        size_t pack_oc_size = pack_size(param.filter_meta.format);
        size_t im2col = 0, packb = 0, bias_temp = 0;
        bool default_pack = matmul_algo->packmode() == Pack_Mode::DEFAULT;
        megdnn_assert(default_pack, "only support default packa");
        size_t im2col_dst_size =
                IC * FH * FW * ohw_tile_size * sizeof(param.src_type);
        size_t matmul_dst_size = pack_oc_size * oc_tile_size * ohw_tile_size *
                                 sizeof(param.bias_type);
        //! matmul_dst and im2col_dst use the same memory
        WorkspaceBundle wb = matmul_algo->get_bundle(im2col_kern_param);
        packb = wb.get_size(1);
        im2col = std::max(im2col_dst_size, matmul_dst_size);
        if (param.bias_mode == megdnn::BiasMode::BIAS) {
            bias_temp = oc_tile_size * ohw_tile_size * sizeof(param.bias_type);
        }
        return {nullptr, {packb, im2col, bias_temp}};
    }
};

template <>
class Im2colKerns<Pack_Mode::ONLY_PACKA> {
public:
    //! conv kernel
    static void kerns(
            const WorkspaceBundle& bundle, WorkspaceBundle bundle_thread,
            const ConvBiasImpl::NCBKernParam& param,
            fallback::MatrixMulImpl::KernSizeParam matmul_kernsize_param,
            const fallback::MatrixMulImpl::AlgoBase* matmul_algo,
            const fallback::MatrixMulImpl::AlgoBase::MatmulDescription&
                    matmul_desc,
            StrategyParam strategyparam,
            fallback::ConvBiasImpl::NCBKernIndex ncb_index,
            size_t ohw_tile_size, StrategyBase* im2colstrategy) {
        size_t OC = param.filter_meta.ocpg;
        size_t output_block_size = std::min(
                ohw_tile_size,
                strategyparam.ohw - ncb_index.ndrange_id[2] * ohw_tile_size);
        size_t output_block_oc_size = std::min(
                strategyparam.oc_tile_size,
                OC - ncb_index.ndrange_id[3] * strategyparam.oc_tile_size);

        bundle_thread.set(
                static_cast<int8_t*>(
                        bundle.get(Im2colBundelIndex::BUNDLE_THREAD_INDEX)) +
                bundle_thread.total_size_in_bytes() * ncb_index.thread_id);

        fallback::MatrixMulImpl::KernParam matmul_param;
        static_cast<fallback::MatrixMulImpl::KernSizeParam&>(matmul_param) =
                matmul_kernsize_param;

        strategyparam.batch_id = ncb_index.ndrange_id[0];
        strategyparam.group_id = ncb_index.ndrange_id[1];
        strategyparam.oc_cur_index =
                ncb_index.ndrange_id[3] *
                strategyparam.oc_tile_size;
        strategyparam.oc_end_index = strategyparam.oc_cur_index +
                                     output_block_oc_size;
        strategyparam.ohw_cur_index =
                ncb_index.ndrange_id[2] * ohw_tile_size;
        strategyparam.output_block_oc_size = output_block_oc_size;
        strategyparam.output_block_size = output_block_size;

        //! 1.Im2col
        im2colstrategy->exec_im2col(bundle, bundle_thread, strategyparam, param,
                                    matmul_param, matmul_algo);

        //! 2.packb and matmul compute
        im2colstrategy->exec_matmul(param, strategyparam, bundle, bundle_thread,
                                    matmul_param, matmul_algo, ncb_index,
                                    matmul_desc);

        //! 3.postprocess and copy dst if need
        im2colstrategy->exec_postprocess(param, strategyparam, bundle_thread);
    }
    WorkspaceBundle get_thread_bundle(
            const fallback::ConvBiasImpl::NCBKernSizeParam& param,
            fallback::MatrixMulImpl::KernSizeParam im2col_kern_param,
            MatrixMulImpl::AlgoBase* matmul_algo, size_t ohw_tile_size,
            size_t oc_tile_size) {
        size_t IC = param.filter_meta.icpg, FH = param.filter_meta.spatial[0],
               FW = param.filter_meta.spatial[1];

        size_t im2col = 0, packb = 0, matmul_dst = 0, bias_temp = 0;
        bool only_packA = matmul_algo->packmode() == Pack_Mode::ONLY_PACKA;
        megdnn_assert(only_packA, "onlysupport onlypackA mode");
        size_t im2col_dst_size =
                IC * FH * FW * ohw_tile_size * sizeof(param.src_type);
        size_t matmul_dst_size =
                oc_tile_size * ohw_tile_size * sizeof(param.bias_type);
        //! matmul_dst and im2col_dst use the same memory
        WorkspaceBundle wb = matmul_algo->get_bundle(im2col_kern_param);
        packb = wb.get_size(1);
        im2col = im2col_dst_size;
        matmul_dst = matmul_dst_size;
        if (param.bias_mode == megdnn::BiasMode::BIAS) {
            bias_temp = oc_tile_size * ohw_tile_size * sizeof(param.bias_type);
        }

        return {nullptr, {packb, im2col, matmul_dst, bias_temp}};
    }
};

template <>
class Im2colKerns<Pack_Mode::NO_PACK> {
public:
    //! conv kernel
    static void kerns(
            const WorkspaceBundle& bundle, WorkspaceBundle bundle_thread,
            const ConvBiasImpl::NCBKernParam& param,
            fallback::MatrixMulImpl::KernSizeParam matmul_kernsize_param,
            const fallback::MatrixMulImpl::AlgoBase* matmul_algo,
            const fallback::MatrixMulImpl::AlgoBase::MatmulDescription&
                    matmul_desc,
            StrategyParam strategyparam,
            fallback::ConvBiasImpl::NCBKernIndex ncb_index,
            size_t ohw_tile_size, StrategyBase* im2colstrategy) {
        size_t OC = param.filter_meta.ocpg;
        size_t output_block_size = std::min(
                ohw_tile_size,
                strategyparam.ohw - ncb_index.ndrange_id[2] * ohw_tile_size);
        size_t output_block_oc_size = std::min(
                strategyparam.oc_tile_size,
                OC - ncb_index.ndrange_id[3] * strategyparam.oc_tile_size);

        strategyparam.batch_id = ncb_index.ndrange_id[0];
        strategyparam.group_id = ncb_index.ndrange_id[1];
        strategyparam.oc_cur_index =
                ncb_index.ndrange_id[3] *
                strategyparam.oc_tile_size;
        strategyparam.oc_end_index = strategyparam.oc_cur_index +
                                     output_block_oc_size;
        strategyparam.ohw_cur_index =
                ncb_index.ndrange_id[2] * ohw_tile_size;
        strategyparam.output_block_oc_size = output_block_oc_size;
        strategyparam.output_block_size = output_block_size;

        bundle_thread.set(
                static_cast<int8_t*>(
                        bundle.get(Im2colBundelIndex::BUNDLE_THREAD_INDEX)) +
                bundle_thread.total_size_in_bytes() * ncb_index.thread_id);

        fallback::MatrixMulImpl::KernParam matmul_param;
        static_cast<fallback::MatrixMulImpl::KernSizeParam&>(matmul_param) =
                matmul_kernsize_param;

        //! 1.Im2col
        im2colstrategy->exec_im2col(bundle, bundle_thread, strategyparam, param,
                                    matmul_param, matmul_algo);

        //! 2.packb and matmul compute
        im2colstrategy->exec_matmul(param, strategyparam, bundle, bundle_thread,
                                    matmul_param, matmul_algo, ncb_index,
                                    matmul_desc);

        //! 3.postprocess and copy dst if need
        im2colstrategy->exec_postprocess(param, strategyparam, bundle_thread);
    }
    WorkspaceBundle get_thread_bundle(
            const fallback::ConvBiasImpl::NCBKernSizeParam& param,
            fallback::MatrixMulImpl::KernSizeParam im2col_kern_param,
            MatrixMulImpl::AlgoBase* matmul_algo, size_t ohw_tile_size,
            size_t oc_tile_size) {
        size_t IC = param.filter_meta.icpg, FH = param.filter_meta.spatial[0],
               FW = param.filter_meta.spatial[1];
        size_t ohw = param.osz[0] * param.osz[1];

        size_t im2col = 0, matmul_dst = 0, bias_temp = 0, matmul_compute = 0;
        bool no_pack = matmul_algo->packmode() == Pack_Mode::NO_PACK;
        megdnn_assert(no_pack, "only support no pack");
        bool is_dst_8bit =
                (param.src_type.enumv() == DTypeEnum::QuantizedS8 &&
                 param.dst_type.enumv() == DTypeEnum::QuantizedS8) ||
                (param.src_type.enumv() == DTypeEnum::Quantized8Asymm &&
                 param.dst_type.enumv() == DTypeEnum::Quantized8Asymm);
        size_t im2col_dst_size =
                IC * FH * FW * ohw_tile_size * sizeof(param.src_type);
        size_t matmul_dst_size =
                oc_tile_size * ohw_tile_size * sizeof(param.bias_type);
        im2col = im2col_dst_size;
        if (is_dst_8bit) {
            matmul_dst = matmul_dst_size;
        } else {
            matmul_dst = ohw_tile_size >= ohw ? 0 : matmul_dst_size;
        }
        matmul_compute = matmul_algo->get_workspace(im2col_kern_param);
        if (param.bias_mode == megdnn::BiasMode::BIAS) {
            bias_temp = oc_tile_size * ohw_tile_size * sizeof(param.bias_type);
        }

        return {nullptr, {im2col, matmul_dst, bias_temp, matmul_compute}};
    }
};

fallback::MatrixMulImpl::KernSizeParam
ConvBiasImpl::AlgoIm2col ::get_matmul_kern_param(const NCBKernSizeParam& param,
                                                 size_t ohw_tile_size,
                                                 size_t oc_tile_size) const {
    auto format = param::MatrixMul::Format::DEFAULT;
    size_t pack_oc_size = pack_size(param.filter_meta.format);
    if (param.filter_meta.format == param::ConvBias::Format::NCHW44) {
        format = param::MatrixMul::Format::MK4;
    } else if(param.filter_meta.format == param::ConvBias::Format::NCHW44_DOT){
        format = param::MatrixMul::Format::MK4_DOT;
    }
    size_t M = oc_tile_size;
    size_t N = ohw_tile_size;
    size_t K = param.filter_meta.icpg * param.filter_meta.spatial[0] *
               param.filter_meta.spatial[1];
    size_t LDA = pack_oc_size * K, LDB = pack_oc_size * N,
           LDC = N * pack_oc_size;
    bool is_dst_8bit = (param.src_type.enumv() == DTypeEnum::QuantizedS8 &&
                        param.dst_type.enumv() == DTypeEnum::QuantizedS8) ||
                       (param.src_type.enumv() == DTypeEnum::Quantized8Asymm &&
                        param.dst_type.enumv() == DTypeEnum::Quantized8Asymm);
    return {param.filter_type,
            param.src_type,
            is_dst_8bit ? param.bias_type : param.dst_type,
            M,
            N,
            K,
            LDA,
            LDB,
            LDC,
            false,
            false,
            param::MatrixMul::ComputeMode::DEFAULT,
            format};
}

void ConvBiasImpl::AlgoIm2col::choice_ohw_oc_block(
        const NCBKernSizeParam& param, size_t& oc_tile_size,
        size_t& ohw_tile_size, size_t block_m, size_t block_n,
        fallback::MatrixMulImpl::AlgoBase::PackMode pack_mode) const {
    size_t nr_threads = param.nr_threads;
    size_t OC = param.filter_meta.ocpg;
    size_t ohw = param.osz[0] * param.osz[1];
    oc_tile_size = DEFAULT_OC_TILE_SIZE;
    ohw_tile_size = m_ohw_tile_size;

    oc_tile_size = std::min(oc_tile_size, OC);
    ohw_tile_size = std::min(ohw_tile_size, ohw);

    if (nr_threads > 1) {
        if (ohw / ohw_tile_size < nr_threads) {
            ohw_tile_size = round_up(div_ceil(ohw, nr_threads), block_n);
            if (ohw_tile_size < DEFAULT_OHW_MIN_TILE_SIZE) {
                ohw_tile_size = ohw;
                oc_tile_size = round_up(div_ceil(OC, nr_threads), block_m);
                if (oc_tile_size > DEFAULT_OC_MAX_TILE_SIZE) {
                    oc_tile_size = DEFAULT_OC_MAX_TILE_SIZE;
                } else if (oc_tile_size < DEFAULT_OC_MIN_TILE_SIZE) {
                    oc_tile_size = DEFAULT_OC_MIN_TILE_SIZE;
                }
            }
        }
    } else {
        //! in no_pack mode don't do block operation when using single thread
        if (pack_mode == fallback::MatrixMulImpl::AlgoBase::PackMode::NO_PACK) {
            ohw_tile_size = ohw;
            oc_tile_size = OC;
        }
    }
}

WorkspaceBundle ConvBiasImpl::AlgoIm2col::get_bundle(
        const NCBKernSizeParam& param) const {
    UNPACK_CONV_F32_NCB_KERN_SIZES(param);
    MEGDNN_MARK_USED_VAR(OC);
    MEGDNN_MARK_USED_VAR(OH);
    MEGDNN_MARK_USED_VAR(OW);
    MEGDNN_MARK_USED_VAR(FH);
    MEGDNN_MARK_USED_VAR(FW);
    MEGDNN_MARK_USED_VAR(SW);
    MEGDNN_MARK_USED_VAR(SH);

    auto IW2 = IH + 2 * PH;
    auto IH2 = IW + 2 * PW;
    bool no_need_pading = (PH == 0 && PW == 0);
    size_t padding = 0, packa_size = 0, packa_group_size = 0;
    size_t nr_threads = param.nr_threads;
    size_t GROUP = param.filter_meta.group;
    fallback::MatrixMulImpl::AlgoBase::MatmulDescription mdesc =
            m_matmul_algo->matmul_description();
    bool need_pack = mdesc.packmode == Pack_Mode::DEFAULT;
    bool only_packA = mdesc.packmode == Pack_Mode::ONLY_PACKA;
    size_t oc_tile_size = 0, ohw_tile_size = 0;
    choice_ohw_oc_block(param, oc_tile_size, ohw_tile_size,
                        mdesc.innerblocksize.m, mdesc.innerblocksize.n,
                        mdesc.packmode);
    if (need_pack || only_packA) {
        auto im2col_kern_param = get_matmul_kern_param(
                param, ohw_tile_size, only_packA ? oc_tile_size : OC);
        size_t oc_parallel_times = div_ceil<size_t>(OC, oc_tile_size);
        WorkspaceBundle wb = m_matmul_algo->get_bundle(im2col_kern_param);
        packa_group_size = only_packA ? oc_parallel_times * wb.get_size(0)
                                      : wb.get_size(0);
    } else {  //! not support pack,not need pack
        packa_group_size = 0;
    }

    if (no_need_pading) {
        padding = 0;  //! not need  padding
    } else {
        padding = (GROUP * N * IC * IH2 * IW2) *
                  sizeof(param.src_type);  //! for padding
    }

    packa_size = GROUP * packa_group_size;  //! for packA  size = GROUP * a_size
    WorkspaceBundle ws = {nullptr, {}};
    auto im2col_kern_param =
            get_matmul_kern_param(param, ohw_tile_size, oc_tile_size);

    if (m_matmul_algo->packmode() == Pack_Mode::DEFAULT) {
        MIDOUT_BEGIN(
                megdnn_fallback_im2col,
                midout_iv("ConvBiasImpl::AlgoIm2col::get_bundle_dft"_hash)) {
            Im2colKerns<Pack_Mode::DEFAULT> defaultkern;
            ws = defaultkern.get_thread_bundle(param, im2col_kern_param,
                                               m_matmul_algo, ohw_tile_size,
                                               oc_tile_size);
        }
        MIDOUT_END();
    } else if (m_matmul_algo->packmode() == Pack_Mode::ONLY_PACKA) {
        MIDOUT_BEGIN(
                megdnn_fallback_im2col,
                midout_iv("ConvBiasImpl::AlgoIm2col::get_bundle_packa"_hash)) {
            Im2colKerns<Pack_Mode::ONLY_PACKA> onlypackakern;
            ws = onlypackakern.get_thread_bundle(param, im2col_kern_param,
                                                 m_matmul_algo, ohw_tile_size,
                                                 oc_tile_size);
        }
        MIDOUT_END();
    } else {
        MIDOUT_BEGIN(
                megdnn_fallback_im2col,
                midout_iv("ConvBiasImpl::AlgoIm2col::get_bundle_other"_hash)) {
            Im2colKerns<Pack_Mode::NO_PACK> nopackkern;
            ws = nopackkern.get_thread_bundle(param, im2col_kern_param,
                                              m_matmul_algo, ohw_tile_size,
                                              oc_tile_size);
        }
        MIDOUT_END();
    }

    return {nullptr,
            {padding, packa_size, ws.total_size_in_bytes() * nr_threads}};
}

size_t ConvBiasImpl::AlgoIm2col::get_workspace(
        ConvBiasImpl*, const NCBKernSizeParam& p) const {
    MIDOUT_BEGIN(megdnn_fallback_im2col, 0, 0) {
        return get_bundle(p).total_size_in_bytes();
    }
    MIDOUT_END();
    return 0;
}

SmallVector<ConvBiasImpl::NCBKern> ConvBiasImpl::AlgoIm2col::dispatch_kerns(
        ConvBiasImpl*, const NCBKernSizeParam& param) const {
    MIDOUT_BEGIN(megdnn_fallback_im2col, 0, 1) {
        UNPACK_CONV_F32_NCB_KERN_SIZES(param);
        MEGDNN_MARK_USED_VAR(SH);
        MEGDNN_MARK_USED_VAR(SW);
        MEGDNN_MARK_USED_VAR(IH);
        MEGDNN_MARK_USED_VAR(IW);
        MEGDNN_MARK_USED_VAR(FH);
        MEGDNN_MARK_USED_VAR(FW);
        size_t oc_tile_size = 0, ohw_tile_size = 0;
        size_t ohw = OH * OW;
        size_t GROUP = param.filter_meta.group;
        WorkspaceBundle bundle = get_bundle(param);
        WorkspaceBundle bundle_thread = {nullptr, {}};
        bool need_padding = (PH != 0 || PW != 0);

        fallback::MatrixMulImpl::AlgoBase::MatmulDescription mdesc =
                m_matmul_algo->matmul_description();

        Pack_Mode packmode = mdesc.packmode;
        bool default_pack = packmode == Pack_Mode::DEFAULT;
        bool no_pack = packmode == Pack_Mode::NO_PACK;
        bool only_packA = packmode == Pack_Mode::ONLY_PACKA;

        choice_ohw_oc_block(param, oc_tile_size, ohw_tile_size,
                            mdesc.innerblocksize.m, mdesc.innerblocksize.n,
                            mdesc.packmode);

        size_t ohw_parallel_times = div_ceil(ohw, ohw_tile_size);
        size_t oc_parallel_times = div_ceil<size_t>(OC, oc_tile_size);
        size_t packa_parallel_times = 0;
        size_t pack_oc_size = pack_size(param.filter_meta.format);

        if (only_packA) {
            packa_parallel_times = div_ceil<size_t>(OC, oc_tile_size);
        } else if (default_pack) {
            packa_parallel_times = div_ceil<size_t>(OC, mdesc.innerblocksize.m);
        }

        auto matmul_param = get_matmul_kern_param(
                param, ohw_tile_size, only_packA ? oc_tile_size : OC);
        if (mdesc.packmode == Pack_Mode::DEFAULT) {
            Im2colKerns<Pack_Mode::DEFAULT> defaultkern;
            bundle_thread = defaultkern.get_thread_bundle(
                    param, matmul_param, m_matmul_algo, ohw_tile_size,
                    oc_tile_size);
        } else if (mdesc.packmode == Pack_Mode::ONLY_PACKA) {
            Im2colKerns<Pack_Mode::ONLY_PACKA> onlypackakern;
            bundle_thread = onlypackakern.get_thread_bundle(
                    param, matmul_param, m_matmul_algo, ohw_tile_size,
                    oc_tile_size);
        } else {
            Im2colKerns<Pack_Mode::NO_PACK> nopackkern;
            bundle_thread = nopackkern.get_thread_bundle(
                    param, matmul_param, m_matmul_algo, ohw_tile_size,
                    oc_tile_size);
        }

        StrategyParam strategyparam;
        strategyparam.ohw = ohw;
        strategyparam.is_dst_8bit =
                (param.src_type.enumv() == DTypeEnum::QuantizedS8 &&
                 param.dst_type.enumv() == DTypeEnum::QuantizedS8) ||
                (param.src_type.enumv() == DTypeEnum::Quantized8Asymm &&
                 param.dst_type.enumv() == DTypeEnum::Quantized8Asymm);
        strategyparam.is_ohw_size_bigger = (ohw_tile_size >= ohw);
        strategyparam.skip_copy_dst =
                strategyparam.is_ohw_size_bigger && !strategyparam.is_dst_8bit;
        strategyparam.oc_tile_size = oc_tile_size;
        strategyparam.pack_oc_size = pack_oc_size;

        SmallVector<ConvBiasImpl::NCBKern> ret_kern;
        MIDOUT_BEGIN(
                megdnn_fallback_im2col,
                midout_iv("ConvBiasImpl::AlgoIm2col::dispatch_kerns"_hash)) {
            StrategyBase* im2colstrategy =
                    Factory::get_im2col_strategy(param, m_matmul_algo);
            auto kern_padding = [bundle, im2colstrategy,
                                 pack_oc_size = pack_oc_size](
                                        const NCBKernParam& param,
                                        const NCBKernIndex& ncb_index) mutable {
                bundle.set(param.workspace_ptr);
                copy_padding_kern(bundle, param, ncb_index, im2colstrategy,
                                  pack_oc_size);
            };

            auto kern_packA = [bundle, matmul_algo = m_matmul_algo,
                               matmul_param, im2colstrategy,
                               pack_oc_size = pack_oc_size, mdesc = mdesc](
                                      const NCBKernParam& param,
                                      const NCBKernIndex& ncb_index) mutable {
                bundle.set(param.workspace_ptr);
                packA_kern(bundle, param, matmul_param, matmul_algo, ncb_index,
                           im2colstrategy, mdesc, pack_oc_size);
            };
            if (default_pack) {
                auto kern_compute_default =
                        [bundle, bundle_thread, matmul_param,
                         matmul_algo = m_matmul_algo,
                         ohw_tile_size = ohw_tile_size,
                         strategyparam = strategyparam, matmul_desc = mdesc,
                         im2colstrategy](
                                const NCBKernParam& param,
                                const NCBKernIndex& ncb_index) mutable {
                            bundle.set(param.workspace_ptr);
                            Im2colKerns<Pack_Mode::DEFAULT>::kerns(
                                    bundle, bundle_thread, param, matmul_param,
                                    matmul_algo, matmul_desc, strategyparam,
                                    ncb_index, ohw_tile_size, im2colstrategy);
                        };
                ret_kern.push_back({kern_packA, {GROUP, packa_parallel_times}});

                if (need_padding) {
                    ret_kern.push_back({kern_padding,
                                        {param.n, GROUP, IC / pack_oc_size}});
                }
                ret_kern.push_back(
                        {kern_compute_default,
                         {N, GROUP, ohw_parallel_times, oc_parallel_times}});
            } else if (only_packA) {
                auto kern_compute_onlypackA =
                        [bundle, bundle_thread, matmul_param,
                         matmul_algo = m_matmul_algo,
                         strategyparam = strategyparam,
                         ohw_tile_size = ohw_tile_size, matmul_desc = mdesc,
                         im2colstrategy](
                                const NCBKernParam& param,
                                const NCBKernIndex& ncb_index) mutable {
                            bundle.set(param.workspace_ptr);
                            Im2colKerns<Pack_Mode::ONLY_PACKA>::kerns(
                                    bundle, bundle_thread, param, matmul_param,
                                    matmul_algo, matmul_desc, strategyparam,
                                    ncb_index, ohw_tile_size, im2colstrategy);
                        };
                ret_kern.push_back({kern_packA, {GROUP, packa_parallel_times}});
                if (need_padding) {
                    ret_kern.push_back({kern_padding, {param.n, GROUP, IC}});
                }
                ret_kern.push_back(
                        {kern_compute_onlypackA,
                         {N, GROUP, ohw_parallel_times, oc_parallel_times}});
            } else if (no_pack) {
                auto kern_compute_nopack =
                        [bundle, bundle_thread, matmul_param,
                         matmul_algo = m_matmul_algo,
                         strategyparam = strategyparam,
                         ohw_tile_size = ohw_tile_size, matmul_desc = mdesc,
                         im2colstrategy](
                                const NCBKernParam& param,
                                const NCBKernIndex& ncb_index) mutable {
                            bundle.set(param.workspace_ptr);
                            Im2colKerns<Pack_Mode::NO_PACK>::kerns(
                                    bundle, bundle_thread, param, matmul_param,
                                    matmul_algo, matmul_desc, strategyparam,
                                    ncb_index, ohw_tile_size, im2colstrategy);
                        };
                if (need_padding) {
                    ret_kern.push_back({kern_padding, {param.n, GROUP, IC}});
                }
                ret_kern.push_back(
                        {kern_compute_nopack,
                         {N, GROUP, ohw_parallel_times, oc_parallel_times}});
            }
            return ret_kern;
        }
        MIDOUT_END();
        return {};
    }
    MIDOUT_END();
    return {};
}

bool ConvBiasImpl::AlgoIm2col::usable(
        ConvBiasImpl* opr, const NCBKernSizeParam& param,
        AlgoSelectionStrategy /*algo_selection_strategy*/) const {
    MIDOUT_BEGIN(megdnn_fallback_im2col, 0, 2) {
        if (opr->param().format != param::ConvBias::Format::NCHW &&
            opr->param().format != param::ConvBias::Format::NCHW44_DOT &&
            opr->param().format != param::ConvBias::Format::NCHW44) {
            return false;
        }

        if(param.src_type.enumv() != param.filter_type.enumv()) {
            return false;
        }

        if (param.src_type.enumv() != DTypeEnum::Int8 &&
            param.src_type.enumv() != DTypeEnum::QuantizedS8 &&
            param.src_type.enumv() != DTypeEnum::Quantized8Asymm &&
#if !MEGDNN_DISABLE_FLOAT16
            param.src_type.enumv() != DTypeEnum::Float16 &&
#endif
            param.src_type.enumv() != DTypeEnum::Float32) {
            return false;
        }
        //! make sure 8x8x16 and 8x8x32 biasmode is  nobias and nonlineMode is
        //! identity otherwise return false mean that 8x8x32 and 8x8x16 not
        //! support PostProcess
        if (param.dst_type.enumv() == DTypeEnum::Int16 ||
            param.dst_type.enumv() == DTypeEnum::Int32 ||
            param.dst_type.enumv() == DTypeEnum::QuantizedS32) {
            if (param.bias_mode != megdnn::BiasMode::NO_BIAS ||
                param.nonlineMode != megdnn::NonlineMode::IDENTITY) {
                return false;
            }
        }
        fallback::MatrixMulImpl::AlgoBase::MatmulDescription mdesc =
                m_matmul_algo->matmul_description();
        if (opr->param().format == param::ConvBias::Format::NCHW44 ||
            opr->param().format == param::ConvBias::Format::NCHW44_DOT) {
            //! current NCHW44 im2col only support DEFAULT mode matmul
            if (mdesc.packmode != Pack_Mode::DEFAULT) {
                return false;
                //! nchw44 hybird mode and channel wise is not support
            } else if (param.filter_meta.icpg < 4_z ||
                       param.filter_meta.icpg == 1 ||
                       param.filter_meta.ocpg == 1) {
                return false;
            }
        }

        size_t oc_tile_size = 0, ohw_tile_size = 0;
        choice_ohw_oc_block(param, oc_tile_size, ohw_tile_size,
                            mdesc.innerblocksize.m, mdesc.innerblocksize.n,
                            m_matmul_algo->packmode());
        fallback::MatrixMulImpl::KernSizeParam matmul_param =
                get_matmul_kern_param(param, ohw_tile_size, oc_tile_size);
        bool matmulusable = m_matmul_algo->usable(matmul_param);
        return matmulusable &&
               (!(param.filter_meta.spatial[0] ==
                          param.filter_meta.spatial[1] &&
                  param.filter_meta.spatial[0] == 1 &&
                  param.filter_meta.stride[0] == param.filter_meta.stride[1] &&
                  param.filter_meta.stride[0] == 1)) &&
               (param.filter_meta.dilation[0] ==
                        param.filter_meta.dilation[1] &&
                param.filter_meta.dilation[0] == 1) &&
               param.compute_mode == param::ConvBias::ComputeMode::DEFAULT;
    }
    MIDOUT_END();
    return false;
}

// vim: syntax=cpp.doxygen
