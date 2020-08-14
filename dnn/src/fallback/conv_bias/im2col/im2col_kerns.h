/**
 * \file dnn/src/fallback/conv_bias/im2col/im2col_kerns.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/fallback/conv_bias/opr_impl.h"
#include "src/naive/convolution/helper.h"
#include "src/fallback/conv_bias/im2col/factory.h"

#include "midout.h"

MIDOUT_DECL(megdnn_fallback_im2col)

namespace megdnn {
namespace fallback {
namespace im2col {

/*!
 *  *\brief The index of all parts workspace in im2col workspace bundel
 *  *Through witch can convenient get the needed ptr
 */
struct Im2colBundelIndex {
    static constexpr size_t BUNDLE_THREAD_INDEX = 2_z;
};

using Pack_Mode=fallback::MatrixMulImpl::AlgoBase::PackMode;
/*!
 * *\brief Im2colKerns collects all the im2col kerns in it
 */
namespace{
//! conv kernel
static void kerns(
        const WorkspaceBundle& bundle, WorkspaceBundle bundle_thread,
        const ConvBiasImpl::NCBKernParam& param,
        fallback::MatrixMulImpl::KernSizeParam matmul_kernsize_param,
        const fallback::MatrixMulImpl::AlgoBase* matmul_algo,
        const fallback::MatrixMulImpl::AlgoBase::MatmulDescription& matmul_desc,
        StrategyParam strategyparam,
        fallback::ConvBiasImpl::NCBKernIndex ncb_index, size_t ohw_tile_size,
        StrategyBase* im2colstrategy) {
    size_t OC = param.filter_meta.ocpg;
    size_t output_block_size = std::min(
            ohw_tile_size,
            strategyparam.ohw - ncb_index.ndrange_id[2] * ohw_tile_size);
    size_t output_block_oc_size =
            std::min(strategyparam.oc_tile_size,
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
            ncb_index.ndrange_id[3] * strategyparam.oc_tile_size;
    strategyparam.oc_end_index =
            strategyparam.oc_cur_index + output_block_oc_size;
    strategyparam.ohw_cur_index = ncb_index.ndrange_id[2] * ohw_tile_size;
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
}  // namespace

template <Pack_Mode packmode>
class Im2colKerns;

template <>
class Im2colKerns<Pack_Mode::DEFAULT> {
public:
    SmallVector<ConvBiasImpl::NCBKern> get_kerns(
            const ConvBiasImpl::NCBKernSizeParam& param,
            WorkspaceBundle& bundle, WorkspaceBundle& bundle_thread,
            const StrategyParam& strategyparam,
            fallback::MatrixMulImpl::KernSizeParam& matmul_param,
            StrategyBase* im2colstrategy, MatrixMulImpl::AlgoBase* matmul_algo,
            size_t ohw_tile_size, size_t oc_tile_size, size_t pack_oc_size) {
        auto matmul_desc = matmul_algo->matmul_description();
        auto kern_padding =
                [bundle, im2colstrategy, pack_oc_size = pack_oc_size](
                        const ConvBiasImpl::NCBKernParam& param,
                        const ConvBiasImpl::NCBKernIndex& ncb_index) mutable {
                    bundle.set(param.workspace_ptr);
                    im2colstrategy->copy_padding_kern(bundle, param, ncb_index,
                                                      pack_oc_size);
                };

        auto kern_packA =
                [bundle, matmul_algo, matmul_param, im2colstrategy,
                 strategyparam = strategyparam, matmul_desc = matmul_desc](
                        const ConvBiasImpl::NCBKernParam& param,
                        const ConvBiasImpl::NCBKernIndex& ncb_index) mutable {
                    bundle.set(param.workspace_ptr);
                    im2colstrategy->packA_kern(bundle, param, matmul_param,
                                               matmul_algo, ncb_index,
                                               matmul_desc, strategyparam);
                };
        auto kern_compute_default =
                [bundle, bundle_thread, matmul_param, matmul_algo,
                 ohw_tile_size, strategyparam, matmul_desc = matmul_desc,
                 im2colstrategy](
                        const ConvBiasImpl::NCBKernParam& param,
                        const ConvBiasImpl::NCBKernIndex& ncb_index) mutable {
                    bundle.set(param.workspace_ptr);
                    kerns(bundle, bundle_thread, param, matmul_param,
                          matmul_algo, matmul_desc, strategyparam, ncb_index,
                          ohw_tile_size, im2colstrategy);
                };
        size_t OH = param.osz[0];
        size_t OW = param.osz[1];
        size_t BATCH = param.n;
        size_t OC = param.filter_meta.ocpg;
        size_t IC = param.filter_meta.icpg;
        size_t PH = param.filter_meta.padding[0];
        size_t PW = param.filter_meta.padding[1];
        size_t GROUP = param.filter_meta.group;
        size_t packa_parallel_times =
                div_ceil<size_t>(OC, matmul_desc.innerblocksize.m);
        size_t ohw_parallel_times = div_ceil(OH * OW, ohw_tile_size);
        size_t oc_parallel_times = div_ceil<size_t>(OC, oc_tile_size);
        SmallVector<ConvBiasImpl::NCBKern> ret_kern;
        if (!is_enable_filter_preprocess(param)) {
            ret_kern.push_back({kern_packA, {GROUP, packa_parallel_times}});
        }
        if (PH != 0 || PW != 0) {
            ret_kern.push_back(
                    {kern_padding, {BATCH, GROUP, IC / pack_oc_size}});
        }
        ret_kern.push_back(
                {kern_compute_default,
                 {BATCH, GROUP, ohw_parallel_times, oc_parallel_times}});
        return ret_kern;
    }

    WorkspaceBundle get_thread_bundle(
            const fallback::ConvBiasImpl::NCBKernSizeParam& param,
            const fallback::MatrixMulImpl::KernSizeParam& im2col_kern_param,
            const MatrixMulImpl::AlgoBase* matmul_algo, size_t ohw_tile_size,
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
    SmallVector<ConvBiasImpl::NCBKern> get_kerns(
            const ConvBiasImpl::NCBKernSizeParam& param,
            WorkspaceBundle& bundle, WorkspaceBundle& bundle_thread,
            const StrategyParam& strategyparam,
            fallback::MatrixMulImpl::KernSizeParam& matmul_param,
            StrategyBase* im2colstrategy, MatrixMulImpl::AlgoBase* matmul_algo,
            size_t ohw_tile_size, size_t oc_tile_size, size_t pack_oc_size) {
        auto matmul_desc = matmul_algo->matmul_description();
        auto kern_padding =
                [bundle, im2colstrategy, pack_oc_size = pack_oc_size](
                        const ConvBiasImpl::NCBKernParam& param,
                        const ConvBiasImpl::NCBKernIndex& ncb_index) mutable {
                    bundle.set(param.workspace_ptr);
                    im2colstrategy->copy_padding_kern(bundle, param, ncb_index,
                                                      pack_oc_size);
                };

        auto kern_packA =
                [bundle, matmul_algo, matmul_param, im2colstrategy,
                 strategyparam = strategyparam, matmul_desc = matmul_desc](
                        const ConvBiasImpl::NCBKernParam& param,
                        const ConvBiasImpl::NCBKernIndex& ncb_index) mutable {
                    bundle.set(param.workspace_ptr);
                    im2colstrategy->packA_kern(bundle, param, matmul_param,
                                               matmul_algo, ncb_index,
                                               matmul_desc, strategyparam);
                };
        auto kern_compute_onlypackA =
                [bundle, bundle_thread, matmul_param, matmul_algo,
                 strategyparam, ohw_tile_size, matmul_desc, im2colstrategy](
                        const ConvBiasImpl::NCBKernParam& param,
                        const ConvBiasImpl::NCBKernIndex& ncb_index) mutable {
                    bundle.set(param.workspace_ptr);
                    kerns(bundle, bundle_thread, param, matmul_param,
                          matmul_algo, matmul_desc, strategyparam, ncb_index,
                          ohw_tile_size, im2colstrategy);
                };
        size_t OH = param.osz[0];
        size_t OW = param.osz[1];
        size_t BATCH = param.n;
        size_t OC = param.filter_meta.ocpg;
        size_t IC = param.filter_meta.icpg;
        size_t PH = param.filter_meta.padding[0];
        size_t PW = param.filter_meta.padding[1];
        size_t GROUP = param.filter_meta.group;
        size_t ohw_parallel_times = div_ceil(OH * OW, ohw_tile_size);
        size_t oc_parallel_times = div_ceil<size_t>(OC, oc_tile_size);
        SmallVector<ConvBiasImpl::NCBKern> ret_kern;
        if (!is_enable_filter_preprocess(param)) {
            ret_kern.push_back({kern_packA, {GROUP, oc_parallel_times}});
        }
        if (PH != 0 || PW != 0) {
            ret_kern.push_back(
                    {kern_padding, {BATCH, GROUP, IC / pack_oc_size}});
        }
        ret_kern.push_back(
                {kern_compute_onlypackA,
                 {BATCH, GROUP, ohw_parallel_times, oc_parallel_times}});
        return ret_kern;
    }
    WorkspaceBundle get_thread_bundle(
            const fallback::ConvBiasImpl::NCBKernSizeParam& param,
            const fallback::MatrixMulImpl::KernSizeParam& im2col_kern_param,
            const MatrixMulImpl::AlgoBase* matmul_algo, size_t ohw_tile_size,
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
    SmallVector<ConvBiasImpl::NCBKern> get_kerns(
            const ConvBiasImpl::NCBKernSizeParam& param,
            WorkspaceBundle& bundle, WorkspaceBundle& bundle_thread,
            const StrategyParam& strategyparam,
            fallback::MatrixMulImpl::KernSizeParam& matmul_param,
            StrategyBase* im2colstrategy, MatrixMulImpl::AlgoBase* matmul_algo,
            size_t ohw_tile_size, size_t oc_tile_size, size_t pack_oc_size) {
        auto matmul_desc = matmul_algo->matmul_description();
        auto kern_padding =
                [bundle, im2colstrategy, pack_oc_size = pack_oc_size](
                        const ConvBiasImpl::NCBKernParam& param,
                        const ConvBiasImpl::NCBKernIndex& ncb_index) mutable {
                    bundle.set(param.workspace_ptr);
                    im2colstrategy->copy_padding_kern(bundle, param, ncb_index,
                                                      pack_oc_size);
                };
        auto kern_compute_nopack =
                [bundle, bundle_thread, matmul_param, matmul_algo,
                 strategyparam, ohw_tile_size, matmul_desc, im2colstrategy](
                        const ConvBiasImpl::NCBKernParam& param,
                        const ConvBiasImpl::NCBKernIndex& ncb_index) mutable {
                    bundle.set(param.workspace_ptr);
                    kerns(bundle, bundle_thread, param, matmul_param,
                          matmul_algo, matmul_desc, strategyparam, ncb_index,
                          ohw_tile_size, im2colstrategy);
                };
        size_t OH = param.osz[0];
        size_t OW = param.osz[1];
        size_t BATCH = param.n;
        size_t OC = param.filter_meta.ocpg;
        size_t IC = param.filter_meta.icpg;
        size_t PH = param.filter_meta.padding[0];
        size_t PW = param.filter_meta.padding[1];
        size_t GROUP = param.filter_meta.group;
        size_t ohw_parallel_times = div_ceil(OH * OW, ohw_tile_size);
        size_t oc_parallel_times = div_ceil<size_t>(OC, oc_tile_size);
        SmallVector<ConvBiasImpl::NCBKern> ret_kern;
        if (PH != 0 || PW != 0) {
            ret_kern.push_back(
                    {kern_padding, {BATCH, GROUP, IC / pack_oc_size}});
        }
        ret_kern.push_back(
                {kern_compute_nopack,
                 {BATCH, GROUP, ohw_parallel_times, oc_parallel_times}});
        return ret_kern;
    }
    WorkspaceBundle get_thread_bundle(
            const fallback::ConvBiasImpl::NCBKernSizeParam& param,
            const fallback::MatrixMulImpl::KernSizeParam& im2col_kern_param,
            const MatrixMulImpl::AlgoBase* matmul_algo, size_t ohw_tile_size,
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

}  // namespace im2col
}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
