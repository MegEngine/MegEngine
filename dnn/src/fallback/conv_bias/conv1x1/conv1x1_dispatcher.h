/**
 * \file dnn/src/fallback/conv_bias/conv1x1/conv1x1_dispatcher.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "src/fallback/conv_bias/conv1x1/conv1x1_strategy.h"
#include "src/fallback/conv_bias/conv1x1/conv1x1_utils.h"
#include "src/fallback/conv_bias/opr_impl.h"

namespace megdnn {
namespace fallback {
namespace conv1x1 {

template <MatrixMulImpl::AlgoBase::PackMode pack_mode>
class Conv1x1Kerns;

template <>
class Conv1x1Kerns<MatrixMulImpl::AlgoBase::PackMode::DEFAULT> {
public:
    //! get_bundle
    WorkspaceBundle get_bundle(const ConvBiasImpl::NCBKernSizeParam& param,
                               const MatrixMulImpl::KernSizeParam& matmul_param,
                               const MatrixMulImpl::AlgoBase* matmul_algo,
                               size_t oc_tile_size) {
        size_t GROUP = param.filter_meta.group;
        size_t OC = param.filter_meta.ocpg;
        size_t BATCH = param.n;
        //! bundle per thread
        //! matmul_param records a matmul with M = oc_tile_size, K = IC, N = OH
        //! * OW this does not bother packb bytes
        auto matmul_bundle = matmul_algo->get_bundle(matmul_param);
        auto thread_bundle = utils::get_thread_bundle(
                param, matmul_bundle.get_size(2), oc_tile_size);
        //! size per thread
        size_t all_threads_bytes =
                thread_bundle.total_size_in_bytes() * param.nr_threads;

        //! packa size = GROUP * packa_size_each_group
        size_t packa_bytes_per_oc_tile = matmul_bundle.get_size(0);
        size_t oc_tiles_per_group = div_ceil(OC, oc_tile_size);
        size_t all_packa_bytes =
                is_enable_filter_preprocess(param)
                        ? 0
                        : packa_bytes_per_oc_tile * oc_tiles_per_group * GROUP;
        //! packb size = N * GROUP * packb_size_per_group
        size_t packb_bytes_per_group = matmul_bundle.get_size(1);
        size_t all_packb_bytes = packb_bytes_per_group * GROUP * BATCH;

        return WorkspaceBundle{
                nullptr, {all_packa_bytes, all_packb_bytes, all_threads_bytes}};
    }

    SmallVector<ConvBiasImpl::NCBKern> get_kern(
            const ConvBiasImpl::NCBKernSizeParam& param,
            WorkspaceBundle& whole_bundle, WorkspaceBundle& matmul_bundle,
            WorkspaceBundle& thread_bundle,
            Conv1x1StrategyBase* conv1x1_strategy,
            const MatrixMulImpl::AlgoBase* matmul_algo, size_t oc_block_size) {
        auto kern_packA =
                [whole_bundle, matmul_bundle, param, matmul_algo, oc_block_size,
                 conv1x1_strategy](
                        const ConvBiasImpl::NCBKernParam& ncb_param,
                        const ConvBiasImpl::NCBKernIndex& ncb_index) mutable {
                    conv1x1_strategy->packA(whole_bundle, matmul_bundle,
                                            oc_block_size, matmul_algo, param,
                                            ncb_param, std::move(ncb_index));
                };
        auto kern_packB =
                [whole_bundle, matmul_bundle, param, matmul_algo,
                 conv1x1_strategy](
                        const ConvBiasImpl::NCBKernParam& ncb_param,
                        const ConvBiasImpl::NCBKernIndex& ncb_index) mutable {
                    conv1x1_strategy->packB(whole_bundle, matmul_bundle,
                                            matmul_algo, param, ncb_param,
                                            std::move(ncb_index));
                };
        auto kern_compt =
                [whole_bundle, matmul_bundle, thread_bundle, matmul_algo, param,
                 oc_block_size, conv1x1_strategy](
                        const ConvBiasImpl::NCBKernParam& ncb_param,
                        const ConvBiasImpl::NCBKernIndex& ncb_index) mutable {
                    conv1x1_strategy->exec(whole_bundle, matmul_bundle,
                                           thread_bundle, oc_block_size,
                                           matmul_algo, param, ncb_param,
                                           std::move(ncb_index));
                };
        size_t GROUP = param.filter_meta.group;
        size_t BATCH = param.n;
        size_t OC = param.filter_meta.ocpg;
        size_t oc_blocks_per_group = div_ceil(OC, oc_block_size);
        SmallVector<ConvBiasImpl::NCBKern> ret_kern;
        if (!is_enable_filter_preprocess(param)) {
            ret_kern.push_back({kern_packA, {GROUP, oc_blocks_per_group}});
        }
        ret_kern.push_back({kern_packB, {BATCH}});
        ret_kern.push_back({kern_compt, {BATCH, GROUP, oc_blocks_per_group}});
        return ret_kern;
    }
    SmallVector<ConvBiasImpl::NCBKern> get_kern_preprocess(
            const ConvBiasImpl::NCBKernSizeParam& param,
            WorkspaceBundle& whole_bundle, WorkspaceBundle& matmul_bundle,
            Conv1x1StrategyBase* conv1x1_strategy,
            const MatrixMulImpl::AlgoBase* matmul_algo, size_t oc_block_size) {
        auto kern_packA =
                [whole_bundle, matmul_bundle, param, matmul_algo, oc_block_size,
                 conv1x1_strategy](
                        const ConvBiasImpl::NCBKernParam& ncb_param,
                        const ConvBiasImpl::NCBKernIndex& ncb_index) mutable {
                    conv1x1_strategy->packA(whole_bundle, matmul_bundle,
                                            oc_block_size, matmul_algo, param,
                                            ncb_param, std::move(ncb_index));
                };
        size_t GROUP = param.filter_meta.group;
        size_t OC = param.filter_meta.ocpg;
        size_t oc_blocks_per_group = div_ceil(OC, oc_block_size);
        SmallVector<ConvBiasImpl::NCBKern> ret_kern;
        ret_kern.push_back({kern_packA, {GROUP, oc_blocks_per_group}});
        return ret_kern;
    }

};

template<>
class Conv1x1Kerns<MatrixMulImpl::AlgoBase::PackMode::ONLY_PACKA> {
public:
    //! get_bundle
    WorkspaceBundle get_bundle(const ConvBiasImpl::NCBKernSizeParam& param,
                               const MatrixMulImpl::KernSizeParam& matmul_param,
                               const MatrixMulImpl::AlgoBase* matmul_algo,
                               size_t oc_tile_size) {
        size_t GROUP = param.filter_meta.group;
        size_t OC = param.filter_meta.ocpg;
        //! bundle per thread
        //! matmul_param records a matmul with M = oc_tile_size, K = IC, N = OH
        //! * OW this does not bother packb bytes
        auto matmul_bundle = matmul_algo->get_bundle(matmul_param);
        auto thread_bundle = utils::get_thread_bundle(
                param, matmul_bundle.get_size(2), oc_tile_size);
        //! size per thread
        size_t all_threads_bytes =
                thread_bundle.total_size_in_bytes() * param.nr_threads;

        //! packa size = GROUP * packa_size_each_group
        size_t packa_bytes_per_oc_tile = matmul_bundle.get_size(0);
        size_t oc_tiles_per_group = div_ceil(OC, oc_tile_size);
        size_t all_packa_bytes =
                is_enable_filter_preprocess(param)
                        ? 0
                        : packa_bytes_per_oc_tile * oc_tiles_per_group * GROUP;

        return WorkspaceBundle{nullptr,
                               {all_packa_bytes, 0, all_threads_bytes}};
    }
    SmallVector<ConvBiasImpl::NCBKern> get_kern(
            const ConvBiasImpl::NCBKernSizeParam& param,
            WorkspaceBundle& whole_bundle, WorkspaceBundle& matmul_bundle,
            WorkspaceBundle& thread_bundle,
            Conv1x1StrategyBase* conv1x1_strategy,
            const MatrixMulImpl::AlgoBase* matmul_algo, size_t oc_block_size) {
        auto kern_packA =
                [whole_bundle, matmul_bundle, param, matmul_algo, oc_block_size,
                 conv1x1_strategy](
                        const ConvBiasImpl::NCBKernParam& ncb_param,
                        const ConvBiasImpl::NCBKernIndex& ncb_index) mutable {
                    conv1x1_strategy->packA(whole_bundle, matmul_bundle,
                                            oc_block_size, matmul_algo, param,
                                            ncb_param, std::move(ncb_index));
                };
        auto kern_compt =
                [whole_bundle, matmul_bundle, thread_bundle, matmul_algo, param,
                 oc_block_size, conv1x1_strategy](
                        const ConvBiasImpl::NCBKernParam& ncb_param,
                        const ConvBiasImpl::NCBKernIndex& ncb_index) mutable {
                    conv1x1_strategy->exec(whole_bundle, matmul_bundle,
                                           thread_bundle, oc_block_size,
                                           matmul_algo, param, ncb_param,
                                           std::move(ncb_index));
                };
        size_t GROUP = param.filter_meta.group;
        size_t BATCH = param.n;
        size_t OC = param.filter_meta.ocpg;
        size_t oc_blocks_per_group = div_ceil(OC, oc_block_size);
        SmallVector<ConvBiasImpl::NCBKern> ret_kern;
        if (!is_enable_filter_preprocess(param)) {
            ret_kern.push_back({kern_packA, {GROUP, oc_blocks_per_group}});
        }
        ret_kern.push_back({kern_compt, {BATCH, GROUP, oc_blocks_per_group}});
        return ret_kern;
    }
    SmallVector<ConvBiasImpl::NCBKern> get_kern_preprocess(
            const ConvBiasImpl::NCBKernSizeParam& param,
            WorkspaceBundle& whole_bundle, WorkspaceBundle& matmul_bundle,
            Conv1x1StrategyBase* conv1x1_strategy,
            const MatrixMulImpl::AlgoBase* matmul_algo, size_t oc_block_size) {
        auto kern_packA =
                [whole_bundle, matmul_bundle, param, matmul_algo, oc_block_size,
                 conv1x1_strategy](
                        const ConvBiasImpl::NCBKernParam& ncb_param,
                        const ConvBiasImpl::NCBKernIndex& ncb_index) mutable {
                    conv1x1_strategy->packA(whole_bundle, matmul_bundle,
                                            oc_block_size, matmul_algo, param,
                                            ncb_param, std::move(ncb_index));
                };
        size_t GROUP = param.filter_meta.group;
        size_t OC = param.filter_meta.ocpg;
        size_t oc_blocks_per_group = div_ceil(OC, oc_block_size);
        SmallVector<ConvBiasImpl::NCBKern> ret_kern;
        ret_kern.push_back({kern_packA, {GROUP, oc_blocks_per_group}});
        return ret_kern;
    }
};

template<>
class Conv1x1Kerns<MatrixMulImpl::AlgoBase::PackMode::NO_PACK> {
public:
    //! get_bundle
    WorkspaceBundle get_bundle(const ConvBiasImpl::NCBKernSizeParam& param,
                               const MatrixMulImpl::KernSizeParam& matmul_param,
                               const MatrixMulImpl::AlgoBase* matmul_algo,
                               size_t oc_tile_size) {
        size_t matmul_size = matmul_algo->get_workspace(matmul_param);
        auto thread_bundle =
                utils::get_thread_bundle(param, matmul_size, oc_tile_size);
        //! size per thread
        size_t all_threads_bytes =
                thread_bundle.total_size_in_bytes() * param.nr_threads;
        return WorkspaceBundle{nullptr, {0, 0, all_threads_bytes}};
    }
    SmallVector<ConvBiasImpl::NCBKern> get_kern(
            const ConvBiasImpl::NCBKernSizeParam& param,
            WorkspaceBundle& whole_bundle, WorkspaceBundle& matmul_bundle,
            WorkspaceBundle& thread_bundle,
            Conv1x1StrategyBase* conv1x1_strategy,
            const MatrixMulImpl::AlgoBase* matmul_algo, size_t oc_block_size) {
        auto kern_compt =
                [whole_bundle, matmul_bundle, thread_bundle, matmul_algo, param,
                 oc_block_size, conv1x1_strategy](
                        const ConvBiasImpl::NCBKernParam& ncb_param,
                        const ConvBiasImpl::NCBKernIndex& ncb_index) mutable {
                    conv1x1_strategy->exec(whole_bundle, matmul_bundle,
                                           thread_bundle, oc_block_size,
                                           matmul_algo, param, ncb_param,
                                           std::move(ncb_index));
                };
        size_t GROUP = param.filter_meta.group;
        size_t BATCH = param.n;
        size_t OC = param.filter_meta.ocpg;
        size_t oc_blocks_per_group = div_ceil(OC, oc_block_size);
        SmallVector<ConvBiasImpl::NCBKern> ret_kern;
        ret_kern.push_back({kern_compt, {BATCH, GROUP, oc_blocks_per_group}});
        return ret_kern;
    }
    SmallVector<ConvBiasImpl::NCBKern> get_kern_preprocess(
            const ConvBiasImpl::NCBKernSizeParam&, WorkspaceBundle&,
            WorkspaceBundle&, Conv1x1StrategyBase*,
            const MatrixMulImpl::AlgoBase*, size_t) {
        return {};
    }
};

}  // namespace conv1x1
}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
