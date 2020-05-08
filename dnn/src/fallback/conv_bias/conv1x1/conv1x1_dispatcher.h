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

namespace megdnn {
namespace fallback {
namespace conv1x1 {

namespace {
//! get_thread_bundle
WorkspaceBundle get_thread_bundle(const ConvBiasImpl::NCBKernSizeParam& param,
                                  size_t matmul_c_size, size_t oc_tile_size) {
    //! for some cases, matmul result need temp space to store
    size_t OH = param.osz[0];
    size_t OW = param.osz[1];
    bool is_dst_8bit = (param.src_type.enumv() == DTypeEnum::QuantizedS8 &&
                        param.dst_type.enumv() == DTypeEnum::QuantizedS8) ||
                       (param.src_type.enumv() == DTypeEnum::Quantized8Asymm &&
                        param.dst_type.enumv() == DTypeEnum::Quantized8Asymm);
    size_t matmul_dst_bytes_per_thread =
            is_dst_8bit ? oc_tile_size * OH * OW * sizeof(param.bias_type) : 0;
    return WorkspaceBundle{nullptr,
                           {matmul_c_size, matmul_dst_bytes_per_thread}};
}
} // anonymous namespace

template <MatrixMulImpl::AlgoBase::PackMode pack_mode>
class Conv1x1Kerns {
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
        auto thread_bundle = get_thread_bundle(param, matmul_bundle.get_size(2),
                                               oc_tile_size);

        //! size per thread
        size_t all_threads_bytes =
                thread_bundle.total_size_in_bytes() * param.nr_threads;

        //! packa size = GROUP * packa_size_each_group
        size_t packa_bytes_per_oc_tile = matmul_bundle.get_size(0);
        size_t oc_tiles_per_group = div_ceil(OC, oc_tile_size);
        size_t all_packa_bytes =
                packa_bytes_per_oc_tile * oc_tiles_per_group * GROUP;

        if (pack_mode == MatrixMulImpl::AlgoBase::PackMode::ONLY_PACKA)
            return WorkspaceBundle{nullptr,
                                   {all_packa_bytes, 0, all_threads_bytes}};

        //! packb size = N * GROUP * packb_size_per_group
        size_t packb_bytes_per_group = matmul_bundle.get_size(1);
        size_t all_packb_bytes = packb_bytes_per_group * GROUP * BATCH;

        return WorkspaceBundle{
                nullptr, {all_packa_bytes, all_packb_bytes, all_threads_bytes}};
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
        auto thread_bundle = get_thread_bundle(param, matmul_size, oc_tile_size);
        //! size per thread
        size_t all_threads_bytes =
                thread_bundle.total_size_in_bytes() * param.nr_threads;
        return WorkspaceBundle{nullptr, {0, 0, all_threads_bytes}};
    }
};

}  // namespace conv1x1
}  // namespace fallback
}  // namespace megdnn
