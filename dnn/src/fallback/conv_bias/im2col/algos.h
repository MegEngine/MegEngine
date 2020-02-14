/**
 * \file dnn/src/fallback/conv_bias/im2col/algos.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megdnn/thin/small_vector.h"
#include "src/common/utils.h"
#include "src/fallback/conv_bias/opr_impl.h"
#include "src/fallback/matrix_mul/opr_impl.h"

namespace megdnn {
namespace fallback {

class ConvBiasImpl::AlgoIm2col final : public AlgoBase {
    //! calculate m_oc_tile_size in choice_ohw_oc_block() fucntion,
    //! when m_oc_tile_size < this value m_oc_tile_size = ohw
    static constexpr size_t DEFAULT_OHW_MIN_TILE_SIZE = 32;
    //! when nr_threads > 1 and round(ohw,nr_threads)>nr_threads,
    //! m_oc_tile_size = DEFAULT_OC_TILE_SIZE
    static constexpr size_t DEFAULT_OC_TILE_SIZE = 512;
    //! when m_oc_tile_size > this value m_oc_tile_size =
    //! DEFAULT_OC_MAX_TILE_SIZE
    static constexpr size_t DEFAULT_OC_MAX_TILE_SIZE = 1024;
    //! when m_oc_tile_size < this value m_oc_tile_size =
    //! DEFAULT_OC_MIN_TILE_SIZE the purpose is aligning the calculation
    static constexpr size_t DEFAULT_OC_MIN_TILE_SIZE = 128;
    fallback::MatrixMulImpl::KernSizeParam get_matmul_kern_param(
            const NCBKernSizeParam& param, size_t ohw_tile_size,
            size_t oc_tile_size) const;
    WorkspaceBundle get_bundle(const NCBKernSizeParam& param) const;
    WorkspaceBundle get_thread_bundle(const NCBKernSizeParam& param) const;
    void choice_ohw_oc_block(const NCBKernSizeParam& param, size_t block_m,
                             size_t block_n, bool pack_default) const;

public:
    AlgoIm2col(MatrixMulImpl::AlgoBase* matmul_algo, size_t ohw_tile_size)
            : m_matmul_algo(matmul_algo),
              m_ohw_tile_origin(ohw_tile_size),
              m_ohw_tile_size(ohw_tile_size) {}

    bool is_reproducible() const override { return true; }
    const char* name() const override {
        if (m_name.empty()) {
            m_name = ssprintf("IM2COLMATMUL:%s:%zu", m_matmul_algo->name(),
                              m_ohw_tile_origin);
        }
        return m_name.c_str();
    }
    bool usable(ConvBiasImpl* opr, const NCBKernSizeParam& param,
                AlgoSelectionStrategy algo_selection_strategy) const override;
    size_t get_workspace(ConvBiasImpl*,
                         const NCBKernSizeParam& param) const override;
    SmallVector<NCBKern> dispatch_kerns(
            ConvBiasImpl* opr, const NCBKernSizeParam& param) const override;
    bool is_preferred(fallback::ConvBiasImpl* opr,
                      const NCBKernSizeParam& param) const override {
        if (param.src_type.category() == DTypeCategory::QUANTIZED) {
            return opr->is_matmul_quantized_prefer(param);
        }
        auto&& fm = param.filter_meta;
        auto OC = fm.ocpg, IC = fm.icpg;
        return (fm.spatial[0] == fm.spatial[1] && fm.spatial[0] == 1) ||
               OC >= 32 || IC >= 32;
    }

private:
    MatrixMulImpl::AlgoBase* m_matmul_algo;
    mutable std::string m_name;
    const size_t m_ohw_tile_origin;
    mutable size_t m_ohw_tile_size;
    mutable size_t m_oc_tile_size = DEFAULT_OC_TILE_SIZE;
};

}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
