/**
 * \file dnn/src/fallback/conv_bias/conv1x1/algos.h
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

class ConvBiasImpl::AlgoConv1x1 final : public AlgoBase {
    WorkspaceBundle get_bundle_according_packmode(
            const NCBKernSizeParam& param) const;
    SmallVector<NCBKern> get_kerns_according_packmode(
            const NCBKernSizeParam& param, bool weight_preprocess) const;

public:
    AlgoConv1x1(MatrixMulImpl::AlgoBase* matmul_algo, size_t oc_block_size)
            : m_matmul_algo(matmul_algo), m_oc_block_size(oc_block_size) {}

    bool is_reproducible() const override { return true; }

    const char* name() const override {
        if (m_name.empty()) {
            m_name = ssprintf("CONV1x1:%s:%zu", m_matmul_algo->name(),
                              m_oc_block_size);
        }
        return m_name.c_str();
    }

    bool usable(const NCBKernSizeParam& param,
                AlgoSelectionStrategy algo_selection_strategy) const override;
    size_t get_workspace(const NCBKernSizeParam& param) const override;
    SmallVector<NCBKern> dispatch_kerns(
            const NCBKernSizeParam& param) const override;

    bool is_preferred(const NCBKernSizeParam&) const override;

    SmallVector<TensorLayout> deduce_preprocessed_filter_layout(
            const NCBKernSizeParam& param) const override;
    size_t get_preprocess_workspace(
            const NCBKernSizeParam& /*param*/) const override {
        return 0;
    }
    SmallVector<NCBKern> dispatch_preprocess_kerns(
            const NCBKernSizeParam& param) const override;

    ConvAlgoTypePack get_algo_type() const override{
        return {m_matmul_algo->matmul_description().algo_type.data_type,
                AlgoCategory::IM2COL};
    }
    MEGDNN_DECL_ALGO_TYPE(FB_WINOGRAD_8X8_QS8)
    std::string param() const override {
        std::string ret;
        serialize_write_pod(m_matmul_algo, ret);
        serialize_write_pod(m_oc_block_size, ret);
        return ret;
    }

protected:
    size_t get_oc_tile_size_heuristic(const NCBKernSizeParam& param) const;

private:
    MatrixMulImpl::AlgoBase* m_matmul_algo;
    mutable std::string m_name;
    const size_t m_oc_block_size = 0;
};

}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
