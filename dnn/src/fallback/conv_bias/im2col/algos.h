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
#include "src/common/opr_delegate.h"


namespace megdnn {
namespace fallback {

class ConvBiasImpl::AlgoIm2col final : public AlgoBase {
public:
    AlgoIm2col(MatrixMulImpl::AlgoBase* matmul_algo, size_t ohw_tile_size)
            : m_matmul_algo(matmul_algo),
              m_ohw_tile_size(ohw_tile_size) {}

    bool is_reproducible() const override { return true; }
    const char* name() const override {
        if (m_name.empty()) {
            m_name = ssprintf("IM2COLMATMUL:%s:%zu", m_matmul_algo->name(),
                              m_ohw_tile_size);
        }
        return m_name.c_str();
    }
    bool usable(const NCBKernSizeParam& param,
                AlgoSelectionStrategy algo_selection_strategy) const override;
    size_t get_workspace(const NCBKernSizeParam& param) const override;
    SmallVector<NCBKern> dispatch_kerns(const NCBKernSizeParam& param) const override;
    SmallVector<TensorLayout> deduce_preprocessed_filter_layout(
            const NCBKernSizeParam& param) const override;
    size_t get_preprocess_workspace(
            const NCBKernSizeParam& /*param*/) const override {
        return 0;
    }
    SmallVector<NCBKern> dispatch_preprocess_kerns(
            const NCBKernSizeParam& param) const override;
    bool is_preferred(const NCBKernSizeParam& param) const override {
        size_t OH = param.osz[0];
        size_t OW = param.osz[1];
        //! gemm and oh * ow > 1 is prefer
        //! gemv and oh * ow == 1 is prefer
        if ((m_matmul_algo->algoset() !=
                     MatrixMulImpl::AlgoBase::AlgoSet::ALGO_TYPE_GEMV &&
             OH * OW > 1) ||
            (m_matmul_algo->algoset() ==
                     MatrixMulImpl::AlgoBase::AlgoSet::ALGO_TYPE_GEMV &&
             OH * OW == 1)) {
            return true;
        } else {
            return false;
        }
    }

    ConvAlgoTypePack get_algo_type() const override {
        return {m_matmul_algo->matmul_description().algo_type.data_type,
                AlgoCategory::IM2COL};
    }
    MEGDNN_DECL_ALGO_TYPE(FB_IM2COL)

    std::string param() const override {
        std::string ret;
        serialize_write_pod(m_matmul_algo, ret);
        serialize_write_pod(m_ohw_tile_size, ret);
        return ret;
    }

private:
    MatrixMulImpl::AlgoBase* m_matmul_algo;
    mutable std::string m_name;
    const size_t m_ohw_tile_size;
};

}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
