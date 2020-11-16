/**
 * \file dnn/src/aarch64/conv_bias/quint8/algos.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "src/aarch64/conv_bias/opr_impl.h"
#include "src/fallback/conv_bias/opr_impl.h"
#include "src/common/opr_delegate.h"

namespace megdnn {
namespace aarch64 {

using FallbackConvBiasImpl = fallback::ConvBiasImpl;

class ConvBiasImpl::AlgoQU8MatrixMul final : public AlgoBase {
    static WorkspaceBundle get_bundle(const NCBKernSizeParam& param);
    static void kimpl(const NCBKernParam& param, const NCBKernIndex&);

public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "QU8MATMUL"; }

    bool usable(const NCBKernSizeParam& param,
                AlgoSelectionStrategy algo_selection_strategy) const override;
    size_t get_workspace(const NCBKernSizeParam& param) const override {
        return get_bundle(param).total_size_in_bytes();
    }
    SmallVector<NCBKern> dispatch_kerns(
            const NCBKernSizeParam& param) const override {
        size_t group = param.filter_meta.group;
        return {{kimpl, {group, 1_z, 1_z}}};
    }
    //! select matmul to the highest preference
    bool is_preferred(const NCBKernSizeParam& param) const override {
        static CpuOprDelegationStorage<1> storage;
        auto conv_bias_opr = storage.get<ConvBias, 0>();
        return static_cast<ConvBiasImpl*>(conv_bias_opr)
                ->is_matmul_quantized_prefer(param);
    }
    ConvAlgoTypePack get_algo_type() const override {
        return {AlgoDataType::QUINT8X8X32, AlgoCategory::IM2COL};
    }
    MEGDNN_DECL_ALGO_TYPE(AARCH64_MATMUL_QU8)
};
}  // namespace aarch64
}  // namespace megdnn

// vim: syntax=cpp.doxygen
