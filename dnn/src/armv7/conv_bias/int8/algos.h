/**
 * \file dnn/src/armv7/conv_bias/int8/algos.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "src/armv7/conv_bias/opr_impl.h"
#include "src/fallback/conv_bias/opr_impl.h"

namespace megdnn {
namespace armv7 {

using FallbackConvBiasImpl = fallback::ConvBiasImpl;

class ConvBiasImpl::AlgoS8MatrixMul final : public AlgoBase {
    static WorkspaceBundle get_bundle(const NCBKernSizeParam& param);
    static void kimpl(const NCBKernParam& param, const NCBKernIndex&);

public:
    AlgoAttribute attribute() const override { return AlgoAttribute::REPRODUCIBLE; }
    const char* name() const override { return "S8MATMUL"; }

    bool usable(
            const NCBKernSizeParam& param,
            AlgoSelectionStrategy algo_selection_strategy) const override;
    size_t get_workspace(const NCBKernSizeParam& param) const override {
        return get_bundle(param).total_size_in_bytes();
    }
    SmallVector<NCBKern> dispatch_kerns(const NCBKernSizeParam& param) const override {
        size_t group = param.filter_meta.group;
        return {{kimpl, {group, 1_z, 1_z}}};
    }
    ConvAlgoTypePack get_algo_type() const override {
        return {AlgoDataType::QINT8X8X32, AlgoCategory::IM2COL};
    }
    MEGDNN_DECL_ALGO_TYPE(ARMV7_MATMUL_S8)
};

}  // namespace armv7
}  // namespace megdnn

// vim: syntax=cpp.doxygen
