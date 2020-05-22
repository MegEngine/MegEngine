/**
 * \file dnn/src/aarch64/conv_bias/int8/algos.h
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

namespace megdnn {
namespace aarch64 {

using FallbackConvBiasImpl = fallback::ConvBiasImpl;

class ConvBiasImpl::AlgoS8MatrixMul final : public AlgoBase {
    static WorkspaceBundle get_bundle(const NCBKernSizeParam& param);
    static void kimpl(const NCBKernParam& param, const NCBKernIndex& ncb_index);

public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "S8MATMUL"; }

    bool usable(FallbackConvBiasImpl* opr, const NCBKernSizeParam& param,
                AlgoSelectionStrategy algo_selection_strategy) const override;
    size_t get_workspace(FallbackConvBiasImpl*,
                         const NCBKernSizeParam& param) const override {
        return get_bundle(param).total_size_in_bytes();
    }
    SmallVector<NCBKern> dispatch_kerns(
            FallbackConvBiasImpl*, const NCBKernSizeParam& param) const override {
        size_t group = param.filter_meta.group;
        return {{kimpl, {group, 1_z, 1_z}}};
    }
    //! select matmul to the highest preference
    bool is_preferred(FallbackConvBiasImpl* opr,
                      const NCBKernSizeParam& param) const override {
        return static_cast<arm_common::ConvBiasImpl*>(opr)
                ->is_matmul_quantized_prefer(param);
    }
};

}  // namespace aarch64
}  // namespace megdnn

// vim: syntax=cpp.doxygen
