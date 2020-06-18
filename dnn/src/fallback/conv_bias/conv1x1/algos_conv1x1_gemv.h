/**
 * \file dnn/src/fallback/conv_bias/conv1x1/algos_conv1x1_gemv.h
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

namespace megdnn {
namespace fallback {

class ConvBiasImpl::AlgoConv1x1Gemv final : public AlgoBase {
public:
    AlgoConv1x1Gemv() = default;

    bool is_reproducible() const override { return true; }

    const char* name() const override {
        return "CONV1x1_GEMV";
    }

    bool usable(ConvBiasImpl* opr, const NCBKernSizeParam& param,
                AlgoSelectionStrategy algo_selection_strategy) const override;
    size_t get_workspace(ConvBiasImpl*,
                         const NCBKernSizeParam& param) const override;
    SmallVector<NCBKern> dispatch_kerns(
            ConvBiasImpl* opr, const NCBKernSizeParam& param) const override;

    bool is_preferred(ConvBiasImpl*, const NCBKernSizeParam&) const override;

protected:
    size_t get_oc_tile_size_heuristic(const NCBKernSizeParam& param) const;
};

}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
