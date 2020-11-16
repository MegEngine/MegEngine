/**
 * \file dnn/src/aarch64/conv_bias/fp32/algos.h
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
/* ===================== stride-2 algo ===================== */

class ConvBiasImpl::AlgoF32DirectStride2 final : public AlgoBase {
    SmallVector<NCBKern> get_kimpls(const NCBKernSizeParam& param) const;
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "ARMV8F32STRD2"; }

    bool usable(const NCBKernSizeParam& param,
                AlgoSelectionStrategy algo_selection_strategy) const override;

    size_t get_workspace(const NCBKernSizeParam& param) const override;

    SmallVector<NCBKern> dispatch_kerns(const NCBKernSizeParam&) const override;

    ConvAlgoTypePack get_algo_type() const override {
        return {AlgoDataType::FLOAT32, AlgoCategory::DIRECT};
    }
    MEGDNN_DECL_ALGO_TYPE(AARCH64_DIRECT_STRD2_FP32)
};

}  // namespace aarch64
}  // namespace megdnn

// vim: syntax=cpp.doxygen
