/**
 * \file dnn/src/arm_common/conv_bias/quint8/algos.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#pragma once

#include "src/arm_common/conv_bias/opr_impl.h"

namespace megdnn {
namespace arm_common {

class ConvBiasImpl::AlgoQU8DirectStride1 final : public AlgoBase {

public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "QU8STRD1"; }

    bool usable(const NCBKernSizeParam& param,
                AlgoSelectionStrategy algo_selection_strategy) const override;

    size_t get_workspace(const NCBKernSizeParam& param) const override;
    virtual SmallVector<NCBKern> dispatch_kerns(
            const NCBKernSizeParam& param) const override;
    ConvAlgoTypePack get_algo_type() const override {
        return {AlgoDataType::QUINT8X8X32, AlgoCategory::DIRECT};
    }
    MEGDNN_DECL_ALGO_TYPE(ARM_COMMON_DIRECT_STRD1_QU8)
};

class ConvBiasImpl::AlgoQU8DirectStride2 final : public AlgoBase {

public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "QU8STRD2"; }
    bool usable(const NCBKernSizeParam& param,
                AlgoSelectionStrategy algo_selection_strategy) const override;

    size_t get_workspace(const NCBKernSizeParam& param) const override;
    virtual SmallVector<NCBKern> dispatch_kerns(
            const NCBKernSizeParam& param) const override;
    ConvAlgoTypePack get_algo_type() const override {
        return {AlgoDataType::QUINT8X8X32, AlgoCategory::DIRECT};
    }
    MEGDNN_DECL_ALGO_TYPE(ARM_COMMON_DIRECT_STRD2_QU8)
};
#if __ARM_FEATURE_DOTPROD
class ConvBiasImpl::AlgoDotU8DirectStride1 final : public AlgoBase {

public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "ARMDOTU8STRD1"; }

    bool usable(const NCBKernSizeParam& param,
                AlgoSelectionStrategy algo_selection_strategy) const override;

    size_t get_workspace(const NCBKernSizeParam& param) const override;
    virtual SmallVector<NCBKern> dispatch_kerns(
            const NCBKernSizeParam& param) const override;
    ConvAlgoTypePack get_algo_type() const override {
        return {AlgoDataType::QUINT8X8X32, AlgoCategory::DIRECT};
    }
    MEGDNN_DECL_ALGO_TYPE(ARM_COMMON_DIRECT_STRD1_DOT_QU8)
};

class ConvBiasImpl::AlgoDotU8DirectStride2 final : public AlgoBase {

public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "ARMDOTU8STRD2"; }
    bool usable(const NCBKernSizeParam& param,
                AlgoSelectionStrategy algo_selection_strategy) const override;

    size_t get_workspace(const NCBKernSizeParam& param) const override;
    virtual SmallVector<NCBKern> dispatch_kerns(
            const NCBKernSizeParam& param) const override;
    ConvAlgoTypePack get_algo_type() const override {
        return {AlgoDataType::QUINT8X8X32, AlgoCategory::DIRECT};
    }
    MEGDNN_DECL_ALGO_TYPE(ARM_COMMON_DIRECT_STRD2_DOT_QU8)
};
#endif
}  // namespace arm_common
}  // namespace megdnn
   // vim: syntax=cpp.doxygen
