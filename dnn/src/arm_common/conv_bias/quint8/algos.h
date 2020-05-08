/**
 * \file dnn/src/arm_common/conv_bias/quint8/algos.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "src/arm_common/conv_bias/opr_impl.h"

namespace megdnn {
namespace arm_common {

class ConvBiasImpl::AlgoQU8DirectStride1 final : public AlgoBase {
    bool m_large_group;

public:
    AlgoQU8DirectStride1(bool large_group) : m_large_group(large_group) {}
    bool is_reproducible() const override { return true; }
    const char* name() const override {
        return m_large_group ? "QU8STRD1_LARGE_GROUP" : "QU8STRD1_SMALL_GROUP";
    }

    bool usable(fallback::ConvBiasImpl* opr, const NCBKernSizeParam& param,
                AlgoSelectionStrategy algo_selection_strategy) const override;

    size_t get_workspace(fallback::ConvBiasImpl*,
                         const NCBKernSizeParam& param) const override;
    virtual SmallVector<NCBKern> dispatch_kerns(
            fallback::ConvBiasImpl* opr,
            const NCBKernSizeParam& param) const override;
};

class ConvBiasImpl::AlgoQU8DirectStride2 final : public AlgoBase {
    bool m_large_group;

public:
    AlgoQU8DirectStride2(bool large_group) : m_large_group(large_group) {}
    bool is_reproducible() const override { return true; }
    const char* name() const override {
        return m_large_group ? "QU8STRD2_LARGE_GROUP" : "QU8STRD2_SMALL_GROUP";
    }
    bool usable(fallback::ConvBiasImpl* opr, const NCBKernSizeParam& param,
                AlgoSelectionStrategy algo_selection_strategy) const override;

    size_t get_workspace(fallback::ConvBiasImpl*,
                         const NCBKernSizeParam& param) const override;
    virtual SmallVector<NCBKern> dispatch_kerns(
            fallback::ConvBiasImpl* opr,
            const NCBKernSizeParam& param) const override;
};
#if  __ARM_FEATURE_DOTPROD
class ConvBiasImpl::AlgoDotU8DirectStride1 final : public AlgoBase {
    bool m_large_group;

public:
    AlgoDotU8DirectStride1(bool large_group) : m_large_group(large_group) {}
    bool is_reproducible() const override { return true; }
    const char* name() const override {
        return m_large_group ? "ARMDOTU8STRD1_LARGE_GROUP"
                             : "ARMDOTU8STRD1_SMALL_GROUP";
    }

    bool usable(fallback::ConvBiasImpl* opr, const NCBKernSizeParam& param,
                AlgoSelectionStrategy algo_selection_strategy) const override;

    size_t get_workspace(fallback::ConvBiasImpl*,
                         const NCBKernSizeParam& param) const override;
    virtual SmallVector<NCBKern> dispatch_kerns(
            fallback::ConvBiasImpl* opr,
            const NCBKernSizeParam& param) const override;
};

class ConvBiasImpl::AlgoDotU8DirectStride2 final : public AlgoBase {
    bool m_large_group;

public:
    AlgoDotU8DirectStride2(bool large_group) : m_large_group(large_group) {}
    bool is_reproducible() const override { return true; }
    const char* name() const override {
        return m_large_group ? "ARMDOTU8STRD2_LARGE_GROUP"
                             : "ARMDOTU8STRD2_SMALL_GROUP";
    }
    bool usable(fallback::ConvBiasImpl* opr, const NCBKernSizeParam& param,
                AlgoSelectionStrategy algo_selection_strategy) const override;

    size_t get_workspace(fallback::ConvBiasImpl*,
                         const NCBKernSizeParam& param) const override;
    virtual SmallVector<NCBKern> dispatch_kerns(
            fallback::ConvBiasImpl* opr,
            const NCBKernSizeParam& param) const override;
};
#endif
}  // namespace arm_common
}  // namespace megdnn
   // vim: syntax=cpp.doxygen
