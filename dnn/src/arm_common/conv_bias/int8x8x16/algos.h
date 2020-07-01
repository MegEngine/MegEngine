/**
 * \file dnn/src/arm_common/conv_bias/int8x8x16/algos.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "../opr_impl.h"

namespace megdnn {
namespace arm_common {
class ConvBiasImpl::AlgoI8x8x16Direct final : public AlgoBase {
    SmallVector<NCBKern> get_kimpls(const NCBKernSizeParam& param) const;
    WorkspaceBundle get_bundle(const NCBKernSizeParam& param) const;
    static void copy_padding_kern(const WorkspaceBundle& bundle,
                                  const NCBKernParam& kern_param,
                                  const NCBKernIndex& ncb_index,
                                  const CpuNDRange& workspace_ids);
    static void do_conv_kern(const WorkspaceBundle& bundle,
                             const NCBKernParam& kern_param,
                             const NCBKernIndex& ncb_index,
                             const CpuNDRange& workspace_ids);
    bool m_large_group;

public:
    AlgoI8x8x16Direct(bool large_group) : m_large_group(large_group) {}
    bool is_reproducible() const override { return true; }
    const char* name() const override {
        return m_large_group ? "I8816DIRECT_LARGE_GROUP"
                             : "I8816DIRECT_SMALL_GROUP";
    }
    bool usable(fallback::ConvBiasImpl*, const NCBKernSizeParam& param,
                AlgoSelectionStrategy algo_selection_strategy) const override;
    size_t get_workspace(fallback::ConvBiasImpl*,
                         const NCBKernSizeParam& param) const override;
    virtual SmallVector<NCBKern> dispatch_kerns(
            fallback::ConvBiasImpl* opr,
            const NCBKernSizeParam& param) const override;
};

class ConvBiasImpl::AlgoI8x8x16Stride2 final : public AlgoBase {
    SmallVector<NCBKern> get_kimpls(const NCBKernSizeParam& param) const;
    WorkspaceBundle get_bundle(const NCBKernSizeParam& param) const;
    static void copy_padding_kern(const WorkspaceBundle& bundle,
                                  const NCBKernParam& kern_param,
                                  const NCBKernIndex& ncb_index,
                                  const CpuNDRange& workspace_ids);
    static void do_conv_kern(const WorkspaceBundle& bundle,
                             const NCBKernParam& kern_param,
                             const NCBKernIndex& ncb_index,
                             const CpuNDRange& workspace_ids);
    bool m_large_group;

public:
    AlgoI8x8x16Stride2(bool large_group) : m_large_group(large_group) {}
    bool is_reproducible() const override { return true; }
    const char* name() const override {
        return m_large_group ? "I8816STRD2_LARGE_GROUP"
                             : "I8816STRD2_SMALL_GROUP";
    }
    bool usable(fallback::ConvBiasImpl* opr, const NCBKernSizeParam& param,
                AlgoSelectionStrategy algo_selection_strategy) const override;

    size_t get_workspace(fallback::ConvBiasImpl*,
                         const NCBKernSizeParam& param) const override;
    virtual SmallVector<NCBKern> dispatch_kerns(
            fallback::ConvBiasImpl* opr,
            const NCBKernSizeParam& param) const override;
};

class ConvBiasImpl::AlgoI8x8x16Stride2Filter2 final : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "I8816STRD2F2"; }

    bool usable(fallback::ConvBiasImpl* opr, const NCBKernSizeParam& param,
                AlgoSelectionStrategy algo_selection_strategy) const override;

    size_t get_workspace(fallback::ConvBiasImpl*,
                         const NCBKernSizeParam& param) const override;
    virtual SmallVector<NCBKern> dispatch_kerns(
            fallback::ConvBiasImpl* opr,
            const NCBKernSizeParam& param) const override;
};

}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
