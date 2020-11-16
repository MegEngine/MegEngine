/**
 * \file dnn/src/arm_common/conv_bias/int8x8x16/algos.h
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

public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "I8816DIRECT"; }
    bool usable(const NCBKernSizeParam& param,
                AlgoSelectionStrategy algo_selection_strategy) const override;
    size_t get_workspace(const NCBKernSizeParam& param) const override;
    virtual SmallVector<NCBKern> dispatch_kerns(
            const NCBKernSizeParam& param) const override;
    ConvAlgoTypePack get_algo_type() const override {
        return {AlgoDataType::INT8X8X16, AlgoCategory::DIRECT};
    }
    MEGDNN_DECL_ALGO_TYPE(ARM_COMMON_DIRECT_INT8X8X16)
};

class ConvBiasImpl::AlgoS8x8x16DirectNCHW44 final : public AlgoBase {
public:
    AlgoS8x8x16DirectNCHW44() {}
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "S8x8x16_NCHW44_DIRECT"; }
    bool usable(const NCBKernSizeParam& param,
                AlgoSelectionStrategy algo_selection_strategy) const override;
    size_t get_workspace(const NCBKernSizeParam& param) const override;
    virtual SmallVector<NCBKern> dispatch_kerns(
            const NCBKernSizeParam& param) const override;
    ConvAlgoTypePack get_algo_type() const override {
        return {AlgoDataType::INT8X8X16, AlgoCategory::DIRECT};
    }
    MEGDNN_DECL_ALGO_TYPE(ARM_COMMON_DIRECT_NCHW44_INT8X8X16)
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

public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "I8816STRD2"; }
    bool usable(const NCBKernSizeParam& param,
                AlgoSelectionStrategy algo_selection_strategy) const override;

    size_t get_workspace(const NCBKernSizeParam& param) const override;
    virtual SmallVector<NCBKern> dispatch_kerns(
            const NCBKernSizeParam& param) const override;
    ConvAlgoTypePack get_algo_type() const override {
        return {AlgoDataType::INT8X8X16, AlgoCategory::DIRECT};
    }
    MEGDNN_DECL_ALGO_TYPE(ARM_COMMON_DIRECT_STRD2_INT8X8X16)
};

class ConvBiasImpl::AlgoI8x8x16Stride2Filter2 final : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "I8816STRD2F2"; }

    bool usable(const NCBKernSizeParam& param,
                AlgoSelectionStrategy algo_selection_strategy) const override;

    size_t get_workspace(const NCBKernSizeParam& param) const override;
    virtual SmallVector<NCBKern> dispatch_kerns(
            const NCBKernSizeParam& param) const override;
    ConvAlgoTypePack get_algo_type() const override {
        return {AlgoDataType::INT8X8X16, AlgoCategory::DIRECT};
    }
    MEGDNN_DECL_ALGO_TYPE(ARM_COMMON_DIRECT_STRD2_F2_INT8X8X16)
};

class ConvBiasImpl::AlgoS8x8x16ChanWiseStride1Stride2NCHW44 final
        : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override {
        return "S8x8x16_CHAN_WISE_STRD1_STRD2_NCHW44";
    }
    bool usable(const NCBKernSizeParam& param,
                AlgoSelectionStrategy algo_selection_strategy) const override;
    size_t get_workspace(
                         const NCBKernSizeParam& param) const override;
    virtual SmallVector<NCBKern> dispatch_kerns(
            const NCBKernSizeParam& param) const override;
    ConvAlgoTypePack get_algo_type() const override {
        return {AlgoDataType::INT8X8X16, AlgoCategory::DIRECT};
    }
    MEGDNN_DECL_ALGO_TYPE(ARM_COMMON_CHWNWISE_STRD1_STRD2_NCHW44_INT8X8X16)
};

class ConvBiasImpl::AlgoI8x8x16DirectNCHWNCHW44 final : public AlgoBase {
    SmallVector<NCBKern> get_kimpls(const NCBKernSizeParam& param) const;

public:
    AlgoI8x8x16DirectNCHWNCHW44() {}
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "I8816_CONV_NCHW_NCHW44"; }
    bool usable(const NCBKernSizeParam& param,
                AlgoSelectionStrategy algo_selection_strategy) const override;

    size_t get_workspace(const NCBKernSizeParam& param) const override;
    virtual SmallVector<NCBKern> dispatch_kerns(
            const NCBKernSizeParam& param) const override;
    ConvAlgoTypePack get_algo_type() const override {
        return {AlgoDataType::INT8X8X16, AlgoCategory::DIRECT};
    }
    MEGDNN_DECL_ALGO_TYPE(ARM_COMMON_DIRECT_NCHW_NCHW44_INT8X8X16)
};

}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
