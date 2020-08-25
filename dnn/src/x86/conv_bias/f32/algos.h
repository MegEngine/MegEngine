/**
 * \file dnn/src/x86/conv_bias/f32/algos.h
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
#include "src/common/nchw_nchwxx_valid.h"
#include "src/x86/conv_bias/opr_impl.h"

using namespace megdnn;
using namespace x86;

/* ===================== direct algo ===================== */
class ConvBiasImpl::AlgoDirect final : public AlgoBase {
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
    const char* name() const override {
        return "X86_CONV_BIAS_DIRECT_STRIDE1_LARGE_GROUP";
    }
    bool usable(const NCBKernSizeParam& param,
                AlgoSelectionStrategy algo_selection_strategy) const override;

    size_t get_workspace(const NCBKernSizeParam& param) const override;

    virtual SmallVector<NCBKern> dispatch_kerns(

            const NCBKernSizeParam& param) const override {
        return get_kimpls(param);
    }

    void* type() const override;
};

/* ===================== direct-stride2 algo ===================== */
class ConvBiasImpl::AlgoDirectStride2 final : public AlgoBase {
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
    const char* name() const override {
        return "X86_CONV_BIAS_DIRECT_STRIDE2_LARGE_GROUP";
    }
    bool usable(const NCBKernSizeParam& param,
                AlgoSelectionStrategy algo_selection_strategy) const override;

    size_t get_workspace(const NCBKernSizeParam& param) const override;

    virtual SmallVector<NCBKern> dispatch_kerns(

            const NCBKernSizeParam& param) const override {
        return get_kimpls(param);
    }

    void* type() const override;
};
/* =========================== winograd ======================== */
class ConvBiasImpl::AlgoFP32WinogradF63_8x8 final : public AlgoBase {
public:
    AlgoFP32WinogradF63_8x8(fallback::MatrixMulImpl::AlgoBase* matmul_algo,
                            uint32_t tile_size)
            : m_matmul_algo{matmul_algo}, m_tile_size{tile_size} {}
    const char* name() const override {
        if (m_name.empty()) {
            m_name = ConvBiasImpl::algo_name<ConvBias::WinogradParam>(
                    m_matmul_algo->name(), {8, 6, m_tile_size});
        }
        return m_name.c_str();
    }
    void* type() const override;
    MEGDNN_WINOGRAD_ALGO_FUN_DECLARE();
};

class ConvBiasImpl::AlgoFP32WinogradF23_8x8 final : public AlgoBase {
public:
    AlgoFP32WinogradF23_8x8(fallback::MatrixMulImpl::AlgoBase* matmul_algo,
                            uint32_t tile_size)
            : m_matmul_algo{matmul_algo}, m_tile_size{tile_size} {}
    const char* name() const override {
        if (m_name.empty()) {
            m_name = ConvBiasImpl::algo_name<ConvBias::WinogradParam>(
                    m_matmul_algo->name(), {8, 2, m_tile_size});
        }
        return m_name.c_str();
    }
    void* type() const override;
    MEGDNN_WINOGRAD_ALGO_FUN_DECLARE();
};

/* ===================== matmul algo ===================== */
class ConvBiasImpl::AlgoMatrixMul final : public AlgoBase {
    static MatrixMul* get_matmul_opr();
    static WorkspaceBundle get_bundle(const NCBKernSizeParam& param);
    static void kimpl(const NCBKernParam& param, const NCBKernIndex&);

public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "X86_CONV_BIAS_MATMUL"; }

    bool usable(const NCBKernSizeParam& param,
                AlgoSelectionStrategy) const override {
        auto&& fm = param.filter_meta;
        return fm.format == Param::Format::NCHW && fm.spatial_ndim == 2 &&
               param.src_type.enumv() == DTypeEnum::Float32 &&
               param.filter_type.enumv() == DTypeEnum::Float32 &&
               param.dst_type.enumv() == DTypeEnum::Float32 &&
               fm.dilation[0] == 1 && fm.dilation[1] == 1 &&
               //! The matmul opr is only used in single thread
               //! TODO:support the no pack matmul algo in fallback im2col +
               //! matmul
               param.nr_threads == 1_z;
    }

    bool is_preferred(const NCBKernSizeParam&) const override;

    size_t get_workspace(const NCBKernSizeParam& param) const override {
        return get_bundle(param).total_size_in_bytes();
    }
    SmallVector<NCBKern> dispatch_kerns(
            const NCBKernSizeParam& param) const override {
        size_t group = param.filter_meta.group;
        return {{kimpl, {group, 1_z, 1_z}}};
    }

    void* type() const override;
};

#if MEGDNN_X86_WITH_MKL_DNN
class ConvBiasImpl::AlgoMkldnnConv final : public AlgoBase {
    static void kern_mkldnn_fp32(const NCBKernParam& param,
                                 const NCBKernIndex&);

public:
    AlgoMkldnnConv() {}
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "MKLDNN_CONV_FP32"; }
    bool usable(const NCBKernSizeParam& param,
                AlgoSelectionStrategy) const override {
        auto&& fm = param.filter_meta;

        bool nchw_nchw88_ok = nchw_nchwxx_valid<NchwNchwxxType::NCHW88>(
                param.src_type.enumv(), param.filter_type.enumv(),
                param.dst_type.enumv(), param.filter_meta, param.bias_mode,
                param.nonlineMode);

        bool normal_conv_ok = (fm.format == param::ConvBias::Format::NCHW88) &&
                              fm.spatial_ndim == 2 &&
                              param.src_type.enumv() == DTypeEnum::Float32 &&
                              param.filter_type.enumv() == DTypeEnum::Float32 &&
                              param.dst_type.enumv() == DTypeEnum::Float32 &&
                              fm.dilation[0] == 1 && fm.dilation[1] == 1;

        return nchw_nchw88_ok || normal_conv_ok;
    };

    size_t get_workspace(const NCBKernSizeParam&) const override { return 0; }

    SmallVector<NCBKern> dispatch_kerns(
            const NCBKernSizeParam& /*param*/) const override {
        auto kern = [](const NCBKernParam& param,
                       const NCBKernIndex& ncb_index) {
            kern_mkldnn_fp32(param, ncb_index);
        };
        return {{kern, {1_z, 1_z, 1_z}}};
    }
    void* type() const override;
};
#endif
// vim: syntax=cpp.doxygen
