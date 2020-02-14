/**
 * \file dnn/src/x86/conv_bias/int8/algos.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "src/x86/conv_bias/opr_impl.h"

namespace megdnn {
namespace x86 {
/* ===================== avx2 stride1 direct algo ===================== */
class ConvBiasImpl::AlgoDirectAvx2Stride1Int8 final : public AlgoBase {
    SmallVector<NCBKern> get_kimpls(const NCBKernSizeParam& param) const;
    static WorkspaceBundle get_bundle(const NCBKernSizeParam& param);

public:
    bool is_reproducible() const override { return true; }
    const char* name() const override {
        return "X86_CONV_BIAS_DIRECT_AVX2_INT8_STRIDE1";
    }
    bool usable(FallbackConvBiasImpl* opr, const NCBKernSizeParam& param,
                AlgoSelectionStrategy algo_selection_strategy) const override;
    size_t get_workspace(FallbackConvBiasImpl* opr,
                         const NCBKernSizeParam& param) const override;
    virtual SmallVector<NCBKern> dispatch_kerns(
            fallback::ConvBiasImpl*,
            const NCBKernSizeParam& param) const override {
        return get_kimpls(param);
    }
    void* type() const override;
};

#if defined(MEGDNN_X86_WITH_MKL_DNN)
/* ===================== mkldnn qint8 algo ===================== */
class ConvBiasImpl::AlgoMkldnnQint8 final : public AlgoBase {
    static void kern_mkldnn_s8x8x32(const NCBKernParam& param,
                                    const NCBKernIndex&);
    static WorkspaceBundle get_bundle(const NCBKernSizeParam& param);

public:
    AlgoMkldnnQint8() {}
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "MKLDNN_INT8"; }
    bool usable(FallbackConvBiasImpl* opr, const NCBKernSizeParam& param,
                AlgoSelectionStrategy) const override;

    size_t get_workspace(FallbackConvBiasImpl* /*opr*/,
                         const NCBKernSizeParam& param) const override {
        size_t nr_threads = param.nr_threads;
        return get_bundle(param).total_size_in_bytes() * nr_threads;
    }
    SmallVector<NCBKern> dispatch_kerns(
            FallbackConvBiasImpl* /*opr*/,
            const NCBKernSizeParam& param) const override {
        size_t group = param.filter_meta.group;
        size_t n = param.n;
        auto workspace_per_thread = get_bundle(param).total_size_in_bytes();
        auto kern = [workspace_per_thread](const NCBKernParam& param,
                                           const NCBKernIndex& ncb_index) {
            auto thread_param = param;
            thread_param.workspace_ptr = reinterpret_cast<void*>(
                    reinterpret_cast<ptrdiff_t>(param.workspace_ptr) +
                    ncb_index.thread_id * workspace_per_thread);
            kern_mkldnn_s8x8x32(thread_param, std::move(ncb_index));
        };
        return {{kern, {group, n, 1_z}}};
    }
    void* type() const override;
};
/* ===================== mkldnn qint8 matmul algo ===================== */
class ConvBiasImpl::AlgoMkldnnMatmulQint8 final : public AlgoBase {
    static MatrixMul* get_matmul_opr();
    static void kern_mkldnn_matmul_s8x8x32(const NCBKernParam& param,
                                           const NCBKernIndex&);
    static WorkspaceBundle get_bundle(const NCBKernSizeParam& param);

public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "MKLDNN_MATMUL_INT8"; }
    bool usable(FallbackConvBiasImpl* opr, const NCBKernSizeParam& param,
                AlgoSelectionStrategy) const override;

    size_t get_workspace(FallbackConvBiasImpl* /*opr*/,
                         const NCBKernSizeParam& param) const override {
        return get_bundle(param).total_size_in_bytes();
    }
    SmallVector<NCBKern> dispatch_kerns(
            FallbackConvBiasImpl* /*opr*/,
            const NCBKernSizeParam& param) const override {
        size_t group = param.filter_meta.group;
        return {{kern_mkldnn_matmul_s8x8x32, {group, 1_z, 1_z}}};
    }
    //! select matmul to the highest preference
    bool is_preferred(FallbackConvBiasImpl*,
                      const NCBKernSizeParam& param) const override;

    void* type() const override;
};
#endif
/* ===================== avx2 int8 direct conv stride2 algo ===================== */
class ConvBiasImpl::AlgoAVX2DirectConvStride2 final : public AlgoBase {
    SmallVector<NCBKern> get_kimpls(const NCBKernSizeParam& param) const;
    static WorkspaceBundle get_bundle(const NCBKernSizeParam& param);

public:
    bool is_reproducible() const override { return true; }
    const char* name() const override {
        return "X86_CONV_BIAS_DIRECT_AVX2_INT8_STRIDE2";
    }
    bool usable(FallbackConvBiasImpl* opr, const NCBKernSizeParam& param,
                AlgoSelectionStrategy algo_selection_strategy) const override;
    size_t get_workspace(FallbackConvBiasImpl* opr,
                         const NCBKernSizeParam& param) const override;
    SmallVector<NCBKern> dispatch_kerns(
            fallback::ConvBiasImpl*,
            const NCBKernSizeParam& param) const override {
        return get_kimpls(param);
    }
    void* type() const override;
};
}  // namespace x86
}  // namespace megdnn

// vim: syntax=cpp.doxygen
