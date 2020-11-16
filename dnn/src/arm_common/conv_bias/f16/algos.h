/**
 * \file dnn/src/arm_common/conv_bias/f16/algos.h
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
#include "src/fallback/matrix_mul/opr_impl.h"
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
namespace megdnn {
namespace arm_common {

class ConvBiasImpl::AlgoFP16WinogradF23 final : public AlgoBase {
public:
    AlgoFP16WinogradF23(fallback::MatrixMulImpl::AlgoBase* matmul_algo,
                        uint32_t tile_size)
            : m_matmul_algo{matmul_algo}, m_tile_size{tile_size} {}
    const char* name() const override {
        if (m_name.empty()) {
            m_name = ConvBiasImpl::algo_name<ConvBias::WinogradParam>(
                    m_matmul_algo->name(), {1, 2, m_tile_size});
        }
        return m_name.c_str();
    }
    MEGDNN_WINOGRAD_ALGO_FUN_DECLARE(AlgoDataType::FLOAT16);
    MEGDNN_DECL_ALGO_TYPE(ARM_COMMON_WINOGRAD_F23_FP16)
};

class ConvBiasImpl::AlgoFP16WinogradF45 final : public AlgoBase {
public:
    AlgoFP16WinogradF45(fallback::MatrixMulImpl::AlgoBase* matmul_algo,
                        uint32_t tile_size)
            : m_matmul_algo{matmul_algo}, m_tile_size{tile_size} {}
    const char* name() const override {
        if (m_name.empty()) {
            m_name = ConvBiasImpl::algo_name<ConvBias::WinogradParam>(
                    m_matmul_algo->name(), {1, 4, m_tile_size});
        }
        return m_name.c_str();
    }
    MEGDNN_WINOGRAD_ALGO_FUN_DECLARE(AlgoDataType::FLOAT16);
    MEGDNN_DECL_ALGO_TYPE(ARM_COMMON_WINOGRAD_F45_FP16)
};
class ConvBiasImpl::AlgoFP16WinogradF63 final : public AlgoBase {
public:
    AlgoFP16WinogradF63(fallback::MatrixMulImpl::AlgoBase* matmul_algo,
                        uint32_t tile_size)
            : m_matmul_algo{matmul_algo}, m_tile_size{tile_size} {}
    const char* name() const override {
        if (m_name.empty()) {
            m_name = ConvBiasImpl::algo_name<ConvBias::WinogradParam>(
                    m_matmul_algo->name(), {1, 6, m_tile_size});
        }
        return m_name.c_str();
    }

    MEGDNN_WINOGRAD_ALGO_FUN_DECLARE(AlgoDataType::FLOAT16);
    MEGDNN_DECL_ALGO_TYPE(ARM_COMMON_WINOGRAD_F63_FP16)
};
class ConvBiasImpl::AlgoFP16WinogradF23_8x8 final : public AlgoBase {
public:
    AlgoFP16WinogradF23_8x8(fallback::MatrixMulImpl::AlgoBase* matmul_algo,
                            uint32_t tile_size)
            : m_matmul_algo{matmul_algo}, m_tile_size{tile_size} {}
    const char* name() const override {
        if (m_name.empty()) {
            m_name = ConvBiasImpl::algo_name<ConvBias::WinogradParam>(
                    m_matmul_algo->name(), {8, 2, m_tile_size});
        }
        return m_name.c_str();
    }
    MEGDNN_WINOGRAD_ALGO_FUN_DECLARE(AlgoDataType::FLOAT16);
    MEGDNN_DECL_ALGO_TYPE(ARM_COMMON_WINOGRAD_F23_8X8_FP16)
};

class ConvBiasImpl::AlgoF16Direct final : public AlgoBase {
    SmallVector<NCBKern> get_kimpls(const NCBKernSizeParam& param) const;

public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "F16DIRECT"; }
    bool usable(const NCBKernSizeParam& param,
                AlgoSelectionStrategy algo_selection_strategy) const override;

    size_t get_workspace(const NCBKernSizeParam& param) const override;

    virtual SmallVector<NCBKern> dispatch_kerns(
            const NCBKernSizeParam& param) const override;

    ConvAlgoTypePack get_algo_type() const override{
        return {AlgoDataType::FLOAT16, AlgoCategory::DIRECT};
    }
    MEGDNN_DECL_ALGO_TYPE(ARM_COMMON_DIRECT_FP16)
};

class ConvBiasImpl::AlgoF16DirectStride1 final : public AlgoBase {
    SmallVector<NCBKern> get_kimpls(const NCBKernSizeParam& param) const;

public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "F16STRD1"; }
    bool usable(const NCBKernSizeParam& param,
                AlgoSelectionStrategy algo_selection_strategy) const override;
    size_t get_workspace(const NCBKernSizeParam& param) const override;
    virtual SmallVector<NCBKern> dispatch_kerns(
            const NCBKernSizeParam& param) const override;
    ConvAlgoTypePack get_algo_type() const override {
        return {AlgoDataType::FLOAT16, AlgoCategory::DIRECT};
    }
    MEGDNN_DECL_ALGO_TYPE(ARM_COMMON_DIRECT_STRD1_FP16)
};

}  // namespace arm_common
}  // namespace megdnn
#endif
// vim: syntax=cpp.doxygen
