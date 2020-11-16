/**
 * \file dnn/src/fallback/conv_bias/algos.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "src/fallback/conv_bias/opr_impl.h"
#include "src/fallback/matrix_mul/opr_impl.h"
#include "megdnn/thin/small_vector.h"

namespace megdnn {
namespace fallback {

class ConvBiasImpl::AlgoNaive final : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "FALLBACK_NAIVE"; }
    bool usable(const NCBKernSizeParam& param,
                AlgoSelectionStrategy algo_selection_strategy) const override;
    size_t get_workspace(const NCBKernSizeParam& param) const override;
    SmallVector<NCBKern> dispatch_kerns(const NCBKernSizeParam&) const override;

    ConvAlgoTypePack get_algo_type() const override {
        auto support_data_type = static_cast<AlgoDataType>(
                static_cast<uint32_t>(AlgoDataType::FLOAT16) |
                static_cast<uint32_t>(AlgoDataType::FLOAT32) |
                static_cast<uint32_t>(AlgoDataType::INT8X8X16) |
                static_cast<uint32_t>(AlgoDataType::QINT8X8X32) |
                static_cast<uint32_t>(AlgoDataType::QUINT8X8X32));
        return {support_data_type, AlgoCategory::NAIVE};
    }
    MEGDNN_DECL_ALGO_TYPE(FB_NAIVE)
};

class ConvBiasImpl::AlgoWinogradF32 final : public AlgoBase {
public:
    AlgoWinogradF32(MatrixMulImpl::AlgoBase* matmul_algo)
            : m_matmul_algo{matmul_algo} {}
    bool is_reproducible() const override { return true; }
    const char* name() const override {
        if (m_name.empty()) {
            m_name = ConvBiasImpl::algo_name<ConvBias::WinogradParam>(
                    ssprintf("FALLBACK_WINOGRAD_F32-%s", m_matmul_algo->name()),
                    {1, 2, UNIT_TILE_SIZE});
        }
        return m_name.c_str();
    }
    bool usable(const NCBKernSizeParam& param,
                AlgoSelectionStrategy algo_selection_strategy) const override;
    size_t get_workspace(const NCBKernSizeParam& param) const override;
    SmallVector<NCBKern> dispatch_kerns(const NCBKernSizeParam&) const override;

    ConvAlgoTypePack get_algo_type() const override {
        return {AlgoDataType::FLOAT32, AlgoCategory::WINOGRAD};
    }
    MEGDNN_DECL_ALGO_TYPE(FB_WINOGRAD_F32)
    std::string param() const override {
        std::string ret;
        serialize_write_pod(m_matmul_algo, ret);
        return ret;
    }

private:
    MatrixMulImpl::AlgoBase* m_matmul_algo;
    mutable std::string m_name;
    constexpr size_t static UNIT_TILE_SIZE = 32;
};

class ConvBiasImpl::AlgoWinogradF32_4x4 final : public AlgoBase {
public:
    AlgoWinogradF32_4x4(MatrixMulImpl::AlgoBase* matmul_algo)
            : m_matmul_algo{matmul_algo} {}
    bool is_reproducible() const override { return true; }
    const char* name() const override {
        if (m_name.empty()) {
            m_name = ConvBiasImpl::algo_name<ConvBias::WinogradParam>(
                    ssprintf("FALLBACK_WINOGRAD_F32-%s", m_matmul_algo->name()),
                    {4, 2, UNIT_TILE_SIZE});
        }
        return m_name.c_str();
    }
    bool usable(const NCBKernSizeParam& param,
                AlgoSelectionStrategy algo_selection_strategy) const override;
    size_t get_workspace(const NCBKernSizeParam& param) const override;
    SmallVector<NCBKern> dispatch_kerns(const NCBKernSizeParam&) const override;

    ConvAlgoTypePack get_algo_type() const override {
        return {AlgoDataType::FLOAT32, AlgoCategory::WINOGRAD};
    }
    MEGDNN_DECL_ALGO_TYPE(FB_WINOGRAD_4X4_F32)
    std::string param() const override {
        std::string ret;
        serialize_write_pod(m_matmul_algo, ret);
        return ret;
    }

private:
    MatrixMulImpl::AlgoBase* m_matmul_algo;
    mutable std::string m_name;
    constexpr size_t static UNIT_TILE_SIZE = 32;
};

class ConvBiasImpl::AlgoWinogradQS8 final : public AlgoBase {
public:
    AlgoWinogradQS8(MatrixMulImpl::AlgoBase* matmul_algo)
            : m_matmul_algo{matmul_algo} {}
    bool is_reproducible() const override { return true; }
    const char* name() const override {
        if (m_name.empty()) {
            m_name = ConvBiasImpl::algo_name<ConvBias::WinogradParam>(
                    ssprintf("FALLBACK_WINOGRAD_QS8-%s", m_matmul_algo->name()),
                    {1, 2, UNIT_TILE_SIZE});
        }
        return m_name.c_str();
    }
    bool usable(const NCBKernSizeParam& param,
                AlgoSelectionStrategy algo_selection_strategy) const override;
    size_t get_workspace(const NCBKernSizeParam& param) const override;
    SmallVector<NCBKern> dispatch_kerns(const NCBKernSizeParam&) const override;

    ConvAlgoTypePack get_algo_type() const override {
        return {AlgoDataType::QINT8X8X32, AlgoCategory::WINOGRAD};
    }
    MEGDNN_DECL_ALGO_TYPE(FB_WINOGRAD_QS8)
    std::string param() const override {
        std::string ret;
        serialize_write_pod(m_matmul_algo, ret);
        return ret;
    }

private:
    MatrixMulImpl::AlgoBase* m_matmul_algo;
    mutable std::string m_name;
    constexpr size_t static UNIT_TILE_SIZE = 32;
};

class ConvBiasImpl::AlgoWinogradQS8_8x8 final : public AlgoBase {
public:
    AlgoWinogradQS8_8x8(MatrixMulImpl::AlgoBase* matmul_algo)
            : m_matmul_algo{matmul_algo} {}
    bool is_reproducible() const override { return true; }
    const char* name() const override {
        if (m_name.empty()) {
            m_name = ConvBiasImpl::algo_name<ConvBias::WinogradParam>(
                    ssprintf("FALLBACK_WINOGRAD_QS8-%s", m_matmul_algo->name()),
                    {8, 2, UNIT_TILE_SIZE});
        }
        return m_name.c_str();
    }
    bool usable(const NCBKernSizeParam& param,
                AlgoSelectionStrategy algo_selection_strategy) const override;
    size_t get_workspace(const NCBKernSizeParam& param) const override;
    SmallVector<NCBKern> dispatch_kerns(const NCBKernSizeParam&) const override;

    ConvAlgoTypePack get_algo_type() const override {
        return {AlgoDataType::QINT8X8X32, AlgoCategory::WINOGRAD};
    }
    MEGDNN_DECL_ALGO_TYPE(FB_WINOGRAD_8X8_QS8)
    std::string param() const override {
        std::string ret;
        serialize_write_pod(m_matmul_algo, ret);
        return ret;
    }

private:
    MatrixMulImpl::AlgoBase* m_matmul_algo;
    mutable std::string m_name;
    constexpr size_t static UNIT_TILE_SIZE = 32;
};

}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
