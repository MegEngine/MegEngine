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
    bool usable(ConvBiasImpl* opr, const NCBKernSizeParam& param,
                AlgoSelectionStrategy algo_selection_strategy) const override;
    size_t get_workspace(ConvBiasImpl*,
                         const NCBKernSizeParam& param) const override;
    SmallVector<NCBKern> dispatch_kerns(ConvBiasImpl*,
                                        const NCBKernSizeParam&) const override;
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
    bool usable(ConvBiasImpl* opr, const NCBKernSizeParam& param,
                AlgoSelectionStrategy algo_selection_strategy) const override;
    size_t get_workspace(ConvBiasImpl*,
                         const NCBKernSizeParam& param) const override;
    SmallVector<NCBKern> dispatch_kerns(ConvBiasImpl*,
                                        const NCBKernSizeParam&) const override;

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
    bool usable(ConvBiasImpl* opr, const NCBKernSizeParam& param,
                AlgoSelectionStrategy algo_selection_strategy) const override;
    size_t get_workspace(ConvBiasImpl*,
                         const NCBKernSizeParam& param) const override;
    SmallVector<NCBKern> dispatch_kerns(ConvBiasImpl*,
                                        const NCBKernSizeParam&) const override;

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
    bool usable(ConvBiasImpl* opr, const NCBKernSizeParam& param,
                AlgoSelectionStrategy algo_selection_strategy) const override;
    size_t get_workspace(ConvBiasImpl*,
                         const NCBKernSizeParam& param) const override;
    SmallVector<NCBKern> dispatch_kerns(ConvBiasImpl*,
                                        const NCBKernSizeParam&) const override;

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
    bool usable(ConvBiasImpl* opr, const NCBKernSizeParam& param,
                AlgoSelectionStrategy algo_selection_strategy) const override;
    size_t get_workspace(ConvBiasImpl*,
                         const NCBKernSizeParam& param) const override;
    SmallVector<NCBKern> dispatch_kerns(ConvBiasImpl*,
                                        const NCBKernSizeParam&) const override;

private:
    MatrixMulImpl::AlgoBase* m_matmul_algo;
    mutable std::string m_name;
    constexpr size_t static UNIT_TILE_SIZE = 32;
};

}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
