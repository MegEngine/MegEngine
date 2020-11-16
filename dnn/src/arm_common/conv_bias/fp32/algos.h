/**
 * \file dnn/src/arm_common/conv_bias/fp32/algos.h
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
#include "src/fallback/matrix_mul/opr_impl.h"

namespace megdnn {
namespace arm_common {
class ConvBiasImpl::AlgoFP32WinogradF23_4x4 final : public AlgoBase {
public:
    AlgoFP32WinogradF23_4x4(fallback::MatrixMulImpl::AlgoBase* matmul_algo,
                            uint32_t tile_size)
            : m_matmul_algo{matmul_algo}, m_tile_size{tile_size} {}
    const char* name() const override {
        if (m_name.empty()) {
            m_name = ConvBiasImpl::algo_name<ConvBias::WinogradParam>(
                    m_matmul_algo->name(), {4, 2, m_tile_size});
        }
        return m_name.c_str();
    }
    MEGDNN_WINOGRAD_ALGO_FUN_DECLARE(AlgoDataType::FLOAT32);
    MEGDNN_DECL_ALGO_TYPE(ARM_COMMON_WINOGRAD_F23_4X4_FP32)
};

class ConvBiasImpl::AlgoFP32WinogradF63 final : public AlgoBase {
public:
    AlgoFP32WinogradF63(fallback::MatrixMulImpl::AlgoBase* matmul_algo,
                        uint32_t tile_size)
            : m_matmul_algo{matmul_algo}, m_tile_size{tile_size} {}
    const char* name() const override {
        if (m_name.empty()) {
            m_name = ConvBiasImpl::algo_name<ConvBias::WinogradParam>(
                    m_matmul_algo->name(), {1, 6, m_tile_size});
        }
        return m_name.c_str();
    }
    MEGDNN_WINOGRAD_ALGO_FUN_DECLARE(AlgoDataType::FLOAT32);
    MEGDNN_DECL_ALGO_TYPE(ARM_COMMON_WINOGRAD_F63_FP32)
};

class ConvBiasImpl::AlgoFP32WinogradF63_4x4 final : public AlgoBase {
public:
    AlgoFP32WinogradF63_4x4(fallback::MatrixMulImpl::AlgoBase* matmul_algo,
                            uint32_t tile_size)
            : m_matmul_algo{matmul_algo}, m_tile_size{tile_size} {}
    const char* name() const override {
        if (m_name.empty()) {
            m_name = ConvBiasImpl::algo_name<ConvBias::WinogradParam>(
                    m_matmul_algo->name(), {4, 6, m_tile_size});
        }
        return m_name.c_str();
    }
    MEGDNN_WINOGRAD_ALGO_FUN_DECLARE(AlgoDataType::FLOAT32);
    MEGDNN_DECL_ALGO_TYPE(ARM_COMMON_WINOGRAD_F63_4X4_FP32)
};

class ConvBiasImpl::AlgoFP32WinogradF54 final : public AlgoBase {
public:
    AlgoFP32WinogradF54(fallback::MatrixMulImpl::AlgoBase* matmul_algo,
                        uint32_t tile_size)
            : m_matmul_algo{matmul_algo}, m_tile_size{tile_size} {}
    const char* name() const override {
        if (m_name.empty()) {
            m_name = ConvBiasImpl::algo_name<ConvBias::WinogradParam>(
                    m_matmul_algo->name(), {1, 5, m_tile_size});
        }
        return m_name.c_str();
    }
    MEGDNN_WINOGRAD_ALGO_FUN_DECLARE(AlgoDataType::FLOAT32);
    MEGDNN_DECL_ALGO_TYPE(ARM_COMMON_WINOGRAD_F54_FP32)
};

class ConvBiasImpl::AlgoFP32WinogradF45 final : public AlgoBase {
public:
    AlgoFP32WinogradF45(fallback::MatrixMulImpl::AlgoBase* matmul_algo,
                        uint32_t tile_size)
            : m_matmul_algo{matmul_algo}, m_tile_size{tile_size} {}
    const char* name() const override {
        if (m_name.empty()) {
            m_name = ConvBiasImpl::algo_name<ConvBias::WinogradParam>(
                    m_matmul_algo->name(), {1, 4, m_tile_size});
        }
        return m_name.c_str();
    }
    MEGDNN_WINOGRAD_ALGO_FUN_DECLARE(AlgoDataType::FLOAT32);
    MEGDNN_DECL_ALGO_TYPE(ARM_COMMON_WINOGRAD_F45_FP32)
};

//===================== NCHW44 Winograd Support =====================//
class ConvBiasImpl::AlgoFP32WinogradF23_4x4_NCHW44 final : public AlgoBase {
public:
    AlgoFP32WinogradF23_4x4_NCHW44(
            fallback::MatrixMulImpl::AlgoBase* matmul_algo, uint32_t tile_size)
            : m_matmul_algo{matmul_algo}, m_tile_size{tile_size} {}
    const char* name() const override {
        if (m_name.empty()) {
            m_name = ConvBiasImpl::algo_name<ConvBias::WinogradParam>(
                    m_matmul_algo->name(), {4, 2, m_tile_size},
                    param::ConvBias::Format::NCHW44);
        }
        return m_name.c_str();
    }
    MEGDNN_WINOGRAD_ALGO_FUN_DECLARE(AlgoDataType::FLOAT32);
    MEGDNN_DECL_ALGO_TYPE(ARM_COMMON_WINOGRAD_F23_4X4_NCHW44_F32)
};

class ConvBiasImpl::AlgoFP32WinogradF63_4x4_NCHW44 final : public AlgoBase {
public:
    AlgoFP32WinogradF63_4x4_NCHW44(
            fallback::MatrixMulImpl::AlgoBase* matmul_algo, uint32_t tile_size)
            : m_matmul_algo{matmul_algo}, m_tile_size{tile_size} {}
    const char* name() const override {
        if (m_name.empty()) {
            m_name = ConvBiasImpl::algo_name<ConvBias::WinogradParam>(
                    m_matmul_algo->name(), {4, 6, m_tile_size},
                    param::ConvBias::Format::NCHW44);
        }
        return m_name.c_str();
    }
    MEGDNN_WINOGRAD_ALGO_FUN_DECLARE(AlgoDataType::FLOAT32);
    MEGDNN_DECL_ALGO_TYPE(ARM_COMMON_WINOGRAD_F63_4X4_NCHW44_F32)
};

class ConvBiasImpl::AlgoFP32WinogradF73_4x4_NCHW44 final : public AlgoBase {
public:
    AlgoFP32WinogradF73_4x4_NCHW44(
            fallback::MatrixMulImpl::AlgoBase* matmul_algo, uint32_t tile_size)
            : m_matmul_algo{matmul_algo}, m_tile_size{tile_size} {}
    const char* name() const override {
        if (m_name.empty()) {
            m_name = ConvBiasImpl::algo_name<ConvBias::WinogradParam>(
                    m_matmul_algo->name(), {4, 7, m_tile_size},
                    param::ConvBias::Format::NCHW44);
        }
        return m_name.c_str();
    }
    MEGDNN_WINOGRAD_ALGO_FUN_DECLARE(AlgoDataType::FLOAT32);
    MEGDNN_DECL_ALGO_TYPE(ARM_COMMON_WINOGRAD_F73_4X4_NCHW44_F32)
};
// ================================================================= //

class ConvBiasImpl::AlgoF32Direct final : public AlgoBase {
    SmallVector<NCBKern> get_kimpls(const NCBKernSizeParam& param) const;

public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "F32DIRECT"; }
    bool usable(const NCBKernSizeParam& param,
                AlgoSelectionStrategy algo_selection_strategy) const override;

    size_t get_workspace(const NCBKernSizeParam& param) const override;
    virtual SmallVector<NCBKern> dispatch_kerns(
            const NCBKernSizeParam& param) const override;
    ConvAlgoTypePack get_algo_type() const override {
        return {AlgoDataType::FLOAT32, AlgoCategory::DIRECT};
    }
    MEGDNN_DECL_ALGO_TYPE(ARM_COMMON_DIRECT_FP32)
};

class ConvBiasImpl::AlgoF32DirectStride1 final : public AlgoBase {
    SmallVector<NCBKern> get_kimpls(const NCBKernSizeParam& param) const;

public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "F32STRD1"; }
    bool usable(const NCBKernSizeParam& param,
                AlgoSelectionStrategy algo_selection_strategy) const override;

    size_t get_workspace(const NCBKernSizeParam& param) const override;
    virtual SmallVector<NCBKern> dispatch_kerns(
            const NCBKernSizeParam& param) const override;
    ConvAlgoTypePack get_algo_type() const override {
        return {AlgoDataType::FLOAT32, AlgoCategory::DIRECT};
    }
    MEGDNN_DECL_ALGO_TYPE(ARM_COMMON_DIRECT_STRD1_FP32)
};

class ConvBiasImpl::AlgoF32DirectStride2 final : public AlgoBase {
    SmallVector<NCBKern> get_kimpls(const NCBKernSizeParam& param) const;

public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "F32STRD2"; }
    bool usable(const NCBKernSizeParam& param,
                AlgoSelectionStrategy algo_selection_strategy) const override;

    size_t get_workspace(const NCBKernSizeParam& param) const override;
    virtual SmallVector<NCBKern> dispatch_kerns(
            const NCBKernSizeParam& param) const override;
    ConvAlgoTypePack get_algo_type() const override {
        return {AlgoDataType::FLOAT32, AlgoCategory::DIRECT};
    }
    MEGDNN_DECL_ALGO_TYPE(ARM_COMMON_DIRECT_STRD2_FP32)
};

class ConvBiasImpl::AlgoF32DirectNCHW44 final : public AlgoBase {
    SmallVector<NCBKern> get_kimpls(const NCBKernSizeParam& param) const;

public:
    AlgoF32DirectNCHW44() {}
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "F32_CONV_NCHW44_DIRECT"; }
    bool usable(const NCBKernSizeParam& param,
                AlgoSelectionStrategy algo_selection_strategy) const override;

    size_t get_workspace(const NCBKernSizeParam& param) const override;
    virtual SmallVector<NCBKern> dispatch_kerns(
            const NCBKernSizeParam& param) const override;
    ConvAlgoTypePack get_algo_type() const override {
        return {AlgoDataType::FLOAT32, AlgoCategory::DIRECT};
    }
    MEGDNN_DECL_ALGO_TYPE(ARM_COMMON_DIRECT_NCHW44_FP32)
};

class ConvBiasImpl::AlgoF32DirectNCHWNCHW44 final : public AlgoBase {
    SmallVector<NCBKern> get_kimpls(const NCBKernSizeParam& param) const;

public:
    AlgoF32DirectNCHWNCHW44() {}
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "F32_CONV_NCHW_NCHW44"; }
    bool usable(const NCBKernSizeParam& param,
                AlgoSelectionStrategy algo_selection_strategy) const override;

    size_t get_workspace(const NCBKernSizeParam& param) const override;
    virtual SmallVector<NCBKern> dispatch_kerns(
            const NCBKernSizeParam& param) const override;
    ConvAlgoTypePack get_algo_type() const override {
        return {AlgoDataType::FLOAT32, AlgoCategory::DIRECT};
    }
    MEGDNN_DECL_ALGO_TYPE(ARM_COMMON_DIRECT_NCHW_NCHW44_FP32)
};

class ConvBiasImpl::AlgoF32ChannelWiseNCHW44 final : public AlgoBase {
    SmallVector<NCBKern> get_kimpls(const NCBKernSizeParam& param) const;

public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "F32_CHANNEL_WISE_NCHW44"; }
    bool usable(const NCBKernSizeParam& param,
                AlgoSelectionStrategy algo_selection_strategy) const override;

    size_t get_workspace(const NCBKernSizeParam& param) const override;
    virtual SmallVector<NCBKern> dispatch_kerns(
            const NCBKernSizeParam& param) const override;
    ConvAlgoTypePack get_algo_type() const override {
        return {AlgoDataType::FLOAT32, AlgoCategory::DIRECT};
    }
    MEGDNN_DECL_ALGO_TYPE(ARM_COMMON_CHWNWISE_NCHW44_F32)
};

}  // namespace arm_common
}  // namespace megdnn

#undef MEGDNN_WINOGRAD_ALGO_FUN_DECLARE

// vim: syntax=cpp.doxygen
