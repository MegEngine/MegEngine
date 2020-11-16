/**
 * \file dnn/src/arm_common/matrix_mul/algos.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "src/arm_common/matrix_mul/opr_impl.h"
#include "src/fallback/matrix_mul/gemm_common.h"

namespace megdnn {
namespace arm_common {

class MatrixMulImpl::AlgoInt8x8x16 final : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "ARM_COMMON_INT8X8X16"; }
    bool usable(const KernSizeParam&) const override;
    size_t get_workspace(const KernSizeParam&) const override;
    kern_t get_kern(const KernSizeParam&) const override;
    PackMode packmode() const override { return PackMode::NO_PACK; }
    MEGDNN_OVERRIDE_MATMUL_DESC(8, 16, 1, 4, AlgoDataType::INT8X8X16, DEFAULT)
    MEGDNN_DECL_ALGO_TYPE(ARM_COMMON_INT8X8X16)
};

class MatrixMulImpl::AlgoInt8x8x32Gemv : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "ARM_COMMON_INT8X8X32_GEMV"; }
    bool usable(const KernSizeParam&) const override;
    bool preferred(const KernSizeParam&) const override;
    size_t get_workspace(const KernSizeParam&) const override { return 0; }
    kern_t get_kern(const KernSizeParam&) const override;
    AlgoSet algoset() const override { return AlgoSet::ALGO_TYPE_GEMV; }
    PackMode packmode() const override { return PackMode::NO_PACK; }
    MEGDNN_OVERRIDE_MATMUL_DESC(8, 16, 1, 2, AlgoDataType::QINT8X8X32, DEFAULT)
    MEGDNN_DECL_ALGO_TYPE(ARM_COMMON_INT8X8X32_GEMV)
};

class MatrixMulImpl::AlgoInt8x8x32GemvMK4 : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "ARM_COMMON_INT8X8X32_GEMV_MK4"; }
    bool usable(const KernSizeParam&) const override;
    bool preferred(const KernSizeParam&) const override;
    size_t get_workspace(const KernSizeParam&) const override { return 0; }
    kern_t get_kern(const KernSizeParam&) const override;
    AlgoSet algoset() const override { return AlgoSet::ALGO_TYPE_GEMV; }
    PackMode packmode() const override { return PackMode::NO_PACK; }
    MEGDNN_OVERRIDE_MATMUL_DESC(8, 16, 1, 2, AlgoDataType::QINT8X8X32, MK4)
    MEGDNN_DECL_ALGO_TYPE(ARM_COMMON_INT8X8X32_GEMV_MK4)
};

#if __ARM_FEATURE_DOTPROD
class MatrixMulImpl::AlgoInt8x8x32GemvMK4Dot : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "ARM_COMMON_INT8X8X32_GEMV_MK4_DOT"; }
    bool usable(const KernSizeParam&) const override;
    bool preferred(const KernSizeParam&) const override;
    size_t get_workspace(const KernSizeParam&) const override { return 0; }
    kern_t get_kern(const KernSizeParam&) const override;
    AlgoSet algoset() const override { return AlgoSet::ALGO_TYPE_GEMV; }
    PackMode packmode() const override { return PackMode::NO_PACK; }
    MEGDNN_OVERRIDE_MATMUL_DESC(8, 16, 1, 2, AlgoDataType::QINT8X8X32, MK4_DOT)
    MEGDNN_DECL_ALGO_TYPE(ARM_COMMON_INT8X8X32_GEMV_MK4_DOT)
};
#endif

class MatrixMulImpl::AlgoF32Gemv : public AlgoBase {
protected:
    ~AlgoF32Gemv() = default;

public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "ARM_COMMON_F32_GEMV"; }
    bool usable(const KernSizeParam&) const override;
    bool preferred(const KernSizeParam&) const override;
    size_t get_workspace(const KernSizeParam&) const override { return 0; }
    kern_t get_kern(const KernSizeParam&) const override;
    AlgoSet algoset() const override { return AlgoSet::ALGO_TYPE_GEMV; }
    PackMode packmode() const override { return PackMode::NO_PACK; }
    MEGDNN_OVERRIDE_MATMUL_DESC(8, 16, 1, 4, AlgoDataType::FLOAT32, DEFAULT)
};

class MatrixMulImpl::AlgoF32GemvMK4 : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "ARM_COMMON_F32_GEMV_MK4"; }
    bool usable(const KernSizeParam&) const override;
    bool preferred(const KernSizeParam&) const override;
    size_t get_workspace(const KernSizeParam&) const override { return 0; }
    kern_t get_kern(const KernSizeParam&) const override;
    AlgoSet algoset() const override { return AlgoSet::ALGO_TYPE_GEMV; }
    PackMode packmode() const override { return PackMode::NO_PACK; }
    MEGDNN_OVERRIDE_MATMUL_DESC(4, 1, 1, 4, AlgoDataType::FLOAT32, MK4)
    MEGDNN_DECL_ALGO_TYPE(ARM_COMMON_F32_GEMV_MK4)
};

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
class MatrixMulImpl::AlgoF16Gemv : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "ARM_COMMON_F16_GEMV"; }
    bool usable(const KernSizeParam&) const override;
    bool preferred(const KernSizeParam&) const override;
    size_t get_workspace(const KernSizeParam&) const override { return 0; }
    kern_t get_kern(const KernSizeParam&) const override;
    AlgoSet algoset() const override { return AlgoSet::ALGO_TYPE_GEMV; }
    PackMode packmode() const override { return PackMode::NO_PACK; }
    MEGDNN_OVERRIDE_MATMUL_DESC(8, 16, 1, 2, AlgoDataType::FLOAT16, DEFAULT)
    MEGDNN_DECL_ALGO_TYPE(ARM_COMMON_F16_GEMV)
};
#endif

class MatrixMulImpl::AlgoGevm : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "ARM_COMMON_GEVM"; }
    bool usable(const KernSizeParam&) const override;
    bool preferred(const KernSizeParam&) const override;
    size_t get_workspace(const KernSizeParam&) const override { return 0; }
    kern_t get_kern(const KernSizeParam&) const override;
    AlgoSet algoset() const override { return AlgoSet::ALGO_TYPE_GEMV; }
    PackMode packmode() const override { return PackMode::NO_PACK; }
    MEGDNN_OVERRIDE_MATMUL_DESC(
            1, 1, 1, 4,
            static_cast<AlgoDataType>(
                    static_cast<uint32_t>(AlgoDataType::FLOAT16) |
                    static_cast<uint32_t>(AlgoDataType::FLOAT32) |
                    static_cast<uint32_t>(AlgoDataType::QINT8X8X32)),
            DEFAULT)
    MEGDNN_DECL_ALGO_TYPE(ARM_COMMON_GEVM)
};

}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
