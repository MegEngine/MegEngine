/**
 * \file dnn/src/aarch64/matrix_mul/algos.h
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

#include "src/aarch64/matrix_mul/opr_impl.h"
#include "src/arm_common/matrix_mul/algos.h"
#include "src/fallback/matrix_mul/gemm_common.h"

namespace megdnn {
namespace aarch64 {

class MatrixMulImpl::AlgoF32K8x12x1 final : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "AARCH64_F32K8X12X1"; }
    bool usable(const KernSizeParam&) const override;
    size_t get_workspace(const KernSizeParam&) const override;
    kern_t get_kern(const KernSizeParam&) const override;
    MEGDNN_REG_GEMM_FUNC_FOR_IM2COL();
    MEGDNN_DECL_ALGO_TYPE(AARCH64_F32_K8X12X1)
};

class MatrixMulImpl::AlgoF32MK4_8x12x1 final : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "AARCH64_F32_MK4_K8X12X1"; }
    bool usable(const KernSizeParam&) const override;
    size_t get_workspace(const KernSizeParam&) const override;
    kern_t get_kern(const KernSizeParam&) const override;
    MEGDNN_REG_GEMM_FUNC_FOR_IM2COL();
    MEGDNN_DECL_ALGO_TYPE(AARCH64_F32_MK4_K8X12X1)
};

class MatrixMulImpl::AlgoF32K4x16x1 final : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "AARCH64_F32K4X16X1"; }
    bool usable(const KernSizeParam&) const override;
    size_t get_workspace(const KernSizeParam&) const override;
    kern_t get_kern(const KernSizeParam&) const override;
    MEGDNN_REG_GEMM_FUNC_FOR_IM2COL();
    MEGDNN_DECL_ALGO_TYPE(AARCH64_F32_K4X16X1)
};

class MatrixMulImpl::AlgoF32MK4_4x16 final : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "AARCH64_F32_MK4_4x16"; }
    bool usable(const KernSizeParam&) const override;
    size_t get_workspace(const KernSizeParam&) const override;
    kern_t get_kern(const KernSizeParam&) const override;
    PackMode packmode() const override { return PackMode::NO_PACK; }
    MEGDNN_OVERRIDE_MATMUL_DESC(4, 16, 4, 4, AlgoDataType::FLOAT32, MK4)
    MEGDNN_DECL_ALGO_TYPE(AARCH64_F32_MK4_4x16)
};

class MatrixMulImpl::AlgoF32Gemv final
        : public arm_common::MatrixMulImpl::AlgoF32Gemv {
public:
    AlgoF32Gemv() : arm_common::MatrixMulImpl::AlgoF32Gemv() {
        m_handle_type = Handle::HandleType::AARCH64;
    }
    MEGDNN_DECL_ALGO_TYPE(AARCH64_F32_GEMV)
};

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
class MatrixMulImpl::AlgoF16K8x24x1 final : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "AARCH64_F16_K8X24X1"; }
    bool usable(const KernSizeParam&) const override;
    size_t get_workspace(const KernSizeParam&) const override;
    kern_t get_kern(const KernSizeParam&) const override;
    MEGDNN_REG_GEMM_FUNC_FOR_IM2COL();
    MEGDNN_DECL_ALGO_TYPE(AARCH64_F16_K8X24X1)
};

class MatrixMulImpl::AlgoF16MK8_8x8 final : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "AARCH64_F16_MK8_8X8"; }
    bool usable(const KernSizeParam&) const override;
    size_t get_workspace(const KernSizeParam&) const override;
    kern_t get_kern(const KernSizeParam&) const override;
    PackMode packmode() const override { return PackMode::NO_PACK; }
    MEGDNN_OVERRIDE_MATMUL_DESC(8, 8, 8, 2, AlgoDataType::FLOAT16, MK8)
    MEGDNN_DECL_ALGO_TYPE(AARCH64_F16_MK8_8X8)
};

#endif

#if __ARM_FEATURE_DOTPROD
class MatrixMulImpl::AlgoInt8x8x32K8x12x4DotProd final : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override {
        return "AARCH64_INT8X8X32_K8X12X4_DOTPROD";
    }
    bool usable(const KernSizeParam&) const override;
    size_t get_workspace(const KernSizeParam&) const override;
    kern_t get_kern(const KernSizeParam&) const override;
    MEGDNN_REG_GEMM_FUNC_FOR_IM2COL();
    MEGDNN_DECL_ALGO_TYPE(AARCH64_INT8X8X32_K8X12X4_DOTPROD)
};

class MatrixMulImpl::AlgoInt8x8x32MK4_8x12x4DotProd final : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override {
        return "AARCH64_INT8X8X32_MK4_8X12X4_DOTPROD";
    }
    bool usable(const KernSizeParam&) const override;
    size_t get_workspace(const KernSizeParam&) const override;
    kern_t get_kern(const KernSizeParam&) const override;
    MEGDNN_REG_GEMM_FUNC_FOR_IM2COL();
    MEGDNN_DECL_ALGO_TYPE(AARCH64_INT8X8X32_MK4_8X12X4_DOTPROD)
};
#else

class MatrixMulImpl::AlgoInt8x8x32MK4_4x4x16 final : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "AARCH64_INT8X8X32_MK4_4X4X16"; }
    bool usable(const KernSizeParam&) const override;
    bool preferred(const KernSizeParam&) const override;
    size_t get_workspace(const KernSizeParam&) const override;
    kern_t get_kern(const KernSizeParam&) const override;
    PackMode packmode() const override { return PackMode::DEFAULT; }

    MEGDNN_REG_GEMM_FUNC_FOR_IM2COL();
    MEGDNN_DECL_ALGO_TYPE(AARCH64_INT8X8X32_MK4_4X4X16)
};

class MatrixMulImpl::AlgoInt8x8x32K4x4x16 final : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "AARCH64_INT8X8X32_K4X4X16"; }
    bool usable(const KernSizeParam&) const override;
    bool preferred(const KernSizeParam&) const override;
    size_t get_workspace(const KernSizeParam&) const override;
    kern_t get_kern(const KernSizeParam&) const override;

    MEGDNN_REG_GEMM_FUNC_FOR_IM2COL();
    MEGDNN_DECL_ALGO_TYPE(AARCH64_INT8X8X32_K4X4X16)
};

class MatrixMulImpl::AlgoInt8x8x32K8x8x8 final : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "AARCH64_INT8X8X32_K8X8X8"; }
    bool usable(const KernSizeParam&) const override;
    bool preferred(const KernSizeParam&) const override;
    size_t get_workspace(const KernSizeParam&) const override;
    kern_t get_kern(const KernSizeParam&) const override;
    MEGDNN_REG_GEMM_FUNC_FOR_IM2COL();
    MEGDNN_DECL_ALGO_TYPE(AARCH64_INT8X8X32_K8X8X8)
};
#endif

class MatrixMulImpl::AlgoInt8x8x16K8x8x8 final : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "AARCH64_INT8X8X16_K8X8X8"; }
    bool usable(const KernSizeParam&) const override;
    bool preferred(const KernSizeParam&) const override;
    size_t get_workspace(const KernSizeParam&) const override;
    kern_t get_kern(const KernSizeParam&) const override;

    MEGDNN_REG_GEMM_FUNC_FOR_IM2COL();
    MEGDNN_DECL_ALGO_TYPE(AARCH64_INT8X8X16_K8X8X8)
};

class MatrixMulImpl::AlgoInt8x8x16K4x4x16 final : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "AARCH64_INT8X8X16_K4X4X16"; }
    bool usable(const KernSizeParam&) const override;
    bool preferred(const KernSizeParam&) const override;
    size_t get_workspace(const KernSizeParam&) const override;
    kern_t get_kern(const KernSizeParam&) const override;
    MEGDNN_REG_GEMM_FUNC_FOR_IM2COL();
    MEGDNN_DECL_ALGO_TYPE(AARCH64_INT8X8X16_K4X4X16)
};

class MatrixMulImpl::AlgoInt4x4x16K8x8x8 final : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "AARCH64_INT4X4X16_K8X8X8"; }
    bool usable(const KernSizeParam&) const override;
    bool preferred(const KernSizeParam&) const override;
    size_t get_workspace(const KernSizeParam&) const override;
    kern_t get_kern(const KernSizeParam&) const override;
    PackMode packmode() const override { return PackMode::DEFAULT; }
    MEGDNN_REG_GEMM_FUNC_FOR_IM2COL();
    MEGDNN_DECL_ALGO_TYPE(AARCH64_INT4X4X16_K8X8X8)
};

class MatrixMulImpl::AlgoInt8x8x16MK4_16x12x4 final : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override {
        return "AARCH64_INT8X8X16_MK4_16X12X4";
    }
    bool usable(const KernSizeParam&) const override;
    bool preferred(const KernSizeParam&) const override;
    size_t get_workspace(const KernSizeParam&) const override;
    kern_t get_kern(const KernSizeParam&) const override;
    PackMode packmode() const override { return PackMode::DEFAULT; }

    MEGDNN_REG_GEMM_FUNC_FOR_IM2COL();
    MEGDNN_DECL_ALGO_TYPE(AARCH64_INT8X8X16_MK4_16X12X4)
};

class MatrixMulImpl::AlgoInt8x8x16MK4_K8x8x8 final : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override {
        return "AARCH64_INT8X8X16_MK4_K8X8X8";
    }
    bool usable(const KernSizeParam&) const override;
    bool preferred(const KernSizeParam&) const override;
    size_t get_workspace(const KernSizeParam&) const override;
    kern_t get_kern(const KernSizeParam&) const override;
    PackMode packmode() const override { return PackMode::DEFAULT; }

    MEGDNN_REG_GEMM_FUNC_FOR_IM2COL();
    MEGDNN_DECL_ALGO_TYPE(AARCH64_INT8X8X16_MK4_K8X8X8)
};

class MatrixMulImpl::AlgoInt8x8x16MK4_4x4x8 final : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "AARCH64_INT8X8X16_MK4_4X4X8"; }
    bool usable(const KernSizeParam&) const override;
    bool preferred(const KernSizeParam&) const override;
    size_t get_workspace(const KernSizeParam&) const override;
    kern_t get_kern(const KernSizeParam&) const override;
    PackMode packmode() const override { return PackMode::DEFAULT; }

    MEGDNN_REG_GEMM_FUNC_FOR_IM2COL();
    MEGDNN_DECL_ALGO_TYPE(AARCH64_INT8X8X16_MK4_4X4X8)
};

class MatrixMulImpl::AlgoInt16x16x32K12x8x1 final : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "AARCH64_INT16X16X32_K12X8X1"; }
    bool usable(const KernSizeParam&) const override;
    bool preferred(const KernSizeParam&) const override;
    size_t get_workspace(const KernSizeParam&) const override;
    kern_t get_kern(const KernSizeParam&) const override;
    MEGDNN_REG_GEMM_FUNC_FOR_IM2COL();
    MEGDNN_DECL_ALGO_TYPE(AARCH64_INT16X16X32_K12X8X1)
};

class MatrixMulImpl::AlgoInt16x16x32MK8_8x8 final : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "AARCH64_INT16X16X32_MK8_8X8"; }
    bool usable(const KernSizeParam&) const override;
    size_t get_workspace(const KernSizeParam&) const override;
    kern_t get_kern(const KernSizeParam&) const override;
    PackMode packmode() const override { return PackMode::NO_PACK; }
    MEGDNN_OVERRIDE_MATMUL_DESC(8, 8, 8, 2, AlgoDataType::INT16X16X32, MK8)
    MEGDNN_DECL_ALGO_TYPE(AARCH64_INT16X16X32_MK8_8X8)
};

#if __ARM_FEATURE_DOTPROD
class MatrixMulImpl::AlgoQuint8K8x8x4DotProd final : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override {
        return "AARCH64_QUINT8_K8X8X4_DOTPROD";
    }
    bool usable(const KernSizeParam&) const override;
    size_t get_workspace(const KernSizeParam&) const override;
    kern_t get_kern(const KernSizeParam&) const override;
    MEGDNN_REG_GEMM_FUNC_FOR_IM2COL();
    MEGDNN_DECL_ALGO_TYPE(AARCH64_QUINT8_K8X8X4_DOTPROD)
};

class MatrixMulImpl::AlgoQuint8GemvDotProd final : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "AARCH64_QUINT8_GEMV_DOTPROD"; }
    bool usable(const KernSizeParam&) const override;
    bool preferred(const KernSizeParam&) const override;
    size_t get_workspace(const KernSizeParam&) const override { return 0; }
    kern_t get_kern(const KernSizeParam&) const override;
    AlgoSet algoset() const override { return AlgoSet::ALGO_TYPE_GEMV; }
    PackMode packmode() const override { return PackMode::NO_PACK; }
    MEGDNN_OVERRIDE_MATMUL_DESC(8, 16, 1, 2, AlgoDataType::QUINT8X8X32, DEFAULT)
    MEGDNN_DECL_ALGO_TYPE(AARCH64_QUINT8_GEMV_DOTPROD)
};
#else

class MatrixMulImpl::AlgoQuint8K8x8x8 final : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "AARCH64_QUINT8_K8X8X8"; }
    bool usable(const KernSizeParam&) const override;
    size_t get_workspace(const KernSizeParam&) const override;
    kern_t get_kern(const KernSizeParam&) const override;
    MEGDNN_REG_GEMM_FUNC_FOR_IM2COL();
    MEGDNN_DECL_ALGO_TYPE(AARCH64_QUINT8_K8X8X8)
};
#endif

}  // namespace aarch64
}  // namespace megdnn

// vim: syntax=cpp.doxygen
