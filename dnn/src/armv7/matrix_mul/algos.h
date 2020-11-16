/**
 * \file dnn/src/armv7/matrix_mul/algos.h
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

#include "src/arm_common/matrix_mul/algos.h"
#include "src/armv7/matrix_mul/opr_impl.h"
#include "src/fallback/matrix_mul/gemm_common.h"

namespace megdnn {
namespace armv7 {

class MatrixMulImpl::AlgoF32 final : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "ARMV7_F32"; }
    bool usable(const KernSizeParam&) const override;
    size_t get_workspace(const KernSizeParam&) const override;
    kern_t get_kern(const KernSizeParam&) const override;
    MEGDNN_REG_GEMM_FUNC_FOR_IM2COL();
    MEGDNN_DECL_ALGO_TYPE(ARMV7_F32)
};

class MatrixMulImpl::AlgoF32MK4Pack4x12 final : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "ARMV7_F32_MK4_PACK_4X12"; }
    bool usable(const KernSizeParam&) const override;
    size_t get_workspace(const KernSizeParam&) const override;
    kern_t get_kern(const KernSizeParam&) const override;
    MEGDNN_REG_GEMM_FUNC_FOR_IM2COL();
    MEGDNN_DECL_ALGO_TYPE(ARMV7_F32_MK4_PACK_4X12)
};

class MatrixMulImpl::AlgoF32MK4_4x8 final : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "ARMV7_F32_MK4_4x8"; }
    bool usable(const KernSizeParam&) const override;
    size_t get_workspace(const KernSizeParam&) const override;
    kern_t get_kern(const KernSizeParam&) const override;
    PackMode packmode() const override { return PackMode::NO_PACK; }
    MEGDNN_OVERRIDE_MATMUL_DESC(4, 8, 4, 4, AlgoDataType::FLOAT32, MK4)
    MEGDNN_DECL_ALGO_TYPE(ARMV7_F32_MK4_4x8)
};

#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
class MatrixMulImpl::AlgoF16K4x16x1 final : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "AARCH32_F16_K4X16X1"; }
    bool usable(const KernSizeParam&) const override;
    size_t get_workspace(const KernSizeParam&) const override;
    kern_t get_kern(const KernSizeParam&) const override;
    MEGDNN_REG_GEMM_FUNC_FOR_IM2COL();
    MEGDNN_DECL_ALGO_TYPE(ARMV7_F16_K4X16X1)
};
class MatrixMulImpl::AlgoF16MK8_4x8 final : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "AARCH32_F16_MK8_4X8"; }
    bool usable(const KernSizeParam&) const override;
    size_t get_workspace(const KernSizeParam&) const override;
    kern_t get_kern(const KernSizeParam&) const override;
    PackMode packmode() const override { return PackMode::NO_PACK; }
    MEGDNN_OVERRIDE_MATMUL_DESC(4, 8, 8, 2, AlgoDataType::FLOAT16, MK8)
    MEGDNN_DECL_ALGO_TYPE(ARMV7_F16_MK8_4X8)
};
#endif
#if __ARM_FEATURE_DOTPROD
class MatrixMulImpl::AlgoInt8x8x32K6x8x4 final : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "AARCH32_INT8_K6X8X4"; }
    bool usable(const KernSizeParam&) const override;
    size_t get_workspace(const KernSizeParam&) const override;
    kern_t get_kern(const KernSizeParam&) const override;
    MEGDNN_REG_GEMM_FUNC_FOR_IM2COL();
    MEGDNN_DECL_ALGO_TYPE(ARMV7_INT8_K6X8X4)
};

class MatrixMulImpl::AlgoQuint8DotK4x8x4 final : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "AARCH32_QUINT8_K4X8X4"; }
    bool usable(const KernSizeParam&) const override;
    size_t get_workspace(const KernSizeParam&) const override;
    kern_t get_kern(const KernSizeParam&) const override;
    MEGDNN_REG_GEMM_FUNC_FOR_IM2COL();
    MEGDNN_DECL_ALGO_TYPE(ARMV7_QUINT8_K4X8X4)
};

class MatrixMulImpl::AlgoInt8x8x32MK4_8x4x4DotProd final : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override {
        return "AARCH32_INT8_MK4_8X4X4_DOTPROD";
    }
    bool usable(const KernSizeParam&) const override;
    size_t get_workspace(const KernSizeParam&) const override;
    kern_t get_kern(const KernSizeParam&) const override;
    MEGDNN_REG_GEMM_FUNC_FOR_IM2COL();
    MEGDNN_DECL_ALGO_TYPE(ARMV7_INT8_MK4_8X4X4_DOTPROD)
};
#endif

class MatrixMulImpl::AlgoF32Gemv final
        : public arm_common::MatrixMulImpl::AlgoF32Gemv {
public:
    AlgoF32Gemv() : arm_common::MatrixMulImpl::AlgoF32Gemv() {
        m_handle_type = Handle::HandleType::ARMV7;
    }
    MEGDNN_DECL_ALGO_TYPE(ARMV7_F32_GEMV)
};

class MatrixMulImpl::AlgoInt8x8x32K4x2x16 final : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "ARMV7_INT8X8X32_K4X2X16"; }
    bool usable(const KernSizeParam&) const override;
    bool preferred(const KernSizeParam&) const override;
    size_t get_workspace(const KernSizeParam&) const override;
    kern_t get_kern(const KernSizeParam&) const override;
    MEGDNN_REG_GEMM_FUNC_FOR_IM2COL();
    MEGDNN_DECL_ALGO_TYPE(ARMV7_INT8X8X32_K4X2X16)
};

class MatrixMulImpl::AlgoInt8x8x32K4x8x8 final : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "ARMV7_INT8X8X32_K4X8X8"; }
    bool usable(const KernSizeParam&) const override;
    bool preferred(const KernSizeParam&) const override;
    size_t get_workspace(const KernSizeParam&) const override;
    kern_t get_kern(const KernSizeParam&) const override;
    MEGDNN_REG_GEMM_FUNC_FOR_IM2COL();
    MEGDNN_DECL_ALGO_TYPE(ARMV7_INT8X8X32_K4X8X8)
};

class MatrixMulImpl::AlgoQuint8K4x8x8 final : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "ARMV7_QUINT8_K4X8X8"; }
    bool usable(const KernSizeParam&) const override;
    size_t get_workspace(const KernSizeParam&) const override;
    kern_t get_kern(const KernSizeParam&) const override;
    MEGDNN_REG_GEMM_FUNC_FOR_IM2COL();
    MEGDNN_DECL_ALGO_TYPE(ARMV7_QUINT8_K4X8X8)
};

class MatrixMulImpl::AlgoInt8x8x16K4x2x16 final : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "ARMV7_INT8X8X16_K4X2X16"; }
    bool usable(const KernSizeParam&) const override;
    bool preferred(const KernSizeParam&) const override;
    size_t get_workspace(const KernSizeParam&) const override;
    kern_t get_kern(const KernSizeParam&) const override;
    MEGDNN_REG_GEMM_FUNC_FOR_IM2COL();
    MEGDNN_DECL_ALGO_TYPE(ARMV7_INT8X8X16_K4X2X16)
};

class MatrixMulImpl::AlgoInt8x8x16K4x8x8 final : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "ARMV7_INT8X8X16_K4X8X8"; }
    bool usable(const KernSizeParam&) const override;
    bool preferred(const KernSizeParam&) const override;
    size_t get_workspace(const KernSizeParam&) const override;
    kern_t get_kern(const KernSizeParam&) const override;
    MEGDNN_REG_GEMM_FUNC_FOR_IM2COL();
    MEGDNN_DECL_ALGO_TYPE(ARMV7_INT8X8X16_K4X8X8)
};

class MatrixMulImpl::AlgoInt8x8x16MK4_8x8x4 final : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "ARMV7_INT8X8X16_MK4_K8X8X4"; }
    bool usable(const KernSizeParam&) const override;
    bool preferred(const KernSizeParam&) const override;
    size_t get_workspace(const KernSizeParam&) const override;
    kern_t get_kern(const KernSizeParam&) const override;
    MEGDNN_REG_GEMM_FUNC_FOR_IM2COL();
    MEGDNN_DECL_ALGO_TYPE(ARMV7_INT8X8X16_MK4_K8X8X4)
};

class MatrixMulImpl::AlgoInt16x16x32K12x4x1 final : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "ARMV7_INT16X16X32_K12X4X1"; }
    bool usable(const KernSizeParam&) const override;
    bool preferred(const KernSizeParam&) const override;
    size_t get_workspace(const KernSizeParam&) const override;
    kern_t get_kern(const KernSizeParam&) const override;
    MEGDNN_REG_GEMM_FUNC_FOR_IM2COL();
    MEGDNN_DECL_ALGO_TYPE(ARMV7_INT16X16X32_K12X4X1)
};

class MatrixMulImpl::AlgoInt16x16x32MK8_4x8 final : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "ARMV7_INT16X16X32_MK8_4X8"; }
    bool usable(const KernSizeParam&) const override;
    size_t get_workspace(const KernSizeParam&) const override;
    kern_t get_kern(const KernSizeParam&) const override;
    PackMode packmode() const override { return PackMode::NO_PACK; }
    MEGDNN_OVERRIDE_MATMUL_DESC(4, 8, 8, 2, AlgoDataType::INT16X16X32, MK8)
    MEGDNN_DECL_ALGO_TYPE(ARMV7_INT16X16X32_MK8_4X8)
};

class MatrixMulImpl::AlgoInt8x8x32MK4_4x2x16 final : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "ARMV7_INT8X8X32_MK4_4X2X16"; }
    bool usable(const KernSizeParam&) const override;
    bool preferred(const KernSizeParam&) const override;
    size_t get_workspace(const KernSizeParam&) const override;
    kern_t get_kern(const KernSizeParam&) const override;
    MEGDNN_REG_GEMM_FUNC_FOR_IM2COL();
    MEGDNN_DECL_ALGO_TYPE(ARMV7_INT8X8X32_MK4_4X2X16)
};

}  // namespace armv7
}  // namespace megdnn

// vim: syntax=cpp.doxygen
