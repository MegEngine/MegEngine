/**
 * \file dnn/src/fallback/matrix_mul/algos.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include <type_traits>
#include "src/fallback/matrix_mul/opr_impl.h"
#include "src/fallback/matrix_mul/gemm_common.h"
#include "src/common/algo_base.h"

namespace megdnn {
namespace fallback {

class MatrixMulImpl::AlgoF32K8x12x1 final : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "FB_F32_K8X12X1"; }
    bool usable(const KernSizeParam&) const override;
    size_t get_workspace(const KernSizeParam&) const override;
    kern_t get_kern(const KernSizeParam&) const override;
    MEGDNN_DECL_ALGO_TYPE(FB_F32K8x12x1)
    MEGDNN_REG_GEMM_FUNC_FOR_IM2COL();
};

class MatrixMulImpl::AlgoGemv final : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "FB_GEMV"; }
    bool usable(const KernSizeParam&) const override;
    bool preferred(const KernSizeParam&) const override;
    size_t get_workspace(const KernSizeParam&) const override { return 0; }
    kern_t get_kern(const KernSizeParam&) const override;
    AlgoSet algoset() const override { return AlgoSet::ALGO_TYPE_GEMV; }
    PackMode packmode() const override { return PackMode::NO_PACK; }
    MEGDNN_DECL_ALGO_TYPE(FB_GEMV)
    MEGDNN_OVERRIDE_MATMUL_DESC(
            8, 16, 1, 4,
            static_cast<AlgoDataType>(
                    static_cast<uint32_t>(AlgoDataType::FLOAT16) |
                    static_cast<uint32_t>(AlgoDataType::FLOAT32) |
                    static_cast<uint32_t>(AlgoDataType::INT8X8X16) |
                    static_cast<uint32_t>(AlgoDataType::QINT8X8X32) |
                    static_cast<uint32_t>(AlgoDataType::QUINT8X8X32)),
            DEFAULT)
};

}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
