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

namespace megdnn {
namespace arm_common {

class MatrixMulImpl::AlgoInt8x8x16 final : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "ARM_COMMON_INT8X8X16"; }
    bool usable(const KernSizeParam&) const override;
    size_t get_workspace(const KernSizeParam&) const override;
    kern_t get_kern(const KernSizeParam&) const override;
    void* type() const override { return sm_arm_common_algo_type; }
    PackMode packmode() const override { return PackMode::NO_PACK; }
};

class MatrixMulImpl::AlgoInt8x8x32Gemv : public AlgoBase {
public:
    bool is_reproducible() const override { return true; }
    const char* name() const override { return "ARM_COMMON_INT8X8X32_GEMV"; }
    bool usable(const KernSizeParam&) const override;
    bool preferred(const KernSizeParam&) const override;
    size_t get_workspace(const KernSizeParam&) const override { return 0; }
    kern_t get_kern(const KernSizeParam&) const override;
    void* type() const override { return sm_arm_common_algo_type; }
    AlgoSet algoset() const override { return AlgoSet::ALGO_TYPE_GEMV; }
    PackMode packmode() const override { return PackMode::NO_PACK; }
};

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
    void* type() const override { return sm_arm_common_algo_type; }
    AlgoSet algoset() const override { return AlgoSet::ALGO_TYPE_GEMV; }
    PackMode packmode() const override { return PackMode::NO_PACK; }
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
    void* type() const override { return sm_arm_common_algo_type; }
    AlgoSet algoset() const override { return AlgoSet::ALGO_TYPE_GEMV; }
    PackMode packmode() const override { return PackMode::NO_PACK; }
};
#endif
}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
