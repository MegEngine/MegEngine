/**
 * \file dnn/src/arm_common/matrix_mul/opr_impl.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "src/common/utils.h"
#include "src/fallback/matrix_mul/opr_impl.h"

namespace megdnn {
namespace arm_common {

class MatrixMulImpl : public fallback::MatrixMulImpl {
public:
    using fallback::MatrixMulImpl::MatrixMulImpl;

    bool is_thread_safe() const override { return true; }

    SmallVector<AlgoBase*> algo_pack() override;

protected:
    static void* const sm_arm_common_algo_type;
    class AlgoInt8x8x32Gemv;  // Arm_common Int 8x8x32 Gemv
    class AlgoF32Gemv;        // Arm_common F32 Gemv
    class AlgoGevm;        // Arm_common Gemv(support int8 and fp32)
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    class AlgoF16Gemv;
#endif
    class AlgoInt8x8x16;  // Arm_common Int 8x8x16
    class AlgoPack;
};

}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
