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
#include "src/common/algo_base.h"

namespace megdnn {
namespace arm_common {

class MatrixMulImpl : public fallback::MatrixMulImpl {
public:
    using fallback::MatrixMulImpl::MatrixMulImpl;
    bool is_thread_safe() const override { return true; }

    class AlgoBase : public fallback::MatrixMulImpl::AlgoBase {
        public:
            AlgoBase() : fallback::MatrixMulImpl::AlgoBase() {
                m_handle_type = Handle::HandleType::ARM_COMMON;
            }
    };

    SmallVector<fallback::MatrixMulImpl::AlgoBase*> get_all_packed_algo()
            override;

    MEGDNN_FB_DECL_GET_ALGO_FROM_DESC(MatrixMulImpl);

protected:
    class AlgoF32Gemv;               // Arm_common F32 Gemv
    class AlgoF32GemvMK4;         // Arm_common F32 Gemv NCHW44
    class AlgoInt8x8x32Gemv;         // Arm_common Int8x8x32 Gemv
    class AlgoInt8x8x32GemvMK4;   // Arm_common Int8x8x32 Gemv NCHW44
    class AlgoGevm;                  // Arm_common Gevm(support int8 and fp32)
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    class AlgoF16Gemv;
#endif
#if __ARM_FEATURE_DOTPROD
    class AlgoInt8x8x32GemvMK4Dot;// Arm_common Int8x8x32 Gemv NCHW44_DOT
#endif
    class AlgoInt8x8x16;  // Arm_common Int 8x8x16
    class AlgoPack;

public:
    static const AlgoPack& algo_pack();
};

}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
