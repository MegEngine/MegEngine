/**
 * \file dnn/src/aarch64/matrix_mul/opr_impl.h
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
namespace aarch64 {

class MatrixMulImpl : public arm_common::MatrixMulImpl {
public:
    using arm_common::MatrixMulImpl::MatrixMulImpl;

    SmallVector<AlgoBase*> algo_pack() override;

private:
    class AlgoF32K8x12x1;   // Aarch64 F32 Kernel 8X12X1
    class AlgoF32K4x16x1;   // Aarch64 F32 Kernel 4x16x1
    class AlgoF32MK4_4x16;  // Aarch64 F32 Format MK4 block 16x4
    class AlgoF32Gemv;      // Aarch64 F32 Gemv
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    class AlgoF16K8x24x1;  // Aarch64 F16 Kernel 8x24x1
    class AlgoF16MK8_8x8;  // Aarch64 F16 Format MK8 block 16x8
#endif

#if __ARM_FEATURE_DOTPROD
    class AlgoInt8x8x32K8x12x4DotProd;  // Aarch64 Int8x8x32 Kernel
                                        // 8x12x4 DotProduct
    class AlgoInt8x8x32GemvDotProd;     // Aarch64 Int8x8x32 Gemv DotProduct
#else
    class AlgoInt8x8x32MK4_4x4x16;  // Aarch64 nchw44 Int8x8x32 Kernel 4x4x16
    class AlgoInt8x8x32K4x4x16;  // Aarch64 Int8x8x32 Kernel 4x4x16
    class AlgoInt8x8x32K8x8x8;   // Aarch64 Int8x8x32 Kernel 8x8x8
    class AlgoInt8x8x32Gemv;     // Aarch64 Int8x8x32 Gemv
#endif
    class AlgoInt8x8x16K8x8x8;   // Aarch64 Int8x8x16 Kernel 8x8x8
    class AlgoInt8x8x16K4x4x16;  // Aarch64 Int8x8x16 Kernel 4x4x16

    class AlgoInt16x16x32K12x8x1;  // Aarch64 Int16x16x32 Kernel 12x8x1
    class AlgoInt16x16x32MK8_8x8;  // Aarch64 Int16x16x32 Format MK8 block 8x8

#if __ARM_FEATURE_DOTPROD
    class AlgoQuint8K8x8x4DotProd;  // Aarch64 Quint8 Kernel
                                    // 8x8x4 DotProduct
    class AlgoQuint8GemvDotProd;    // Aarch64 Quint8 Gemv DotProduct
#else
    class AlgoQuint8K8x8x8;      // Aarch64 Quint8 Kernel 8x8x8
#endif

    class AlgoPack;
};

}  // namespace aarch64
}  // namespace megdnn

// vim: syntax=cpp.doxygen
