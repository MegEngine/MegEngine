/**
 * \file dnn/src/aarch64/matrix_mul/opr_impl.h
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
#include "src/arm_common/matrix_mul/opr_impl.h"

namespace megdnn {
namespace aarch64 {

class MatrixMulImpl : public arm_common::MatrixMulImpl {
public:
    using arm_common::MatrixMulImpl::MatrixMulImpl;
    class AlgoBase : public arm_common::MatrixMulImpl::AlgoBase {
    public:
        AlgoBase() : arm_common::MatrixMulImpl::AlgoBase() {
            m_handle_type = Handle::HandleType::AARCH64;
        }
    };

    SmallVector<fallback::MatrixMulImpl::AlgoBase*> get_all_packed_algo()
            override;

    MEGDNN_FB_DECL_GET_ALGO_FROM_DESC(MatrixMulImpl);

private:
    class AlgoF32K8x12x1;     // Aarch64 F32 Kernel 8X12X1
    class AlgoF32MK4_8x12x1;  // Aarch64 F32 Kernel MK4 8x12x1
    class AlgoF32K4x16x1;     // Aarch64 F32 Kernel 4x16x1
    class AlgoF32MK4_4x16;    // Aarch64 F32 Format MK4 block 16x4
    class AlgoF32Gemv;        // Aarch64 F32 Gemv
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    class AlgoF16K8x24x1;  // Aarch64 F16 Kernel 8x24x1
    class AlgoF16MK8_8x8;  // Aarch64 F16 Format MK8 block 16x8
#endif

#if __ARM_FEATURE_DOTPROD
    class AlgoInt8x8x32K8x12x4DotProd;     // Aarch64 Int8x8x32 Kernel
                                           // 8x12x4 DotProduct
    class AlgoInt8x8x32MK4_8x12x4DotProd;  // Aarch64 nchw44 Int8x8x32 Kernel
                                           // 8x12x4 DotProduct
#else
    class AlgoInt8x8x32MK4_4x4x16;  // Aarch64 nchw44 Int8x8x32 Kernel 4x4x16
    class AlgoInt8x8x32K4x4x16;     // Aarch64 Int8x8x32 Kernel 4x4x16
    class AlgoInt8x8x32K8x8x8;      // Aarch64 Int8x8x32 Kernel 8x8x8
#endif
    class AlgoInt8x8x16K8x8x8;       // Aarch64 Int8x8x16 Kernel 8x8x8
    class AlgoInt8x8x16K4x4x16;      // Aarch64 Int8x8x16 Kernel 4x4x16
    class AlgoInt8x8x16MK4_16x12x4;  // Aarch64 Int8x8x16 Kernel 16x12x16
    class AlgoInt8x8x16MK4_4x4x8;    // Aarch64 Int8x8x16 Kernel 4x4x8

    class AlgoInt16x16x32K12x8x1;  // Aarch64 Int16x16x32 Kernel 12x8x1
    class AlgoInt16x16x32MK8_8x8;  // Aarch64 Int16x16x32 Format MK8 block 8x8

#if __ARM_FEATURE_DOTPROD
    class AlgoQuint8K8x8x4DotProd;  // Aarch64 Quint8 Kernel
                                    // 8x8x4 DotProduct
    class AlgoQuint8GemvDotProd;    // Aarch64 Quint8 Gemv DotProduct
#else
    class AlgoQuint8K8x8x8;         // Aarch64 Quint8 Kernel 8x8x8
#endif
    class AlgoInt8x8x16MK4_K8x8x8;  // Aarch64 Int8x8x16 Kernel 4x4x16
    class AlgoInt4x4x16K8x8x8;      // Aarch64 Int4x4x16 Kernel 4x4x16
    class AlgoPack;
public:
    static const AlgoPack& algo_pack();
};

}  // namespace aarch64
}  // namespace megdnn

// vim: syntax=cpp.doxygen
