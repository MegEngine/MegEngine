/**
 * \file dnn/src/armv7/matrix_mul/opr_impl.h
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
namespace armv7 {

class MatrixMulImpl : public arm_common::MatrixMulImpl {
public:
    using arm_common::MatrixMulImpl::MatrixMulImpl;
    class AlgoBase : public arm_common::MatrixMulImpl::AlgoBase {
    public:
        AlgoBase() : arm_common::MatrixMulImpl::AlgoBase() {
            m_handle_type = Handle::HandleType::ARMV7;
        }
    };

    SmallVector<fallback::MatrixMulImpl::AlgoBase*> get_all_packed_algo()
            override;

    MEGDNN_FB_DECL_GET_ALGO_FROM_DESC(MatrixMulImpl);

private:
    class AlgoF32;                  // Armv7 F32
    class AlgoF32MK4Pack4x12;       // Armv7 F32 Kernel 4x12 with pack
    class AlgoF32MK4_4x8;           // Armv7 F32 Kernel 4x8 nopack
    class AlgoF32Gemv;              // Armv7 F32 Gemv
    class AlgoInt8x8x32K4x8x8;      // Armv7 Int8x8x32 Kernel 4x8x8
    class AlgoInt8x8x32K4x2x16;     // Armv7 Int8x8x32 Kernel 4x2x16
    class AlgoInt8x8x32MK4_4x2x16;  // Armv7 Int8x8x32 Kernel MK4 4x2x16
    class AlgoQuint8K4x8x8;         // Armv7 Quint8 Kernel 4x8x8
    class AlgoInt8x8x16K4x2x16;     // Armv7 Int8x8x16 Kernel 4x2x16
    class AlgoInt8x8x16K4x8x8;      // Armv7 Int8x8x16 Kernel 4x8x8
    class AlgoInt8x8x16MK4_8x8x4;   // Armv7 Int8x8x16 Kernel 8x8x8
    class AlgoInt16x16x32K12x4x1;   // Armv7 Int16x16x32 Kernel 12x4x1
    class AlgoInt16x16x32MK8_4x8;   // Armv7 Int16x16x32 MK8 Format block 4x8
#if __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
    class AlgoF16K4x16x1;  // Armv7 F16 Kernel 4x16x1
    class AlgoF16MK8_4x8;  // Armv7 F16 MK8 Format block 4x8
#endif
#if __ARM_FEATURE_DOTPROD
    class AlgoInt8x8x32K6x8x4;            // Armv7 Int8 Kernel 6x8x4
    class AlgoQuint8DotK4x8x4;            // Armv7 Quint8 Kernel 6x8x4
    class AlgoInt8x8x32MK4_8x4x4DotProd;  // Armv7 nchw44 Int8x8x32 Kernel 8x4x4
                                          // DotProduct
#endif
    class AlgoPack;

public:
    static const AlgoPack& algo_pack();
};

}  // namespace armv7
}  // namespace megdnn

// vim: syntax=cpp.doxygen
