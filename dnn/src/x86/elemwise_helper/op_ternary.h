/**
 * \file dnn/src/x86/elemwise_helper/op_ternary.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include "src/x86/elemwise_helper/kimpl/fuse_mul_add3.h"
//////////////////// quantization //////////////////////////////
namespace megdnn {
namespace x86 {
#define cb(op, simd_type)                                             \
    template <>                                                       \
    struct op<simd_type, dt_qint8, dt_qint8>                          \
            : TernaryQuantizationOp<simd_type, dt_qint8, dt_qint8,    \
                                    op<simd_type, float, float> > {   \
        using TernaryQuantizationOp<                                  \
                simd_type, dt_qint8, dt_qint8,                        \
                op<simd_type, float, float> >::TernaryQuantizationOp; \
    };                                                                \
    template <>                                                       \
    struct op<simd_type, dt_quint8, dt_quint8>                        \
            : TernaryQuantizationOp<simd_type, dt_quint8, dt_quint8,  \
                                    op<simd_type, float, float> > {   \
        using TernaryQuantizationOp<                                  \
                simd_type, dt_quint8, dt_quint8,                      \
                op<simd_type, float, float> >::TernaryQuantizationOp; \
    };

cb(FuseMulAdd3Op, SIMDType::SSE4_2);
cb(FuseMulAdd3Op, SIMDType::AVX2);
#undef cb
}  // namespace x86
}  // namespace megdnn
   // vim: syntax=cpp.doxygen
