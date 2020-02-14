/**
 * \file dnn/src/x86/elemwise_helper/op_unary.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include "src/x86/elemwise_helper/kimpl/abs.h"
#include "src/x86/elemwise_helper/kimpl/exp.h"
#include "src/x86/elemwise_helper/kimpl/fast_tanh.h"
#include "src/x86/elemwise_helper/kimpl/hswish.h"
#include "src/x86/elemwise_helper/kimpl/relu.h"
#include "src/x86/elemwise_helper/kimpl/sigmoid.h"
#include "src/x86/elemwise_helper/kimpl/hswish.h"
#include "src/x86/elemwise_helper/kimpl/typecvt.h"
#include "src/x86/elemwise_helper/kimpl/none.h"

//////////////////// quantization //////////////////////////////
namespace megdnn {
namespace x86 {
#define cb(op, simd_type)                                           \
    template <>                                                     \
    struct op<simd_type, dt_qint8, dt_qint8>                        \
            : UnaryQuantizationOp<simd_type, dt_qint8, dt_qint8,    \
                                  op<simd_type, float, float> > {   \
        using UnaryQuantizationOp<                                  \
                simd_type, dt_qint8, dt_qint8,                      \
                op<simd_type, float, float> >::UnaryQuantizationOp; \
    };                                                              \
    template <>                                                     \
    struct op<simd_type, dt_quint8, dt_quint8>                      \
            : UnaryQuantizationOp<simd_type, dt_quint8, dt_quint8,  \
                                  op<simd_type, float, float> > {   \
        using UnaryQuantizationOp<                                  \
                simd_type, dt_quint8, dt_quint8,                    \
                op<simd_type, float, float> >::UnaryQuantizationOp; \
    };

cb(SigmoidOp, SIMDType::SSE4_2);
cb(FastTanhOp, SIMDType::SSE4_2);
cb(HSwishOp, SIMDType::SSE4_2);
cb(AbsOp, SIMDType::SSE4_2);
cb(ReluOp, SIMDType::SSE4_2);
cb(ExpOp, SIMDType::SSE4_2);

cb(SigmoidOp, SIMDType::AVX2);
cb(AbsOp, SIMDType::AVX2);
cb(FastTanhOp, SIMDType::AVX2);
cb(HSwishOp, SIMDType::AVX2);
cb(ReluOp, SIMDType::AVX2);
cb(ExpOp, SIMDType::AVX2);
#undef cb
#define cb(op, simd_type)                                           \
    template <>                                                     \
    struct op<simd_type, dt_qint32, dt_qint8>                       \
            : UnaryQuantizationOp<simd_type, dt_qint32, dt_qint8,   \
                                  op<simd_type, float, float> > {   \
        using UnaryQuantizationOp<                                  \
                simd_type, dt_qint32, dt_qint8,                     \
                op<simd_type, float, float> >::UnaryQuantizationOp; \
    };                                                              \
    template <>                                                     \
    struct op<simd_type, dt_qint32, dt_quint8>                      \
            : UnaryQuantizationOp<simd_type, dt_qint32, dt_quint8,  \
                                  op<simd_type, float, float> > {   \
        using UnaryQuantizationOp<                                  \
                simd_type, dt_qint32, dt_quint8,                    \
                op<simd_type, float, float> >::UnaryQuantizationOp; \
    };

cb(HSwishOp, SIMDType::SSE4_2);
cb(ReluOp, SIMDType::SSE4_2);
cb(HSwishOp, SIMDType::AVX2);
cb(ReluOp, SIMDType::AVX2);

cb(ReluOp, SIMDType::NONE);
cb(HSwishOp, SIMDType::NONE);
cb(TypeCvtOp, SIMDType::NONE);
#undef cb
}  // namespace x86
}  // namespace megdnn
   // vim: syntax=cpp.doxygen
