#pragma once

#include "src/x86/elemwise_helper/kimpl/fuse_mul_add3.h"
//////////////////// quantization //////////////////////////////
namespace megdnn {
namespace x86 {
#define cb(op, simd_type)                                                             \
    template <>                                                                       \
    struct op<simd_type, dt_qint8, dt_qint8>                                          \
            : TernaryQuantizationOp<                                                  \
                      simd_type, dt_qint8, dt_qint8, op<simd_type, float, float>> {   \
        using TernaryQuantizationOp<                                                  \
                simd_type, dt_qint8, dt_qint8,                                        \
                op<simd_type, float, float>>::TernaryQuantizationOp;                  \
    };                                                                                \
    template <>                                                                       \
    struct op<simd_type, dt_quint8, dt_quint8>                                        \
            : TernaryQuantizationOp<                                                  \
                      simd_type, dt_quint8, dt_quint8, op<simd_type, float, float>> { \
        using TernaryQuantizationOp<                                                  \
                simd_type, dt_quint8, dt_quint8,                                      \
                op<simd_type, float, float>>::TernaryQuantizationOp;                  \
    };

cb(FuseMulAdd3Op, SIMDType::SSE4_2);
cb(FuseMulAdd3Op, SIMDType::AVX2);
#undef cb
}  // namespace x86
}  // namespace megdnn
   // vim: syntax=cpp.doxygen
