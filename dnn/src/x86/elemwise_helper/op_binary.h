#pragma once

#include "src/x86/elemwise_helper/kimpl/add.h"
#include "src/x86/elemwise_helper/kimpl/fuse_add_h_swish.h"
#include "src/x86/elemwise_helper/kimpl/fuse_add_relu.h"
#include "src/x86/elemwise_helper/kimpl/fuse_add_sigmoid.h"
#include "src/x86/elemwise_helper/kimpl/max.h"
#include "src/x86/elemwise_helper/kimpl/min.h"
#include "src/x86/elemwise_helper/kimpl/mul.h"
#include "src/x86/elemwise_helper/kimpl/sub.h"

//////////////////// quantization //////////////////////////////
namespace megdnn {
namespace x86 {
#define cb(op, simd_type)                                                             \
    template <>                                                                       \
    struct op<simd_type, dt_qint8, dt_qint8>                                          \
            : BinaryQuantizationOp<                                                   \
                      simd_type, dt_qint8, dt_qint8, op<simd_type, float, float>> {   \
        using BinaryQuantizationOp<                                                   \
                simd_type, dt_qint8, dt_qint8,                                        \
                op<simd_type, float, float>>::BinaryQuantizationOp;                   \
    };                                                                                \
    template <>                                                                       \
    struct op<simd_type, dt_quint8, dt_quint8>                                        \
            : BinaryQuantizationOp<                                                   \
                      simd_type, dt_quint8, dt_quint8, op<simd_type, float, float>> { \
        using BinaryQuantizationOp<                                                   \
                simd_type, dt_quint8, dt_quint8,                                      \
                op<simd_type, float, float>>::BinaryQuantizationOp;                   \
    };

cb(AddOp, SIMDType::SSE4_2);
cb(MaxOp, SIMDType::SSE4_2);
cb(MinOp, SIMDType::SSE4_2);
cb(SubOp, SIMDType::SSE4_2);
cb(MulOp, SIMDType::SSE4_2);
cb(FuseAddReluOp, SIMDType::SSE4_2);
cb(FuseAddSigmoidOp, SIMDType::SSE4_2);
cb(FuseAddHSwishOp, SIMDType::SSE4_2);

cb(AddOp, SIMDType::AVX2);
cb(MaxOp, SIMDType::AVX2);
cb(MinOp, SIMDType::AVX2);
cb(SubOp, SIMDType::AVX2);
cb(MulOp, SIMDType::AVX2);
cb(FuseAddReluOp, SIMDType::AVX2);
cb(FuseAddSigmoidOp, SIMDType::AVX2);
cb(FuseAddHSwishOp, SIMDType::AVX2);
#undef cb
#define cb(op, simd_type)                                                             \
    template <>                                                                       \
    struct op<simd_type, dt_qint32, dt_qint8>                                         \
            : BinaryQuantizationOp<                                                   \
                      simd_type, dt_qint32, dt_qint8, op<simd_type, float, float>> {  \
        using BinaryQuantizationOp<                                                   \
                simd_type, dt_qint32, dt_qint8,                                       \
                op<simd_type, float, float>>::BinaryQuantizationOp;                   \
    };                                                                                \
    template <>                                                                       \
    struct op<simd_type, dt_qint32, dt_quint8>                                        \
            : BinaryQuantizationOp<                                                   \
                      simd_type, dt_qint32, dt_quint8, op<simd_type, float, float>> { \
        using BinaryQuantizationOp<                                                   \
                simd_type, dt_qint32, dt_quint8,                                      \
                op<simd_type, float, float>>::BinaryQuantizationOp;                   \
    };

cb(AddOp, SIMDType::SSE4_2);
cb(FuseAddReluOp, SIMDType::SSE4_2);
cb(FuseAddHSwishOp, SIMDType::SSE4_2);

cb(AddOp, SIMDType::AVX2);
cb(FuseAddReluOp, SIMDType::AVX2);
cb(FuseAddHSwishOp, SIMDType::AVX2);

cb(AddOp, SIMDType::NONE);
cb(FuseAddReluOp, SIMDType::NONE);
cb(FuseAddHSwishOp, SIMDType::NONE);
#undef cb
}  // namespace x86
}  // namespace megdnn
   // vim: syntax=cpp.doxygen
