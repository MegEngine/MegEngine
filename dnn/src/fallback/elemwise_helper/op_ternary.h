/**
 * \file dnn/src/fallback/elemwise_helper/op_ternary.h
 */
#pragma once

#include "src/fallback/elemwise_helper/kimpl/fuse_mul_add3.h"

//////////////////// quantization //////////////////////////////
namespace megdnn {
namespace fallback {
#define cb(op)                                                                \
    template <>                                                               \
    struct op<dt_qint8, dt_qint8>                                             \
            : TernaryQuantizationOp<dt_qint8, dt_qint8, op<float, float>> {   \
        using TernaryQuantizationOp<                                          \
                dt_qint8, dt_qint8, op<float, float>>::TernaryQuantizationOp; \
    };

cb(FuseMulAdd3Op);
#undef cb
}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
