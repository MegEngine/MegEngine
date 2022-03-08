/**
 * \file dnn/src/fallback/elemwise_helper/op_unary.h
 */
#pragma once

#include "src/fallback/elemwise_helper/kimpl/abs.h"
#include "src/fallback/elemwise_helper/kimpl/exp.h"
#include "src/fallback/elemwise_helper/kimpl/fast_tanh.h"
#include "src/fallback/elemwise_helper/kimpl/hswish.h"
#include "src/fallback/elemwise_helper/kimpl/none.h"
#include "src/fallback/elemwise_helper/kimpl/relu.h"
#include "src/fallback/elemwise_helper/kimpl/sigmoid.h"
#include "src/fallback/elemwise_helper/kimpl/tanh.h"
#include "src/fallback/elemwise_helper/kimpl/typecvt.h"

//////////////////// quantization //////////////////////////////
namespace megdnn {
namespace fallback {
#define cb(op)                                                              \
    template <>                                                             \
    struct op<dt_qint8, dt_qint8>                                           \
            : UnaryQuantizationOp<dt_qint8, dt_qint8, op<float, float>> {   \
        using UnaryQuantizationOp<                                          \
                dt_qint8, dt_qint8, op<float, float>>::UnaryQuantizationOp; \
    };

cb(SigmoidOp);
cb(ExpOp);
cb(TanhOp);
cb(FastTanhOp);
cb(HSwishOp);
#undef cb
}  // namespace fallback
}  // namespace megdnn

// vim: syntax=cpp.doxygen
