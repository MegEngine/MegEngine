#pragma once

#include "src/arm_common/elemwise_helper/kimpl/abs.h"
#include "src/arm_common/elemwise_helper/kimpl/exp.h"
#include "src/arm_common/elemwise_helper/kimpl/fast_tanh.h"
#include "src/arm_common/elemwise_helper/kimpl/hswish.h"
#include "src/arm_common/elemwise_helper/kimpl/none.h"
#include "src/arm_common/elemwise_helper/kimpl/relu.h"
#include "src/arm_common/elemwise_helper/kimpl/sigmoid.h"
#include "src/arm_common/elemwise_helper/kimpl/tanh.h"
#include "src/arm_common/elemwise_helper/kimpl/typecvt.h"

//////////////////// quantization //////////////////////////////
namespace megdnn {
namespace arm_common {
#define cb(op)                                                                \
    template <>                                                               \
    struct op<dt_qint8, dt_qint8>                                             \
            : UnaryQuantizationOp<dt_qint8, dt_qint8, op<float, float>> {     \
        using UnaryQuantizationOp<                                            \
                dt_qint8, dt_qint8, op<float, float>>::UnaryQuantizationOp;   \
    };                                                                        \
    template <>                                                               \
    struct op<dt_quint8, dt_quint8>                                           \
            : UnaryQuantizationOp<dt_quint8, dt_quint8, op<float, float>> {   \
        using UnaryQuantizationOp<                                            \
                dt_quint8, dt_quint8, op<float, float>>::UnaryQuantizationOp; \
    };

cb(SigmoidOp);
cb(ExpOp);
cb(TanhOp);
cb(FastTanhOp);
cb(HSwishOp);
#undef cb
}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
