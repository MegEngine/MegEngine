#pragma once

#include "src/arm_common/elemwise_helper/kimpl/add.h"
#include "src/arm_common/elemwise_helper/kimpl/fuse_add_h_swish.h"
#include "src/arm_common/elemwise_helper/kimpl/fuse_add_relu.h"
#include "src/arm_common/elemwise_helper/kimpl/fuse_add_sigmoid.h"
#include "src/arm_common/elemwise_helper/kimpl/fuse_add_tanh.h"
#include "src/arm_common/elemwise_helper/kimpl/max.h"
#include "src/arm_common/elemwise_helper/kimpl/min.h"
#include "src/arm_common/elemwise_helper/kimpl/mul.h"
#include "src/arm_common/elemwise_helper/kimpl/rmulh.h"
#include "src/arm_common/elemwise_helper/kimpl/sub.h"
#include "src/arm_common/elemwise_helper/kimpl/true_div.h"

//////////////////// quantization //////////////////////////////
namespace megdnn {
namespace arm_common {
#define cb(op)                                                                 \
    template <>                                                                \
    struct op<dt_qint8, dt_qint8>                                              \
            : BinaryQuantizationOp<dt_qint8, dt_qint8, op<float, float>> {     \
        using BinaryQuantizationOp<                                            \
                dt_qint8, dt_qint8, op<float, float>>::BinaryQuantizationOp;   \
    };                                                                         \
    template <>                                                                \
    struct op<dt_quint8, dt_quint8>                                            \
            : BinaryQuantizationOp<dt_quint8, dt_quint8, op<float, float>> {   \
        using BinaryQuantizationOp<                                            \
                dt_quint8, dt_quint8, op<float, float>>::BinaryQuantizationOp; \
    };

cb(TrueDivOp);
cb(FuseAddSigmoidOp);
cb(FuseAddTanhOp);
cb(FuseAddHSwishOp);

#undef cb
}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
