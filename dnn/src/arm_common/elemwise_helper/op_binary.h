/**
 * \file dnn/src/arm_common/elemwise_helper/op_binary.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include "src/arm_common/elemwise_helper/kimpl/add.h"
#include "src/arm_common/elemwise_helper/kimpl/mul.h"
#include "src/arm_common/elemwise_helper/kimpl/rmulh.h"
#include "src/arm_common/elemwise_helper/kimpl/fuse_add_relu.h"
#include "src/arm_common/elemwise_helper/kimpl/fuse_add_sigmoid.h"
#include "src/arm_common/elemwise_helper/kimpl/fuse_add_tanh.h"
#include "src/arm_common/elemwise_helper/kimpl/fuse_add_h_swish.h"
#include "src/arm_common/elemwise_helper/kimpl/max.h"
#include "src/arm_common/elemwise_helper/kimpl/min.h"
#include "src/arm_common/elemwise_helper/kimpl/pow.h"
#include "src/arm_common/elemwise_helper/kimpl/sub.h"
#include "src/arm_common/elemwise_helper/kimpl/true_div.h"

//////////////////// quantization //////////////////////////////
namespace megdnn {
namespace arm_common {
#define cb(op)                                                                \
    template <>                                                               \
    struct op<dt_qint8, dt_qint8>                                             \
            : BinaryQuantizationOp<dt_qint8, dt_qint8, op<float, float> > {   \
        using BinaryQuantizationOp<dt_qint8, dt_qint8,                        \
                                   op<float, float> >::BinaryQuantizationOp;  \
    };                                                                        \
    template <>                                                               \
    struct op<dt_quint8, dt_quint8>                                           \
            : BinaryQuantizationOp<dt_quint8, dt_quint8, op<float, float> > { \
        using BinaryQuantizationOp<dt_quint8, dt_quint8,                      \
                                   op<float, float> >::BinaryQuantizationOp;  \
    };

cb(TrueDivOp);
cb(FuseAddSigmoidOp);
cb(FuseAddTanhOp);
cb(FuseAddHSwishOp);

#undef cb
}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
