/**
 * \file dnn/src/arm_common/elemwise_helper/op_ternary.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include "src/arm_common/elemwise_helper/kimpl/fuse_mul_add3.h"

//////////////////// quantization //////////////////////////////
namespace megdnn {
namespace arm_common {
#define cb(op)                                                                 \
    template <>                                                                \
    struct op<dt_qint8, dt_qint8>                                              \
            : TernaryQuantizationOp<dt_qint8, dt_qint8, op<float, float> > {   \
        using TernaryQuantizationOp<dt_qint8, dt_qint8,                        \
                                    op<float, float> >::TernaryQuantizationOp; \
    };                                                                         \
    template <>                                                                \
    struct op<dt_quint8, dt_quint8>                                            \
            : TernaryQuantizationOp<dt_quint8, dt_quint8, op<float, float> > { \
        using TernaryQuantizationOp<dt_quint8, dt_quint8,                      \
                                    op<float, float> >::TernaryQuantizationOp; \
    };

cb(FuseMulAdd3Op);
#undef cb
}  // namespace arm_common
}  // namespace megdnn

// vim: syntax=cpp.doxygen
