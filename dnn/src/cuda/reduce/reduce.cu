/**
 * \file dnn/src/cuda/reduce/reduce.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/common/reduce_helper_device.h"

#include "megdnn/dtype.h"
#include "src/cuda/reduce_helper.cuh"

namespace megdnn {
namespace cuda {

using namespace device_reduce;

#define COMMA ,

#define INST(sctype, dctype, wtype)                                \
    INST_REDUCE(SumOp<sctype COMMA dctype COMMA wtype>, false);    \
    INST_REDUCE(SumSqrOp<sctype COMMA dctype COMMA wtype>, false); \
    INST_REDUCE(ProdOp<sctype COMMA dctype COMMA wtype>, false);   \
    INST_REDUCE(MinOp<sctype COMMA dctype COMMA wtype>, false);    \
    INST_REDUCE(MaxOp<sctype COMMA dctype COMMA wtype>, false);    \
    INST_REDUCE(MeanOp<sctype COMMA dctype COMMA wtype>, false);

#define cb(_dt) \
    INST(DTypeTrait<_dt>::ctype, DTypeTrait<_dt>::ctype, DTypeTrait<_dt>::ctype)

MEGDNN_FOREACH_COMPUTING_DTYPE(cb)

INST(dt_float16, dt_float16, float)
INST(dt_float16, float, float)
INST(float, dt_float16, float)
INST(int, float, float)

#undef cb
#undef INST
#undef COMMA

}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
