/**
 * \file dnn/src/cuda/argmxx/argmxx.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/common/argmxx_helper.h"

#include "src/cuda/reduce_helper.cuh"
#include "megdnn/dtype.h"

namespace megdnn {
namespace cuda {

#define INST(_dt) \
    INST_REDUCE(argmxx::ArgmxxOp<DTypeTrait<_dt>::ctype MEGDNN_COMMA false>, false); \
    INST_REDUCE(argmxx::ArgmxxOp<DTypeTrait<_dt>::ctype MEGDNN_COMMA true>, false); \

    MEGDNN_FOREACH_COMPUTING_DTYPE(INST)

} // namespace argmxx
} // namespace megdnn
