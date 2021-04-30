/**
 * \file dnn/src/rocm/argmxx/argmxx.cpp.hip
 *
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "hcc_detail/hcc_defs_prologue.h"
#include "hip_header.h"
#include "src/common/argmxx_helper.h"

#include "src/rocm/reduce_helper.h.hip"
#include "megdnn/dtype.h"

namespace megdnn {
namespace rocm {

#define INST(_dt) \
    INST_REDUCE(argmxx::ArgmxxOp<DTypeTrait<_dt>::ctype MEGDNN_COMMA false>, false); \
    INST_REDUCE(argmxx::ArgmxxOp<DTypeTrait<_dt>::ctype MEGDNN_COMMA true>, false); \

    MEGDNN_FOREACH_COMPUTING_DTYPE(INST)

} // namespace rocm
} // namespace megdnn
