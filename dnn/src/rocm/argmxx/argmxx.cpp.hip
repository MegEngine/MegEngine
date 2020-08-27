/**
 * \file src/rocm/argmxx/argmxx.cpp.hip
 *
 * This file is part of MegDNN, a deep neural network run-time library
 * developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2019 Megvii Inc. All rights reserved.
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
