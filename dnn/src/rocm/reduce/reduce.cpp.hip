/**
 * \file src/rocm/reduce/reduce.cpp.hip
 *
 * This file is part of MegDNN, a deep neural network run-time library
 * developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2019 Megvii Inc. All rights reserved.
 */
#include "hcc_detail/hcc_defs_prologue.h"
#include "hip_header.h"
#include "src/common/reduce_helper.h"

#include "megdnn/dtype.h"
#include "src/rocm/reduce_helper.h.hip"

namespace megdnn {
namespace rocm {

using namespace reduce;

#define COMMOA ,

#define INST(sctype, dctype, wtype)                                  \
    INST_REDUCE(SumOp<sctype COMMOA dctype COMMOA wtype>, false);    \
    INST_REDUCE(SumSqrOp<sctype COMMOA dctype COMMOA wtype>, false); \
    INST_REDUCE(ProdOp<sctype COMMOA dctype COMMOA wtype>, false);   \
    INST_REDUCE(MinOp<sctype COMMOA dctype COMMOA wtype>, false);    \
    INST_REDUCE(MaxOp<sctype COMMOA dctype COMMOA wtype>, false);    \
    INST_REDUCE(MeanOp<sctype COMMOA dctype COMMOA wtype>, false);

#define cb(_dt) \
    INST(DTypeTrait<_dt>::ctype, DTypeTrait<_dt>::ctype, DTypeTrait<_dt>::ctype)

MEGDNN_FOREACH_COMPUTING_DTYPE(cb)

INST(dt_float16, dt_float16, float)
INST(dt_float16, float, float)
INST(float, dt_float16, float)
INST(int, float, float)
#undef cb
#undef INST

}  // namespace rocm
}  // namespace megdnn


// vim: syntax=cpp.doxygen
