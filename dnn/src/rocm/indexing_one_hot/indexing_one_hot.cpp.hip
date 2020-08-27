/**
 * \file src/rocm/indexing_one_hot/indexing_one_hot.cpp.hip
 *
 * This file is part of MegDNN, a deep neural network run-time library
 * developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2016 Megvii Inc. All rights reserved.
 */
#include "hcc_detail/hcc_defs_prologue.h"
#include "./indexing_one_hot.h.hip"
#include "src/rocm/elemwise_helper.h.hip"

namespace megdnn {
namespace rocm {

#define cb(_dt) \
    typedef indexing_one_hot::OpGet<DTypeTrait<dtype::_dt>::ctype, dt_int32> \
            OpGet##_dt; \
    typedef indexing_one_hot::OpSet<DTypeTrait<dtype::_dt>::ctype, dt_int32> \
            OpSet##_dt; \
    INST_RUN_ELEMWISE(OpGet##_dt, void, 0); \
    INST_RUN_ELEMWISE(OpSet##_dt, void, 0);

    MEGDNN_FOREACH_DTYPE_NAME(cb)
    MEGDNN_FOREACH_PARAMETERIZED_DTYPE(cb)

#undef cb

} // namespace rocm
} // namespace megdnn


// vim: ft=cpp syntax=cpp.doxygen

