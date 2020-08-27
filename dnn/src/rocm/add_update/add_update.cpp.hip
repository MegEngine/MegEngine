/**
 * \file src/rocm/add_update/add_update.cpp.hip
 *
 * This file is part of MegDNN, a deep neural network run-time library
 * developed by Megvii.
 *
 * \copyright Copyright (c) 2014-2019 Megvii Inc. All rights reserved.
 */
#include "hcc_detail/hcc_defs_prologue.h"
#include "./add_update.h.hip"

namespace megdnn {
namespace rocm {

#define cb(_dtype)                                                         \
    INST_RUN_ELEMWISE(AddUpdateKernOp<DTypeTrait<_dtype>::ctype>,          \
                      DTypeTrait<_dtype>::ctype, 1);                       \
    INST_RUN_ELEMWISE(AddUpdateKernOpNonContig<DTypeTrait<_dtype>::ctype>, \
                      DTypeTrait<_dtype>::ctype, 2);

MEGDNN_FOREACH_COMPUTING_DTYPE(cb)

}  // namespace rocm
}  // namespace megdnn


// vim: ft=cpp syntax=cpp.doxygen

