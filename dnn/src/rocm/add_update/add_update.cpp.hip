/**
 * \file dnn/src/rocm/add_update/add_update.cpp.hip
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

