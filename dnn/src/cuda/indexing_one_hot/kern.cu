/**
 * \file dnn/src/cuda/indexing_one_hot/kern.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./kern.cuh"
#include "src/cuda/utils.cuh"
#include "src/cuda/elemwise_helper.cuh"

namespace megdnn {
namespace cuda {

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

} // namespace cuda
} // namespace megdnn

// vim: ft=cpp syntax=cpp.doxygen

