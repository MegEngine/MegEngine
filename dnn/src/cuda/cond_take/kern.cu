/**
 * \file dnn/src/cuda/cond_take/kern.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "./kern.cuh"
#include "src/cuda/cumsum/kern_impl.cuinl"
#include "src/cuda/query_blocksize.cuh"
#include "src/common/cond_take/predicate.cuh"
#include <limits>

using namespace megdnn;
using namespace megdnn::cond_take;
using namespace megdnn::cuda::cond_take;

size_t cuda::cond_take::gen_idx_get_workspace_size(size_t size) {
    megdnn_assert(size < std::numeric_limits<uint32_t>::max());
    return cumsum::get_workspace_in_bytes(1, size, 1, sizeof(IdxType));
}

// vim: ft=cuda syntax=cuda.doxygen
