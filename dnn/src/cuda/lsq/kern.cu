/**
 * \file dnn/src/cuda/lsq/kern.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "./kern.cuh"

namespace megdnn {
namespace cuda {

#define cb(_dtype)                                                                    \
    INST_RUN_ELEMWISE(                                                                \
            LSQKernOp<DTypeTrait<_dtype>::ctype>, DTypeTrait<_dtype>::ctype, 3);      \
    INST_RUN_ELEMWISE(                                                                \
            LSQBwdKernOp<DTypeTrait<_dtype>::ctype>, DTypeTrait<_dtype>::ctype, 3);   \
    INST_RUN_ELEMWISE(                                                                \
            LSQKernOpNonContig<DTypeTrait<_dtype>::ctype>, DTypeTrait<_dtype>::ctype, \
            5);                                                                       \
    INST_RUN_ELEMWISE(                                                                \
            LSQBwdKernOpNonContig<DTypeTrait<_dtype>::ctype>,                         \
            DTypeTrait<_dtype>::ctype, 7);
cb(megdnn::dtype::Float32)

}  // namespace cuda
}  // namespace megdnn
