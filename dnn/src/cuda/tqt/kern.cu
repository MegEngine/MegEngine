/**
 * \file dnn/src/cuda/tqt/kern.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "./kern.cuh"

namespace megdnn {
namespace cuda {

#define cb(_dtype)                                                      \
    INST_RUN_ELEMWISE(TQTKernOp<DTypeTrait<_dtype>::ctype>,             \
                      DTypeTrait<_dtype>::ctype, 1);                    \
    INST_RUN_ELEMWISE(TQTBwdKernOp<DTypeTrait<_dtype>::ctype>,          \
                      DTypeTrait<_dtype>::ctype, 1);                    \
    INST_RUN_ELEMWISE(TQTKernOpNonContig<DTypeTrait<_dtype>::ctype>,    \
                      DTypeTrait<_dtype>::ctype, 3);                    \
    INST_RUN_ELEMWISE(TQTBwdKernOpNonContig<DTypeTrait<_dtype>::ctype>, \
                      DTypeTrait<_dtype>::ctype, 5);
cb(megdnn::dtype::Float32)

}  // namespace cuda
}  // namespace megdnn
