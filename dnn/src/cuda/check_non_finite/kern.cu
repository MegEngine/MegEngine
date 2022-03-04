/**
 * \file dnn/src/cuda/check_non_finite/kern.cu
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/common/reduce_helper_device.h"

#include "megdnn/dtype.h"
#include "src/cuda/reduce_helper.cuh"

namespace megdnn {
namespace cuda {

#define COMMA ,

#define cb(_dtype)                                                      \
    INST_REDUCE(                                                        \
            device_reduce::CheckNonFiniteOp<                            \
                    _dtype COMMA size_t COMMA dt_int32 COMMA dt_int32>, \
            false);

cb(dt_float32);
cb(dt_float16);
#undef cb
#undef COMMA
}  // namespace cuda
}  // namespace megdnn

// vim: ft=cpp syntax=cpp.doxygen
