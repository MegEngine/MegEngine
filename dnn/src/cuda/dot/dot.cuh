/**
 * \file dnn/src/cuda/dot/dot.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "megdnn/dtype.h"

namespace megdnn {
namespace cuda {
namespace dot {

template <typename T> void run(const T *a, const T *b, T *c, 
        float *workspace,
        uint32_t n,
        int32_t strideA, int32_t strideB,
        cudaStream_t stream);

} // namespace dot
} // namespace cuda
} // namespace megdnn

// vim: syntax=cpp.doxygen
