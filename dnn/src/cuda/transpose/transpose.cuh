/**
 * \file dnn/src/cuda/transpose/transpose.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include <cstddef>
#include <cuda_runtime_api.h>

namespace megdnn {
namespace cuda {

// (m, n) to (n, m)
template <typename T>
void transpose(const T *A, T *B, size_t m, size_t n,
        size_t LDA, size_t LDB, cudaStream_t stream);

} // namespace cuda
} // namespace megdnn

// vim: syntax=cpp.doxygen
