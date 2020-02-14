/**
 * \file dnn/src/cuda/flip/flip.cuh
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
namespace flip {

template <typename T, bool vertical, bool horizontal>
void flip(const T *src, T *dst, size_t N, size_t H, size_t W, size_t IC,
          size_t stride1, size_t stride2, size_t stride3, cudaStream_t stream);

}  // namespace flip
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
