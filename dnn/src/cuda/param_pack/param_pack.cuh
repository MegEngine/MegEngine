/**
 * \file dnn/src/cuda/param_pack/param_pack.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once
#include <cuda_runtime.h>

#include <stdint.h>
#include <stdio.h>

namespace megdnn {
namespace cuda {
namespace param_pack {

template <typename T>
void concat_proxy(const T** srcs, T* dst, size_t srcs_size, size_t total_size,
                  const int32_t* offsets, cudaStream_t stream);

}  // namespace param_pack
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
