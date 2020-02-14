/**
 * \file dnn/src/cuda/argsort/backward.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include <cuda_runtime_api.h>
#include <stdint.h>

namespace megdnn {
namespace cuda {
namespace argsort {

template <typename T>
void backward_proxy(uint32_t dst_h, uint32_t dst_w, uint32_t src_w, T* dst,
                    const T* src_data, const int* src_idx, cudaStream_t stream);

}  // namespace argsort
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen

