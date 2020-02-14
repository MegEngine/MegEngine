/**
 * \file dnn/src/cuda/relayout/kern.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megdnn/basic_types.h"
#include "src/cuda/elemwise_helper.cuh"
#include "src/cuda/utils.cuh"
#include "src/cuda/relayout/kern.cuh"

namespace megdnn {
namespace cuda {

void get_launch_spec_unroll16(const void* kern, size_t size, int* grid_size, int* block_size) {
    safe_size_in_kern(size);
    auto config = query_launch_config_for_kernel(kern);
    *block_size = config.block_size;
    *grid_size = (size - 1) / (config.block_size * 16) + 1;
    if (!*grid_size) {
        *block_size = std::min<int>(std::max<int>(size / 64, 1) * 32, 1024);
        *grid_size = std::max<int>(size / *block_size, 1);
    }
    megdnn_assert(static_cast<size_t>(*block_size) * *grid_size * 16 >= size);
}

void get_launch_spec_unroll4(const void* kern, size_t size, int* grid_size, int* block_size) {
    safe_size_in_kern(size);
    auto config = query_launch_config_for_kernel(kern);
    *block_size = config.block_size;
    *grid_size = (size - 1) / (config.block_size * 4) + 1;
    if (!*grid_size) {
        *block_size = std::min<int>(std::max<int>(size / 64, 1) * 32, 1024);
        *grid_size = std::max<int>(size / *block_size, 1);
    }
    megdnn_assert(static_cast<size_t>(*block_size) * *grid_size * 4 >= size);
}

} // namespace cuda
} // namespace megdnn

// vim: ft=cpp syntax=cpp.doxygen
