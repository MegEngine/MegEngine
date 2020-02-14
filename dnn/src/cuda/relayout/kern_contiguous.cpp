/**
 * \file dnn/src/cuda/relayout/kern_contiguous.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/common/utils.h"
#include "src/cuda/query_blocksize.cuh"
#include "src/cuda/relayout/kern_contiguous.cuh"

namespace megdnn {
namespace cuda {


void get_last_contiguous_launch_spec(const void *kern, size_t size,
                                     size_t contiguous_size, int *grid_size,
                                     int *block_size) {
    safe_size_in_kern(size);
    LaunchConfig config = query_launch_config_for_kernel(kern);
    *block_size = config.block_size;

    int a = size / (config.block_size * (contiguous_size - 1)),
        b = (size - 1) / (config.block_size * contiguous_size) + 1;
    *grid_size = std::max(a, b);

    if (!*grid_size) {
        *block_size = std::min<int>(std::max<int>(size / 64, 1) * 32, 1024);
        *grid_size = std::max<int>(size / *block_size, 1);
    }

    // because we unroll contiguous_size times in the kernel
    megdnn_assert(static_cast<size_t>(*block_size) * *grid_size *
                      contiguous_size >=
                  size);
}

}  // cuda
}  // megdnn

// vim: ft=cpp syntax=cpp.doxygen
