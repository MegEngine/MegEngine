/**
 * \file dnn/src/cuda/conv_bias/quint4x4x32_wmma/activation_u4.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *
 * This file has been modified by Megvii ("Megvii Modifications").
 * All Megvii Modifications are Copyright (C) 2014-2019 Megvii Inc. All rights reserved.
 */

/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "src/cuda/utils.h"
#include "src/cuda/query_blocksize.cuh"

namespace megdnn {
namespace cuda {
namespace activation_u4 {
/*
 * \note: The following code copied from TensorFlow. Used for calculating the
 * Cuda 3D launch config to ensure maximize occupancy we should use for a kernel
 * launch.
 */
void get_launch_config(const void* kern, int dimx, int dimy, int dimz,
                       dim3& blocks, dim3& grids) {
    auto config =
            query_launch_config_for_kernel(reinterpret_cast<const void*>(kern));
    int block_size = config.block_size;
    int grid_size = config.grid_size;
    auto&& device_prop = current_device_prop();
    int x_thread_limit = device_prop.maxThreadsDim[0];
    int y_thread_limit = device_prop.maxThreadsDim[1];
    int z_thread_limit = device_prop.maxThreadsDim[2];
    int x_grid_limit = device_prop.maxGridSize[0];
    int y_grid_limit = device_prop.maxGridSize[1];
    int z_grid_limit = device_prop.maxGridSize[2];
#define MIN3(a, b, c) std::min({(a), (b), (c)})
    uint32_t blkx = MIN3(dimx, block_size, x_thread_limit);
    uint32_t blky =
            MIN3(dimy, std::max(block_size / (int)(blkx), 1), y_thread_limit);
    uint32_t blkz =
            MIN3(dimz, std::max(block_size / ((int)blkx * (int)blky), 1),
                 z_thread_limit);
    uint32_t gridx = MIN3(grid_size, DIVUP((int)dimx, (int)blkx), x_grid_limit);
    uint32_t gridy = MIN3(DIVUP(grid_size, (int)gridx), DIVUP(dimy, (int)blky),
                          y_grid_limit);
    uint32_t gridz = MIN3(DIVUP(grid_size, (int)(gridx * gridy)),
                          DIVUP(dimz, (int)blkz), z_grid_limit);
#undef MIN3

    grids = dim3{gridx, gridy, gridz};
    blocks = dim3{blkx, blky, blkz};
}
}  // namespace activation_u4
}  // namespace cuda
}  // namespace megdnn

// vim: ft=cpp syntax=cuda.doxygen
