/**
 * \file dnn/src/cuda/local_share/helper.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "./helper.cuh"
#include "src/cuda/query_blocksize.cuh"
#include "src/cuda/utils.h"

namespace megdnn {
namespace cuda {
namespace local_share {

void _check_launch_config(const local_share::LaunchConfig& launch_config) {
    auto&& device_prop = current_device_prop();
    int x_thread_limit = device_prop.maxThreadsDim[0];
    int y_thread_limit = device_prop.maxThreadsDim[1];
    int z_thread_limit = device_prop.maxThreadsDim[2];
    int x_grid_limit = device_prop.maxGridSize[0];
    int y_grid_limit = device_prop.maxGridSize[1];
    int z_grid_limit = device_prop.maxGridSize[2];
    int sh_mem_size_limit = device_prop.sharedMemPerBlock;
    MEGDNN_MARK_USED_VAR(x_thread_limit);
    MEGDNN_MARK_USED_VAR(y_thread_limit);
    MEGDNN_MARK_USED_VAR(z_thread_limit);
    MEGDNN_MARK_USED_VAR(x_grid_limit);
    MEGDNN_MARK_USED_VAR(y_grid_limit);
    MEGDNN_MARK_USED_VAR(z_grid_limit);
    MEGDNN_MARK_USED_VAR(sh_mem_size_limit);
    megdnn_assert(launch_config.nr_threads_x <= x_thread_limit);
    megdnn_assert(launch_config.nr_threads_y <= y_thread_limit);
    megdnn_assert(launch_config.nr_threads_z <= z_thread_limit);
    megdnn_assert(launch_config.nr_blocks_x <= x_grid_limit);
    megdnn_assert(launch_config.nr_blocks_y <= y_grid_limit);
    megdnn_assert(launch_config.nr_blocks_z <= z_grid_limit);
    megdnn_assert(launch_config.smem_size_in_bytes <= sh_mem_size_limit);
}

uint32_t _get_kern_block_size(const void* kern) {
    uint32_t ret = query_blocksize_for_kernel(kern);
    return ret;
}

}  // namespace local_share
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
