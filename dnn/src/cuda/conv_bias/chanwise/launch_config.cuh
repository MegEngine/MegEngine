/**
 * \file dnn/src/cuda/conv_bias/chanwise/launch_config.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/common/utils.cuh"

namespace megdnn {
namespace cuda {
namespace conv_bias {
namespace chanwise {

int GetFixedBlockSize1(
        int work_element_count, const void* func, int dynamic_shared_memory_size,
        int fixed_block_size);

template <typename DeviceFunc>
int GetFixedBlockSize(
        int work_element_count, DeviceFunc func, int dynamic_shared_memory_size,
        int fixed_block_size) {
    return GetFixedBlockSize1(
            work_element_count, reinterpret_cast<const void*>(func),
            dynamic_shared_memory_size, fixed_block_size);
}

}  // namespace chanwise
}  // namespace conv_bias
}  // namespace cuda
}  // namespace megdnn

// vim: ft=cpp syntax=cuda.doxygen
