/**
 * \file dnn/src/cuda/cumsum/kern_helper.cuh
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
namespace cumsum {

void get_BX_BY(uint32_t A, uint32_t B, uint32_t C, uint32_t& BX, uint32_t& BY);

uint32_t get_workspace_bytes_for_cub_1d(uint32_t nr_item, uint32_t item_size);

}  // namespace cumsum
}  // namespace cuda
}  // namespace megdnn

// vim: ft=cpp syntax=cpp.doxygen
