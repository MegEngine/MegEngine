/**
 * \file dnn/src/cuda/pooling/pooling2d_int8_cdiv4hwn4.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "./pooling2d_int8_cdiv4hwn4.cuh"
#include "src/cuda/query_blocksize.cuh"

namespace megdnn {
namespace cuda {
namespace pooling2d {

uint32_t _get_kern_block_size(const void* kern) {
    uint32_t ret = query_blocksize_for_kernel(kern);
    return ret;
}

}  // namespace pooling2d
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
