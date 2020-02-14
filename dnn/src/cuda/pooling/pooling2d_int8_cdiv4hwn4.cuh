/**
 * \file dnn/src/cuda/pooling/pooling2d_int8_cdiv4hwn4.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

#include "src/cuda/utils.cuh"

namespace megdnn {
namespace cuda {
namespace pooling2d {

struct Param {
    int n, c, hi, wi, ho, wo, ph, pw, window_h, window_w, sh, sw;
};

uint32_t _get_kern_block_size(const void* kern);

void _do_pooling2d_int8_cdiv4hwn4(const int8_t* d_src, int8_t* d_dst,
                                  const Param& param, cudaStream_t stream,
                                  uint32_t mode);

}  // namespace pooling2d
}  // namespace cuda
}  // namespace megdnn
   

// vim: syntax=cuda.doxygen
