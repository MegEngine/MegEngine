/**
 * \file dnn/src/cuda/resize/common.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once

namespace megdnn {
namespace cuda {
namespace resize {

__device__ inline void get_origin_coord(float scale, int size, int idx,
                                        float& alpha, int& origin_idx) {
    alpha = (idx + 0.5f) / scale - 0.5f;
    origin_idx = static_cast<int>(floor(alpha));
    alpha -= origin_idx;
    if (origin_idx < 0) {
        origin_idx = 0;
        alpha = 0;
    } else if (origin_idx + 1 >= size) {
        origin_idx = size - 2;
        alpha = 1;
    }
}

}  // namespace resize
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
