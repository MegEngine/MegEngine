/**
 * \file dnn/src/common/reduce_helper.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "src/common/reduce_helper.h"

#include <algorithm>
#include <numeric>
#include "src/common/utils.h"

namespace megdnn {
namespace reduce {

void get_ABC(const TensorShape& shape, size_t& A, size_t& B, size_t& C,
             size_t axis) {
    auto shape_arr = shape.shape;
    auto ndim = shape.ndim;
    A = std::accumulate(shape_arr, shape_arr + axis, 1_z,
                        SafeMultiplies<size_t>());
    B = shape_arr[axis];
    C = std::accumulate(shape_arr + (axis + 1), shape_arr + ndim, 1_z,
                        SafeMultiplies<size_t>());
}

}  // namespace reduce
}  // namespace megdnn
