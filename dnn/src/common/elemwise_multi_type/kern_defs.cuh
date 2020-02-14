/**
 * \file dnn/src/common/elemwise_multi_type/kern_defs.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2020 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megdnn/dtype.h"
#include "src/common/utils.cuh"
#include "src/common/elemwise_helper.cuh"

#include <cmath>

namespace megdnn {
namespace elemwise_multi_type {

template <typename stype, typename dtype>
struct Fma3iXxf32xf32xiYOp {
    MEGDNN_HOST MEGDNN_DEVICE dtype operator()(stype x, float k, float b) {
        const float MIN = static_cast<float>(DTypeTrait<dtype>::min());
        const float MAX = static_cast<float>(DTypeTrait<dtype>::max());
        float fv = rint(k * static_cast<float>(x) + b);
        return static_cast<dtype>(fv >= MIN ? (fv <= MAX ? fv : MAX) : MIN);
    }
};

template <typename stype, typename dtype> 
MEGDNN_HOST MEGDNN_DEVICE dtype round_shr_saturate(stype x, int k) {
    stype result = rounding_shift_right_away_from_zero(x, k);
    if (!is_same<stype, dtype>::value) {
        result = std::min<stype>(result, std::numeric_limits<dtype>::max());
        result = std::max<stype>(result, std::numeric_limits<dtype>::min());
    }
    return static_cast<dtype>(result);
}

}  // namespace elemwise_multi_type
}  // namespace megdnn

// vim: ft=cpp syntax=cpp.doxygen
