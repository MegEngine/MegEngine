/**
 * \file dnn/src/common/resize.cuh
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megdnn/arch.h"

#if MEGDNN_CC_HOST && !defined(__host__)
#if __GNUC__ || __has_attribute(always_inline)
#define __forceinline__ inline __attribute__((always_inline))
#else
#define __forceinline__ inline
#endif
#endif

namespace megdnn {
namespace resize {

MEGDNN_HOST MEGDNN_DEVICE __forceinline__ void interpolate_cubic(
        float x, float* coeffs) {
    const float A = -0.75f;

    coeffs[0] = ((A * (x + 1) - 5 * A) * (x + 1) + 8 * A) * (x + 1) - 4 * A;
    coeffs[1] = ((A + 2) * x - (A + 3)) * x * x + 1;
    coeffs[2] = ((A + 2) * (1 - x) - (A + 3)) * (1 - x) * (1 - x) + 1;
    coeffs[3] = 1.f - coeffs[0] - coeffs[1] - coeffs[2];
}
}  // namespace resize
}  // namespace megdnn

/* vim: set ft=cpp: */
