#pragma once

#include "megdnn/arch.h"

namespace megdnn {
namespace resize {

MEGDNN_HOST MEGDNN_DEVICE MEGDNN_FORCE_INLINE void interpolate_cubic(
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
