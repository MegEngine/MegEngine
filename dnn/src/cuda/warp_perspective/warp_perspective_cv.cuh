#pragma once

#include <cstdio>
#include "src/cuda/utils.cuh"
#include "src/cuda/warp_perspective/common.h"

namespace megdnn {
namespace cuda {
namespace warp_perspective {

template <typename T, size_t CH>
void warp_perspective_cv_proxy(
        const T* src, T* dst, const size_t src_rows, const size_t src_cols,
        const size_t dst_rows, const size_t dst_cols, const size_t src_step,
        const size_t dst_step, BorderMode bmode, InterpolationMode imode,
        const float* trans, const T border_val, double* workspace, cudaStream_t stream);

}  // namespace warp_perspective
}  // namespace cuda
}  // namespace megdnn

// vim: syntax=cpp.doxygen
